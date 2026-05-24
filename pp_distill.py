#!/usr/bin/env python3

""" PP_DISTILL - distill calibrated image databases into one database
                 of select moving or fixed sources
    v1.0: 2016-01-24, mommermiscience@gmail.com
"""
# Photometry Pipeline
# Copyright (C) 2016-2018  Michael Mommert, mommermiscience@gmail.com

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.


import numpy as np
import os
import sys
import logging
import argparse
import sqlite3
from astroquery.jplhorizons import Horizons
from astropy.io import ascii
from astropy.io import fits
from astropy.wcs import WCS

try:
    from astroquery.vizier import Vizier
except ImportError:
    print('Module astroquery not found. Please install with: pip install '
          'astroquery')
    sys.exit()

# only import if Python3 is used
if sys.version_info > (3, 0):
    from builtins import str
    from builtins import range


# pipeline-specific modules
import _pp_conf
from pp_setup import confdistill as conf
from catalog import *
from toolbox import *
from diagnostics import distill as diag

# setup logging
logging.basicConfig(filename=_pp_conf.log_filename,
                    level=_pp_conf.log_level,
                    format=_pp_conf.log_formatline,
                    datefmt=_pp_conf.log_datefmt)


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _fits_filename_from_catalog(cat):
    try:
        return cat.origin.split(';')[1]
    except IndexError:
        return None


def _sky_jacobian_arcsec_per_pix(fits_filename, x, y):
    """Numerically derive the local WCS Jacobian in arcsec per pixel."""

    if fits_filename is None or not os.path.exists(fits_filename):
        return None

    try:
        wcs = WCS(fits.getheader(fits_filename))
        ra0, dec0 = wcs.all_pix2world([x], [y], 1)
        rax, decx = wcs.all_pix2world([x+1], [y], 1)
        ray, decy = wcs.all_pix2world([x], [y+1], 1)
    except Exception as exc:
        logging.warning('could not derive WCS uncertainty scale for %s: %s',
                        fits_filename, exc)
        return None

    dec_rad = np.deg2rad(dec0[0])

    def delta_ra_arcsec(ra_a, ra_b):
        return ((ra_a-ra_b+180) % 360-180)*3600*np.cos(dec_rad)

    return np.array([[delta_ra_arcsec(rax[0], ra0[0]),
                      delta_ra_arcsec(ray[0], ra0[0])],
                     [(decx[0]-dec0[0])*3600,
                      (decy[0]-dec0[0])*3600]])


def source_position_uncertainty(cat, values):
    """Propagate Source Extractor centroid errors into sky coordinates."""

    err_a = _safe_float(values.get('ERRAWIN_IMAGE'))
    err_b = _safe_float(values.get('ERRBWIN_IMAGE'))
    theta = _safe_float(values.get('ERRTHETAWIN_IMAGE'))
    x = _safe_float(values.get('XWIN_IMAGE'))
    y = _safe_float(values.get('YWIN_IMAGE'))
    if any(np.isnan(val) for val in [err_a, err_b, theta, x, y]):
        return np.nan, np.nan, np.nan

    theta = np.deg2rad(theta)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    cov_pix = rot.dot(np.diag([err_a**2, err_b**2])).dot(rot.T)

    jacobian = _sky_jacobian_arcsec_per_pix(
        _fits_filename_from_catalog(cat), x, y)
    if jacobian is None:
        obsparam = _pp_conf.telescope_parameters[
            cat.origin.split(';')[0].strip()]
        jacobian = np.array([[obsparam['secpix'][0], 0],
                             [0, obsparam['secpix'][1]]])

    cov_sky = jacobian.dot(cov_pix).dot(jacobian.T)
    ra_sig = np.sqrt(max(cov_sky[0, 0], 0))
    dec_sig = np.sqrt(max(cov_sky[1, 1], 0))
    pos_sig = np.sqrt(ra_sig**2+dec_sig**2)
    return ra_sig, dec_sig, pos_sig


def manual_positions(posfile, catalogs, display=True):
    """create targets for manually provided positions (using -positions
    option)"""

    if display:
        print(('# target positions as a function of time manually '
               'provided... '), end=' ')
        sys.stdout.flush()
    logging.info('target positions as a function of time manually provided')

    # handle wildcard symbol
    if posfile == 'all_objects':

        objects = []
        for filename in os.listdir('.'):
            if 'positions_' in filename and '.dat' in filename:
                objects.append(manual_positions(filename, catalogs,
                                                display=False))

        if display:
            print(len(objects)/len(catalogs), 'object(s) found')

        return (list(np.hstack(objects)))

    try:
        positions = np.genfromtxt(posfile, dtype=[('filename', 'S50'),
                                                  ('ra', float),
                                                  ('dec', float),
                                                  ('MJD', float),
                                                  ('name', 'S30')])
    except:
        positions = np.genfromtxt(posfile, dtype=[('filename', 'S50'),
                                                  ('ra', float),
                                                  ('dec', float),
                                                  ('MJD', float)])

    # Single-frame runs produce a scalar record; normalize to an array so the
    # code below treats one and many manual positions identically.
    if len(positions.shape) == 0:
        positions = np.array([positions])

    try:
        assert len(positions) == len(catalogs)
    except AssertionError:
        print(posfile + ' is not complete; has to provide a position ' +
              'for each frame')
        logging.error(posfile+' is not complete; has to provide position ' +
                      'for each frame')
        return []

    objects = []
    for cat_idx, cat in enumerate(catalogs):
        try:
            objects.append({'ident': positions[cat_idx]['name'].decode('utf-8'),
                            'obsdate.jd':  cat.obstime,
                            'cat_idx':  cat_idx,
                            'ra_deg':  positions[cat_idx]['ra'],
                            'dec_deg':  positions[cat_idx]['dec']})
        except:
            objects.append({'ident': 'manual_target',
                            'obsdate.jd':  cat.obstime,
                            'cat_idx':  cat_idx,
                            'ra_deg':  positions[cat_idx]['ra'],
                            'dec_deg':  positions[cat_idx]['dec']})

    if display:
        print(len(objects)/len(catalogs), 'object(s) found')

    return objects


def pick_controlstar(catalogs, display=True):
    """match the first and the last catalog and pick a bright star"""

    if display:
        print('# pick control star... ', end=' ')
        sys.stdout.flush()
    logging.info('pick control star')

    match = catalogs[0].match_with(catalogs[-1],
                                   match_keys_this_catalog=[
                                       'ra_deg', 'dec_deg'],
                                   match_keys_other_catalog=[
                                       'ra_deg', 'dec_deg'],
                                   extract_this_catalog=[
                                       'ra_deg', 'dec_deg', 'FLAGS'],
                                   extract_other_catalog=[
                                       'ra_deg', 'dec_deg', 'FLAGS', 'MAG_APER'],
                                   tolerance=1/3600)

    objects = []
    if len(match[0][0]) > 0:

        ctlstar_idx = np.argsort(match[1][3])[int(0.05*len(match[1][3]))]

        for cat_idx, cat in enumerate(catalogs):
            objects.append({'ident': 'Control Star',
                            'obsdate.jd':  cat.obstime[0],
                            'cat_idx':  cat_idx,
                            'ra_deg':  match[1][0][ctlstar_idx],
                            'dec_deg':  match[1][1][ctlstar_idx]})
    else:
        print('  no common control star found in first and last frame')
        logging.info('no common control star found in first and last frame')

    if display:
        print('done!')

    return objects


def moving_primary_target(catalogs, man_targetname, offset, is_asteroid=None,
                          display=True):
    """
    is_asteroid == True:  this object is an asteroid
    is_asteroid == False: this object is a planet/moon/spacecraft
    is_asteroid == None:  no information on target nature
    """

    if display:
        print('# check JPL Horizons for primary target... ')
        sys.stdout.flush()
    logging.info('check JPL Horizons for primary target')

    obsparam = _pp_conf.telescope_parameters[
        catalogs[0].origin.split(';')[0].strip()]

    objects = []

    # check for target nature, if unknown
    if is_asteroid is None:
        cat = catalogs[0]
        targetname = cat.obj.replace('_', ' ')
        if man_targetname is not None:
            targetname = man_targetname.replace('_', ' ')
        for smallbody in [True, False]:
            obj = Horizons(targetname.replace('_', ' '),
                           id_type={True: 'smallbody',
                                    False: 'majorbody'}[smallbody],
                           epochs=cat.obstime[0],
                           location=obsparam['observatory_code'])
            n = 0
            try:
                eph = obj.ephemerides()
                n = len(eph)
            except ValueError:
                if display and smallbody is True:
                    print("'%s' is not a small body" % targetname)
                    logging.warning("'%s' is not a small body" %
                                    targetname)
                if display and smallbody is False:
                    print("'%s' is not a Solar System object" % targetname)
                    logging.warning("'%s' is not a Solar System object" %
                                    targetname)
                pass
            if n > 0:
                is_asteroid = smallbody
                break

    # if is_asteroid is still None, this object is not in the Horizons db
    if is_asteroid is None:
        return objects

    message_shown = False

    # query information for each image
    for cat_idx, cat in enumerate(catalogs):
        targetname = cat.obj.replace('_', ' ')
        if man_targetname is not None:
            targetname = man_targetname.replace('_', ' ')
            cat.obj = targetname
        obj = Horizons(targetname.replace('_', ' '),
                       id_type={True: 'smallbody',
                                False: 'majorbody'}[is_asteroid],
                       epochs=cat.obstime[0],
                       location=obsparam['observatory_code'])
        try:
            eph = obj.ephemerides()
            n = len(eph)
        except ValueError:
            # if is_asteroid:
            #     if display and not message_shown:
            #         print 'is \'%s\' an asteroid?' % targetname
            #     logging.warning('Target (%s) is not an asteroid' % targetname)

            # else:
            #     if display and not message_shown:
            #         print ('is \'%s\' a different Solar System object?' %
            #                )targetname
            #     logging.warning('Target (%s) is not a Solar System object' %
            #                     targetname)
            #     n = None
            pass

        if n is None or n == 0:
            logging.warning('WARNING: No position from Horizons! ' +
                            'Name (%s) correct?' % cat.obj.replace('_', ' '))
            logging.warning('HORIZONS call: %s' % obj.uri)
            if display and not message_shown:
                print('  no Horizons data for %s ' % cat.obj.replace('_', ' '))
                message_shown = True

        else:
            objects.append({'ident': eph[0]['targetname'].replace(" ", "_"),
                            'obsdate.jd': cat.obstime[0],
                            'cat_idx': cat_idx,
                            'ra_deg': eph[0]['RA']-offset[0]/3600,
                            'dec_deg': eph[0]['DEC']-offset[1]/3600})
            logging.info('Successfully grabbed Horizons position for %s ' %
                         cat.obj.replace('_', ' '))
            logging.info('HORIZONS call: %s' % obj.uri)
            if display and not message_shown:
                print(cat.obj.replace('_', ' '), "identified")
                message_shown = True

    return objects


def fixed_targets(fixed_targets_file, catalogs, display=True):
    """add fixed target positions to object catalog"""

    if display:
        print('# read fixed target file... ', end=' ')
        sys.stdout.flush()
    logging.info('read fixed target file')

    fixed_targets = np.genfromtxt(fixed_targets_file,
                                  dtype=[('name', 'S20'),
                                         ('ra', float),
                                         ('dec', float)])

    # force array shape even for single line fixed_targets_files
    if len(fixed_targets.shape) == 0:
        fixed_targets = np.array([fixed_targets])

    objects = []
    for obj in fixed_targets:
        for cat_idx, cat in enumerate(catalogs):
            objects.append({'ident': obj['name'].decode('utf-8'),
                            'obsdate.jd': cat.obstime[0],
                            'cat_idx': cat_idx,
                            'ra_deg': obj['ra'],
                            'dec_deg': obj['dec']})

    if display:
        print(len(objects)/len(catalogs), 'targets read')

    return objects


# ---- search for serendipitous targets

def serendipitous_variablestars(catalogs, display=True):
    """match catalogs with VSX catalog using astroquery.Vizier
    """

    if display:
        print('# match frames with variable star database... ', end=' ',
              flush=True)
    logging.info('match frames with variable star database')

    # derive center and radius of field of view of all images
    ra_deg, dec_deg, rad_deg = skycenter(catalogs)
    logging.info('FoV center (%.7f/%+.7f) and radius (%.2f deg) derived' %
                 (ra_deg, dec_deg, rad_deg))

    # derive observation midtime of sequence
    midtime = np.average([cat.obstime[0] for cat in catalogs])

    # setup Vizier query
    # note: column filters uses original Vizier column names
    # -> green column names in Vizier

    logging.info(('query Vizier for VSX at %7.3f/%+8.3f in '
                  + 'a %.2f deg radius') %
                 (ra_deg, dec_deg, rad_deg))

    field = coord.SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg),
                           frame='icrs')

    vquery = Vizier(columns=['Name', 'RAJ2000', 'DEJ2000'])
    try:
        data = vquery.query_region(field,
                                   width=("%fd" % rad_deg),
                                   catalog="B/vsx/vsx")[0]
    except IndexError:
        if display:
            print('no data available from VSX')
            logging.error('no data available from VSX')
            return []

    objects = []
    for cat_idx, cat in enumerate(catalogs):
        for star in data:
            objects.append({'ident': star['Name'],
                            'obsdate.jd': cat.obstime[0],
                            'cat_idx': cat_idx,
                            'ra_deg': star['RAJ2000'],
                            'dec_deg': star['DEJ2000']})

    if display:
        print(len(objects)/len(catalogs), 'variable stars found')

    return objects


def serendipitous_asteroids(catalogs, display=True):

    import requests
    server = 'http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php'

    if display:
        print('# check frames with IMCCE SkyBoT... ', end=' ')
        sys.stdout.flush()
    logging.info('check frames with IMCCE SkyBoT')

    # todo: smarter treatment of limiting magnitude and pos. uncertainty

    # identify limiting magnitude (90 percentile in calibrated magnitude)
    band_key = None
    for key in catalogs[0].fields:
        if 'mag' in key and not '_e_' in key:
            band_key = key

    if band_key is None:
        if display:
            print(('Error: No calibrated magnitudes found '
                   'in catalog "{:s}"').format(catalogs[0].catalogname),
                  flush=True)
        logging.error(('No calibrated magnitudes found '
                       'in catalog "{:s}"').format(catalogs[0].catalogname))
        return []

    maglims = [np.percentile(cat[band_key], 90) for cat in catalogs]

    # maximum positional uncertainty = 5 px (unbinned)
    obsparam = _pp_conf.telescope_parameters[
        catalogs[0].origin.split(';')[0].strip()]
    max_posunc = 5*max(obsparam['secpix'])

    # derive center and radius of field of view of all images
    ra_deg, dec_deg, rad_deg = skycenter(catalogs)
    logging.info('FoV center (%.7f/%+.7f) and radius (%.2f deg) derived' %
                 (ra_deg, dec_deg, rad_deg))

    # derive observation midtime of sequence
    midtime = np.average([cat.obstime[0] for cat in catalogs])

    r = requests.get(server,
                     params={'RA': ra_deg, 'DEC': dec_deg,
                             'SR': rad_deg, 'EPOCH': str(midtime),
                             '-output': 'object',
                             '-mime': 'text'},
                     timeout=180)

    if 'No solar system object was found in the requested FOV' in r.text:
        print('No Solar System object found')
        logging.warning('SkyBot failed: ' + r.text)
        return []

    results = ascii.read(r.text, delimiter='|',
                         names=('number', 'name', 'ra', 'dec', 'type',
                                'V', 'posunc', 'd'))

    objects = []
    for obj in results:
        if obj['posunc'] > max_posunc:
            logging.warning(('asteroid {:s} rejected due to large '
                             'pos. unc ({:f} > {:f} arcsec)').format(
                                 obj['name'],
                                 obj['posunc'],
                                 max_posunc))
            continue
        if obj['V'] > max(maglims):
            logging.warning(('asteroid {:s} rejected; too faint '
                             '({:f} mag)').format(obj['name'],
                                                  obj['V']))
            continue

        objects += moving_primary_target(catalogs, obj['name'], (0, 0),
                                         display=False)

        logging.info(('asteroid {:s} added to '
                      'target pool').format(obj['name']))

    if display:
        print(len(objects)/len(catalogs), 'asteroids found')

    return objects


# -------------------


def distill(catalogs, man_targetname, offset, fixed_targets_file, posfile,
            rejectionfilter='pos', display=False, diagnostics=False,
            variable_stars=False, asteroids=False):
    """
    distill wrapper
    """

    # start logging
    logging.info('starting distill with parameters: %s' %
                 (', '.join([('%s: %s' % (var, str(val))) for
                             var, val in list(locals().items())])))

    output = {}

    # read in database files (if necessary)
    if isinstance(catalogs[0], str):
        filenames = catalogs[:]
        catalogs = []
        for filename in filenames:
            filename = filename[:filename.find('.fit')]+'.ldac.db'
            cat = catalog(filename)
            try:
                cat.read_database(filename)
            except IOError:
                logging.error('Cannot find database', filename)
                print('Cannot find database', filename)
                continue
            except sqlite3.OperationalError:
                logging.error('File %s is not a database file' % filename)
                print('File %s is not a database file' % filename)
                continue
            catalogs.append(cat)

    # identify target names and types

    objects = []  # one dictionary for each target

    if display:
        print('#------ Identify Targets')

    # check for positions file
    if posfile is not None:
        objects += manual_positions(posfile, catalogs, display=display)

    # check Horizons for primary target (if a moving target)
    if posfile is None and fixed_targets_file is None and asteroids is False:
        objects += moving_primary_target(catalogs, man_targetname, offset,
                                         display=display)

    # add fixed target
    if fixed_targets_file is not None:
        objects += fixed_targets(fixed_targets_file, catalogs, display=display)

    # serendipitous asteroids
    if asteroids:
        objects += serendipitous_asteroids(catalogs, display=display)

    # serendipitous variable stars
    if variable_stars:
        objects += serendipitous_variablestars(catalogs, display=display)

    # select a sufficiently bright star as control star
    objects += pick_controlstar(catalogs, display=display)

    if display:
        print('#-----------------------')

    if display:
        print('{:d} potential target(s) per frame identified.'.format(
            int(len(objects)/len(catalogs))))

    # extract source data for identified targets

    data = []
    targetnames = {}

    # sort objects by catalog idx
    for cat_idx, cat in enumerate(catalogs):

        # check for each target if it is within the image boundaries
        min_ra = np.min(cat['ra_deg'])
        max_ra = np.max(cat['ra_deg'])
        min_dec = np.min(cat['dec_deg'])
        max_dec = np.max(cat['dec_deg'])
        objects_thiscat = []
        for obj in objects:
            if obj['cat_idx'] != cat_idx:
                continue
            # # not required since there is exactly one entry per catalog
            # if (obj['ra_deg'] > max_ra or obj['ra_deg'] < min_ra or
            #         obj['dec_deg'] > max_dec or obj['dec_deg'] < min_dec):
            #     if display:
            #         print('\"%s\" not in image %s' % (obj['ident'],
            #                                           cat.catalogname))
            #         logging.info('\"%s\" not in image %s' % (obj['ident'],
            #                                                  cat.catalogname))

            #     continue
            objects_thiscat.append(obj)

        if len(objects_thiscat) == 0:
            continue

        # create a new catalog
        target_cat = catalog('targetlist:_'+cat.catalogname)

        target_cat.add_fields(['ident', 'ra_deg', 'dec_deg'],
                              [[obj['ident'] for obj in objects_thiscat],
                               [obj['ra_deg']
                                for obj in objects_thiscat],
                               [obj['dec_deg'] for obj in objects_thiscat]])

        # identify filtername
        filtername = cat.filtername

        # identify magnitudes
        mag_keys = ['MAG_APER', 'MAGERR_APER']
        if filtername is not None:
            for key in cat.fields:
                if filtername+'mag' in key:
                    mag_keys.append(key)
        # if both catalog magnitudes and transformated magnitudes in mag_keys
        # use the transformed ones (the target is most probably not in the
        # catalog used for photometric calibration)
        fixed_mag_keys = []
        for band in mag_keys:
            if '_'+band in mag_keys:
                continue
            else:
                fixed_mag_keys.append(band)
        mag_keys = fixed_mag_keys

        match_keys_other_catalog, extract_other_catalog = [], []

        for key in ['ra_deg', 'dec_deg', 'XWIN_IMAGE', 'YWIN_IMAGE',
                    'FLAGS', 'FWHM_WORLD', 'ERRAWIN_IMAGE',
                    'ERRBWIN_IMAGE', 'ERRTHETAWIN_IMAGE',
                    'ASTR_SIG_RA', 'ASTR_SIG_DEC', 'ASTR_SIG_POS']:
            if key in cat.fields:
                match_keys_other_catalog.append(key)
                extract_other_catalog.append(key)

        match = target_cat.match_with(
            cat,
            match_keys_this_catalog=('ra_deg', 'dec_deg'),
            match_keys_other_catalog=match_keys_other_catalog,
            extract_this_catalog=['ra_deg', 'dec_deg', 'ident'],
            extract_other_catalog=extract_other_catalog+mag_keys,
            tolerance=None)

        for i in range(len(match[0][0])):
            values = {key: match[1][idx][i] for idx, key in
                      enumerate(extract_other_catalog)}
            # derive calibrated magnitudes, if available
            try:
                cal_mag = match[1][len(extract_other_catalog)+2][i]
                cal_magerr = match[1][len(extract_other_catalog)+3][i]
            except IndexError:
                # use instrumental magnitudes
                cal_mag = match[1][len(extract_other_catalog)][i]
                cal_magerr = match[1][len(extract_other_catalog)+1][i]

            source_ra_sig, source_dec_sig, source_pos_sig = (
                source_position_uncertainty(cat, values))
            astrom_ra_sig = _safe_float(values.get('ASTR_SIG_RA'))
            astrom_dec_sig = _safe_float(values.get('ASTR_SIG_DEC'))
            astrom_pos_sig = _safe_float(values.get('ASTR_SIG_POS'))
            total_ra_sig = np.sqrt(source_ra_sig**2+astrom_ra_sig**2)
            total_dec_sig = np.sqrt(source_dec_sig**2+astrom_dec_sig**2)
            total_pos_sig = np.sqrt(source_pos_sig**2+astrom_pos_sig**2)

            data.append([match[0][2][i], match[0][0][i], match[0][1][i],
                         values.get('ra_deg'), values.get('dec_deg'),
                         match[1][len(extract_other_catalog)][i],
                         match[1][len(extract_other_catalog)+1][i],
                         cal_mag, cal_magerr,
                         cat.obstime, cat.catalogname,
                         values.get('XWIN_IMAGE'), values.get('YWIN_IMAGE'),
                         cat.origin, values.get('FLAGS'),
                         values.get('FWHM_WORLD'),
                         source_ra_sig, source_dec_sig, source_pos_sig,
                         astrom_ra_sig, astrom_dec_sig, astrom_pos_sig,
                         total_ra_sig, total_dec_sig, total_pos_sig])
            # format: ident, RA_exp, Dec_exp, RA_img, Dec_img,
            #         mag_inst, sigmag_instr, mag_cal, sigmag_cal
            #         obstime, filename, img_x, img_y, origin, flags
            #         fwhm, source/astrometric/total position sigmas
            targetnames[match[0][2][i]] = 1

    # list of targets
    output['targetnames'] = targetnames

    # dictionary: list of frames per target
    output['targetframes'] = {}

    # write results to ASCII file

    for target in targetnames:

        if sys.version_info < (3, 0):
            target = str(target)

        output[target] = []
        output['targetframes'][target] = []

        if display:
            print('write photometry results for %s' % target)

        outf = open('photometry_%s.dat' %
                    target.translate(_pp_conf.target2filename), 'w')
        outf.write('#                           filename     julian_date      ' +
                   'mag    sig     source_ra    source_dec   [1]   [2]   ' +
                   '[3]   [4]    [5]       ZP ZP_sig inst_mag ' +
                   'in_sig               [6] [7] [8]    [9]          [10] ' +
                   'FWHM mag_src_sig mag_cal_sig mag_tot_sig ' +
                   'ra_src_sig dec_src_sig pos_src_sig ra_ast_sig ' +
                   'dec_ast_sig pos_ast_sig ra_tot_sig dec_tot_sig ' +
                   'pos_tot_sig\n')

        for dat in data:
            # sort measured magnitudes by target
            if dat[0] == target:
                reject_this_target = False
                try:
                    filtername = dat[13].split(';')[3]
                    if 'manual_zp' in dat[13].split(';')[2]:
                        filtername = dat[13].split(';')[2][0]
                except IndexError:
                    filtername = '-'
                    if (len(dat[13].split(';')) > 2 and
                            'manual_zp' in dat[13].split(';')[2]):
                        filtername = dat[13].split(';')[2][0]
                try:
                    catalogname = dat[13].split(';')[2]
                    if 'manual_zp' in catalogname:
                        catalogname = 'manual_zp'
                except IndexError:
                    catalogname = dat[13].split(';')[1]
                    if 'manual_zp' in catalogname:
                        catalogname = 'manual_zp'

                # apply rejectionfilter
                for reject in rejectionfilter.split(','):
                    if conf.rejection[reject](dat):
                        logging.info(('reject photometry for target {:s} '
                                      'from frame {:s} due to rejection '
                                      'schema {:s}').format(
                                          dat[0],
                                          dat[10].replace(' ', '_'),
                                          reject))
                        reject_this_target = True

                if reject_this_target:
                    outf.write('#')
                else:
                    outf.write(' ')
                    output[target].append(dat)
                # Existing calibrated sigma is source and zeropoint in quadrature.
                mag_cal_sig = np.sqrt(max(dat[8]**2-dat[6]**2, 0))
                outf.write(('%35.35s ' % dat[10].replace(' ', '_')) +
                           ('%15.7f ' % dat[9][0]) +
                           ('%8.4f ' % dat[7]) +
                           ('%6.4f ' % dat[8]) +
                           ('%13.8f ' % dat[3]) +
                           ('%+13.8f ' % dat[4]) +
                           ('%5.2f ' % ((dat[1] - dat[3]) * 3600.)) +
                           ('%5.2f ' % ((dat[2] - dat[4]) * 3600.)) +
                           ('%5.2f ' % offset[0]) +
                           ('%5.2f ' % offset[1]) +
                           ('%5.2f ' % dat[9][1]) +
                           ('%8.4f ' % (dat[7] - dat[5])) +
                           ('%6.4f ' % np.sqrt(dat[8]**2 - dat[6]**2)) +
                           ('%8.4f ' % dat[5]) +
                           ('%6.4f ' % dat[6]) +
                           ('%s ' % catalogname) +
                           ('%s ' % filtername) +
                           ('%3d ' % dat[14]) +
                           ('%s' % dat[13].split(';')[0]) +
                           ('%10s ' % _pp_conf.photmode) +
                           ('%4.2f ' % (dat[15]*3600)) +
                           ('%7.4f ' % dat[6]) +
                           ('%7.4f ' % mag_cal_sig) +
                           ('%7.4f ' % dat[8]) +
                           ('%7.4f ' % dat[16]) +
                           ('%7.4f ' % dat[17]) +
                           ('%7.4f ' % dat[18]) +
                           ('%7.4f ' % dat[19]) +
                           ('%7.4f ' % dat[20]) +
                           ('%7.4f ' % dat[21]) +
                           ('%7.4f ' % dat[22]) +
                           ('%7.4f ' % dat[23]) +
                           ('%7.4f\n' % dat[24]))
                output['targetframes'][target].append(dat[10][:-4]+'fits')

        outf.writelines('#\n# [1]: predicted_RA - source_RA [arcsec]\n' +
                        '# [2]: predicted_Dec - source_Dec [arcsec]\n' +
                        '# [3,4]: manual target offsets in RA and DEC ' +
                        '[arcsec]\n' +
                        '# [5]: exposure time (s)\n' +
                        '# [6]: photometric catalog\n' +
                        '# [7]: photometric band\n' +
                        '# [8]: Source Extractor flag\n' +
                        '# [9]: telescope/instrument\n' +
                        '# [10]: photometry method\n' +
                        '# *_src_sig: source measurement uncertainty\n' +
                        '# *_cal_sig: photometric calibration uncertainty\n' +
                        '# *_ast_sig: astrometric field uncertainty\n' +
                        '# *_tot_sig: quadrature-combined uncertainty\n' +
                        '# position uncertainties are in arcsec\n')
        outf.close()

    # output content
    #
    # { 'targetnames': list of all targets,
    #   'targetframes': lists of frames per target,
    #   '(individual targetname)': [ident, RA_exp, Dec_exp, RA_img, Dec_img,
    #                               mag_inst, sigmag_instr, mag_cal,
    #                               sigmag_cal, obstime, filename, img_x,
    #                               img_y, origin, flags, fwhm],
    # }
    ###

    # create diagnostics
    if diagnostics:
        if display:
            print('extracting thumbnail images')
        logging.info(' ~~~~~~~~~ creating diagnostic output')
        diag.add_results(output)

    return output


if __name__ == '__main__':

    # command line arguments
    parser = argparse.ArgumentParser(description='distill sources of interest')
    parser.add_argument('-target', help='target name', default=None)
    parser.add_argument('-offset', help='primary target offset (arcsec)',
                        nargs=2, default=[0, 0])
    parser.add_argument('-positions', help='positions file', default=None)
    parser.add_argument('-fixedtargets', help='target file', default=None)
    parser.add_argument('-variable_stars',
                        help=('search for serendipitous variable star '
                              'observations'),
                        action="store_true")
    parser.add_argument('-asteroids',
                        help='search for serendipitous asteroids',
                        action="store_true")
    parser.add_argument('-reject',
                        help='schemas for target rejection',
                        nargs=1, default='pos')
    parser.add_argument('images', help='images to process', nargs='+')
    args = parser.parse_args()
    man_targetname = args.target
    man_offset = [float(coo) for coo in args.offset]
    fixed_targets_file = args.fixedtargets
    variable_stars = args.variable_stars
    asteroids = args.asteroids
    posfile = args.positions
    rejectionfilter = args.reject
    filenames = args.images

    # check if input filenames is actually a list
    if len(filenames) == 1:
        if filenames[0].find('.lst') > -1 or filenames[0].find('.list') > -1:
            filenames = [filename[:-1] for filename in
                         open(filenames[0], 'r').readlines()]

    distillate = distill(filenames, man_targetname, man_offset,
                         fixed_targets_file,
                         posfile, rejectionfilter,
                         display=True, diagnostics=True,
                         variable_stars=variable_stars,
                         asteroids=asteroids)
