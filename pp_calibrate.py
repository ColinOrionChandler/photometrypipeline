#!/usr/bin/env python3

from diagnostics import calibration as diag
from toolbox import *
from catalog import *
from diagnostics import calibration as diag
from pp_setup import confcalibrate as conf
import _pp_conf
""" PP_CALIBRATE - match image databases against photometry catalogs
                   and derive magnitude zeropoint
    v1.0: 2016-01-15, mommermiscience@gmail.com
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


from copy import deepcopy
import sys
import numpy as np
import argparse
import logging
from astropy.io import fits
from scipy.optimize import minimize
from astropy.table import join

# only import if Python3 is used
if sys.version_info > (3, 0):
    from builtins import range


# pipeline-specific modules

# setup logging
logging.basicConfig(filename=_pp_conf.log_filename,
                    level=_pp_conf.log_level,
                    format=_pp_conf.log_formatline,
                    datefmt=_pp_conf.log_datefmt)

# photometric fitting routines


def ldac_filename_for_fits(filename):
    """Return the LDAC filename produced for a FITS image."""

    return filename[:filename.find('.fit')]+'.ldac'


def manual_zeropoints_from_headers(filenames, magzp=None,
                                   magzp_keyword=None,
                                   magzp_sig_keyword=None,
                                   magzp_sig=None):
    """Build per-frame zeropoints from CLI values or FITS headers."""

    if magzp is not None and magzp_keyword is not None:
        raise ValueError('use either -magzp or -magzp_keyword, not both')

    if magzp is not None:
        zp, zp_sig = float(magzp[0]), float(magzp[1])
        return {ldac_filename_for_fits(filename):
                {'filename': ldac_filename_for_fits(filename),
                 'zp': zp, 'zp_sig': zp_sig,
                 'zp_nstars': 0, 'zp_usedstars': 0,
                 'obstime': 0, 'match': 0,
                 'clipping_steps': 0, 'zp_idx': 0,
                 'success': True,
                 'source': 'manual_zp'}
                for filename in filenames}

    if magzp_keyword is None:
        return None

    if magzp_sig_keyword is None and magzp_sig is None:
        raise ValueError('-magzp_keyword requires -magzp_sig_keyword or '
                         '-magzp_sig')

    zeropoints = {}
    for filename in filenames:
        header = fits.getheader(filename, ignore_missing_end=True)
        try:
            zp = float(header[magzp_keyword])
        except KeyError:
            raise KeyError('zeropoint keyword %s not found in %s' %
                           (magzp_keyword, filename))

        if magzp_sig_keyword is not None:
            try:
                zp_sig = float(header[magzp_sig_keyword])
            except KeyError:
                if magzp_sig is None:
                    raise KeyError('zeropoint uncertainty keyword %s not '
                                   'found in %s' %
                                   (magzp_sig_keyword, filename))
                zp_sig = float(magzp_sig)
        else:
            zp_sig = float(magzp_sig)

        ldac_filename = ldac_filename_for_fits(filename)
        zeropoints[ldac_filename] = {
            'filename': ldac_filename,
            'zp': zp,
            'zp_sig': zp_sig,
            'zp_nstars': 0,
            'zp_usedstars': 0,
            'obstime': 0,
            'match': 0,
            'clipping_steps': 0,
            'zp_idx': 0,
            'success': True,
            'source': magzp_keyword,
        }

    return zeropoints


def apply_manual_zeropoints(catalogs, filtername, zeropoints):
    """Apply fixed or header-derived zeropoints to extracted catalogs."""

    filterkey = filtername+'mag'
    efilterkey = 'e_' + filtername + 'mag'

    for cat in catalogs:
        zeropoint = zeropoints[cat.catalogname]
        cat.add_fields([filterkey, efilterkey],
                       [cat['MAG_'+_pp_conf.photmode] + zeropoint['zp'],
                        np.sqrt(cat['MAGERR_'+_pp_conf.photmode]**2 +
                                zeropoint['zp_sig']**2)],
                       ['F', 'F'])
        cat.origin = (cat.origin.strip() +
                      ';'+filtername+'_manual_zp;')
        cat.history += 'calibrated using manual zeropoint'

    return catalogs


def create_photometrycatalog(ra_deg, dec_deg, rad_deg, filtername,
                             preferred_catalogs,
                             min_sources=_pp_conf.min_sources_photometric_catalog,
                             max_sources=1e4, mag_accuracy=0.1,
                             solar=False, use_all_stars=False,
                             display=False):
    """create a photometric catalog of the field of view"""

    for catalogname in preferred_catalogs:
        cat = catalog(catalogname, display)

        # load catalog
        n_sources = cat.download_catalog(ra_deg, dec_deg, rad_deg,
                                         max_sources,
                                         use_all_stars=use_all_stars,
                                         save_catalog=True)

        if display:
            print(n_sources, 'sources downloaded from', catalogname)
        if n_sources < min_sources:
            continue

        if 'URAT' in catalogname:
            print(catalogname + ' should only be used as an astrometric '
                  'catalog; please use APASS9 instead')
            logging.error(catalogname + ' should only be used as an '
                          'astrometric catalog; please use APASS9 instead')
            return None

        # reject non-solar colors, if requested by user
        if solar and not use_all_stars:
            sol_gr = 0.44  # g-r
            sol_ri = 0.11  # r-i
            sol_JH = 0.29  # J-H
            n_rejected = 0
            n_raw = cat.shape[0]
            if ('SDSS' in cat.catalogname or
                    'APASS' in cat.catalogname):
                n_rejected += cat.reject_sources_with(
                    (cat['gmag']-cat['rmag']) < sol_gr-_pp_conf.solcol)
                n_rejected += cat.reject_sources_with(
                    (cat['gmag']-cat['rmag']) > sol_gr+_pp_conf.solcol)
                n_rejected += cat.reject_sources_with(
                    (cat['rmag']-cat['imag']) < sol_ri-_pp_conf.solcol)
                n_rejected += cat.reject_sources_with(
                    (cat['rmag']-cat['imag']) > sol_ri+_pp_conf.solcol)
            elif 'SkyMapper' in cat.catalogname:
                cat.transform_filters('g')  # derive Sloan griz
                n_rejected += cat.reject_sources_with(
                    (cat['_gmag']-cat['_rmag']) < sol_gr-_pp_conf.solcol)
                n_rejected += cat.reject_sources_with(
                    (cat['_gmag']-cat['_rmag']) > sol_gr+_pp_conf.solcol)
                n_rejected += cat.reject_sources_with(
                    (cat['_rmag']-cat['_imag']) < sol_ri-_pp_conf.solcol)
                n_rejected += cat.reject_sources_with(
                    (cat['_rmag']-cat['_imag']) > sol_ri+_pp_conf.solcol)
            elif 'PANSTARRS' in cat.catalogname:
                cat.transform_filters('g')
                # derive Sloan griz
                n_rejected += cat.reject_sources_with(
                    (cat['_gmag']-cat['_rmag']) < sol_gr-_pp_conf.solcol)
                n_rejected += cat.reject_sources_with(
                    (cat['_gmag']-cat['_rmag']) > sol_gr+_pp_conf.solcol)
                n_rejected += cat.reject_sources_with(
                    (cat['_rmag']-cat['_imag']) < sol_ri-_pp_conf.solcol)
                n_rejected += cat.reject_sources_with(
                    (cat['_rmag']-cat['_imag']) > sol_ri+_pp_conf.solcol)
            elif 'GAIA' in cat.catalogname:
                cat.transform_filters('g')  # derive Sloan gri
                n_rejected += cat.reject_sources_with(
                    (cat['_gmag']-cat['_rmag']) < sol_gr-_pp_conf.solcol)
                n_rejected += cat.reject_sources_with(
                    (cat['_gmag']-cat['_rmag']) > sol_gr+_pp_conf.solcol)
                n_rejected += cat.reject_sources_with(
                    (cat['_rmag']-cat['_imag']) < sol_ri-_pp_conf.solcol)
                n_rejected += cat.reject_sources_with(
                    (cat['_rmag']-cat['_imag']) > sol_ri+_pp_conf.solcol)
            elif '2MASS' in cat.catalogname:
                # derive UKIRT ZJHK (Casagrande et al. 2012)
                cat.transform_filters('K')
                n_rejected += cat.reject_sources_with(
                    (cat['_Jmag']-cat['_Hmag']) < sol_JH-_pp_conf.solcol)
            else:
                if display:
                    print('Warning: solar colors not supported for catalog',
                          cat.catalogname)
                    logging.warning(('Warning: solar colors not supported ' +
                                     'for catalog'), cat.catalogname)
            if display:
                print('%d/%d sources left with solar-like colors' %
                      (n_raw-n_rejected, n_raw))
                logging.info('%d/%d sources left with solar-like colors' %
                             (n_raw-n_rejected, n_raw))
            cat.history += (', {:d} stars left with solar colors'.
                            format(n_raw-n_rejected))
            cat.catalogname += '_solar'

        print(catalogname, filtername)

        # transform catalog to requested filtername, if necessesary
        if (n_sources > 0 and
            ('SDSS' in catalogname and
             filtername not in {'u', 'g', 'r', 'i', 'z'}) or
            ('URAT' in catalogname and
             filtername not in {'B', 'V', 'g', 'r', 'i'}) or
            ('APASS' in catalogname and
             filtername not in {'B', 'V', 'g', 'r', 'i'}) or
            ('2MASS' in catalogname and
             filtername not in {'J', 'H', 'Ks'}) or
            ('PANSTARRS' in catalogname and
             filtername not in {'gp1', 'rp1', 'ip1', 'zp1', 'yp1'}) or
            ('SkyMapper' in catalogname and
             filtername not in {'gsm', 'rsm', 'ism', 'zsm'}) or

            ('GAIA' in catalogname and
             filtername not in {'G', 'RP', 'BP'})):

            n_transformed = cat.transform_filters(filtername,
                                                  use_all_stars=use_all_stars)
            if n_transformed == 0:
                raise ValueError(('unable to transform {:s} to {:s}'.format(
                    cat.catalogname, filtername) +
                    '; refer to LOG file for details'))
            n_transformed -= cat.reject_sources_with(
                cat['_e_'+filtername+'mag'] > mag_accuracy)

            if display and n_transformed > 0:
                print('%s transformed to %s-band: %d sources' %
                      (catalogname, filtername, n_transformed))
            if n_transformed > min_sources:
                logging.info('more than %d sources (%d), use this catalog' %
                             (min_sources, n_transformed))
                return cat
        # no transformation necessary
        else:
            # reject sources that do not have measured magnitudes
            logging.info('rejecting sources with no magnitude information')

            n_sources = n_sources - cat.reject_sources_with(
                np.isnan(cat[filtername+'mag'])) \
                - cat.reject_sources_with(
                cat['e_'+filtername+'mag'] > mag_accuracy)

            if display:
                logging.info('%d sources with accurate magnitudes in %s band' %
                             (n_sources, filtername))
                print('%d sources with accurate magnitudes in %s band' %
                      (n_sources, filtername))

            if n_sources > min_sources:
                logging.info('more than %d sources (%d), use this catalog' %
                             (min_sources, n_sources))
                return cat
            else:
                logging.info('less than %d sources (%d), try other catalog' %
                             (min_sources, n_sources))
                continue

    # end up here if none of the catalogs has n_sources > min_sources
    if display:
        print('ERROR: not enough sources in reference catalog %s (%d)' %
              (catalogname, n_sources))
    logging.warning('not enough sources in reference catalog %s (%d)' %
                    (catalogname, n_sources))
    return None


def derive_zeropoints(ref_cat, catalogs, filtername, minstars_external,
                      use_all_stars=False,
                      display=False, diagnostics=False):
    """derive zeropoint for a number of catalogs based on a reference catalog"""

    output = {'filtername': filtername, 'minstars': minstars_external,
              'zeropoints': [], 'clipping_steps': []}

    def astrometric_residual_sigmas(ref_cat, cat, match, used_indices):
        """Estimate frame-level WCS scatter from calibration-star residuals."""

        if len(used_indices) < 2:
            return np.nan, np.nan, np.nan

        used_indices = np.asarray(used_indices, dtype=int)
        ref_ra = np.asarray(match[0][3][used_indices], dtype=float)
        ref_dec = np.asarray(match[0][4][used_indices], dtype=float)
        cat_idx = np.asarray(match[1][2][used_indices], dtype=int)
        cat_ra = np.asarray(cat['ra_deg'][cat_idx], dtype=float)
        cat_dec = np.asarray(cat['dec_deg'][cat_idx], dtype=float)

        dra = (ref_ra-cat_ra)*3600*np.cos(np.deg2rad(ref_dec))
        ddec = (ref_dec-cat_dec)*3600
        # Remove the bulk catalog offset; this column reports random scatter.
        dra -= np.nanmedian(dra)
        ddec -= np.nanmedian(ddec)

        ra_sig = np.nanstd(dra, ddof=1)
        dec_sig = np.nanstd(ddec, ddof=1)
        pos_sig = np.sqrt(ra_sig**2+dec_sig**2)
        return ra_sig, dec_sig, pos_sig

    # match catalogs based on coordinates
    for cat in catalogs:

        logging.info('derive zeropoint for catalog: %s based on %s' %
                     (" | ".join([cat.catalogname, cat.origin, cat.history]),
                      " | ".join([ref_cat.catalogname, ref_cat.origin,
                                  ref_cat.history])))

        if display:
            print('zeropoint for %s:' % cat.catalogname, end=' ')

        filterkey = filtername+'mag' if filtername+'mag' \
            in ref_cat.fields else '_'+filtername+'mag'
        efilterkey = 'e_'+filtername+'mag' if 'e_'+filtername+'mag' \
                     in ref_cat.fields else '_e_'+filtername+'mag'

        # reject sources with MAG_APER/MAGERR_APER = 99 or nan

        # read this: if there is a
        # ValueError: boolean index array should have 1 dimension
        # or
        # IndexError: too many indices for array
        # pointing here, the problem is that pp_extract has not been
        # properly run using a single aperture
        # currently it seems like pp_photometry (maybe callhorizons)
        # has not finished properly

        cat.reject_sources_other_than(cat.data['MAG_'+_pp_conf.photmode] != 99)
        cat.reject_sources_other_than(cat.data['MAGERR_'
                                               + _pp_conf.photmode] != 99)
        cat.reject_sources_with(np.isnan(
            cat.data['MAG_'+_pp_conf.photmode]))
        cat.reject_sources_with(np.isnan(cat.data['MAGERR_' +
                                                  _pp_conf.photmode]))

        # add idx columns to both catalogs
        if 'idx' not in ref_cat.fields:
            ref_cat.add_field('idx',
                              list(range(ref_cat.shape[0])),
                              field_type=int)
        if 'idx' not in cat.fields:
            cat.add_field('idx', list(range(cat.shape[0])),
                          field_type=int)

        match = ref_cat.match_with(
            cat,
            match_keys_this_catalog=[
                'ra_deg', 'dec_deg'],
            match_keys_other_catalog=[
                'ra_deg', 'dec_deg'],
            extract_this_catalog=[filterkey,
                                  efilterkey,
                                  'ident',
                                  'ra_deg',
                                  'dec_deg',
                                  'idx'],
            extract_other_catalog=['MAG_'+_pp_conf.photmode,
                                   'MAGERR_' +
                                   _pp_conf.photmode,
                                   'idx'],
            tolerance=_pp_conf.pos_epsilon/3600.)

        logging.info('{:d} sources matched within {:.2f} arcsec'.format(
            len(match[0][0]), _pp_conf.pos_epsilon))

        # artificially blow up incredibly small ref_cat uncertainties
        for i in np.where(match[0][1] < 0.01):
            match[0][1][i] = 0.01

        residuals = match[0][0]-match[1][0]  # ref - instr
        residuals_sig = match[0][1]**2+match[1][1]**2
        m_idc = list(range(len(match[0][0])))

        clipping_steps = []
        #  [zeropoint, sigma, chi2, source indices in match array, match]

        # fewer than 3 reference stars -> skip this catalog
        if len(residuals) < 3:
            if display:
                print(('Warning: {:d} stars left after source matching '
                       'for frame {:s}; report instrumental magnitudes').
                      format(len(residuals), cat.catalogname))
                logging.warning(
                    ('Warning: {:d} stars left after source matching '
                     'for frame {:s}; report instrumental magnitudes').
                    format(len(residuals), cat.catalogname))
                clipping_steps = [[0, 0, 1e-10, [], [[], []]]]

                output['zeropoints'].append({'filename': cat.catalogname,
                                             'zp': np.nan,
                                             'zp_sig': np.nan,
                                             'zp_nstars': 0,
                                             'zp_usedstars': 0,
                                             'obstime': cat.obstime,
                                             'match': match,
                                             'clipping_steps': clipping_steps,
                                             'zp_idx': np.nan,
                                             'success': False})
                continue

        # if minstars is a fraction, use minstars*len(match[0][0])
        if minstars_external < 1:
            minstars = int(minstars_external*len(match[0][0]))
        else:
            minstars = int(minstars_external)

        # max 100 minstars
        if minstars > 100:
            minstars = 100

        # perform clipping to reject one outlier at a time
        zeropoint = 25  # initialize zeropoint
        popped_idc = []  # list of 'match' indices of rejected stars
        while len(residuals) >= 3:
            def fchi2(zp): return np.sum([(zp-residuals)**2/residuals_sig])
            # fchi2 = lambda zp: np.sum((zp-residuals)**2) # unweighted

            minchi2 = minimize(fchi2, zeropoint, method='Nelder-Mead')
            red_chi2 = minchi2.fun/(len(residuals)-2)
            # reduced chi2: chi2/(N-observations-N_fit_variables-1)
            zeropoint = minchi2.x[0]

            # derive weighted standard deviation
            var = np.average((residuals-zeropoint)**2,
                             weights=1/residuals_sig)
            # sigma = np.sqrt(var/(len(residuals)-1)) # weighted std of mean
            # weighted std + rms of individual sigmas
            # residuals_sig is already squared!
            sigma = np.sqrt(var + np.mean(residuals_sig))
            # sigma = np.std(residuals-zeropoint)

            clipping_steps.append([zeropoint, sigma, red_chi2, m_idc,
                                   match])

            # identify most significant outliers (not weighted) and remove them
            for repeat in range(max([1, int(len(residuals)/50)])):
                popidx = np.argmax(np.absolute(residuals
                                               - zeropoint))
                residuals = np.delete(residuals, popidx)
                residuals_sig = np.delete(residuals_sig, popidx)
                popped_idc.append(m_idc[popidx])
                m_idc = np.delete(m_idc, popidx)

        # select best-fit zeropoint based on minimum chi2
        idx = np.nanargmin([step[2] for step in clipping_steps])
        # # select best-fit zeropoint based on minimum sigma
        # idx = np.nanargmin([step[1] for step in clipping_steps])

        # reduce/increase idx to increase the number of sources until
        # minstars is met
        if len(clipping_steps[idx][3]) < minstars:
            while len(clipping_steps[idx][3]) < minstars and idx > 0:
                idx -= 1
        else:
            while len(clipping_steps[idx][3]) < minstars and idx > 0:
                idx += 1

        output['zeropoints'].append({'filename': cat.catalogname,
                                     'zp': clipping_steps[idx][0],
                                     'zp_sig': clipping_steps[idx][1],
                                     'zp_nstars': len(clipping_steps[idx][3]),
                                     'zp_usedstars': clipping_steps[idx][3],
                                     'obstime': cat.obstime,
                                     'match': match,
                                     'clipping_steps': clipping_steps,
                                     'zp_idx': idx,
                                     'success': True})

        if display:
            print('%6.3f+-%.3f (%d/%d reference stars)' %
                  (clipping_steps[idx][0], clipping_steps[idx][1],
                   len(clipping_steps[idx][3]), len(clipping_steps[0][3])))

        # write calibration catalog to file
        if conf.save_caldata:
            caldata_filename = cat.catalogname[:-5]+conf.save_caldata_suffix
            matched_ref_cat = ref_cat[match[0][5].data]
            # build `fit` column that indicates whether star is used in fit
            used_in_fit = np.zeros(len(matched_ref_cat), dtype=int)
            used_in_fit[clipping_steps[idx][3]] = 1
            matched_ref_cat.add_column(Column(used_in_fit, 'fit'))
            matched_ref_cat.remove_column('idx')
            # add instrumental magnitudes
            matched_ref_cat.add_column(
                cat['MAG_'+_pp_conf.photmode][match[1][2]])
            matched_ref_cat.add_column(
                cat['MAGERR_'+_pp_conf.photmode][match[1][2]])

            if conf.save_caldata_usedonly:
                matched_ref_cat = matched_ref_cat[used_in_fit == 1]
            matched_ref_cat.write(caldata_filename,
                                  format=conf.save_caldata_format,
                                  overwrite=True)

        # append calibrated magnitudes to catalog
        if filterkey[0] != '_':
            filterkey = '_' + filterkey
            efilterkey = '_' + efilterkey

        # add calibration data to catalog, so that it ends up in database
        if conf.caldata_in_db:
            db_ref_cat = deepcopy(ref_cat)
            # replace `idx` column in ref_cat with one that points to cat
            db_ref_cat.data.remove_column('idx')
            cat_idc = np.ones(ref_cat.shape[0], dtype=int)*-1
            for i in range(len(match[0][0])):
                cat_idc[match[0][5][i]] = match[1][2][i]
            db_ref_cat.add_field('idx', cat_idc)
            cat.data = join(cat.data, db_ref_cat.data,
                            keys='idx',
                            join_type='left')
            # remove unnecessary fields
            cat.data.remove_columns(['idx', 'ra_deg_2', 'dec_deg_2'])
            cat.data.rename_column('ra_deg_1', 'ra_deg')
            cat.data.rename_column('dec_deg_1', 'dec_deg')

        # remove columns for filterkey for matched sources
        if filterkey in cat.fields:
            cat.data.remove_column(filterkey)
            cat.data.remove_column(efilterkey)

        cat.add_fields([filterkey, efilterkey],
                       [cat['MAG_'+_pp_conf.photmode] +
                        clipping_steps[idx][0],
                        np.sqrt(cat['MAGERR_'+_pp_conf.photmode]**2 +
                                clipping_steps[idx][1]**2)],
                       ['F', 'F'])

        ra_sig, dec_sig, pos_sig = astrometric_residual_sigmas(
            ref_cat, cat, match, clipping_steps[idx][3])
        cat.add_fields(['ASTR_SIG_RA', 'ASTR_SIG_DEC', 'ASTR_SIG_POS'],
                       [np.ones(cat.shape[0])*ra_sig,
                        np.ones(cat.shape[0])*dec_sig,
                        np.ones(cat.shape[0])*pos_sig],
                       ['F', 'F', 'F'])

        # add ref_cat identifier to catalog
        cat.origin = cat.origin.strip() + ";" + ref_cat.catalogname + ";"\
            + filtername
        cat.history += 'calibrated using ' + ref_cat.history
        cat.magsys = getattr(ref_cat, 'magsystem', ref_cat.magsys)

    output['catalogs'] = catalogs
    output['ref_cat'] = ref_cat

    # output content
    #
    # { 'filtername'      : filter name,
    #   'minstars'        : requested minimum number/fraction of ref stars,
    #   'zeropoints'      : for each frame:
    #                       {'filename' : catalog name,
    #                        'zp'       : derived zeropoint,
    #                        'zp_sig'   : uncertainty,
    #                        'zp_nstars': number of reference stars available,
    #                        'zp_usedstars': numer used stars,
    #                        'obstime'  : observation midtime (JD),
    #                        'match'    : match array (see above),
    #                        'clipping_steps'  : clipping_steps (see above),
    #                        'zp_idx'   : zeropoint index
    #                       },
    #   'catalogs'        : ldac catalogs,
    #   'ref_cat'         : reference catalog
    # }
    ###

    return output


def calibrate(filenames, minstars, manfilter, manualcatalog,
              obsparam, maxflag=3,
              magzp=None, magzp_keyword=None,
              magzp_sig_keyword=None, magzp_sig=None, solar=False,
              use_all_stars=False,
              display=False, diagnostics=False):
    """
    wrapper for photometric calibration
    """

    # read in ldac data into catalogs
    catalogs, filternames = [], {}
    for filename in filenames:
        hdulist = fits.open(filename, ignore_missing_end=True)
        try:
            filtername = hdulist[0].header[obsparam['filter']]
        except KeyError:
            print('Cannot read filter name from file %s' % filename)
            logging.error('Cannot read filter name from file %s' % filename)
            return None

        # translate filtername, if available
        try:
            filtername = obsparam['filter_translations'][filtername]
        except:
            pass

        if filtername in filternames:
            filternames[filtername].append(filename)
        else:
            filternames[filtername] = [filename]
        ldac_filename = filename[:filename.find('.fit')]+'.ldac'
        cat = catalog(filename)
        if manfilter is not False and manfilter is not None:
            cat.filtername = manfilter
        else:
            cat.filtername = filtername
        if display:
            print(cat.read_ldac(ldac_filename, filename, maxflag=maxflag,
                                object_keyword=obsparam['object'],
                                exptime_keyword=obsparam['exptime'],
                                time_keyword='MIDTIMJD'),
                  '(sources, columns) read from', filename)
        else:
            cat.read_ldac(ldac_filename, filename, maxflag=maxflag,
                          object_keyword=obsparam['object'],
                          exptime_keyword=obsparam['exptime'],
                          time_keyword='MIDTIMJD')

        if cat.shape[0] > 0:
            catalogs.append(cat)
        else:
            logging.warning(('catalog {:s} is empty; '
                             'ignore').format(ldac_filename))
            if display:
                print('catalog {:s} is empty; ignore'.format(ldac_filename))

    # derive center and radius of field of view of all images
    ra_deg, dec_deg, rad_deg = skycenter(catalogs)
    logging.info('FoV center (%.7f/%+.7f) and radius (%.2f deg) derived' %
                 (ra_deg, dec_deg, rad_deg))

    # obtain photometric catalog(s) of the field based on settings in
    # setup/telescope.py and the image filter
    if manfilter is not False:
        filtername = manfilter
    else:
        if len(filternames) == 1:
            filtername = list(filternames.keys())[0]
        else:
            logging.error(('ERROR: ambiguous filters in this '
                           + 'image sample (%s)') %
                          ", ".join(['%s: %s' % (key, val)
                                     for key, val in list(filternames.items())]))
            if display:
                print('ERROR: ambiguous filters in this image sample (%s)' %
                      ", ".join(['%s: %s' % (key, val)
                                 for key, val in list(filternames.items())]))
            return []

    if manualcatalog is not None:
        preferred_catalogs = [manualcatalog]
    else:
        preferred_catalogs = obsparam['photometry_catalogs']

    manual_zps = manual_zeropoints_from_headers(
        filenames, magzp=magzp, magzp_keyword=magzp_keyword,
        magzp_sig_keyword=magzp_sig_keyword, magzp_sig=magzp_sig)

    ref_cat = None
    zp_data = None
    if filtername is not None and manual_zps is None:
        best_ref_cat = None
        best_zp_data = None
        best_score = (-1, -1)

        for catalog_idx, catalogname in enumerate(preferred_catalogs):
            try:
                candidate_ref_cat = create_photometrycatalog(
                    ra_deg, dec_deg, rad_deg, filtername, [catalogname],
                    max_sources=2e4, solar=solar,
                    use_all_stars=use_all_stars, display=display)
            except Exception as exc:
                message = ('Warning: photometric catalog {:s} failed: {}'
                           .format(catalogname, exc))
                if display:
                    print(message)
                logging.warning(message)
                continue

            if candidate_ref_cat is None:
                continue

            # Zeropoint fitting mutates the LDAC catalogs; keep each catalog
            # attempt independent so later fallbacks start from clean data.
            candidate_zp_data = derive_zeropoints(
                candidate_ref_cat, deepcopy(catalogs), filtername, minstars,
                use_all_stars=use_all_stars, display=display,
                diagnostics=diagnostics)
            successes = [frame.get('success', False)
                         for frame in candidate_zp_data['zeropoints']]
            n_success = sum(successes)
            def finite_int(value):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0
            n_stars = sum([finite_int(frame.get('zp_nstars', 0))
                           for frame in candidate_zp_data['zeropoints']])
            score = (n_success, n_stars)
            if score > best_score:
                best_ref_cat = candidate_ref_cat
                best_zp_data = candidate_zp_data
                best_score = score

            if len(successes) > 0 and all(successes):
                ref_cat = candidate_ref_cat
                zp_data = candidate_zp_data
                break

            if catalog_idx < len(preferred_catalogs)-1:
                message = (
                    'Warning: {:s} produced zeropoints for {:d}/{:d} '
                    'frames; trying next catalog').format(
                        candidate_ref_cat.catalogname, n_success,
                        len(successes))
                if display:
                    print(message)
                logging.warning(message)

        if zp_data is None:
            ref_cat = best_ref_cat
            zp_data = best_zp_data

    if ref_cat is None:
        if manual_zps is None:
            print('Skip calibration - report instrumental magnitudes')
            logging.info('Skip calibration - report instrumental magnitudes')
        else:
            if filtername is None:
                raise ValueError('manual zeropoints require a filter name')
            zps = [frame['zp'] for frame in manual_zps.values()]
            zp_errs = [frame['zp_sig'] for frame in manual_zps.values()]
            print(('use externally provided magnitude zeropoints: ' +
                   '%5.2f+-%4.2f average') %
                  (np.average(zps), np.average(zp_errs)))
            logging.info(('use externally provided magnitude zeropoints: ' +
                          '%5.2f+-%4.2f average') %
                         (np.average(zps), np.average(zp_errs)))

            catalogs = apply_manual_zeropoints(catalogs, filtername,
                                               manual_zps)

        # write calibrated database files
        logging.info('write calibrated data into database files')
        if display:
            print('write calibrated data into database files')
        for cat in catalogs:
            cat.write_database(cat.catalogname+'.db')

        logging.info('Done! ------------------------------------------------')

        zeropoints = ([manual_zps[cat.catalogname] for cat in catalogs]
                      if manual_zps is not None
                      else [{'filename': 'stuff',
                             'zp': 0,
                             'zp_sig': 0,
                             'zp_nstars': 0,
                             'zp_usedstars': 0,
                             'obstime': 0,
                             'match': 0,
                             'clipping_steps': 0,
                             'zp_idx': 0}
                            for i in range(len(filenames))])

        output = {'filtername': filtername,
                  'minstars': 0,
                  'zeropoints': zeropoints,
                  'catalogs': catalogs,
                  'ref_cat': None}

        # output content
        #
        # { 'filtername'      : filter name,
        #   'minstars'        : requested minimum number/fraction of ref stars,
        #   'zeropoints'      : for each frame:
        #                       {'filename' : catalog name,
        #                        'zp'       : derived zeropoint,
        #                        'zp_sig'   : uncertainty,
        #                        'zp_nstars': number of reference stars available,
        #                        'zp_usedstars': number of used stars,
        #                        'obstime'  : observation midtime (JD),
        #                        'match'    : match array (see above),
        #                        'clipping_steps'  : clipping_steps (see above),
        #                        'zp_idx'   : zeropoint index
        #                       },
        #   'catalogs'        : ldac catalogs,
        #   'ref_cat'         : reference catalog
        # }
        ###

        # update diagnostics website
        if diagnostics:
            if display:
                print('creating diagnostic output')
            logging.info(' ~~~~~~~~~ creating diagnostic output')
            diag.add_calibration(output, instrumental=True)

        return output

    # match catalogs and derive magnitude zeropoint
    if zp_data is None:
        zp_data = derive_zeropoints(ref_cat, catalogs, filtername,
                                    minstars,
                                    use_all_stars=use_all_stars,
                                    display=display,
                                    diagnostics=diagnostics)

    # zp_data content
    #
    # derive_zeropoints.output
    #
    ###

    catalogs = zp_data['catalogs']

    # update diagnostics website
    diag.add_calibration(zp_data)

    # write calibrated database files
    logging.info('write calibrated data into database files')
    if display:
        print('write calibrated data into database files')
    for cat in catalogs:
        cat.write_database(cat.catalogname+'.db')

    logging.info('Done! -----------------------------------------------------')

    return zp_data


if __name__ == '__main__':

    # define command line arguments
    parser = argparse.ArgumentParser(description='photometric calibration')
    parser.add_argument('-minstars', help='min number of calibration stars ' +
                        'or fraction', default=0.5)
    parser.add_argument("-cat",
                        choices=_pp_conf.allcatalogs,
                        help="use this catalog instead of default one")
    parser.add_argument("-filter", help="manual filter override")
    parser.add_argument("-maxflag", help="maximum flag for all sources",
                        default=3)
    parser.add_argument('-instrumental',
                        help='skip calibration, ' +
                             'only report instrumental magnitudes',
                        action="store_true")
    parser.add_argument('-magzp', help=('provide external magnitude zeropoint' +
                                        ' and uncertainty'),
                        nargs=2)
    parser.add_argument('-magzp_keyword',
                        help='read external magnitude zeropoints from this '
                             'FITS header keyword')
    parser.add_argument('-magzp_sig_keyword',
                        help='read magnitude zeropoint uncertainties from '
                             'this FITS header keyword')
    parser.add_argument('-magzp_sig',
                        help='fixed magnitude zeropoint uncertainty to use '
                             'with -magzp_keyword')
    parser.add_argument('-solar',
                        help='restrict to solar-color stars',
                        action="store_true", default=False)
    parser.add_argument('-use_all_stars',
                        help='ignore all quality checks and use all stars',
                        action="store_true", default=False)
    parser.add_argument('images', help='images to process', nargs='+')
    args = parser.parse_args()
    minstars = float(args.minstars)
    manfilter = args.filter
    maxflag = int(float(args.maxflag))
    manualcatalog = args.cat
    instrumental = args.instrumental
    man_magzp = args.magzp
    magzp_keyword = args.magzp_keyword
    magzp_sig_keyword = args.magzp_sig_keyword
    magzp_sig = args.magzp_sig
    solar = args.solar
    use_all_stars = args.use_all_stars
    filenames = args.images

    # manfilter: None: instrumental magnitudes, False: no manfilter provided
    if instrumental:
        manfilter = None
    else:
        if manfilter is None:
            manfilter = False

    # check if input filenames is actually a list
    if len(filenames) == 1:
        if filenames[0].find('.lst') > -1 or filenames[0].find('.list') > -1:
            filenames = [filename[:-1] for filename in open(filenames[0], 'r').
                         readlines()]

    # obtain telescope information
    hdulist = fits.open(filenames[0], ignore_missing_end=True)
    try:
        telescope = hdulist[0].header['TEL_KEYW']
    except KeyError:
        print('ERROR: cannot find telescope keyword in image header;',
              'has this image run through pp_prepare?')
        sys.exit(0)
    obsparam = _pp_conf.telescope_parameters[telescope]

    if man_magzp is not None:
        man_magzp = (float(man_magzp[0]), float(man_magzp[1]))
    if magzp_sig is not None:
        magzp_sig = float(magzp_sig)

    calibration = calibrate(filenames, minstars, manfilter,
                            manualcatalog, obsparam, maxflag=maxflag,
                            magzp=man_magzp,
                            magzp_keyword=magzp_keyword,
                            magzp_sig_keyword=magzp_sig_keyword,
                            magzp_sig=magzp_sig,
                            solar=solar,
                            use_all_stars=use_all_stars,
                            display=True, diagnostics=conf.diagnostics)
