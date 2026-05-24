#!/usr/bin/env python3

""" PPTOOL_MPCREPORT - produce a file for submission of asteroid astrometry
                       to the Minor Planet center
    v1.0: 2017-05-25, mommermiscience@gmail.com
"""
from __future__ import print_function

import argparse
import os
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

# pipeline-specific modules
import _pp_conf
from catalog import *
import toolbox

# only import if Python3 is used
if sys.version_info > (3, 0):
    from builtins import str

# setup logging
logging.basicConfig(filename=_pp_conf.log_filename,
                    level=_pp_conf.log_level,
                    format=_pp_conf.log_formatline,
                    datefmt=_pp_conf.log_datefmt)


def resolve_fitsfilename(photometryfile, catalog_token):
    """Find the FITS image that produced a photometry row."""

    photometrydir = os.path.dirname(os.path.abspath(photometryfile))
    candidates = [
        catalog_token,
        catalog_token+'.fits',
        catalog_token.replace('.ldac', '.fits'),
        os.path.join(photometrydir, catalog_token),
        os.path.join(photometrydir, catalog_token+'.fits'),
        os.path.join(photometrydir, catalog_token.replace('.ldac', '.fits')),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    fits_candidates = [os.path.join(photometrydir, filename)
                       for filename in os.listdir(photometrydir)
                       if filename.lower().endswith(('.fits', '.fit',
                                                     '.fts'))]
    if len(fits_candidates) == 1:
        return fits_candidates[0]

    raise FileNotFoundError('cannot identify FITS file for %s row %s' %
                            (photometryfile, catalog_token))


def write_observations(outf, filename, targetname):
    """Append MPC 80-column observations from one PP photometry file."""

    first_observation = next(obs for obs in open(filename, 'r').readlines()
                             if obs.strip() and not
                             obs.lstrip().startswith('#'))
    fitsfilename = resolve_fitsfilename(filename,
                                        first_observation.split()[0])

    # read telescope and filter information from fits headers
    instrument = None
    hdulist = fits.open(fitsfilename, ignore_missing_end=True,
                        verify='silentfix')
    header = hdulist[0].header
    instrument = hdulist[0].header['TEL_KEYW']

    if instrument is None == 0:
        raise KeyError('cannot identify telescope/instrument; '
                       'please update '
                       '_pp_conf.instrument_keys accordingly')

    # assign telescope parameters (telescopes.py)
    obsparam = _pp_conf.telescope_parameters[instrument]

    observatory_code = obsparam['observatory_code']

    # loop over photometry file
    for obs in open(filename, 'r').readlines():
        if not obs.strip() or obs.lstrip().startswith('#'):
            continue

        obs = obs.split()

        # convert observation midtime
        date = Time(float(obs[1]), format='jd', scale='utc').to_datetime()
        date = '{0:4d} {1:02d} {2:08.5f} '.format(date.year, date.month,
                                                  date.day+(date.hour/24 +
                                                            date.minute/1440 +
                                                            date.second/86400))

        filtername = obs[16]
        if filtername == '-':
            filtername = 'C'

        pos = SkyCoord(ra=float(obs[4])*u.degree,
                       dec=float(obs[5])*u.degree, frame='icrs')
        ra = pos.ra.hms
        dec = pos.dec.signed_dms

        mag = float(obs[2])

        outf.write(('     {:7s}'.format(targetname)) +
                   (' ') +  # not a discovery
                   (' ') +  # no note1
                   ('C') +  # note2: CCD observation
                   ('{0:17s}'.format(date)) +
                   ('{0:02d} {1:02d} {2:05.2f} '.format(int(ra.h),
                                                        int(ra.m),
                                                        ra.s)) +
                   ('{0:+03d} {1:02d} {2:04.1f}'.format(int(dec.sign*dec.d),
                                                        int(dec.m),
                                                        dec.s)) +
                   ('         ') +  # blank
                   ('{0:5.1f} {1:1s}'.format(mag, filtername)) +
                   ('      ') +  # blank
                   ('{0:3s}'.format(observatory_code)) +
                   ('\n'))


if __name__ == '__main__':

    # command line arguments
    parser = argparse.ArgumentParser(description='prepare MPC submission')
    parser.add_argument('photometryfile', nargs='+',
                        help='photometry file(s) to process')
    parser.add_argument('targetname',
                        help='target identifier (<=7 chars)')
    parser.add_argument('-output', default='mpc_astrometry.dat',
                        help='output file')

    args = parser.parse_args()
    filenames = args.photometryfile
    targetname = args.targetname
    if len(targetname) > 7:
        targetname = targetname[:7]

    outf = open(args.output, 'w')

    # write header
    outf.write('COD #observatory_code: 3 char#\n'
               'CON #contact person: J. Doe#\n'
               'CON [#email address#]\n'
               'OBS #observer name: J. Doe#\n'
               'MEA #measurer name: J. Doe#\n'
               'TEL #telescope name, aperture, camera#\n'
               'NET #astrometry catalog#\n'
               'BND #calibration band#\n'
               'NUM #number of observations submitted#\n'
               'COM #comments#\n'
               'ACK #email address\n')

    for filename in filenames:
        write_observations(outf, filename, targetname)

    outf.close()
