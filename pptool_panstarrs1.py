#!/usr/bin/env python3

"""Workflow helper for Pan-STARRS1/GPC1 CADC image products."""

from __future__ import print_function

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
from contextlib import contextmanager

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS


TELESCOPE_KEY = 'PANSTARRS1'
BACKUP_NAME = 'originals_backup.tar'
MANIFEST_NAME = 'panstarrs1_workflow_manifest.json'
WORKING_PREFIX = 'ps1'
TARGET_TRANSLATION = str.maketrans(' /()', '____')
DEFAULT_XY = (283.0, 337.0)
DEFAULT_PIXEL_ORIGIN = 1


@contextmanager
def working_directory(path):
    """Temporarily run PP in the image directory."""

    old_cwd = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old_cwd)


def repo_root():
    """Return the photometrypipeline checkout that contains this script."""

    return Path(__file__).resolve().parent


def configure_pp_environment():
    """Set the environment needed before importing PP modules."""

    root = repo_root()
    os.environ['PHOTPIPEDIR'] = str(root)
    bin_dir = str(Path(sys.executable).resolve().parent)
    path_parts = os.environ.get('PATH', '').split(os.pathsep)
    if bin_dir not in path_parts:
        os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def discover_raw_images(image_dir):
    """Find source Pan-STARRS1 full-image FITS products in one directory."""

    return sorted(path for path in image_dir.iterdir()
                  if path.is_file()
                  and path.name.endswith('_image.fits')
                  and not path.name.startswith(WORKING_PREFIX+'_'))


def discover_cutouts(image_dir):
    """Find Pan-STARRS1 cutout FITS products used for target positions."""

    return sorted(path for path in image_dir.iterdir()
                  if path.is_file() and path.name.endswith('_cutout.fits'))


def backup_candidates(image_dir, raw_images, cutouts):
    """Return source files and sidecars that should be archived."""

    sidecars = ['downloads.jsonl', 'download_plan.sh',
                'download_negative_delta_mag.sh', 'download_plan_specifici.sh']
    candidates = list(raw_images) + list(cutouts)
    for name in sidecars:
        path = image_dir / name
        if path.exists():
            candidates.append(path)
    return sorted(set(candidates))


def create_originals_backup(image_dir, raw_images, cutouts, refresh=False):
    """Create or reuse the tar backup of the raw source files."""

    backup_path = image_dir / BACKUP_NAME
    if backup_path.exists() and not refresh:
        return {'path': str(backup_path), 'created': False}

    if backup_path.exists():
        backup_path.unlink()

    with tarfile.open(str(backup_path), 'w') as tar:
        for path in backup_candidates(image_dir, raw_images, cutouts):
            tar.add(str(path), arcname=path.name)

    return {'path': str(backup_path), 'created': True}


def science_hdu_index(hdulist):
    """Locate the first image-bearing HDU in a FITS file."""

    for idx, hdu in enumerate(hdulist):
        data = getattr(hdu, 'data', None)
        if data is not None and len(data.shape) >= 2:
            return idx
    raise ValueError('no image-bearing HDU found')


def strip_extension_only_cards(header):
    """Remove cards that should not be copied into a primary image HDU."""

    cleaned = header.copy()
    remove_exact = {
        'XTENSION', 'EXTNAME', 'PCOUNT', 'GCOUNT', 'CHECKSUM', 'DATASUM',
        'THEAP', 'TFIELDS', 'BZERO', 'BSCALE', 'BLANK',
    }
    for key in list(cleaned.keys()):
        if key in remove_exact or key.startswith('Z'):
            try:
                cleaned.remove(key, remove_all=True, ignore_missing=True)
            except TypeError:
                cleaned.remove(key)
    return cleaned


def normalize_filter(value):
    """Translate common Pan-STARRS1 filter labels to PP filter names."""

    text = str(value).strip()
    if text.endswith('.00000'):
        text = text.split('.', 1)[0]
    mapping = {'gp1': 'g', 'rp1': 'r', 'ip1': 'i',
               'zp1': 'z', 'yp1': 'y'}
    return mapping.get(text, text)


def filter_from_header_or_name(header, path):
    """Read the PS1 filter from a FITS header, falling back to the filename."""

    for key in ('FPA.FILTER', 'FILTER'):
        if key in header:
            return normalize_filter(header[key])

    match = re.search(r'_([grizy])_exp', path.name)
    if match:
        return match.group(1)

    match = re.search(r'_([grizy]p1)_', path.name)
    if match:
        return normalize_filter(match.group(1))

    raise KeyError('cannot identify Pan-STARRS1 filter for %s' % path)


def midtime_values(header):
    """Return MJD start, JD midtime, and ISO start time from PS1 headers."""

    mjd_obs = float(header['MJD-OBS'])
    exptime = float(header.get('EXPTIME', header.get('EXPOSURE', 0.0)))
    mid_mjd = mjd_obs + exptime/2.0/86400.0
    date_obs = Time(mjd_obs, format='mjd', scale='utc').isot
    mid_jd = Time(mid_mjd, format='mjd', scale='utc').jd
    return mjd_obs, mid_mjd, mid_jd, date_obs


def repair_nonfinite_pixels(data):
    """Replace NaN/Inf pixels with the finite median sky level."""

    image = np.asarray(data, dtype=np.float32).copy()
    bad = ~np.isfinite(image)
    n_bad = int(np.count_nonzero(bad))
    if n_bad == 0:
        return image, n_bad, None

    good = image[~bad]
    fill_value = float(np.median(good)) if good.size else 0.0
    image[bad] = fill_value
    return image, n_bad, fill_value


def convert_to_pp_fits(raw_path, working_path, original_index,
                       target, magzp_sig):
    """Flatten a PS1 compressed image-extension FITS into a PP-ready FITS."""

    with fits.open(str(raw_path), memmap=False,
                   ignore_missing_end=True) as hdulist:
        hdulist.verify('fix')
        hdu_index = science_hdu_index(hdulist)
        image_hdu = hdulist[hdu_index]
        header = strip_extension_only_cards(image_hdu.header)
        image, n_bad, fill_value = repair_nonfinite_pixels(image_hdu.data)

        filt = filter_from_header_or_name(header, raw_path)
        mjd_obs, mid_mjd, mid_jd, date_obs = midtime_values(header)

        header['PPINSTRU'] = (TELESCOPE_KEY, 'PP telescope override')
        header['TELESCOP'] = ('Pan-STARRS1', 'PP normalized telescope')
        header['INSTRUME'] = ('GPC1', 'PP normalized instrument')
        header['FILTER'] = (filt, 'PP normalized filter')
        header['OBJECT'] = (target, 'PP target name')
        header['DATE-OBS'] = (date_obs, 'exposure start UTC from MJD-OBS')
        header['MJD-MID'] = (mid_mjd, 'exposure midtime MJD')
        header['MIDTIMJD'] = (mid_jd, 'exposure midtime JD')
        header['CELL.XBIN'] = (header.get('CELL.XBIN', 1), 'PP x binning')
        header['CELL.YBIN'] = (header.get('CELL.YBIN', 1), 'PP y binning')
        header['RA'] = (header.get('CRVAL1'), 'PP copied WCS reference RA')
        header['DEC'] = (header.get('CRVAL2'), 'PP copied WCS reference Dec')
        header['ORIGFILE'] = (raw_path.name[:68], 'source FITS filename')
        header['ORIGEXT'] = (hdu_index, 'source FITS extension')
        header['PPSEQ'] = (original_index, 'Pan-STARRS1 workflow index')
        header['NANNPIX'] = (n_bad, 'non-finite pixels replaced for PP')
        if fill_value is not None:
            header['NANFILL'] = (fill_value, 'replacement value for NaNs')

        if 'FPA.ZP' in header:
            # PS1 warp pixels are total counts, while FPA.ZP is calibrated for
            # count rates.  PP measures total-count instrumental magnitudes.
            exptime = float(header.get('EXPTIME',
                                       header.get('EXPOSURE', 0.0)) or 0.0)
            magzp = float(header['FPA.ZP'])
            if exptime > 0:
                magzp += 2.5*np.log10(exptime)
            header['MAGZP'] = (magzp, 'PS1 exposure-normalized zeropoint')
            header['MAGZPSIG'] = (float(magzp_sig), 'assumed zeropoint sigma')

        fits.PrimaryHDU(data=image, header=header).writeto(
            str(working_path), overwrite=True, output_verify='silentfix')

    with fits.open(str(working_path), ignore_missing_end=True) as hdulist:
        header = hdulist[0].header
        return {
            'raw': str(raw_path),
            'working': str(working_path),
            'working_name': working_path.name,
            'source_hdu': hdu_index,
            'image_index': original_index - 1,
            'filter': header.get('FILTER'),
            'mjd_obs': float(header['MJD-OBS']),
            'midtimjd': float(header['MIDTIMJD']),
            'date_obs': header.get('DATE-OBS'),
            'exptime': float(header['EXPTIME']),
            'magzp': header.get('MAGZP'),
            'magzp_sig': header.get('MAGZPSIG'),
            'nan_pixels_replaced': int(header.get('NANNPIX', 0)),
            'nan_fill': header.get('NANFILL'),
        }


def prepare_working_fits(image_dir, raw_images, target, magzp_sig):
    """Create fresh PP-ready FITS files from the raw PS1 images."""

    records = []
    for idx, raw_path in enumerate(raw_images, start=1):
        with fits.open(str(raw_path), memmap=False,
                       ignore_missing_end=True) as hdulist:
            filt = filter_from_header_or_name(
                hdulist[science_hdu_index(hdulist)].header, raw_path)
        working_path = image_dir / ('%s_%s_%04d.fits' %
                                    (WORKING_PREFIX, filt, idx))
        records.append(convert_to_pp_fits(raw_path, working_path, idx,
                                          target, magzp_sig))
    return records


def derive_cutout_position(cutout_path, xy=DEFAULT_XY,
                           pixel_origin=DEFAULT_PIXEL_ORIGIN):
    """Derive the target sky position from one PS1 cutout at pixel x, y."""

    with fits.open(str(cutout_path), memmap=False,
                   ignore_missing_end=True) as hdulist:
        header = hdulist[0].header
        wcs = WCS(header)
        ra_deg, dec_deg = wcs.all_pix2world([[float(xy[0]), float(xy[1])]],
                                            int(pixel_origin))[0]
        mjd_obs, mid_mjd, mid_jd, date_obs = midtime_values(header)
        return {
            'cutout': str(cutout_path),
            'cutout_name': cutout_path.name,
            'filter': filter_from_header_or_name(header, cutout_path),
            'mjd_obs': mjd_obs,
            'mjd_mid': mid_mjd,
            'midtimjd': mid_jd,
            'date_obs': date_obs,
            'x': float(xy[0]),
            'y': float(xy[1]),
            'pixel_origin': int(pixel_origin),
            'ra_deg': float(ra_deg),
            'dec_deg': float(dec_deg),
        }


def derive_cutout_positions(cutouts, xy=DEFAULT_XY,
                            pixel_origin=DEFAULT_PIXEL_ORIGIN):
    """Derive target sky positions from all matching PS1 cutouts."""

    return [derive_cutout_position(path, xy=xy, pixel_origin=pixel_origin)
            for path in sorted(cutouts)]


def attach_positions_to_records(records, positions, tolerance_seconds=5.0):
    """Match cutout-derived target coordinates to PP-ready full images."""

    tolerance_days = tolerance_seconds / 86400.0
    remaining = list(positions)
    matched_records = []

    for record in records:
        candidates = [pos for pos in remaining
                      if pos['filter'] == record['filter']]
        if not candidates:
            raise ValueError('no cutout position for %s' %
                             record['working_name'])
        best = min(candidates,
                   key=lambda pos: abs(pos['mjd_obs']-record['mjd_obs']))
        dt = abs(best['mjd_obs']-record['mjd_obs'])
        if dt > tolerance_days:
            raise ValueError('nearest cutout for %s is %.2f seconds away' %
                             (record['working_name'], dt*86400.0))
        remaining.remove(best)
        merged = dict(record)
        merged['target_position'] = best
        matched_records.append(merged)

    return matched_records


def positions_filename(target, filt=None):
    """Return a stable PP positions filename for a target/filter."""

    ident = target.translate(TARGET_TRANSLATION)
    if filt is None:
        return 'positions_%s.dat' % ident
    return 'positions_%s_%s.dat' % (ident, filt)


def write_positions_file(image_dir, records, target, filt=None):
    """Write PP -positions input for the selected records."""

    path = image_dir / positions_filename(target, filt)
    ident = target.translate(TARGET_TRANSLATION)
    with path.open('w') as handle:
        for record in records:
            pos = record['target_position']
            handle.write('%s %.10f %.10f %.8f %s\n' %
                         (record['working_name'], pos['ra_deg'],
                          pos['dec_deg'], record['midtimjd'], ident))
    return path


def target_photometry_filename(target):
    """Return PP's photometry filename for a target name."""

    return 'photometry_%s.dat' % target.translate(TARGET_TRANSLATION)


def filter_photometry_filename(target, filt):
    """Return the per-filter target photometry filename."""

    return 'photometry_%s_%s.dat' % (target.translate(TARGET_TRANSLATION),
                                    filt)


def run_pp_pipeline(image_dir, records, target, posfile,
                    fixed_aprad=0.0, rejectionfilter='pos'):
    """Run pp_run on one filter's PP-ready images with kept WCS."""

    configure_pp_environment()
    import pp_run

    working_names = [record['working_name'] for record in records]
    with working_directory(image_dir):
        pp_run.run_the_pipeline(
            working_names, target, None, float(fixed_aprad), 'high',
            False, False, False, True, telescope=TELESCOPE_KEY,
            posfile=Path(posfile).name, magzp_keyword='MAGZP',
            magzp_sig_keyword='MAGZPSIG', rejectionfilter=rejectionfilter)


def remove_stale_target_output(image_dir, target):
    """Remove PP target photometry outputs before a fresh run."""

    for path in image_dir.glob('photometry_%s*.dat' %
                               target.translate(TARGET_TRANSLATION)):
        path.unlink()


def copy_filter_photometry(image_dir, target, filt):
    """Copy PP's canonical target output to a filter-specific filename."""

    source = image_dir / target_photometry_filename(target)
    if not source.exists():
        return None
    destination = image_dir / filter_photometry_filename(target, filt)
    shutil.copyfile(str(source), str(destination))
    return destination


def split_photometry_file(path):
    """Split a PP photometry file into header, active rows, and footer."""

    header = []
    rows = []
    footer = []
    seen_data = False

    with Path(path).open() as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith('#'):
                seen_data = True
                rows.append(line)
            elif seen_data:
                footer.append(line)
            else:
                header.append(line)

    return header, rows, footer


def all_filters_photometry_filename(target):
    """Return the explicit all-filter target photometry filename."""

    return 'photometry_%s_all_filters.dat' % target.translate(
        TARGET_TRANSLATION)


def combine_filter_photometry(image_dir, target, outputs):
    """Combine per-filter PP photometry files into all-filter products."""

    paths = [Path(output['photometry_file']) for output in outputs
             if output.get('photometry_file')]
    paths = [path for path in paths if path.exists()]
    if not paths:
        return []

    first_header, first_rows, first_footer = split_photometry_file(paths[0])
    rows = list(first_rows)
    for path in paths[1:]:
        rows.extend(split_photometry_file(path)[1])

    combined_text = ''.join(first_header + rows + first_footer)
    destinations = [
        image_dir / target_photometry_filename(target),
        image_dir / all_filters_photometry_filename(target),
    ]
    for destination in destinations:
        destination.write_text(combined_text)

    return [str(destination) for destination in destinations]


def run_pp_by_filter(image_dir, records, target, fixed_aprad=0.0,
                     rejectionfilter='pos'):
    """Run PP once per filter and preserve one photometry file per filter."""

    outputs = []
    filters = sorted(set(record['filter'] for record in records))
    for filt in filters:
        filter_records = [record for record in records
                          if record['filter'] == filt]
        posfile = write_positions_file(image_dir, filter_records,
                                       target, filt=filt)
        run_pp_pipeline(image_dir, filter_records, target, posfile,
                        fixed_aprad=fixed_aprad,
                        rejectionfilter=rejectionfilter)
        photometry_path = copy_filter_photometry(image_dir, target, filt)
        outputs.append({
            'filter': filt,
            'positions_file': str(posfile),
            'photometry_file': (str(photometry_path)
                                if photometry_path is not None else None),
            'working_names': [record['working_name']
                              for record in filter_records],
        })
    return outputs


def write_manifest(image_dir, manifest):
    """Write the workflow manifest JSON."""

    path = image_dir / MANIFEST_NAME
    with path.open('w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return path


def build_workflow(image_dir, target, xy=DEFAULT_XY,
                   pixel_origin=DEFAULT_PIXEL_ORIGIN, magzp_sig=0.03,
                   refresh_backup=False):
    """Prepare PP-ready FITS files and position sidecars."""

    image_dir = Path(image_dir).expanduser().resolve()
    raw_images = discover_raw_images(image_dir)
    cutouts = discover_cutouts(image_dir)
    if not raw_images:
        raise IOError('no Pan-STARRS1 full-image FITS files found in %s' %
                      image_dir)
    if not cutouts:
        raise IOError('no Pan-STARRS1 cutout FITS files found in %s' %
                      image_dir)

    backup = create_originals_backup(image_dir, raw_images, cutouts,
                                     refresh=refresh_backup)
    records = prepare_working_fits(image_dir, raw_images, target, magzp_sig)
    positions = derive_cutout_positions(cutouts, xy=xy,
                                        pixel_origin=pixel_origin)
    records = attach_positions_to_records(records, positions)
    all_positions_path = write_positions_file(image_dir, records, target)

    manifest = {
        'telescope': TELESCOPE_KEY,
        'target': target,
        'image_dir': str(image_dir),
        'backup': backup,
        'raw_images': [str(path) for path in raw_images],
        'cutouts': [str(path) for path in cutouts],
        'xy': [float(xy[0]), float(xy[1])],
        'pixel_origin': int(pixel_origin),
        'magzp_sig': float(magzp_sig),
        'positions_file': str(all_positions_path),
        'records': records,
    }
    manifest_path = write_manifest(image_dir, manifest)
    manifest['manifest_path'] = str(manifest_path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='prepare and run PP for Pan-STARRS1 CADC products')
    parser.add_argument('image_dir', help='directory with PS1 FITS products')
    parser.add_argument('-target', default='2014_FU61',
                        help='target name to use in PP outputs')
    parser.add_argument('--x', type=float, default=DEFAULT_XY[0],
                        help='target x pixel in the cutouts')
    parser.add_argument('--y', type=float, default=DEFAULT_XY[1],
                        help='target y pixel in the cutouts')
    parser.add_argument('--pixel-origin', type=int,
                        default=DEFAULT_PIXEL_ORIGIN,
                        help='pixel origin for WCS conversion; DS9 is 1')
    parser.add_argument('--magzp-sig', type=float, default=0.03,
                        help='assumed uncertainty for FPA.ZP header ZPs')
    parser.add_argument('--fixed-aprad', type=float, default=0.0,
                        help='fixed PP aperture radius; 0 uses curve of growth')
    parser.add_argument('--reject', default='pos',
                        help='PP target rejection schema, e.g. pos or none')
    parser.add_argument('--refresh-backup', action='store_true',
                        help='recreate originals_backup.tar')
    parser.add_argument('--prepare-only', action='store_true',
                        help='write PP-ready FITS/positions but do not run PP')

    args = parser.parse_args(argv)

    manifest = build_workflow(
        args.image_dir, args.target, xy=(args.x, args.y),
        pixel_origin=args.pixel_origin, magzp_sig=args.magzp_sig,
        refresh_backup=args.refresh_backup)

    if not args.prepare_only:
        image_dir = Path(args.image_dir).expanduser().resolve()
        remove_stale_target_output(image_dir, args.target)
        outputs = run_pp_by_filter(
            image_dir, manifest['records'], args.target,
            fixed_aprad=args.fixed_aprad, rejectionfilter=args.reject)
        manifest['pp_outputs'] = outputs
        manifest['combined_photometry_files'] = combine_filter_photometry(
            image_dir, args.target, outputs)
        from pptool_pp_cutouts import record_pp_cutouts
        record_pp_cutouts(
            manifest, image_dir, args.target,
            photometry_file=(manifest['combined_photometry_files'][0]
                             if manifest['combined_photometry_files']
                             else None))
        manifest_path = write_manifest(image_dir, manifest)
        manifest['manifest_path'] = str(manifest_path)

    print(json.dumps({
        'manifest_path': manifest['manifest_path'],
        'positions_file': manifest['positions_file'],
        'n_records': len(manifest['records']),
        'filters': sorted(set(record['filter']
                              for record in manifest['records'])),
        'pp_outputs': manifest.get('pp_outputs', []),
        'combined_photometry_files': manifest.get(
            'combined_photometry_files', []),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
