#!/usr/bin/env python3

"""Workflow helper for CFHT MegaCam click-points photometry."""

from __future__ import print_function

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import tarfile
from contextlib import contextmanager

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS


TELESCOPE_KEY = 'CFHTMEGAPRIME'
BACKUP_NAME = 'originals_backup.tar'
MANIFEST_NAME = 'cfht_megacam_workflow_manifest.json'
WORKING_PREFIX = 'cfht'
TARGET_TRANSLATION = str.maketrans(' /()', '____')
DEFAULT_PIXEL_ORIGIN = 0
DEFAULT_MAGZP_SIG = 0.03


@contextmanager
def working_directory(path):
    """Temporarily run PP in one image directory."""

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


def target_to_filename(target):
    """Return PP's target-safe filename token."""

    return target.translate(TARGET_TRANSLATION)


def target_photometry_filename(target):
    """Return PP's canonical photometry filename for a target."""

    return 'photometry_%s.dat' % target_to_filename(target)


def all_filters_photometry_filename(target):
    """Return the explicit all-filter photometry filename."""

    return 'photometry_%s_all_filters.dat' % target_to_filename(target)


def parse_click_image_name(image_name):
    """Parse a CFHT click PNG name into image and chip identifiers."""

    match = re.search(r'_(?P<image_id>\d+p)_chip(?P<chip>\d+)'
                      r'(?:-ccd(?P<ccd>\d+))?', Path(image_name).name)
    if match is None:
        raise ValueError('cannot parse CFHT image id/chip from %s' %
                         image_name)
    return {
        'image_id': match.group('image_id'),
        'chip_hdu': int(match.group('chip')),
        'ccd': (int(match.group('ccd'))
                if match.group('ccd') is not None else None),
    }


def resolve_raw_path(base_dir, image_id):
    """Find the raw CADC MegaCam FITS file for one image id."""

    candidates = [
        base_dir / image_id / (image_id + '.fits.fz'),
        base_dir / image_id / (image_id + '.fits'),
        base_dir / (image_id + '.fits.fz'),
        base_dir / (image_id + '.fits'),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError('cannot find raw FITS for %s below %s' %
                            (image_id, base_dir))


def resolve_cutout_path(base_dir, image_name):
    """Find the cutout FITS file corresponding to one clicked PNG."""

    parsed = parse_click_image_name(image_name)
    stem = Path(image_name).with_suffix('.fits').name
    candidates = [
        base_dir / parsed['image_id'] / 'FITS' / stem,
        base_dir / 'fits_thumbs_all' / stem,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError('cannot find cutout FITS for %s below %s' %
                            (image_name, base_dir))


def read_clicked_rows(click_points_csv):
    """Load clicked rows from a click_points.csv file."""

    csv_path = Path(click_points_csv).expanduser().resolve()
    base_dir = csv_path.parent
    rows = []
    with csv_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if str(row.get('status', '')).strip().lower() != 'clicked':
                continue
            parsed = parse_click_image_name(row['image'])
            click = {
                'row_index': index,
                'image': row['image'],
                'image_id': parsed['image_id'],
                'chip_hdu': parsed['chip_hdu'],
                'ccd': parsed['ccd'],
                'x': float(row['x']),
                'y': float(row['y']),
                'width': float(row['width']),
                'height': float(row['height']),
                'status': row.get('status', ''),
            }
            click['raw_path'] = str(resolve_raw_path(
                base_dir, click['image_id']))
            click['cutout_path'] = str(resolve_cutout_path(
                base_dir, click['image']))
            rows.append(click)
    if not rows:
        raise ValueError('no clicked rows found in %s' % csv_path)
    return rows


def normalize_filter(value):
    """Translate common CFHT MegaCam filter labels to PP filter names."""

    text = str(value).strip()
    mapping = {
        'u.MP9302': 'u',
        'g.MP9401': 'g',
        'r.MP9602': 'r',
        'i.MP9701': 'i',
        'z.MP9801': 'z',
        'U.MP9302': 'u',
        'G.MP9401': 'g',
        'R.MP9602': 'r',
        'I.MP9701': 'i',
        'Z.MP9801': 'z',
        'u': 'u',
        'g': 'g',
        'r': 'r',
        'i': 'i',
        'z': 'z',
    }
    if text in mapping:
        return mapping[text]
    if len(text) > 1 and text[1:].upper().startswith('.MP'):
        first = text[0].lower()
        if first in 'ugriz':
            return first
    return text


def filter_from_header(header):
    """Read and normalize a MegaCam filter header value."""

    if 'FILTER' not in header:
        raise KeyError('FILTER keyword is missing from CFHT header')
    return normalize_filter(header['FILTER'])


def first_image_hdu_index(hdulist):
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


def merged_image_header(hdulist, hdu_index):
    """Merge primary metadata with the selected chip extension header."""

    header = strip_extension_only_cards(hdulist[0].header)
    extension_header = strip_extension_only_cards(hdulist[hdu_index].header)
    header.extend(extension_header, update=True, useblanks=False)
    return header


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


def midtime_values(header):
    """Return MJD start, MJD midpoint, JD midpoint, and ISO start time."""

    if 'MJD-OBS' in header:
        mjd_obs = float(header['MJD-OBS'])
    else:
        mjd_obs = float(header['MJDATE'])
    exptime = float(header.get('EXPTIME', 0.0))
    mid_mjd = mjd_obs + exptime / 2.0 / 86400.0
    date_obs = Time(mjd_obs, format='mjd', scale='utc').isot
    mid_jd = Time(mid_mjd, format='mjd', scale='utc').jd
    return mjd_obs, mid_mjd, mid_jd, date_obs


def verify_requested_chip(hdulist, click):
    """Ensure the clicked chip maps to the requested FITS extension."""

    hdu_index = int(click['chip_hdu'])
    if hdu_index >= len(hdulist):
        raise IndexError('chip%d is not present in %s-HDU FITS file' %
                         (hdu_index, len(hdulist)))
    hdu = hdulist[hdu_index]
    if getattr(hdu, 'data', None) is None:
        raise ValueError('chip%d does not contain image data' % hdu_index)
    ccd = click.get('ccd')
    if ccd is not None:
        extname = str(hdu.header.get('EXTNAME', '')).strip().lower()
        if extname and extname != 'ccd%d' % int(ccd):
            raise ValueError('chip%d expected ccd%d but found %s' %
                             (hdu_index, int(ccd), extname))
    return hdu_index


def raw_chip_filter(raw_path, click):
    """Read the normalized filter for a clicked chip."""

    with fits.open(str(raw_path), memmap=False,
                   ignore_missing_end=True) as hdulist:
        hdu_index = verify_requested_chip(hdulist, click)
        return filter_from_header(hdulist[hdu_index].header)


def working_filename(click, filt, sequence):
    """Return a short PP-ready filename for one extracted chip."""

    return '%s_%s_%04d_%s_chip%02d.fits' % (
        WORKING_PREFIX, filt, sequence, click['image_id'],
        int(click['chip_hdu']))


def convert_to_pp_fits(raw_path, working_path, click, target, sequence,
                       magzp_sig=DEFAULT_MAGZP_SIG):
    """Extract one MegaCam chip into a single-extension PP-ready FITS."""

    raw_path = Path(raw_path)
    working_path = Path(working_path)
    with fits.open(str(raw_path), memmap=False,
                   ignore_missing_end=True) as hdulist:
        hdulist.verify('fix')
        hdu_index = verify_requested_chip(hdulist, click)
        image_hdu = hdulist[hdu_index]
        header = merged_image_header(hdulist, hdu_index)
        image, n_bad, fill_value = repair_nonfinite_pixels(image_hdu.data)

        original_filter = str(header.get('FILTER', '')).strip()
        filt = normalize_filter(original_filter)
        mjd_obs, mid_mjd, mid_jd, date_obs = midtime_values(header)

        if 'PHOT_C' not in header:
            raise KeyError('PHOT_C zeropoint not found in %s chip%d' %
                           (raw_path, hdu_index))
        zp_sig = float(header['PHOT_CS']) if 'PHOT_CS' in header else float(
            magzp_sig)

        header['PPINSTRU'] = (TELESCOPE_KEY, 'PP telescope override')
        header['TELESCOP'] = (header.get('TELESCOP', 'CFHT'),
                              'PP normalized telescope')
        header['INSTRUME'] = (header.get('INSTRUME', 'MegaPrime'),
                              'PP normalized instrument')
        header['ORIGFILT'] = (original_filter, 'source CFHT filter')
        header['FILTER'] = (filt, 'PP normalized filter')
        header['OBJECT'] = (target, 'PP target name')
        header['DATE-OBS'] = (date_obs, 'exposure start UTC from MJD-OBS')
        if 'MJD-OBS' not in header:
            header['MJD-OBS'] = (mjd_obs, 'exposure start MJD')
        header['MJD-MID'] = (mid_mjd, 'exposure midtime MJD')
        header['MIDTIMJD'] = (mid_jd, 'exposure midtime JD')
        header['MAGZP'] = (float(header['PHOT_C']), 'CFHT Elixir zeropoint')
        header['MAGZPSIG'] = (zp_sig, 'CFHT zeropoint uncertainty')
        header['ORIGFILE'] = (raw_path.name[:68], 'source FITS filename')
        header['ORIGEXT'] = (hdu_index, 'source FITS extension')
        header['ORIGEXTN'] = (str(image_hdu.header.get('EXTNAME', ''))[:68],
                              'source FITS extension name')
        header['PPSEQ'] = (sequence, 'CFHT MegaCam workflow index')
        header['CLICKX'] = (float(click['x']), 'click x in display pixels')
        header['CLICKY'] = (float(click['y']), 'click y in display pixels')
        header['CLICKW'] = (float(click['width']), 'click display width')
        header['CLICKH'] = (float(click['height']), 'click display height')
        header['NANNPIX'] = (n_bad, 'non-finite pixels replaced for PP')
        if fill_value is not None:
            header['NANFILL'] = (fill_value, 'replacement value for NaNs')

        working_path.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(data=image, header=header).writeto(
            str(working_path), overwrite=True, output_verify='silentfix')

    with fits.open(str(working_path), ignore_missing_end=True) as hdulist:
        header = hdulist[0].header
        return {
            'raw': str(raw_path),
            'working': str(working_path),
            'working_name': working_path.name,
            'source_hdu': int(header['ORIGEXT']),
            'source_extname': header.get('ORIGEXTN'),
            'image_id': click['image_id'],
            'chip_hdu': int(click['chip_hdu']),
            'ccd': click.get('ccd'),
            'filter': header.get('FILTER'),
            'mjd_obs': float(header.get('MJD-OBS', header.get('MJDATE'))),
            'mjd_mid': float(header['MJD-MID']),
            'midtimjd': float(header['MIDTIMJD']),
            'date_obs': header.get('DATE-OBS'),
            'exptime': float(header['EXPTIME']),
            'magzp': float(header['MAGZP']),
            'magzp_sig': float(header['MAGZPSIG']),
            'nan_pixels_replaced': int(header.get('NANNPIX', 0)),
            'nan_fill': header.get('NANFILL'),
        }


def derive_click_position(cutout_path, click,
                          pixel_origin=DEFAULT_PIXEL_ORIGIN, flip_y=False):
    """Convert one click-points row into a sky position using cutout WCS."""

    with fits.open(str(cutout_path), memmap=False,
                   ignore_missing_end=True) as hdulist:
        hdu_index = first_image_hdu_index(hdulist)
        hdu = hdulist[hdu_index]
        ny, nx = hdu.data.shape[-2:]
        scaled_x = float(click['x']) * float(nx) / float(click['width'])
        if flip_y:
            scaled_y = ((float(click['height']) - float(click['y'])) *
                        float(ny) / float(click['height']))
        else:
            scaled_y = float(click['y']) * float(ny) / float(click['height'])

        wcs = WCS(hdu.header)
        ra_deg, dec_deg = wcs.all_pix2world(
            [[scaled_x, scaled_y]], int(pixel_origin))[0]
        return {
            'cutout': str(cutout_path),
            'cutout_name': Path(cutout_path).name,
            'cutout_hdu': hdu_index,
            'display_x': float(click['x']),
            'display_y': float(click['y']),
            'display_width': float(click['width']),
            'display_height': float(click['height']),
            'fits_x': float(scaled_x),
            'fits_y': float(scaled_y),
            'fits_width': int(nx),
            'fits_height': int(ny),
            'pixel_origin': int(pixel_origin),
            'flip_y': bool(flip_y),
            'ra_deg': float(ra_deg),
            'dec_deg': float(dec_deg),
        }


def prepare_records(clicks, pp_dir, target,
                    pixel_origin=DEFAULT_PIXEL_ORIGIN, flip_y=False,
                    magzp_sig=DEFAULT_MAGZP_SIG):
    """Create PP-ready FITS files and attach click-derived positions."""

    records = []
    for sequence, click in enumerate(clicks, start=1):
        raw_path = Path(click['raw_path'])
        filt = raw_chip_filter(raw_path, click)
        filter_dir = Path(pp_dir) / filt
        working_path = filter_dir / working_filename(click, filt, sequence)
        record = convert_to_pp_fits(
            raw_path, working_path, click, target, sequence,
            magzp_sig=magzp_sig)
        record['filter_dir'] = str(filter_dir)
        record['filter_dir_name'] = filt
        record['click'] = dict(click)
        record['target_position'] = derive_click_position(
            Path(click['cutout_path']), click, pixel_origin=pixel_origin,
            flip_y=flip_y)
        records.append(record)
    return records


def backup_candidates(click_points_csv, clicks):
    """Return source files and sidecars that should be archived."""

    csv_path = Path(click_points_csv).expanduser().resolve()
    base_dir = csv_path.parent
    sidecars = ['download_plan.sh', 'download_negative_delta_mag.sh',
                'downloads.jsonl']
    candidates = [csv_path]
    for click in clicks:
        candidates.append(Path(click['raw_path']))
        candidates.append(Path(click['cutout_path']))
    for name in sidecars:
        path = base_dir / name
        if path.exists():
            candidates.append(path)
    return sorted(set(candidates))


def create_originals_backup(pp_dir, click_points_csv, clicks, refresh=False):
    """Create or reuse the PP-side tar backup of source files."""

    pp_dir = Path(pp_dir)
    backup_path = pp_dir / BACKUP_NAME
    if backup_path.exists() and not refresh:
        return {'path': str(backup_path), 'created': False}

    if backup_path.exists():
        backup_path.unlink()

    base_dir = Path(click_points_csv).expanduser().resolve().parent
    pp_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(backup_path), 'w') as tar:
        for path in backup_candidates(click_points_csv, clicks):
            try:
                arcname = str(path.resolve().relative_to(base_dir))
            except ValueError:
                arcname = path.name
            tar.add(str(path), arcname=arcname)

    return {'path': str(backup_path), 'created': True}


def positions_filename(target, filt):
    """Return a stable per-filter PP positions filename."""

    return 'positions_%s_%s.dat' % (target_to_filename(target), filt)


def write_positions_file(filter_dir, records, target, filt):
    """Write PP -positions input for one filter directory."""

    path = Path(filter_dir) / positions_filename(target, filt)
    ident = target_to_filename(target)
    with path.open('w') as handle:
        for record in records:
            pos = record['target_position']
            handle.write('%s %.10f %.10f %.8f %s\n' %
                         (record['working_name'], pos['ra_deg'],
                          pos['dec_deg'], record['midtimjd'], ident))
    return path


def remove_stale_target_output(image_dir, target):
    """Remove target photometry products before a fresh run."""

    for path in Path(image_dir).glob('photometry_%s*.dat' %
                                     target_to_filename(target)):
        path.unlink()


def run_pp_pipeline(filter_dir, records, target, posfile,
                    fixed_aprad=0.0, rejectionfilter='pos'):
    """Run pp_run on one filter's extracted chips with kept WCS."""

    configure_pp_environment()

    working_names = [record['working_name'] for record in records]
    with working_directory(filter_dir):
        import pp_run

        pp_run.run_the_pipeline(
            sorted(working_names), target, None, float(fixed_aprad),
            'high', False, False, False, True, telescope=TELESCOPE_KEY,
            posfile=Path(posfile).name, magzp_keyword='MAGZP',
            magzp_sig_keyword='MAGZPSIG', rejectionfilter=rejectionfilter)


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


def rewrite_row_catalog_token(row, filter_name):
    """Rewrite a per-filter PP row so root-level tools can resolve FITS."""

    values = row.split()
    if not values:
        return row
    values[0] = str(Path(filter_name) / values[0])
    return ' ' + ' '.join(values) + '\n'


def combine_filter_photometry(pp_dir, target, outputs):
    """Combine per-filter PP photometry into root-level products."""

    pp_dir = Path(pp_dir)
    existing = [output for output in outputs
                if output.get('photometry_file') and
                Path(output['photometry_file']).exists()]
    if not existing:
        return []

    first_header, first_rows, first_footer = split_photometry_file(
        existing[0]['photometry_file'])
    rows = [rewrite_row_catalog_token(row, existing[0]['filter'])
            for row in first_rows]
    for output in existing[1:]:
        rows.extend(rewrite_row_catalog_token(row, output['filter'])
                    for row in split_photometry_file(
                        output['photometry_file'])[1])

    combined_text = ''.join(first_header + rows + first_footer)
    destinations = [
        pp_dir / target_photometry_filename(target),
        pp_dir / all_filters_photometry_filename(target),
    ]
    for destination in destinations:
        destination.write_text(combined_text)
    return [str(destination) for destination in destinations]


def run_pp_by_filter(pp_dir, records, target, fixed_aprad=0.0,
                     rejectionfilter='pos'):
    """Run PP once per filter and preserve per-filter outputs."""

    outputs = []
    filters = sorted(set(record['filter'] for record in records))
    for filt in filters:
        filter_records = [record for record in records
                          if record['filter'] == filt]
        filter_dir = Path(pp_dir) / filt
        remove_stale_target_output(filter_dir, target)
        posfile = write_positions_file(filter_dir, filter_records,
                                       target, filt)
        run_pp_pipeline(filter_dir, filter_records, target, posfile,
                        fixed_aprad=fixed_aprad,
                        rejectionfilter=rejectionfilter)
        photometry_path = filter_dir / target_photometry_filename(target)
        outputs.append({
            'filter': filt,
            'positions_file': str(posfile),
            'photometry_file': (str(photometry_path)
                                if photometry_path.exists() else None),
            'working_names': [record['working_name']
                              for record in filter_records],
        })
    return outputs


def write_positions_by_filter(pp_dir, records, target):
    """Write one PP positions file for each filter group."""

    outputs = []
    for filt in sorted(set(record['filter'] for record in records)):
        filter_records = [record for record in records
                          if record['filter'] == filt]
        path = write_positions_file(Path(pp_dir) / filt, filter_records,
                                    target, filt)
        outputs.append({
            'filter': filt,
            'positions_file': str(path),
            'working_names': [record['working_name']
                              for record in filter_records],
        })
    return outputs


def build_mpc_outputs(pp_dir, target, observatory_code='568'):
    """Build ADES PSV and MPC 80-column files from combined photometry."""

    configure_pp_environment()
    import pptool_mpcsubmission as mpcsub

    pp_dir = Path(pp_dir)
    photometry_path = pp_dir / target_photometry_filename(target)
    config = mpcsub.SubmissionConfig(
        target=target, observatory_code=str(observatory_code),
        output_dir=pp_dir)
    bundle = mpcsub.build_submission(photometry_path, config)
    mpcsub.write_submission(bundle)
    return {
        'observations': len(bundle.observations),
        'ades_path': str(bundle.ades_path),
        'obs80_path': str(bundle.obs80_path),
        'summary_path': str(bundle.summary_path),
        'warnings': list(bundle.warnings),
    }


def write_manifest(pp_dir, manifest):
    """Write the workflow manifest JSON."""

    path = Path(pp_dir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return path


def build_workflow(click_points_csv, target, pp_dir=None,
                   pixel_origin=DEFAULT_PIXEL_ORIGIN, flip_y=False,
                   magzp_sig=DEFAULT_MAGZP_SIG, refresh_backup=False):
    """Prepare PP-ready CFHT chip FITS files and position sidecars."""

    csv_path = Path(click_points_csv).expanduser().resolve()
    if pp_dir is None:
        pp_dir = csv_path.parent / 'PP'
    pp_dir = Path(pp_dir).expanduser().resolve()
    pp_dir.mkdir(parents=True, exist_ok=True)

    clicks = read_clicked_rows(csv_path)
    backup = create_originals_backup(pp_dir, csv_path, clicks,
                                     refresh=refresh_backup)
    records = prepare_records(
        clicks, pp_dir, target, pixel_origin=pixel_origin, flip_y=flip_y,
        magzp_sig=magzp_sig)
    positions_files = write_positions_by_filter(pp_dir, records, target)

    manifest = {
        'workflow': 'cfht_megacam_click_points',
        'telescope': TELESCOPE_KEY,
        'target': target,
        'click_points_csv': str(csv_path),
        'pp_dir': str(pp_dir),
        'backup': backup,
        'pixel_origin': int(pixel_origin),
        'flip_y': bool(flip_y),
        'magzp_sig_default': float(magzp_sig),
        'clicks': clicks,
        'records': records,
        'positions_files': positions_files,
        'filters': sorted(set(record['filter'] for record in records)),
    }
    manifest_path = write_manifest(pp_dir, manifest)
    manifest['manifest_path'] = str(manifest_path)
    return manifest


def run_workflow(args):
    """Execute the full CFHT MegaCam click-points PP workflow."""

    manifest = build_workflow(
        args.click_points_csv, args.target, pp_dir=args.pp_dir,
        pixel_origin=args.pixel_origin, flip_y=args.flip_y,
        magzp_sig=args.magzp_sig, refresh_backup=args.refresh_backup)

    pp_dir = Path(manifest['pp_dir'])
    if not args.prepare_only:
        remove_stale_target_output(pp_dir, args.target)
        outputs = run_pp_by_filter(
            pp_dir, manifest['records'], args.target,
            fixed_aprad=args.fixed_aprad, rejectionfilter=args.reject)
        manifest['pp_outputs'] = outputs
        manifest['combined_photometry_files'] = combine_filter_photometry(
            pp_dir, args.target, outputs)
        manifest['mpc_outputs'] = build_mpc_outputs(
            pp_dir, args.target, observatory_code=args.observatory_code)
        manifest_path = write_manifest(pp_dir, manifest)
        manifest['manifest_path'] = str(manifest_path)
    return manifest


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='prepare and run PP for CFHT MegaCam click_points.csv')
    parser.add_argument('click_points_csv',
                        help='click_points.csv produced from CFHT PNGs')
    parser.add_argument('--target', default='2013 SO107',
                        help='target name to use in PP/submission outputs')
    parser.add_argument('--pp-dir',
                        help='output PP directory; defaults to CSV sibling PP')
    parser.add_argument('--pixel-origin', type=int,
                        default=DEFAULT_PIXEL_ORIGIN,
                        help='pixel origin for WCS conversion')
    parser.add_argument('--flip-y', action='store_true',
                        help='flip click y before converting through WCS')
    parser.add_argument('--magzp-sig', type=float,
                        default=DEFAULT_MAGZP_SIG,
                        help='fallback MAGZPSIG when PHOT_CS is absent')
    parser.add_argument('--fixed-aprad', type=float, default=0.0,
                        help='fixed PP aperture radius; 0 uses curve of growth')
    parser.add_argument('--reject', default='pos',
                        help='PP target rejection schema, e.g. pos or none')
    parser.add_argument('--observatory-code', default='568',
                        help='MPC observatory code for submission files')
    parser.add_argument('--refresh-backup', action='store_true',
                        help='recreate originals_backup.tar')
    parser.add_argument('--prepare-only', action='store_true',
                        help='write PP-ready FITS/positions but do not run PP')
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    manifest = run_workflow(args)
    print(json.dumps({
        'manifest_path': manifest['manifest_path'],
        'pp_dir': manifest['pp_dir'],
        'n_records': len(manifest['records']),
        'filters': manifest['filters'],
        'pp_outputs': manifest.get('pp_outputs', []),
        'combined_photometry_files': manifest.get(
            'combined_photometry_files', []),
        'mpc_outputs': manifest.get('mpc_outputs'),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
