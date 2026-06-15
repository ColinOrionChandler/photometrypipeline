#!/usr/bin/env python3

"""Workflow helper for CFHT/CFH12K CADC image products."""

from __future__ import print_function

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS


TELESCOPE_KEY = 'CFHTCFH12K'
MANIFEST_NAME = 'cfht_cfh12k_workflow_manifest.json'
WORKING_PREFIX = 'cfh12k'
TARGET_TRANSLATION = str.maketrans(' /()', '____')
DEFAULT_MAGZP_SIG = 0.03
DEFAULT_CLICK_PIXEL_ORIGIN = 0
DEFAULT_CENTROID_SEARCH_RADIUS = 6.0
DEFAULT_CENTROID_APERTURE_RADIUS = 3.0
DEFAULT_CENTROID_MIN_SNR = 1.5
DEFAULT_MAX_PP_MATCH_RESIDUAL_ARCSEC = 2.0
CLICK_SOLUTION_CENTROID = 'centroid'
CLICK_SOLUTION_PURE = 'pure-click'
CLICK_SOLUTION_BOTH = 'both'
CLICK_BRANCHES = {
    CLICK_SOLUTION_CENTROID: ('click_centroid', True),
    CLICK_SOLUTION_PURE: ('click_raw', False),
}


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


def parse_bool(value):
    """Return True for common truthy CSV flag values."""

    return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y')


def normalize_filter(value):
    """Translate common CFH12K filter labels to PP filter names."""

    text = str(value).strip()
    mapping = {'U': 'U', 'B': 'B', 'V': 'V', 'R': 'R', 'I': 'I',
               'Z': 'z', 'u': 'u', 'g': 'g', 'r': 'r', 'i': 'i',
               'z': 'z'}
    return mapping.get(text, text)


def filter_dir_name(value):
    """Return the on-disk PP filter folder name for a source filter."""

    text = str(value).strip()
    return text if text else normalize_filter(value)


def parse_cutout_name(path):
    """Parse a CFH12K cutout filename into image and chip identifiers."""

    match = re.search(r'_(?P<image_id>\d+p)_chip(?P<hdu>\d+)'
                      r'-chip(?P<detector>\d+)_', Path(path).name)
    if match is None:
        raise ValueError('cannot parse CFH12K cutout chip from %s' % path)
    return {
        'image_id': match.group('image_id'),
        'chip_hdu': int(match.group('hdu')),
        'detector_chip': int(match.group('detector')),
    }


def resolve_raw_path(base_dir, image_id):
    """Find the raw CADC CFH12K FITS file for one image id."""

    candidates = [
        base_dir / image_id / (image_id + '.fits.fz'),
        base_dir / image_id / (image_id + '.fits'),
        base_dir / (image_id + '.fits.fz'),
        base_dir / (image_id + '.fits'),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError('cannot find raw FITS for %s below %s' %
                            (image_id, base_dir))


def resolve_cutout_path(base_dir, image_id, chip_hdu, detector_chip=None):
    """Find the cutout FITS file corresponding to one exposure/chip."""

    detector_values = []
    if detector_chip is not None:
        detector_values.append(int(detector_chip))
    detector_values.append(int(chip_hdu) - 1)

    search_dirs = [
        base_dir / image_id / 'FITS',
        base_dir / image_id / 'fits_thumbs_all',
        base_dir / 'fits_thumbs_all',
    ]
    patterns = []
    for detector in dict.fromkeys(detector_values):
        patterns.append('*_%s_chip%d-chip%02d_*.fits' %
                        (image_id, int(chip_hdu), detector))
        patterns.append('*_%s_chip%02d-chip%02d_*.fits' %
                        (image_id, int(chip_hdu), detector))

    for directory in search_dirs:
        if not directory.exists():
            continue
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0].resolve()
    return None


def first_image_hdu_index(hdulist):
    """Locate the first image-bearing HDU in a FITS file."""

    for idx, hdu in enumerate(hdulist):
        data = getattr(hdu, 'data', None)
        if data is not None and len(data.shape) >= 2:
            return idx
    raise ValueError('no image-bearing HDU found')


def read_clicked_rows(click_points_csv):
    """Load clicked CFH12K PNG rows and resolve their FITS cutouts."""

    csv_path = Path(click_points_csv).expanduser().resolve()
    base_dir = csv_path.parent
    rows = []
    with csv_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if str(row.get('status', '')).strip().lower() != 'clicked':
                continue
            parsed = parse_cutout_name(row['image'])
            cutout_path = base_dir / Path(row['image']).with_suffix('.fits')
            if not cutout_path.exists():
                raise FileNotFoundError('cannot find FITS cutout for %s' %
                                        row['image'])
            rows.append({
                'row_index': index,
                'image': row['image'],
                'image_id': parsed['image_id'],
                'chip_hdu': parsed['chip_hdu'],
                'detector_chip': parsed['detector_chip'],
                'x': float(row['x']),
                'y': float(row['y']),
                'width': float(row['width']),
                'height': float(row['height']),
                'status': row.get('status', ''),
                'cutout_path': str(cutout_path.resolve()),
            })
    if not rows:
        raise ValueError('no clicked rows found in %s' % csv_path)
    return rows


def derive_click_position(cutout_path, click,
                          pixel_origin=DEFAULT_CLICK_PIXEL_ORIGIN):
    """Convert one PNG click into a sky position through cutout WCS."""

    with fits.open(str(cutout_path), memmap=False,
                   ignore_missing_end=True) as hdulist:
        hdu_index = first_image_hdu_index(hdulist)
        hdu = hdulist[hdu_index]
        ny, nx = hdu.data.shape[-2:]
        scaled_x = float(click['x']) * float(nx) / float(click['width'])
        scaled_y = float(click['y']) * float(ny) / float(click['height'])
        ra_deg, dec_deg = WCS(hdu.header, relax=True).all_pix2world(
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
        'ra_deg': float(ra_deg),
        'dec_deg': float(dec_deg),
    }


def apply_click_positions(location_records, click_points_csv,
                          pixel_origin=DEFAULT_CLICK_PIXEL_ORIGIN):
    """Select clicked records and replace their target positions."""

    clicks = read_clicked_rows(click_points_csv)
    click_by_image = {}
    for click in clicks:
        if click['image_id'] in click_by_image:
            raise ValueError('multiple clicked rows for %s' %
                             click['image_id'])
        click_by_image[click['image_id']] = click

    clicked_records = []
    for record in location_records:
        click = click_by_image.get(record['image_id'])
        if click is None:
            continue
        if int(click['chip_hdu']) != int(record['chip_hdu']):
            raise ValueError('clicked chip %d does not match %s HDU %d' %
                             (int(click['chip_hdu']), record['image_id'],
                              int(record['chip_hdu'])))
        position = derive_click_position(click['cutout_path'], click,
                                         pixel_origin=pixel_origin)
        updated = dict(record)
        updated['target_ra_deg'] = position['ra_deg']
        updated['target_dec_deg'] = position['dec_deg']
        updated['click_position'] = position
        updated['position_source'] = 'click_points'
        updated['cutout_path'] = click['cutout_path']
        updated['detector_chip'] = int(click['detector_chip'])
        updated['cutout_chip'] = {
            'image_id': click['image_id'],
            'chip_hdu': int(click['chip_hdu']),
            'detector_chip': int(click['detector_chip']),
        }
        clicked_records.append(updated)

    missing = sorted(set(click_by_image) -
                     set(record['image_id'] for record in clicked_records))
    if missing:
        raise ValueError('clicked image ids not found in locations CSV: %s' %
                         ', '.join(missing))
    if not clicked_records:
        raise ValueError('click CSV did not match any location rows')
    return clicked_records, clicks


def image_shape_from_header(header):
    """Return image dimensions from regular or compressed-image headers."""

    nx = header.get('NAXIS1', header.get('ZNAXIS1'))
    ny = header.get('NAXIS2', header.get('ZNAXIS2'))
    if nx is None or ny is None:
        raise KeyError('cannot determine image dimensions from header')
    return int(nx), int(ny)


def locate_chip_by_wcs(raw_path, ra_deg, dec_deg):
    """Find which image HDU contains the target sky position."""

    with fits.open(str(raw_path), memmap=False, ignore_missing_end=True,
                   do_not_scale_image_data=True) as hdulist:
        for idx in range(1, len(hdulist)):
            header = hdulist[idx].header
            try:
                nx, ny = image_shape_from_header(header)
                x, y = WCS(header).all_world2pix(
                    [[float(ra_deg), float(dec_deg)]], 0)[0]
            except Exception:
                continue
            if -0.5 <= x <= nx - 0.5 and -0.5 <= y <= ny - 0.5:
                return idx
    raise ValueError('cannot locate %.8f %.8f in %s' %
                     (float(ra_deg), float(dec_deg), raw_path))


def read_location_records(base_dir, target):
    """Load present-target rows from CFH12K *_locations.csv files."""

    base_dir = Path(base_dir).expanduser().resolve()
    target_stem = target_to_filename(target)
    paths = sorted(base_dir.glob('*/%s_locations.csv' % target_stem))
    if not paths:
        paths = sorted(base_dir.glob('*/*_locations.csv'))

    records = []
    for path in paths:
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader, start=1):
                if not parse_bool(row.get('Present_Flag', '')):
                    continue
                image_id = Path(path).parent.name
                raw_path = resolve_raw_path(base_dir, image_id)
                ra_deg = float(row['OBJ_RA'])
                dec_deg = float(row['OBJ_Dec'])
                if str(row.get('ChipSliceNum', '')).strip():
                    chip_hdu = int(float(row['ChipSliceNum']))
                else:
                    chip_hdu = locate_chip_by_wcs(raw_path, ra_deg, dec_deg)

                cutout_path = resolve_cutout_path(base_dir, image_id, chip_hdu)
                detector_chip = chip_hdu - 1
                cutout_chip = None
                if cutout_path is not None:
                    cutout_chip = parse_cutout_name(cutout_path)
                    if cutout_chip['image_id'] != image_id:
                        raise ValueError('cutout image id mismatch for %s' %
                                         cutout_path)
                    if cutout_chip['chip_hdu'] != chip_hdu:
                        raise ValueError('cutout chip mismatch for %s' %
                                         cutout_path)
                    detector_chip = cutout_chip['detector_chip']

                source_filter = str(row.get('Filter', '')).strip()
                ut = str(row.get('UT', '')).strip()
                date = ut.split()[0] if ut else None
                records.append({
                    'row_index': row_index,
                    'locations_csv': str(path.resolve()),
                    'image_id': image_id,
                    'raw_path': str(raw_path),
                    'cutout_path': (str(cutout_path)
                                    if cutout_path is not None else None),
                    'cutout_chip': cutout_chip,
                    'chip_hdu': chip_hdu,
                    'detector_chip': detector_chip,
                    'target_ra_deg': ra_deg,
                    'target_dec_deg': dec_deg,
                    'midtimjd': float(row['JD']),
                    'ut': ut,
                    'date': date,
                    'source_filter': source_filter,
                    'filter': normalize_filter(source_filter),
                    'filter_dir_name': filter_dir_name(source_filter),
                    'delta_mag': row.get('deltaMag'),
                })

    if not records:
        raise ValueError('no present CFH12K location rows found below %s' %
                         base_dir)
    return sorted(records, key=lambda item: item['midtimjd'])


def find_imcopy():
    """Return a CFITSIO imcopy executable path."""

    configure_pp_environment()
    candidates = [
        shutil.which('imcopy'),
        str(Path(sys.executable).resolve().parent / 'imcopy'),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError('CFITSIO imcopy executable not found')


def extract_full_chip(raw_path, working_path, chip_hdu):
    """Extract one compressed image HDU to a single-HDU FITS file."""

    raw_path = Path(raw_path)
    working_path = Path(working_path)
    working_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.suffix.lower() == '.fits':
        try:
            with fits.open(str(raw_path), memmap=False,
                           ignore_missing_end=True) as hdulist:
                hdu = hdulist[int(chip_hdu)]
                fits.PrimaryHDU(
                    data=np.asarray(hdu.data, dtype=np.float32),
                    header=hdu.header.copy()).writeto(
                        str(working_path), overwrite=True,
                        output_verify='silentfix')
            return working_path
        except Exception:
            pass
    command = [
        find_imcopy(),
        '%s[%d]' % (raw_path, int(chip_hdu)),
        '!%s' % working_path,
    ]
    subprocess.check_call(command)
    return working_path


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


def robust_sky_sigma(values):
    """Return a robust sky and sigma estimate for a small image patch."""

    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    sky = float(np.median(finite))
    mad = float(np.median(np.abs(finite - sky)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(finite))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return sky, sigma


def refine_position_by_centroid(working_path, ra_deg, dec_deg,
                                search_radius=DEFAULT_CENTROID_SEARCH_RADIUS,
                                aperture_radius=DEFAULT_CENTROID_APERTURE_RADIUS,
                                min_snr=DEFAULT_CENTROID_MIN_SNR):
    """Centroid the closest faint positive source near an input sky position."""

    with fits.open(str(working_path), mode='update', memmap=False,
                   ignore_missing_end=True) as hdulist:
        data = np.asarray(hdulist[0].data, dtype=np.float32)
        header = hdulist[0].header
        wcs = WCS(header, relax=True)
        x0, y0 = wcs.all_world2pix([[float(ra_deg), float(dec_deg)]], 0)[0]
        result = {
            'input_ra_deg': float(ra_deg),
            'input_dec_deg': float(dec_deg),
            'input_x': float(x0),
            'input_y': float(y0),
            'search_radius_px': float(search_radius),
            'aperture_radius_px': float(aperture_radius),
            'min_snr': float(min_snr),
            'used': False,
            'reason': None,
        }
        if (not np.isfinite(x0) or not np.isfinite(y0) or
                x0 < 0 or y0 < 0 or
                x0 >= data.shape[1] or y0 >= data.shape[0]):
            result['reason'] = 'input position outside full chip'
            return result

        radius = max(1, int(np.ceil(float(search_radius))))
        x_min = max(0, int(np.floor(x0)) - radius)
        x_max = min(data.shape[1] - 1, int(np.floor(x0)) + radius)
        y_min = max(0, int(np.floor(y0)) - radius)
        y_max = min(data.shape[0] - 1, int(np.floor(y0)) + radius)
        patch = data[y_min:y_max + 1, x_min:x_max + 1]
        if patch.size == 0 or not np.isfinite(patch).any():
            result['reason'] = 'no finite pixels in centroid search box'
            return result

        sky, sigma = robust_sky_sigma(patch)
        yy, xx = np.indices(patch.shape, dtype=float)
        abs_x = xx + x_min
        abs_y = yy + y_min
        distance = np.hypot(abs_x - x0, abs_y - y0)
        candidates = np.isfinite(patch) & (distance <= float(search_radius))
        positive = candidates & ((patch - sky) >= float(min_snr) * sigma)
        if not positive.any():
            result['reason'] = 'no positive source above SNR threshold'
            result['sky'] = sky
            result['sigma'] = sigma
            return result

        candidate_indices = np.argwhere(positive)
        distances = distance[positive]
        fluxes = (patch - sky)[positive]
        order = np.lexsort((-fluxes, distances))
        peak_y, peak_x = candidate_indices[order[0]]
        peak_abs_x = float(peak_x + x_min)
        peak_abs_y = float(peak_y + y_min)

        aperture = (np.hypot(abs_x - peak_abs_x, abs_y - peak_abs_y) <=
                    float(aperture_radius))
        weights = np.where(aperture & np.isfinite(patch), patch - sky, 0.0)
        weights = np.where(weights > 0, weights, 0.0)
        total = float(weights.sum())
        if total <= 0 or not np.isfinite(total):
            result['reason'] = 'non-positive centroid aperture flux'
            result['sky'] = sky
            result['sigma'] = sigma
            result['peak_x'] = peak_abs_x
            result['peak_y'] = peak_abs_y
            return result

        cx = float((weights * abs_x).sum() / total)
        cy = float((weights * abs_y).sum() / total)
        refined_ra, refined_dec = wcs.all_pix2world([[cx, cy]], 0)[0]
        header['TARGRA'] = (float(refined_ra), 'target RA after centroiding')
        header['TARGDEC'] = (float(refined_dec), 'target Dec after centroiding')
        header['TARGX'] = (cx, 'target centroid x pixel, origin 0')
        header['TARGY'] = (cy, 'target centroid y pixel, origin 0')
        header['CENTROID'] = (True, 'target position was centroid-refined')
        hdulist.flush(output_verify='silentfix')

    result.update({
        'used': True,
        'reason': 'centroided nearest positive source',
        'sky': sky,
        'sigma': sigma,
        'peak_snr': float((patch[int(peak_y), int(peak_x)] - sky) / sigma),
        'peak_x': peak_abs_x,
        'peak_y': peak_abs_y,
        'centroid_x': cx,
        'centroid_y': cy,
        'ra_deg': float(refined_ra),
        'dec_deg': float(refined_dec),
        'offset_from_input_px': float(np.hypot(cx - x0, cy - y0)),
    })
    return result


def midtime_values(header, fallback_mid_jd=None):
    """Return MJD start, MJD midpoint, JD midpoint, and ISO start time."""

    if 'MJD-OBS' in header:
        mjd_obs = float(header['MJD-OBS'])
    else:
        mjd_obs = float(header['MJDATE'])
    exptime = float(header.get('EXPTIME', 0.0))
    mid_mjd = mjd_obs + exptime / 2.0 / 86400.0
    if fallback_mid_jd is not None:
        mid_jd = float(fallback_mid_jd)
    else:
        mid_jd = Time(mid_mjd, format='mjd', scale='utc').jd
    date_obs = Time(mjd_obs, format='mjd', scale='utc').isot
    return mjd_obs, mid_mjd, mid_jd, date_obs


def validate_chip_header(header, location_record):
    """Validate the one-based HDU and zero-based detector chip labels."""

    expected_detector = int(location_record['detector_chip'])
    extname = str(header.get('EXTNAME', '')).strip().lower()
    if extname and extname not in ('chip%d' % expected_detector,
                                   'chip%02d' % expected_detector):
        raise ValueError('HDU %d expected detector chip%02d but found %s' %
                         (int(location_record['chip_hdu']),
                          expected_detector, extname))
    if 'IMAGEID' in header and int(header['IMAGEID']) != expected_detector:
        raise ValueError('HDU %d expected IMAGEID %d but found %s' %
                         (int(location_record['chip_hdu']),
                          expected_detector, header['IMAGEID']))


def normalize_extracted_fits(working_path, location_record, target,
                             sequence, magzp_sig=DEFAULT_MAGZP_SIG):
    """Normalize one extracted CFH12K chip for PP."""

    working_path = Path(working_path)
    with fits.open(str(working_path), memmap=False,
                   ignore_missing_end=True) as hdulist:
        if len(hdulist) != 1:
            raise ValueError('%s should contain one extracted image HDU' %
                             working_path)
        image_hdu = hdulist[0]
        if image_hdu.data is None:
            raise ValueError('%s does not contain image data' % working_path)
        header = image_hdu.header.copy()
        validate_chip_header(header, location_record)
        image, n_bad, fill_value = repair_nonfinite_pixels(image_hdu.data)

    original_filter = str(header.get(
        'FILTER', location_record['source_filter'])).strip()
    filt = normalize_filter(original_filter)
    mjd_obs, mid_mjd, mid_jd, date_obs = midtime_values(
        header, fallback_mid_jd=location_record.get('midtimjd'))

    warnings = []
    has_magzp = 'PHOT_C' in header
    if has_magzp:
        zp = float(header['PHOT_C'])
        zp_sig = float(header['PHOT_CS']) if 'PHOT_CS' in header else float(
            magzp_sig)
    else:
        zp = None
        zp_sig = None
        warnings.append('PHOT_C zeropoint not found in %s; PP will use '
                        'catalog calibration for this filter group' %
                        working_path)

    header['PPINSTRU'] = (TELESCOPE_KEY, 'PP telescope override')
    header['TEL_KEYW'] = (TELESCOPE_KEY, 'PP telescope keyword')
    header['TELESCOP'] = (header.get('TELESCOP', 'CFHT 3.6m'),
                          'PP normalized telescope')
    header['INSTRUME'] = (header.get('INSTRUME', 'CFH12K Mosaic'),
                          'PP normalized instrument')
    header['DETECTOR'] = (header.get('DETECTOR', 'CFH12K'),
                          'PP normalized detector')
    header['ORIGFILT'] = (original_filter, 'source CFH12K filter')
    header['FILTER'] = (filt, 'PP normalized filter')
    header['OBJECT'] = (target, 'PP target name')
    header['DATE-OBS'] = (date_obs, 'exposure start UTC from MJD-OBS')
    header['MJD-OBS'] = (mjd_obs, 'exposure start MJD')
    header['MJD-MID'] = (mid_mjd, 'exposure midtime MJD')
    header['MIDTIMJD'] = (mid_jd, 'exposure midtime JD')
    if has_magzp:
        header['MAGZP'] = (zp, 'CFHT Elixir zeropoint')
        header['MAGZPSIG'] = (zp_sig, 'CFHT zeropoint uncertainty')
    header['ORIGFILE'] = (Path(location_record['raw_path']).name[:68],
                          'source FITS filename')
    header['ORIGEXT'] = (int(location_record['chip_hdu']),
                         'source FITS extension')
    header['ORIGEXTN'] = (str(header.get('EXTNAME', ''))[:68],
                          'source FITS extension name')
    header['ORIGIMID'] = (location_record['image_id'], 'source image id')
    header['PPSEQ'] = (sequence, 'CFH12K workflow index')
    header['TARGRA'] = (float(location_record['target_ra_deg']),
                        'target RA from locations CSV')
    header['TARGDEC'] = (float(location_record['target_dec_deg']),
                         'target Dec from locations CSV')
    header['NANNPIX'] = (n_bad, 'non-finite pixels replaced for PP')
    if fill_value is not None:
        header['NANFILL'] = (fill_value, 'replacement value for NaNs')
    if 'RADESYS' in header and 'RADECSYS' not in header:
        header['RADECSYS'] = (header['RADESYS'], 'copied from RADESYS')

    fits.PrimaryHDU(data=image, header=header).writeto(
        str(working_path), overwrite=True, output_verify='silentfix')

    with fits.open(str(working_path), ignore_missing_end=True) as hdulist:
        header = hdulist[0].header
        return {
            'raw': str(Path(location_record['raw_path'])),
            'cutout': location_record.get('cutout_path'),
            'working': str(working_path),
            'working_name': working_path.name,
            'source_hdu': int(header['ORIGEXT']),
            'source_extname': header.get('ORIGEXTN'),
            'image_id': location_record['image_id'],
            'detector_chip': int(location_record['detector_chip']),
            'filter': header.get('FILTER'),
            'source_filter': location_record['source_filter'],
            'filter_dir_name': location_record['filter_dir_name'],
            'date': location_record['date'],
            'mjd_obs': float(header['MJD-OBS']),
            'mjd_mid': float(header['MJD-MID']),
            'midtimjd': float(header['MIDTIMJD']),
            'date_obs': header.get('DATE-OBS'),
            'exptime': float(header['EXPTIME']),
            'has_magzp': bool(has_magzp),
            'magzp': (float(header['MAGZP']) if has_magzp else None),
            'magzp_sig': (float(header['MAGZPSIG']) if has_magzp else None),
            'target_ra_deg': float(location_record['target_ra_deg']),
            'target_dec_deg': float(location_record['target_dec_deg']),
            'nan_pixels_replaced': int(header.get('NANNPIX', 0)),
            'nan_fill': header.get('NANFILL'),
            'warnings': warnings,
        }


def working_filename(location_record, sequence):
    """Return a short PP-ready filename for one extracted chip."""

    filt = normalize_filter(location_record['source_filter']).lower()
    return '%s_%s_%04d_%s_chip%02d.fits' % (
        WORKING_PREFIX, filt, sequence, location_record['image_id'],
        int(location_record['chip_hdu']))


def prepare_records(base_dir, pp_dir, target,
                    magzp_sig=DEFAULT_MAGZP_SIG, click_points_csv=None,
                    click_pixel_origin=DEFAULT_CLICK_PIXEL_ORIGIN,
                    centroid=True,
                    centroid_search_radius=DEFAULT_CENTROID_SEARCH_RADIUS,
                    centroid_aperture_radius=DEFAULT_CENTROID_APERTURE_RADIUS,
                    centroid_min_snr=DEFAULT_CENTROID_MIN_SNR):
    """Create PP-ready FITS files from CFH12K location records."""

    locations = read_location_records(base_dir, target)
    click_rows = []
    skipped_location_image_ids = []
    if click_points_csv is not None:
        all_image_ids = [record['image_id'] for record in locations]
        locations, click_rows = apply_click_positions(
            locations, click_points_csv, pixel_origin=click_pixel_origin)
        clicked_image_ids = set(record['image_id'] for record in locations)
        skipped_location_image_ids = [
            image_id for image_id in all_image_ids
            if image_id not in clicked_image_ids
        ]

    records = []
    for sequence, location_record in enumerate(locations, start=1):
        date_dir = location_record['date']
        if date_dir is None:
            date_dir = Time(location_record['midtimjd'],
                            format='jd', scale='utc').isot.split('T')[0]
        filter_dir = (Path(pp_dir) / date_dir /
                      location_record['filter_dir_name'])
        working_path = filter_dir / working_filename(location_record, sequence)
        extract_full_chip(location_record['raw_path'], working_path,
                          location_record['chip_hdu'])
        record = normalize_extracted_fits(
            working_path, location_record, target, sequence,
            magzp_sig=magzp_sig)
        record['position_source'] = location_record.get(
            'position_source', 'locations_csv')
        if location_record.get('click_position') is not None:
            record['click_position'] = location_record['click_position']
        if centroid:
            centroid_result = refine_position_by_centroid(
                working_path, record['target_ra_deg'],
                record['target_dec_deg'],
                search_radius=centroid_search_radius,
                aperture_radius=centroid_aperture_radius,
                min_snr=centroid_min_snr)
            record['centroid'] = centroid_result
            if centroid_result.get('used'):
                record['target_ra_deg'] = centroid_result['ra_deg']
                record['target_dec_deg'] = centroid_result['dec_deg']
                record['position_source'] = (
                    record['position_source'] + '+centroid')
        record['filter_dir'] = str(filter_dir)
        record['filter_dir_relative'] = str(filter_dir.relative_to(pp_dir))
        record['location_record'] = dict(location_record)
        records.append(record)
    return records, {
        'click_points_csv': (str(Path(click_points_csv).expanduser().resolve())
                             if click_points_csv is not None else None),
        'click_pixel_origin': int(click_pixel_origin),
        'n_click_rows': len(click_rows),
        'clicked_images': [row['image_id'] for row in click_rows],
        'skipped_location_image_ids': skipped_location_image_ids,
        'centroid': bool(centroid),
        'centroid_search_radius_px': float(centroid_search_radius),
        'centroid_aperture_radius_px': float(centroid_aperture_radius),
        'centroid_min_snr': float(centroid_min_snr),
    }


def positions_filename(target, filt):
    """Return a stable per-filter PP positions filename."""

    return 'positions_%s_%s.dat' % (target_to_filename(target), filt)


def write_positions_file(filter_dir, records, target, filt):
    """Write PP -positions input for one date/filter directory."""

    path = Path(filter_dir) / positions_filename(target, filt)
    ident = target_to_filename(target)
    with path.open('w') as handle:
        for record in records:
            handle.write('%s %.10f %.10f %.8f %s\n' %
                         (record['working_name'],
                          float(record['target_ra_deg']),
                          float(record['target_dec_deg']),
                          float(record['midtimjd']), ident))
    return path


def remove_stale_target_output(image_dir, target):
    """Remove target photometry products before a fresh run."""

    for path in Path(image_dir).glob('photometry_%s*.dat' %
                                     target_to_filename(target)):
        path.unlink()


def run_pp_pipeline(filter_dir, records, target, posfile,
                    fixed_aprad=0.0, rejectionfilter='pos'):
    """Run pp_run on one date/filter's extracted chips with kept WCS."""

    configure_pp_environment()
    working_names = [record['working_name'] for record in records]
    with working_directory(filter_dir):
        import pp_run

        magzp_keyword = 'MAGZP' if all(record.get('has_magzp')
                                       for record in records) else None
        magzp_sig_keyword = ('MAGZPSIG' if magzp_keyword is not None
                             else None)
        pp_run.run_the_pipeline(
            sorted(working_names), target, None, float(fixed_aprad),
            'high', False, False, False, True, telescope=TELESCOPE_KEY,
            posfile=Path(posfile).name, magzp_keyword=magzp_keyword,
            magzp_sig_keyword=magzp_sig_keyword,
            rejectionfilter=rejectionfilter)


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


def row_match_residual_arcsec(values):
    """Return total PP predicted-minus-measured residual from row values."""

    try:
        dra = float(values[6])
        ddec = float(values[7])
    except (IndexError, TypeError, ValueError):
        return None
    if not np.isfinite(dra) or not np.isfinite(ddec):
        return None
    return float(np.hypot(dra, ddec))


def forced_astrometry_only_row(values, target_position):
    """Replace a false PP source match with supplied astrometry-only values."""

    values = list(values)
    values[2] = '99.0000'
    values[3] = '99.0000'
    values[4] = '%.8f' % float(target_position['target_ra_deg'])
    values[5] = '%+.8f' % float(target_position['target_dec_deg'])
    values[6] = '0.00'
    values[7] = '0.00'
    if len(values) > 13:
        values[13] = '99.0000'
    if len(values) > 14:
        values[14] = '99.0000'
    for index in (21, 23):
        if len(values) > index:
            values[index] = '99.0000'
    return values


def rewrite_row_catalog_token(row, relative_dir, target_positions=None,
                              max_match_residual_arcsec=None,
                              rewrites=None):
    """Rewrite a per-directory PP row so root tools can resolve FITS."""

    values = row.split()
    if not values:
        return row
    original_token = values[0]
    if max_match_residual_arcsec is not None and target_positions:
        residual = row_match_residual_arcsec(values)
        target_position = target_positions.get(
            Path(original_token).with_suffix('.fits').name)
        if (residual is not None and
                residual > float(max_match_residual_arcsec) and
                target_position is not None):
            values = forced_astrometry_only_row(values, target_position)
            if rewrites is not None:
                rewrites.append({
                    'catalog_token': str(Path(relative_dir) / original_token),
                    'working_name': target_position.get('working_name'),
                    'residual_arcsec': residual,
                    'max_match_residual_arcsec': float(
                        max_match_residual_arcsec),
                    'ra_deg': float(target_position['target_ra_deg']),
                    'dec_deg': float(target_position['target_dec_deg']),
                    'position_source': target_position.get(
                        'position_source'),
                    'reason': 'PP source match exceeded residual threshold; '
                              'using supplied astrometry-only position',
                })
    values[0] = str(Path(relative_dir) / original_token)
    return ' ' + ' '.join(values) + '\n'


def combine_filter_photometry(pp_dir, target, outputs,
                              max_match_residual_arcsec=None):
    """Combine per-date/filter PP photometry into root-level products."""

    pp_dir = Path(pp_dir)
    existing = [output for output in outputs
                if output.get('photometry_file') and
                Path(output['photometry_file']).exists()]
    if not existing:
        return []

    first_header, first_rows, first_footer = split_photometry_file(
        existing[0]['photometry_file'])
    rewrites = []
    rows = [rewrite_row_catalog_token(
                row, existing[0]['relative_dir'],
                target_positions=existing[0].get('target_positions'),
                max_match_residual_arcsec=max_match_residual_arcsec,
                rewrites=rewrites)
            for row in first_rows]
    for output in existing[1:]:
        rows.extend(rewrite_row_catalog_token(
                    row, output['relative_dir'],
                    target_positions=output.get('target_positions'),
                    max_match_residual_arcsec=max_match_residual_arcsec,
                    rewrites=rewrites)
                    for row in split_photometry_file(
                        output['photometry_file'])[1])

    combined_text = ''.join(first_header + rows + first_footer)
    destinations = [
        pp_dir / target_photometry_filename(target),
        pp_dir / all_filters_photometry_filename(target),
    ]
    for destination in destinations:
        destination.write_text(combined_text)
    return {
        'files': [str(destination) for destination in destinations],
        'forced_astrometry_only_rows': rewrites,
    }


def grouped_records(records):
    """Return records grouped by date/filter output directory."""

    groups = {}
    for record in records:
        key = (record['date'], record['filter_dir_name'])
        groups.setdefault(key, []).append(record)
    return [(key, groups[key]) for key in sorted(groups)]


def run_pp_by_group(pp_dir, records, target, fixed_aprad=0.0,
                    rejectionfilter='pos'):
    """Run PP once per date/filter group and preserve per-group outputs."""

    outputs = []
    for (date, filt), group_records in grouped_records(records):
        filter_dir = Path(pp_dir) / date / filt
        relative_dir = str(filter_dir.relative_to(pp_dir))
        remove_stale_target_output(filter_dir, target)
        posfile = write_positions_file(filter_dir, group_records,
                                       target, filt)
        run_pp_pipeline(filter_dir, group_records, target, posfile,
                        fixed_aprad=fixed_aprad,
                        rejectionfilter=rejectionfilter)
        photometry_path = filter_dir / target_photometry_filename(target)
        outputs.append({
            'date': date,
            'filter': filt,
            'relative_dir': relative_dir,
            'positions_file': str(posfile),
            'uses_header_zeropoints': all(record.get('has_magzp')
                                          for record in group_records),
            'photometry_file': (str(photometry_path)
                                if photometry_path.exists() else None),
            'working_names': [record['working_name']
                              for record in group_records],
            'target_positions': {
                record['working_name']: {
                    'working_name': record['working_name'],
                    'target_ra_deg': float(record['target_ra_deg']),
                    'target_dec_deg': float(record['target_dec_deg']),
                    'position_source': record.get('position_source'),
                } for record in group_records
            },
        })
    return outputs


def write_positions_by_group(pp_dir, records, target):
    """Write one PP positions file for each date/filter group."""

    outputs = []
    for (date, filt), group_records in grouped_records(records):
        filter_dir = Path(pp_dir) / date / filt
        path = write_positions_file(filter_dir, group_records, target, filt)
        outputs.append({
            'date': date,
            'filter': filt,
            'relative_dir': str(filter_dir.relative_to(pp_dir)),
            'positions_file': str(path),
            'working_names': [record['working_name']
                              for record in group_records],
        })
    return outputs


def build_mpc_outputs(pp_dir, target, observatory_code='568',
                      program_code_contact=None):
    """Build ADES PSV and MPC 80-column files from combined photometry."""

    configure_pp_environment()
    import pptool_mpcsubmission as mpcsub

    pp_dir = Path(pp_dir)
    photometry_path = pp_dir / target_photometry_filename(target)
    config_kwargs = {
        'target': target,
        'observatory_code': str(observatory_code),
        'output_dir': pp_dir,
    }
    if program_code_contact is not None:
        config_kwargs['program_code_contact'] = program_code_contact
    config = mpcsub.SubmissionConfig(**config_kwargs)
    bundle = mpcsub.build_submission(photometry_path, config)
    mpcsub.write_submission(bundle)
    submission_manifest_path = pp_dir / ('mpc_%s_submission_manifest.json' %
                                         target_to_filename(target))
    submission_manifest_path.write_text(json.dumps({
        'target': target,
        'photometry_file': str(photometry_path),
        'observatory_code': str(observatory_code),
        'program_code': config.prog,
        'program_code_contact': config.program_code_contact,
        'observations': len(bundle.observations),
        'astrometry_only_observations': sum(
            1 for obs in bundle.observations if obs.astrometry_only),
        'ades_path': str(bundle.ades_path),
        'ades_xml_path': str(bundle.ades_xml_path),
        'obs80_path': str(bundle.obs80_path),
        'summary_path': str(bundle.summary_path),
        'note': ('Submission sidecars were generated from active PP '
                 'photometry rows; for clicked CFH12K workflows the '
                 'clicked+centroid branch is the preferred submission '
                 'branch.'),
    }, indent=2, sort_keys=True) + '\n')
    return {
        'observations': len(bundle.observations),
        'ades_path': str(bundle.ades_path),
        'ades_xml_path': str(bundle.ades_xml_path),
        'obs80_path': str(bundle.obs80_path),
        'summary_path': str(bundle.summary_path),
        'submission_manifest_path': str(submission_manifest_path),
        'program_code': bundle.observations and config.prog or None,
        'warnings': list(bundle.warnings),
    }


def find_find_orb_executable():
    """Return a local Find_Orb command from PATH or common Project Pluto builds."""

    env_path = os.environ.get('FIND_ORB_EXECUTABLE')
    if env_path and Path(env_path).expanduser().exists():
        return str(Path(env_path).expanduser().resolve())
    for name in ('fo', 'find_orb'):
        exe = shutil.which(name)
        if exe:
            return exe
    home = Path.home()
    candidates = []
    for github_name in ('Github', 'GitHub'):
        root = home / github_name / 'find_orb'
        candidates.extend([
            root / '.local' / 'bin' / 'fo',
            root / 'fo',
            root / '.local' / 'bin' / 'find_orb',
            root / 'find_orb',
        ])
    for candidate in candidates:
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            return str(candidate.resolve())
    return None


def run_find_orb_case(executable, input_path, case_dir):
    """Run one Find_Orb case, capturing stdout and stderr."""

    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = case_dir / 'fo_stdout.txt'
    stderr_path = case_dir / 'fo_stderr.txt'
    with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
        proc = subprocess.run([executable, str(input_path)],
                              cwd=str(case_dir), stdout=stdout,
                              stderr=stderr, timeout=300)
    summary = write_find_orb_summary(stdout_path, case_dir)
    return {
        'returncode': proc.returncode,
        'stdout': str(stdout_path),
        'stderr': str(stderr_path),
        'summary': summary,
        'files': sorted(path.name for path in case_dir.iterdir()),
    }


def strip_ansi(text):
    """Remove terminal color escapes from Find_Orb output."""

    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def parse_find_orb_stdout(text):
    """Parse the compact command-line fo orbit summary."""

    clean = strip_ansi(text)
    match = re.search(
        r'^\s*\d+:\s*(?P<object>.*?);\s*'
        r'a=(?P<a>[-+0-9.]+),\s*'
        r'e=(?P<e>[-+0-9.]+),\s*'
        r'i=(?P<i>[-+0-9.]+)\s+'
        r'(?:(?P<used_obs>\d+)\s*/\s*)?'
        r'(?P<n_obs>\d+)\s+obs;\s*'
        r'(?P<arc>.+?)\s*$',
        clean, flags=re.MULTILINE)
    if match is None:
        return None
    return {
        'object': match.group('object').strip(),
        'semimajor_axis_au': float(match.group('a')),
        'eccentricity': float(match.group('e')),
        'inclination_deg': float(match.group('i')),
        'observations': int(match.group('used_obs') or match.group('n_obs')),
        'total_observations': int(match.group('n_obs')),
        'arc': match.group('arc').strip(),
    }


def write_find_orb_summary(stdout_path, case_dir):
    """Write parsed Find_Orb stdout summary sidecars if possible."""

    stdout_path = Path(stdout_path)
    case_dir = Path(case_dir)
    parsed = parse_find_orb_stdout(stdout_path.read_text(errors='replace'))
    if parsed is None:
        return None
    json_path = case_dir / 'orbit_summary.json'
    md_path = case_dir / 'orbit_summary.md'
    json_path.write_text(json.dumps(parsed, indent=2, sort_keys=True) + '\n')
    md_path.write_text(
        '# Find_Orb Summary\n\n'
        '- Object: {object}\n'
        '- Observations used: {observations}\n'
        '- Observations total: {total_observations}\n'
        '- Arc: {arc}\n'
        '- Semimajor axis a: {semimajor_axis_au:.3f} au\n'
        '- Eccentricity e: {eccentricity:.3f}\n'
        '- Inclination i: {inclination_deg:.3f} deg\n'.format(**parsed))
    result = dict(parsed)
    result['json_path'] = str(json_path)
    result['markdown_path'] = str(md_path)
    return result


def prepare_orbit_check(pp_dir, target, mpc_outputs):
    """Prepare Find_Orb-style orbit-check inputs and run if possible."""

    pp_dir = Path(pp_dir)
    safe_target = target_to_filename(target)
    orbit_dir = pp_dir / 'find_orb_runs' / safe_target
    orbit_dir.mkdir(parents=True, exist_ok=True)
    source_obs80 = Path(mpc_outputs['obs80_path'])
    new_obs_path = orbit_dir / ('mpc_%s_cfh12k_only.obs80' % safe_target)
    new_obs_path.write_text(source_obs80.read_text())

    input_summary = orbit_dir / 'input_summary.txt'
    input_summary.write_text(
        'Find_Orb input preparation\n'
        'new CFH12K observations: %d\n'
        'source 80-column file: %s\n' %
        (int(mpc_outputs.get('observations', 0)), source_obs80))

    result = {
        'orbit_dir': str(orbit_dir),
        'new_observations_input': str(new_obs_path),
        'input_summary': str(input_summary),
        'warnings': [],
        'runs': [],
    }

    executable = find_find_orb_executable()
    if executable is None:
        result['warnings'].append('Find_Orb executable not found on PATH')
        return result
    result['executable'] = executable

    try:
        run = run_find_orb_case(executable, new_obs_path,
                                orbit_dir / 'cfh12k_only')
        run['key'] = 'cfh12k_only'
        result['runs'].append(run)
    except Exception as exc:
        result['warnings'].append('Find_Orb run failed: %s' % exc)
    return result


def write_manifest(pp_dir, manifest):
    """Write the workflow manifest JSON."""

    path = Path(pp_dir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return path


def build_workflow(base_dir, target, pp_dir=None,
                   magzp_sig=DEFAULT_MAGZP_SIG, click_points_csv=None,
                   click_pixel_origin=DEFAULT_CLICK_PIXEL_ORIGIN,
                   centroid=True,
                   centroid_search_radius=DEFAULT_CENTROID_SEARCH_RADIUS,
                   centroid_aperture_radius=DEFAULT_CENTROID_APERTURE_RADIUS,
                   centroid_min_snr=DEFAULT_CENTROID_MIN_SNR):
    """Prepare PP-ready CFH12K chip FITS files and positions sidecars."""

    base_dir = Path(base_dir).expanduser().resolve()
    if pp_dir is None:
        pp_dir = base_dir / 'PP'
    pp_dir = Path(pp_dir).expanduser().resolve()
    pp_dir.mkdir(parents=True, exist_ok=True)

    records, position_inputs = prepare_records(
        base_dir, pp_dir, target, magzp_sig=magzp_sig,
        click_points_csv=click_points_csv,
        click_pixel_origin=click_pixel_origin, centroid=centroid,
        centroid_search_radius=centroid_search_radius,
        centroid_aperture_radius=centroid_aperture_radius,
        centroid_min_snr=centroid_min_snr)
    positions_files = write_positions_by_group(pp_dir, records, target)
    manifest = {
        'workflow': 'cfht_cfh12k_locations',
        'telescope': TELESCOPE_KEY,
        'target': target,
        'base_dir': str(base_dir),
        'pp_dir': str(pp_dir),
        'magzp_sig_default': float(magzp_sig),
        'records': records,
        'position_inputs': position_inputs,
        'positions_files': positions_files,
        'filters': sorted(set(record['filter_dir_name']
                              for record in records)),
        'dates': sorted(set(record['date'] for record in records)),
        'warnings': sorted(set(warning
                               for record in records
                               for warning in record.get('warnings', []))),
    }
    manifest_path = write_manifest(pp_dir, manifest)
    manifest['manifest_path'] = str(manifest_path)
    return manifest


def run_single_workflow(args, pp_dir=None, centroid=None, branch_key=None):
    """Execute one CFH12K PP workflow branch."""

    if pp_dir is None:
        pp_dir = args.pp_dir
    if centroid is None:
        centroid = not args.no_centroid
    manifest = build_workflow(
        args.base_dir, args.target, pp_dir=pp_dir,
        magzp_sig=args.magzp_sig,
        click_points_csv=args.click_points_csv,
        click_pixel_origin=args.click_pixel_origin,
        centroid=centroid,
        centroid_search_radius=args.centroid_search_radius,
        centroid_aperture_radius=args.centroid_aperture_radius,
        centroid_min_snr=args.centroid_min_snr)
    if branch_key is not None:
        manifest['click_solution_branch'] = branch_key

    pp_dir = Path(manifest['pp_dir'])
    if not args.prepare_only:
        remove_stale_target_output(pp_dir, args.target)
        outputs = run_pp_by_group(
            pp_dir, manifest['records'], args.target,
            fixed_aprad=args.fixed_aprad, rejectionfilter=args.reject)
        manifest['pp_outputs'] = outputs
        combined = combine_filter_photometry(
            pp_dir, args.target, outputs,
            max_match_residual_arcsec=args.max_match_residual_arcsec)
        manifest['combined_photometry_files'] = combined['files']
        manifest['forced_astrometry_only_rows'] = combined[
            'forced_astrometry_only_rows']
        manifest['max_match_residual_arcsec'] = float(
            args.max_match_residual_arcsec)
        from pptool_pp_cutouts import record_pp_cutouts
        record_pp_cutouts(
            manifest, pp_dir, args.target,
            photometry_file=(manifest['combined_photometry_files'][0]
                             if manifest['combined_photometry_files']
                             else None))
        manifest['mpc_outputs'] = build_mpc_outputs(
            pp_dir, args.target, observatory_code=args.observatory_code,
            program_code_contact=args.program_code_contact)
        manifest['orbit_check'] = prepare_orbit_check(
            pp_dir, args.target, manifest['mpc_outputs'])
        manifest['warnings'].extend(
            manifest['orbit_check'].get('warnings', []))
        if args.recover_photometry:
            from pptool_forced_photometry import build_recovery
            try:
                manifest['photometry_recovery'] = build_recovery(
                    manifest['combined_photometry_files'][0], args.target,
                    output_dir=(Path(args.photometry_recovery_dir)
                                if args.photometry_recovery_dir else
                                pp_dir / 'photometry_recovery'),
                    reference_catalog=args.photometry_reference_catalog,
                    snr_threshold=args.photometry_snr_threshold)
            except Exception as exc:
                manifest.setdefault('warnings', []).append(
                    'photometry-only recovery failed: %s' % exc)
        manifest_path = write_manifest(pp_dir, manifest)
        manifest['manifest_path'] = str(manifest_path)
    return manifest


def mirror_selected_branch_outputs(root_pp_dir, selected_manifest, target):
    """Copy selected branch products to the PP root default locations."""

    root_pp_dir = Path(root_pp_dir)
    source_pp_dir = Path(selected_manifest['pp_dir'])
    mirrored = {}
    for key in ('combined_photometry_files',):
        mirrored[key] = []
        for source in selected_manifest.get(key, []):
            source_path = Path(source)
            destination = root_pp_dir / source_path.name
            shutil.copy2(source_path, destination)
            mirrored[key].append(str(destination))
    mpc_outputs = {}
    for key in ('ades_path', 'ades_xml_path', 'obs80_path', 'summary_path',
                'submission_manifest_path'):
        source = selected_manifest.get('mpc_outputs', {}).get(key)
        if not source:
            continue
        source_path = Path(source)
        destination = root_pp_dir / source_path.name
        shutil.copy2(source_path, destination)
        mpc_outputs[key] = str(destination)
    if mpc_outputs:
        mpc_outputs['observations'] = selected_manifest.get(
            'mpc_outputs', {}).get('observations')
        mpc_outputs['program_code'] = selected_manifest.get(
            'mpc_outputs', {}).get('program_code')
        mpc_outputs['warnings'] = list(selected_manifest.get(
            'mpc_outputs', {}).get('warnings', []))
        mirrored['mpc_outputs'] = mpc_outputs

    note_path = root_pp_dir / ('mpc_%s_submission_manifest.json' %
                              target_to_filename(target))
    note = {
        'target': target,
        'selected_click_solution_branch': selected_manifest.get(
            'click_solution_branch'),
        'selected_pp_dir': str(source_pp_dir),
        'program_code': selected_manifest.get(
            'mpc_outputs', {}).get('program_code'),
        'observations': selected_manifest.get(
            'mpc_outputs', {}).get('observations'),
        'note': ('Default submission files were mirrored from the '
                 'clicked+centroid branch selected for submission and '
                 'the forced MPFit comparison.'),
    }
    note_path.write_text(json.dumps(note, indent=2, sort_keys=True) + '\n')
    mirrored['submission_manifest'] = str(note_path)
    return mirrored


def write_root_branch_manifest(root_pp_dir, manifest):
    """Write the root manifest for a multi-branch click workflow."""

    path = Path(root_pp_dir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return path


def run_click_branch_workflow(args):
    """Run one or both click-position solution branches."""

    base_dir = Path(args.base_dir).expanduser().resolve()
    root_pp_dir = (Path(args.pp_dir).expanduser().resolve()
                   if args.pp_dir else base_dir / 'PP')
    root_pp_dir.mkdir(parents=True, exist_ok=True)

    if args.click_solution_mode == CLICK_SOLUTION_BOTH:
        branch_keys = [CLICK_SOLUTION_CENTROID, CLICK_SOLUTION_PURE]
    else:
        branch_keys = [args.click_solution_mode]

    branches = {}
    for branch_key in branch_keys:
        branch_dir_name, centroid = CLICK_BRANCHES[branch_key]
        branch_manifest = run_single_workflow(
            args, pp_dir=root_pp_dir / branch_dir_name,
            centroid=centroid, branch_key=branch_key)
        branches[branch_key] = branch_manifest

    selected = (CLICK_SOLUTION_CENTROID
                if CLICK_SOLUTION_CENTROID in branches else branch_keys[0])
    root_manifest = {
        'workflow': 'cfht_cfh12k_click_branches',
        'telescope': TELESCOPE_KEY,
        'target': args.target,
        'base_dir': str(base_dir),
        'pp_dir': str(root_pp_dir),
        'click_solution_mode': args.click_solution_mode,
        'selected_click_solution_branch': selected,
        'branches': branches,
        'warnings': [],
    }
    if not args.prepare_only:
        root_manifest['mirrored_selected_outputs'] = (
            mirror_selected_branch_outputs(root_pp_dir, branches[selected],
                                           args.target))
    manifest_path = write_root_branch_manifest(root_pp_dir, root_manifest)
    root_manifest['manifest_path'] = str(manifest_path)
    write_root_branch_manifest(root_pp_dir, root_manifest)
    return root_manifest


def run_workflow(args):
    """Execute the full CFH12K PP workflow."""

    if args.click_points_csv and args.click_solution_mode in CLICK_BRANCHES:
        return run_click_branch_workflow(args)
    if (args.click_points_csv and
            args.click_solution_mode == CLICK_SOLUTION_BOTH):
        return run_click_branch_workflow(args)
    return run_single_workflow(args)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='prepare and run PP for CFHT/CFH12K CADC products')
    parser.add_argument('base_dir',
                        help='CFHT_CFH12K CADC directory')
    parser.add_argument('--target', default='2025 MH348',
                        help='target name to use in PP/submission outputs')
    parser.add_argument('--pp-dir',
                        help='output PP directory; defaults to base_dir/PP')
    parser.add_argument('--magzp-sig', type=float,
                        default=DEFAULT_MAGZP_SIG,
                        help='fallback MAGZPSIG when PHOT_CS is absent')
    parser.add_argument('--fixed-aprad', type=float, default=0.0,
                        help='fixed PP aperture radius; 0 uses curve of growth')
    parser.add_argument('--reject', default='pos',
                        help='PP target rejection schema, e.g. pos or none')
    parser.add_argument('--observatory-code', default='568',
                        help='MPC observatory code for submission files')
    parser.add_argument('--program-code-contact',
                        default='C. O. Chandler',
                        help='MPC program-code contact lookup key')
    parser.add_argument('--click-points-csv',
                        help='clicked PNG positions to translate through '
                             'cutout WCS and use instead of locations CSV '
                             'positions; only clicked rows are processed')
    parser.add_argument('--click-solution-mode',
                        choices=(CLICK_SOLUTION_BOTH,
                                 CLICK_SOLUTION_CENTROID,
                                 CLICK_SOLUTION_PURE),
                        default=CLICK_SOLUTION_BOTH,
                        help='clicked-position branch strategy')
    parser.add_argument('--click-pixel-origin', type=int,
                        default=DEFAULT_CLICK_PIXEL_ORIGIN,
                        help='FITS WCS origin for click coordinates')
    parser.add_argument('--no-centroid', action='store_true',
                        help='use click-derived WCS positions without local '
                             'full-chip centroid refinement')
    parser.add_argument('--centroid-search-radius', type=float,
                        default=DEFAULT_CENTROID_SEARCH_RADIUS,
                        help='full-chip search radius in pixels around click '
                             'position')
    parser.add_argument('--centroid-aperture-radius', type=float,
                        default=DEFAULT_CENTROID_APERTURE_RADIUS,
                        help='centroid aperture radius in pixels')
    parser.add_argument('--centroid-min-snr', type=float,
                        default=DEFAULT_CENTROID_MIN_SNR,
                        help='minimum local SNR for centroid peak selection')
    parser.add_argument('--max-match-residual-arcsec', type=float,
                        default=DEFAULT_MAX_PP_MATCH_RESIDUAL_ARCSEC,
                        help='convert PP source matches farther than this '
                             'from the supplied position into active '
                             'astrometry-only rows')
    parser.add_argument('--prepare-only', action='store_true',
                        help='write PP-ready FITS/positions but do not run PP')
    parser.add_argument('--recover-photometry', action='store_true',
                        help='after PP, run fixed-position forced photometry '
                             'into photometry_recovery without changing '
                             'submission astrometry')
    parser.add_argument('--photometry-recovery-dir',
                        help='override forced-photometry recovery directory')
    parser.add_argument('--photometry-reference-catalog',
                        help='optional CSV reference-star catalog for '
                             'photometry-only zeropoint fitting')
    parser.add_argument('--photometry-snr-threshold', type=float, default=5.0,
                        help='minimum SNR for accepted recovered magnitudes')
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    manifest = run_workflow(args)
    print(json.dumps({
        'manifest_path': manifest['manifest_path'],
        'pp_dir': manifest['pp_dir'],
        'n_records': len(manifest.get('records', [])),
        'filters': manifest.get('filters', []),
        'dates': manifest.get('dates', []),
        'click_solution_mode': manifest.get('click_solution_mode'),
        'selected_click_solution_branch': manifest.get(
            'selected_click_solution_branch'),
        'pp_outputs': manifest.get('pp_outputs', []),
        'combined_photometry_files': manifest.get(
            'combined_photometry_files', []),
        'mpc_outputs': manifest.get('mpc_outputs'),
        'orbit_check': manifest.get('orbit_check'),
        'photometry_recovery': manifest.get('photometry_recovery'),
        'warnings': manifest.get('warnings', []),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
