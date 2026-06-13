#!/usr/bin/env python3

"""Workflow helper for Catalina Lemmon 60-inch archival images."""

from __future__ import print_function

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tarfile
from contextlib import contextmanager

from astropy.io import fits
from astropy.time import Time


TELESCOPE_KEY = 'CATALINALEM60'
BACKUP_NAME = 'originals_backup.tar'
MANIFEST_NAME = 'catalina_lemmon60_workflow_manifest.json'
WORKING_PREFIX = 'cl60'
TARGET_TRANSLATION = str.maketrans(' /()', '____')


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
    """Find raw compressed Catalina/Lemmon FITS images."""

    return sorted(path for path in image_dir.iterdir()
                  if path.is_file() and path.suffix.lower() == '.fz')


def backup_candidates(image_dir, raw_images):
    """Return raw inputs and source sidecars that should be archived."""

    sidecars = ['downloads.jsonl', 'full_images_manifest.json',
                'download_plan.sh']
    candidates = list(raw_images)
    for name in sidecars:
        path = image_dir / name
        if path.exists():
            candidates.append(path)
    return sorted(set(candidates))


def create_originals_backup(image_dir, raw_images, refresh=False):
    """Create or reuse the tar backup of the raw source files."""

    backup_path = image_dir / BACKUP_NAME
    if backup_path.exists() and not refresh:
        return {'path': str(backup_path), 'created': False}

    if backup_path.exists():
        backup_path.unlink()

    with tarfile.open(str(backup_path), 'w') as tar:
        for path in backup_candidates(image_dir, raw_images):
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
    prefixes = ('Z',)
    remove_exact = {
        'XTENSION', 'EXTNAME', 'PCOUNT', 'GCOUNT', 'CHECKSUM', 'DATASUM',
        'THEAP', 'TFIELDS',
    }
    for key in list(cleaned.keys()):
        if key in remove_exact or key.startswith(prefixes):
            try:
                cleaned.remove(key, remove_all=True, ignore_missing=True)
            except TypeError:
                cleaned.remove(key)
    return cleaned


def convert_to_pp_fits(raw_path, working_path, original_index):
    """Flatten a compressed image-extension FITS into a PP-ready FITS."""

    with fits.open(str(raw_path), memmap=False,
                   ignore_missing_end=True) as hdulist:
        hdulist.verify('fix')
        hdu_index = science_hdu_index(hdulist)
        image_hdu = hdulist[hdu_index]
        header = strip_extension_only_cards(image_hdu.header)
        header['PPINSTRU'] = (TELESCOPE_KEY, 'PP telescope override')
        header['ORIGFILE'] = (raw_path.name[:68], 'source FITS filename')
        header['ORIGEXT'] = (hdu_index, 'source FITS extension')
        header['PPSEQ'] = (original_index, 'Catalina Lemmon workflow index')
        fits.PrimaryHDU(data=image_hdu.data, header=header).writeto(
            str(working_path), overwrite=True, output_verify='silentfix')

    with fits.open(str(working_path), ignore_missing_end=True) as hdulist:
        header = hdulist[0].header
        return {
            'raw': str(raw_path),
            'working': str(working_path),
            'working_name': working_path.name,
            'source_hdu': hdu_index,
            'image_index': original_index - 1,
            'date_obs': header.get('DATE-OBS'),
            'time_obs': header.get('TIME-OBS'),
            'filter': header.get('FILTER'),
            'magzp': header.get('MAGZP'),
            'magzp_sig': header.get('PHOTIRMS'),
        }


def prepare_working_fits(image_dir, raw_images):
    """Create fresh PP-ready FITS files from the raw images."""

    records = []
    for idx, raw_path in enumerate(raw_images, start=1):
        working_path = image_dir / ('%s_%04d.fits' % (WORKING_PREFIX, idx))
        records.append(convert_to_pp_fits(raw_path, working_path, idx))
    return records


def write_manifest(image_dir, manifest):
    """Write the workflow manifest JSON."""

    path = image_dir / MANIFEST_NAME
    with path.open('w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return path


def target_photometry_filename(target):
    """Return PP's photometry filename for a target name."""

    return 'photometry_%s.dat' % target.translate(TARGET_TRANSLATION)


def has_usable_target_rows(photometry_path):
    """Check whether PP distilled at least one non-comment target row."""

    if not photometry_path.exists():
        return False
    with photometry_path.open() as handle:
        return any(line.strip() and not line.lstrip().startswith('#')
                   for line in handle)


def remove_stale_target_output(image_dir, target):
    """Remove the canonical target photometry file before a fresh run."""

    photometry_path = image_dir / target_photometry_filename(target)
    if photometry_path.exists():
        photometry_path.unlink()


def run_pp_pipeline(image_dir, working_names, target):
    """Run pp_run on the PP-ready images with kept WCS and header ZPs."""

    configure_pp_environment()
    import pp_run

    with working_directory(image_dir):
        pp_run.run_the_pipeline(
            sorted(working_names), target, None, 0, 'high', False, False,
            False, True, telescope=TELESCOPE_KEY, magzp_keyword='MAGZP',
            magzp_sig_keyword='PHOTIRMS')


def load_group_positions(impact_summary, group_id):
    """Load fallback source positions from the orbit impact summary."""

    rows = []
    with Path(impact_summary).open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row['group_id']) == int(group_id):
                rows.append(row)

    if not rows:
        raise ValueError('no rows found for group %s in %s' %
                         (group_id, impact_summary))

    return sorted(rows, key=lambda row: int(row['image_index']))


def write_positions_file(image_dir, rows, working_by_image_index, target,
                         group_id):
    """Write PP -positions input for the selected fallback group."""

    path = image_dir / ('positions_group%03d.dat' % int(group_id))
    ident = target.translate(TARGET_TRANSLATION)
    with path.open('w') as handle:
        for row in rows:
            image_index = int(row['image_index'])
            if image_index not in working_by_image_index:
                raise KeyError('image index %d is not in the working manifest' %
                               image_index)
            jd = Time(row['isot'], format='isot', scale='utc').jd
            handle.write('%s %.10f %.10f %.8f %s\n' %
                         (working_by_image_index[image_index],
                          float(row['ra_deg']), float(row['dec_deg']),
                          jd, ident))
    return path


def run_group_fallback(image_dir, manifest_records, impact_summary,
                       group_id, target):
    """Rerun only pp_distill using the neutral-impact manual positions."""

    configure_pp_environment()
    import pp_distill

    rows = load_group_positions(impact_summary, group_id)
    working_by_image_index = {record['image_index']: record['working_name']
                              for record in manifest_records}
    positions_path = write_positions_file(
        image_dir, rows, working_by_image_index, target, group_id)
    fallback_files = [working_by_image_index[int(row['image_index'])]
                      for row in rows]

    with working_directory(image_dir):
        pp_distill.distill(
            sorted(fallback_files), target, [0, 0], None,
            positions_path.name, rejectionfilter='none', display=True,
            diagnostics=True)

    return {'positions_file': str(positions_path),
            'working_files': sorted(fallback_files),
            'rejectionfilter': 'none',
            'rows': len(rows)}


def run_workflow(args):
    """Execute the Catalina/Lemmon PP workflow."""

    image_dir = Path(args.full_images_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(str(image_dir))

    raw_images = discover_raw_images(image_dir)
    if not raw_images:
        raise FileNotFoundError('no .fz images found in %s' % image_dir)

    manifest = {
        'workflow': 'catalina_lemmon60',
        'target': args.target,
        'telescope': TELESCOPE_KEY,
        'image_dir': str(image_dir),
    }
    manifest['backup'] = create_originals_backup(
        image_dir, raw_images, refresh=args.refresh_backup)
    manifest['images'] = prepare_working_fits(image_dir, raw_images)
    manifest['pp_run'] = {
        'keep_wcs': True,
        'magzp_keyword': 'MAGZP',
        'magzp_sig_keyword': 'PHOTIRMS',
    }
    write_manifest(image_dir, manifest)

    working_names = [record['working_name'] for record in manifest['images']]
    remove_stale_target_output(image_dir, args.target)
    run_pp_pipeline(image_dir, working_names, args.target)

    photometry_path = image_dir / target_photometry_filename(args.target)
    manifest['photometry_file'] = str(photometry_path)
    manifest['fallback_used'] = False

    if not has_usable_target_rows(photometry_path):
        if args.impact_summary is None:
            raise ValueError('PP did not produce usable target rows; provide '
                             '--impact-summary for group fallback')
        manifest['fallback'] = run_group_fallback(
            image_dir, manifest['images'],
            Path(args.impact_summary).expanduser().resolve(),
            args.fallback_group, args.target)
        manifest['fallback_used'] = True

    from pptool_pp_cutouts import record_pp_cutouts
    record_pp_cutouts(manifest, image_dir, args.target,
                      photometry_file=photometry_path)
    manifest_path = write_manifest(image_dir, manifest)
    return manifest_path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='run the Catalina Lemmon 60-inch PP workflow')
    parser.add_argument('full_images_dir',
                        help='directory containing Catalina/Lemmon .fz files')
    parser.add_argument('--target', default='2020 VS6',
                        help='target name for PP and fallback positions')
    parser.add_argument('--impact-summary',
                        help='orbit impact summary CSV used for fallback')
    parser.add_argument('--fallback-group', type=int, default=3,
                        help='impact-summary group to use for manual fallback')
    parser.add_argument('--refresh-backup', action='store_true',
                        help='recreate originals_backup.tar if it exists')
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    manifest_path = run_workflow(args)
    print('Catalina Lemmon workflow manifest:', manifest_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
