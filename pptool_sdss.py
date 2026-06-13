#!/usr/bin/env python3

"""Workflow helper for SDSS CADC image products."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import sys
from contextlib import contextmanager


RUNTIME_CACHE_ROOT = Path(os.environ.get(
    "TMPDIR", "/private/tmp")) / "photometrypipeline_runtime"
for env_name, child in (
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_CONFIG_HOME", "config"),
        ("MPLCONFIGDIR", "matplotlib"),
):
    cache_path = RUNTIME_CACHE_ROOT / child
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(env_name, str(cache_path))

from astropy import units as u
from astropy.config.paths import set_temp_cache, set_temp_config
from astropy.io import fits
from astropy.time import Time

_ASTROPY_CONFIG_CONTEXT = set_temp_config(
    str(RUNTIME_CACHE_ROOT / "astropy_config"), delete=False)
_ASTROPY_CACHE_CONTEXT = set_temp_cache(
    str(RUNTIME_CACHE_ROOT / "astropy_cache"), delete=False)
for astropy_path in (RUNTIME_CACHE_ROOT / "astropy_config",
                     RUNTIME_CACHE_ROOT / "astropy_cache"):
    astropy_path.mkdir(parents=True, exist_ok=True)
_ASTROPY_CONFIG_CONTEXT.__enter__()
_ASTROPY_CACHE_CONTEXT.__enter__()

from pptool_mpcsubmission import (
    PHOTOMETRY_COLUMNS,
    SubmissionError,
    parse_photometry_file,
    resolve_fits_filename,
    target_to_filename,
)


TELESCOPE_KEY = "SDSS"
MANIFEST_NAME = "sdss_workflow_manifest.json"
CUTOUT_PATTERN = re.compile(r"_chip\d+_126arcsec_cutout\.fits$")
SDSS_FILTERS = {"u", "g", "r", "i", "z"}


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
    os.environ["PHOTPIPEDIR"] = str(root)
    runtime_home = RUNTIME_CACHE_ROOT / "home"
    runtime_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(runtime_home)
    bin_dir = str(Path(sys.executable).resolve().parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in path_parts:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def normalize_filter(value):
    """Translate SDSS filter labels to PP filter names."""

    filt = str(value).strip().lower()
    if filt not in SDSS_FILTERS:
        raise ValueError("unsupported SDSS filter %r" % value)
    return filt


def filter_from_header_or_name(header, path):
    """Read the SDSS filter from a FITS header, falling back to the filename."""

    if "FILTER" in header:
        return normalize_filter(header["FILTER"])

    match = re.search(r"_([ugriz])_exp", Path(path).name)
    if match:
        return normalize_filter(match.group(1))

    match = re.search(r"_frame-([ugriz])-", Path(path).name)
    if match:
        return normalize_filter(match.group(1))

    raise KeyError("cannot identify SDSS filter for %s" % path)


def date_from_header(header):
    """Return YYYY-MM-DD from SDSS DATE-OBS cards."""

    date_obs = str(header["DATE-OBS"]).strip().replace("/", "-")
    return date_obs.split("T", 1)[0][:10]


def midtime_jd_from_header(header):
    """Return exposure midtime JD using the DATE-OBS and TAIHMS cards."""

    start = "%sT%s" % (date_from_header(header), str(header["TAIHMS"]).strip())
    exptime = float(header.get("EXPTIME", 0.0) or 0.0)
    return float((Time(start, format="isot", scale="utc") +
                  (exptime / 2.0) * u.second).jd)


def _coerce_cutout_path(path, sdss_dir):
    """Resolve a cutout path, accepting PNG names when the FITS twin exists."""

    sdss_dir = Path(sdss_dir).expanduser().resolve()
    path = Path(path).expanduser()
    if not path.is_absolute():
        direct = path.resolve()
        candidate = sdss_dir / "cutouts" / path
        path = direct if direct.exists() else candidate

    if path.suffix.lower() != ".fits":
        fits_twin = path.with_suffix(".fits")
        if fits_twin.exists():
            path = fits_twin

    path = path.resolve()
    if not path.exists():
        raise IOError("cutout does not exist: %s" % path)
    if not CUTOUT_PATTERN.search(path.name):
        raise ValueError("not an original 126 arcsec SDSS cutout FITS: %s" %
                         path)
    return path


def discover_cutouts(sdss_dir):
    """Find original SDSS target cutout FITS products."""

    cutout_dir = Path(sdss_dir).expanduser().resolve() / "cutouts"
    if not cutout_dir.is_dir():
        raise IOError("missing SDSS cutout directory: %s" % cutout_dir)
    return sorted(path for path in cutout_dir.iterdir()
                  if path.is_file() and CUTOUT_PATTERN.search(path.name))


def full_image_for_cutout(cutout_path, sdss_dir):
    """Map one CADC cutout filename to its SDSS full-frame image."""

    cutout_path = Path(cutout_path)
    name = cutout_path.name.replace("_CADC_frame-", "_SDSS_frame-")
    name = CUTOUT_PATTERN.sub("_image.fits", name)
    if name == cutout_path.name:
        raise ValueError("cannot derive full-frame image name from %s" %
                         cutout_path)
    full_frame = Path(sdss_dir).expanduser().resolve() / name
    if not full_frame.exists():
        raise IOError("missing full-frame image for %s: %s" %
                      (cutout_path, full_frame))
    return full_frame


def selected_image_records(sdss_dir, cutouts=None):
    """Return cutout/full-frame pairs selected for PP processing."""

    sdss_dir = Path(sdss_dir).expanduser().resolve()
    if cutouts is None:
        cutout_paths = discover_cutouts(sdss_dir)
    else:
        cutout_paths = [_coerce_cutout_path(path, sdss_dir)
                        for path in cutouts]

    records = []
    for cutout in sorted(cutout_paths):
        raw = full_image_for_cutout(cutout, sdss_dir)
        records.append({
            "cutout": cutout,
            "raw": raw,
        })
    return records


def stamp_sdss_header(path, raw_path, cutout_path, target, index):
    """Stamp copied SDSS FITS images with PP and provenance headers."""

    with fits.open(str(path), mode="update", memmap=False,
                   ignore_missing_end=True) as hdulist:
        header = hdulist[0].header
        filt = filter_from_header_or_name(header, path)
        header["PPINSTRU"] = (TELESCOPE_KEY, "PP telescope override")
        header["TEL_KEYW"] = (TELESCOPE_KEY, "PP telescope keyword")
        header["TELESCOP"] = ("2.5-m SDSS telescope", "PP normalized telescope")
        header["INSTRUME"] = ("SDSS imaging camera", "PP normalized instrument")
        header["FILTER"] = (filt, "PP normalized filter")
        header["OBJECT"] = (target, "PP target name")
        header["ORIGFILE"] = (Path(raw_path).name[:68], "source FITS filename")
        header["SDSSCUT"] = (Path(cutout_path).name[:68], "source cutout")
        header["PPSEQ"] = (int(index), "SDSS workflow index")
        hdulist.flush(output_verify="silentfix")


def prepare_pp_tree(sdss_dir, output_root, target, cutouts=None,
                    overwrite=False):
    """Copy selected SDSS full frames into PP/date/filter directories."""

    sdss_dir = Path(sdss_dir).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    selected = selected_image_records(sdss_dir, cutouts=cutouts)
    if not selected:
        raise IOError("no SDSS cutouts selected below %s" % sdss_dir)

    records = []
    for index, record in enumerate(selected, start=1):
        raw = record["raw"]
        cutout = record["cutout"]
        with fits.open(str(raw), memmap=False,
                       ignore_missing_end=True) as hdulist:
            header = hdulist[0].header
            filt = filter_from_header_or_name(header, raw)
            date_obs = date_from_header(header)
            midtimjd = midtime_jd_from_header(header)
            exptime = float(header.get("EXPTIME", 0.0) or 0.0)

        destination_dir = output_root / date_obs / filt
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / raw.name
        copied = False
        if overwrite or not destination.exists():
            shutil.copy2(str(raw), str(destination))
            copied = True
        stamp_sdss_header(destination, raw, cutout, target, index)
        records.append({
            "raw": str(raw),
            "cutout": str(cutout),
            "working": str(destination),
            "working_dir": str(destination_dir),
            "working_name": destination.name,
            "filter": filt,
            "date_obs": date_obs,
            "midtimjd": midtimjd,
            "exptime": exptime,
            "copied": copied,
        })

    return {
        "telescope": TELESCOPE_KEY,
        "target": target,
        "observatory_code": "645",
        "astrometry_catalogs": ["SDSS-R9", "GAIA"],
        "photometry_catalogs": ["SDSS-R9"],
        "keep_wcs": True,
        "registration": "skipped",
        "source_dir": str(sdss_dir),
        "output_root": str(output_root),
        "records": records,
    }


def target_photometry_filename(target):
    """Return PP's photometry filename for a target name."""

    return "photometry_%s.dat" % target_to_filename(target)


def target_photometry_filenames(target):
    """Return canonical and Horizons-decorated PP target photometry names."""

    translation = str.maketrans(" /()", "____")
    stems = [
        target_to_filename(target),
        ("(%s)" % target).translate(translation),
    ]
    return ["photometry_%s.dat" % stem for stem in dict.fromkeys(stems)]


def find_target_photometry_file(image_dir, target):
    """Find PP's target photometry output for one image directory."""

    image_dir = Path(image_dir)
    for name in target_photometry_filenames(target):
        path = image_dir / name
        if path.exists():
            return path
    return None


def ensure_canonical_target_photometry(image_dir, target):
    """Copy PP's target output to the canonical target filename if needed."""

    source = find_target_photometry_file(image_dir, target)
    if source is None:
        return None
    destination = Path(image_dir) / target_photometry_filename(target)
    if source != destination:
        shutil.copyfile(str(source), str(destination))
    return destination


def remove_stale_target_output(image_dir, target):
    """Remove PP target photometry outputs before a fresh run."""

    for name in target_photometry_filenames(target):
        path = Path(image_dir) / name
        if path.exists():
            path.unlink()


def run_pp_pipeline(image_dir, working_names, target,
                    fixed_aprad=0.0, rejectionfilter="pos"):
    """Run PP on one SDSS date/filter directory with kept WCS."""

    configure_pp_environment()
    import pp_run

    with working_directory(image_dir):
        pp_run.run_the_pipeline(
            list(working_names), target, None, float(fixed_aprad), "high",
            False, False, False, True, telescope=TELESCOPE_KEY,
            rejectionfilter=rejectionfilter)


def run_pp_by_directory(records, target, fixed_aprad=0.0,
                        rejectionfilter="pos"):
    """Run PP once for each prepared PP/date/filter directory."""

    grouped = {}
    for record in records:
        grouped.setdefault(record["working_dir"], []).append(record)

    outputs = []
    for image_dir in sorted(grouped):
        image_records = grouped[image_dir]
        remove_stale_target_output(image_dir, target)
        working_names = [record["working_name"] for record in image_records]
        run_pp_pipeline(image_dir, working_names, target,
                        fixed_aprad=fixed_aprad,
                        rejectionfilter=rejectionfilter)
        photometry_path = ensure_canonical_target_photometry(image_dir,
                                                             target)
        outputs.append({
            "image_dir": image_dir,
            "filter": image_records[0]["filter"],
            "working_names": working_names,
            "photometry_file": (str(photometry_path)
                                if photometry_path.exists() else None),
        })
    return outputs


def discover_target_photometry_files(output_root, target):
    """Find PP target photometry files below the SDSS output tree."""

    output_root = Path(output_root).expanduser().resolve()
    name = target_photometry_filename(target)
    canonical = sorted(path for path in output_root.rglob(name)
                       if path.is_file())
    if canonical:
        return canonical

    paths = []
    for candidate in target_photometry_filenames(target):
        paths.extend(path for path in output_root.rglob(candidate)
                     if path.is_file())
    return sorted(set(paths))


def write_combined_photometry_csv(output_root, target, output_path=None):
    """Write a CSV combining active rows from all SDSS PP photometry files."""

    output_root = Path(output_root).expanduser().resolve()
    if output_path is None:
        output_path = output_root / ("photometry_%s_combined.csv" %
                                     target_to_filename(target))
    else:
        output_path = Path(output_path).expanduser().resolve()

    rows = []
    for photometry_file in discover_target_photometry_files(output_root,
                                                            target):
        for row in parse_photometry_file(photometry_file, require_rows=False):
            source_fits = resolve_fits_filename(photometry_file,
                                                row["catalog_token"])
            combined = dict(row)
            combined["source_photometry_file"] = str(photometry_file)
            combined["source_fits_file"] = str(source_fits)
            rows.append(combined)

    fieldnames = list(PHOTOMETRY_COLUMNS) + [
        "source_photometry_file",
        "source_fits_file",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path, rows


def write_manifest(output_root, manifest):
    """Write the workflow manifest JSON."""

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / MANIFEST_NAME
    with path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def build_workflow(sdss_dir, output_root, target, cutouts=None,
                   overwrite=False):
    """Prepare selected SDSS full frames for PP."""

    manifest = prepare_pp_tree(sdss_dir, output_root, target,
                               cutouts=cutouts, overwrite=overwrite)
    manifest_path = write_manifest(output_root, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def run_workflow(sdss_dir, output_root, target, cutouts=None,
                 overwrite=False, prepare_only=False, fixed_aprad=0.0,
                 rejectionfilter="pos"):
    """Prepare SDSS inputs, optionally run PP, and write summary sidecars."""

    manifest = build_workflow(sdss_dir, output_root, target, cutouts=cutouts,
                              overwrite=overwrite)
    if not prepare_only:
        manifest["pp_outputs"] = run_pp_by_directory(
            manifest["records"], target, fixed_aprad=fixed_aprad,
            rejectionfilter=rejectionfilter)
        try:
            combined_path, rows = write_combined_photometry_csv(
                output_root, target)
        except SubmissionError as exc:
            manifest["combined_photometry_error"] = str(exc)
        else:
            manifest["combined_photometry_csv"] = str(combined_path)
            manifest["combined_photometry_rows"] = len(rows)
        from pptool_pp_cutouts import record_pp_cutouts
        record_pp_cutouts(manifest, output_root, target)
        manifest_path = write_manifest(output_root, manifest)
        manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="prepare and run PP for SDSS CADC products")
    parser.add_argument("sdss_dir",
                        help="directory with SDSS full frames and cutouts/")
    parser.add_argument("-target", "--target", default="2003 WW218",
                        help="target name to use in PP outputs")
    parser.add_argument("--output-root", default=str(repo_root() / "PP"),
                        help="PP output root")
    parser.add_argument("--cutout", action="append",
                        help="specific cutout FITS/PNG to process; repeatable")
    parser.add_argument("--overwrite", action="store_true",
                        help="refresh copied FITS files in PP output dirs")
    parser.add_argument("--prepare-only", action="store_true",
                        help="copy/stamp FITS files but do not run PP")
    parser.add_argument("--fixed-aprad", type=float, default=0.0,
                        help="fixed PP aperture radius; 0 uses curve of growth")
    parser.add_argument("--reject", default="pos",
                        help="PP target rejection schema, e.g. pos or none")

    args = parser.parse_args(argv)
    manifest = run_workflow(
        args.sdss_dir, args.output_root, args.target, cutouts=args.cutout,
        overwrite=args.overwrite, prepare_only=args.prepare_only,
        fixed_aprad=args.fixed_aprad, rejectionfilter=args.reject)

    print(json.dumps({
        "manifest_path": manifest["manifest_path"],
        "n_records": len(manifest["records"]),
        "filters": sorted(set(record["filter"]
                              for record in manifest["records"])),
        "pp_outputs": manifest.get("pp_outputs", []),
        "combined_photometry_csv": manifest.get("combined_photometry_csv"),
        "combined_photometry_rows": manifest.get("combined_photometry_rows"),
        "combined_photometry_error": manifest.get(
            "combined_photometry_error"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
