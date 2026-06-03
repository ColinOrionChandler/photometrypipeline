#!/usr/bin/env python3

"""Workflow helper for WHT Prime CASU image products."""

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

from astropy import units as u
from astropy.io import fits
from astropy.time import Time

from pptool_mpcsubmission import (
    PHOTOMETRY_COLUMNS,
    SubmissionError,
    parse_photometry_file,
    resolve_fits_filename,
    target_to_filename,
)


TELESCOPE_KEY = "WHTPFIP"
MANIFEST_NAME = "wht_prime_workflow_manifest.json"
TARGET_TRANSLATION = str.maketrans(" /()", "____")
SOURCE_RE = re.compile(r"wht(?P<date>\d{8})_(?P<recno>\d+)\.fits")
SCAMP_WCS_KEYS = {
    "EQUINOX",
    "RADESYS",
    "RADECSYS",
    "CTYPE1",
    "CTYPE2",
    "CUNIT1",
    "CUNIT2",
    "CRVAL1",
    "CRVAL2",
    "CRPIX1",
    "CRPIX2",
    "CD1_1",
    "CD1_2",
    "CD2_1",
    "CD2_2",
    "FGROUPNO",
    "ASTIRMS1",
    "ASTIRMS2",
    "ASTRRMS1",
    "ASTRRMS2",
    "ASTINST",
    "FLXSCALE",
    "MAGZEROP",
    "PHOTIRMS",
    "PHOTINST",
    "PHOTLINK",
}
SIP_KEY_RE = re.compile(r"^(?:A|B|AP|BP)(?:_ORDER|_\d+_\d+)$")
PV_KEY_RE = re.compile(r"^PV\d+_\d+$")


@contextmanager
def working_directory(path):
    old_cwd = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old_cwd)


def repo_root():
    return Path(__file__).resolve().parent


def configure_pp_environment():
    root = repo_root()
    os.environ["PHOTPIPEDIR"] = str(root)
    runtime_home = Path(os.environ.get(
        "TMPDIR", "/private/tmp")) / "photometrypipeline_runtime" / "home"
    runtime_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(runtime_home)
    bin_dir = str(Path(sys.executable).resolve().parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in path_parts:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_cutout_rows(wht_dir):
    path = Path(wht_dir).expanduser().resolve() / "cutouts.jsonl"
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _row_cutout_names(row):
    names = set()
    for key in ("output_path", "input_path"):
        if row.get(key):
            names.add(Path(row[key]).name)
    return names


def select_target_rows(wht_dir, cutouts=None):
    rows = [
        row for row in load_cutout_rows(wht_dir)
        if row.get("kind") == "target" and row.get("inside", True)
    ]
    if not cutouts:
        return sorted(rows, key=lambda row: row.get("obs_time", ""))

    requested = {Path(path).name for path in cutouts}
    selected = []
    missing = set(requested)
    for row in rows:
        names = _row_cutout_names(row)
        if names.intersection(requested):
            selected.append(row)
            missing.difference_update(names)
    if missing:
        raise KeyError("cutout(s) not found in cutouts.jsonl: %s" %
                       ", ".join(sorted(missing)))
    return sorted(selected, key=lambda row: row.get("obs_time", ""))


def reduced_path_for_row(wht_dir, row):
    wht_dir = Path(wht_dir).expanduser().resolve()
    input_path = row.get("input_path")
    if input_path:
        path = wht_dir / input_path.replace("_reduced_wcs.fits",
                                            "_reduced.fits")
        if path.exists():
            return path

    source_filename = (row.get("source_filename") or
                       row.get("metadata", {}).get("source_filename") or "")
    match = SOURCE_RE.search(source_filename)
    if not match:
        raise ValueError("cannot derive reduced FITS path from row: %s" %
                         source_filename)
    name = "wht%s_%s_reduced.fits" % (
        match.group("date"), match.group("recno"))
    path = wht_dir / "reduced" / name
    if not path.exists():
        raise IOError("missing reduced FITS for row: %s" % path)
    return path


def normalize_filter(row, header):
    raw = str(row.get("filter_name") or header.get("FILTER") or "").strip()
    if raw.lower() in ("", "unknown", "none"):
        object_tokens = str(header.get("OBJECT", "")).strip().split()
        if object_tokens and object_tokens[-1].upper() in ("B", "V", "R", "I"):
            return object_tokens[-1].upper()
        return "R"
    return raw.upper() if len(raw) == 1 else raw


def start_time_from_row_or_header(row, header):
    if row.get("mjd") not in (None, ""):
        return Time(float(row["mjd"]), format="mjd", scale="utc")
    if header.get("MJD-OBS") not in (None, ""):
        return Time(float(header["MJD-OBS"]), format="mjd", scale="utc")
    if row.get("obs_time"):
        return Time(row["obs_time"], format="isot", scale="utc")
    raise KeyError("row/header lacks obs_time or MJD-OBS")


def date_from_time(time):
    return time.utc.isot.split("T", 1)[0]


def target_photometry_filename(target):
    return "photometry_%s.dat" % target_to_filename(target)


def target_photometry_filenames(target):
    stems = [
        target_to_filename(target),
        ("(%s)" % target).translate(TARGET_TRANSLATION),
    ]
    return ["photometry_%s.dat" % stem for stem in dict.fromkeys(stems)]


def find_target_photometry_file(image_dir, target):
    for name in target_photometry_filenames(target):
        path = Path(image_dir) / name
        if path.exists():
            return path
    return None


def ensure_canonical_target_photometry(image_dir, target):
    source = find_target_photometry_file(image_dir, target)
    if source is None:
        return None
    destination = Path(image_dir) / target_photometry_filename(target)
    if source != destination:
        shutil.copyfile(str(source), str(destination))
    return destination


def remove_stale_target_output(image_dir, target):
    for name in target_photometry_filenames(target):
        path = Path(image_dir) / name
        if path.exists():
            path.unlink()


def stamp_wht_header(path, source_path, row, target, index):
    with fits.open(str(path), mode="update", memmap=False,
                   ignore_missing_end=True) as hdulist:
        header = hdulist[0].header
        start = start_time_from_row_or_header(row, header)
        exptime = float(row.get("exposure") or header.get("EXPTIME") or 0.0)
        mid = start + (exptime / 2.0) * u.second
        filt = normalize_filter(row, header)

        header["PPINSTRU"] = (TELESCOPE_KEY, "PP telescope override")
        header["TEL_KEYW"] = (TELESCOPE_KEY, "PP telescope keyword")
        header["TELESCOP"] = ("4.2-m William Herschel Telescope",
                              "PP normalized telescope")
        header["INSTRUME"] = ("WHT Prime Focus Imaging Platform",
                              "PP normalized instrument")
        header["OBJECT"] = (target, "PP target name")
        header["DATE-OBS"] = (start.utc.isot, "exposure start UTC")
        header["MIDTIMJD"] = (float(mid.jd), "exposure midtime JD")
        header["EQUINOX"] = (2000.0, "PP normalized coordinate equinox")
        header["FILTER"] = (filt, "PP normalized filter")
        header["ORIGFILE"] = (Path(source_path).name[:68],
                              "source reduced FITS")
        header["PPSEQ"] = (int(index), "WHT Prime workflow index")
        if row.get("ra_deg") not in (None, ""):
            header["JPLRA"] = (float(row["ra_deg"]), "JPL target RA deg")
        if row.get("dec_deg") not in (None, ""):
            header["JPLDEC"] = (float(row["dec_deg"]), "JPL target Dec deg")
        hdulist.flush(output_verify="silentfix")

        return {
            "filter": filt,
            "date_obs": date_from_time(start),
            "midtimjd": float(mid.jd),
            "exptime": exptime,
        }


def is_scamp_wcs_key(key):
    return key in SCAMP_WCS_KEYS or PV_KEY_RE.match(key) is not None


def remove_existing_wcs_cards(header):
    keys = list(header.keys())
    for key in keys:
        if is_scamp_wcs_key(key) or SIP_KEY_RE.match(key):
            header.remove(key, remove_all=True, ignore_missing=True)


def apply_scamp_head_file(fits_path, head_path=None, refcat="GAIA"):
    fits_path = Path(fits_path).expanduser().resolve()
    if head_path is None:
        head_path = fits_path.with_suffix(".head")
    else:
        head_path = Path(head_path).expanduser().resolve()
    scamp_header = fits.Header.fromstring(head_path.read_text(), sep="\n")

    with fits.open(str(fits_path), mode="update", memmap=False,
                   ignore_missing_end=True) as hdulist:
        header = hdulist[0].header
        remove_existing_wcs_cards(header)
        for card in scamp_header.cards:
            key = card.keyword
            if key in ("", "COMMENT", "HISTORY", "EXTNAME"):
                continue
            if is_scamp_wcs_key(key):
                header[key] = (card.value, card.comment)
        if "RADESYS" in header:
            header["RADECSYS"] = (header["RADESYS"],
                                  "copied from RADESYS")
        header["TEL_KEYW"] = (TELESCOPE_KEY, "pipeline telescope keyword")
        header["REGCAT"] = (refcat, "catalog used in WCS registration")
        hdulist.flush(output_verify="silentfix")
    return fits_path


def prepare_pp_tree(wht_dir, output_root, target, cutouts=None,
                    overwrite=False):
    wht_dir = Path(wht_dir).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    rows = select_target_rows(wht_dir, cutouts=cutouts)
    if not rows:
        raise IOError("no target cutout rows found below %s" % wht_dir)

    records = []
    for index, row in enumerate(rows, start=1):
        source = reduced_path_for_row(wht_dir, row)
        with fits.open(str(source), memmap=False,
                       ignore_missing_end=True) as hdulist:
            start = start_time_from_row_or_header(row, hdulist[0].header)
            filt = normalize_filter(row, hdulist[0].header)

        destination_dir = output_root / date_from_time(start) / filt
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        copied = False
        if overwrite or not destination.exists():
            shutil.copy2(str(source), str(destination))
            copied = True
        stamped = stamp_wht_header(destination, source, row, target, index)
        records.append({
            "raw": str(source),
            "working": str(destination),
            "working_dir": str(destination_dir),
            "working_name": destination.name,
            "filter": stamped["filter"],
            "date_obs": stamped["date_obs"],
            "midtimjd": stamped["midtimjd"],
            "exptime": stamped["exptime"],
            "copied": copied,
            "target_position": {
                "ra_deg": float(row["ra_deg"]),
                "dec_deg": float(row["dec_deg"]),
            },
            "cutout_row": {
                "output_path": row.get("output_path"),
                "source_filename": row.get("source_filename"),
                "obs_time": row.get("obs_time"),
                "mjd": row.get("mjd"),
            },
        })

    return {
        "telescope": TELESCOPE_KEY,
        "target": target,
        "observatory_code": "950",
        "astrometry_catalogs": ["GAIA"],
        "photometry_catalogs": ["PANSTARRS", "SDSS-R9", "APASS9"],
        "registration": "run",
        "keep_wcs": False,
        "source_dir": str(wht_dir),
        "output_root": str(output_root),
        "records": records,
    }


def positions_filename(target):
    return "positions_%s.dat" % target.translate(TARGET_TRANSLATION)


def write_positions_file(image_dir, records, target):
    path = Path(image_dir) / positions_filename(target)
    ident = target.translate(TARGET_TRANSLATION)
    with path.open("w") as handle:
        for record in records:
            pos = record["target_position"]
            handle.write("%s %.10f %.10f %.8f %s\n" %
                         (record["working_name"], pos["ra_deg"],
                          pos["dec_deg"], record["midtimjd"], ident))
    return path


def grouped_records(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["working_dir"], []).append(record)
    return grouped


def write_group_positions(records, target):
    groups = []
    for image_dir, image_records in sorted(grouped_records(records).items()):
        positions_path = write_positions_file(image_dir, image_records, target)
        groups.append({
            "image_dir": image_dir,
            "records": image_records,
            "positions_file": str(positions_path),
        })
    return groups


def apply_scamp_heads(records, refcat="GAIA"):
    applied = []
    for record in records:
        fits_path = Path(record["working"])
        head_path = fits_path.with_suffix(".head")
        if head_path.exists():
            applied.append(str(apply_scamp_head_file(fits_path, head_path,
                                                     refcat=refcat)))
    return applied


def run_pp_pipeline(image_dir, records, target, posfile,
                    fixed_aprad=0.0, rejectionfilter="pos",
                    keep_wcs=False):
    configure_pp_environment()
    import pp_run

    working_names = [record["working_name"] for record in records]
    with working_directory(image_dir):
        pp_run.run_the_pipeline(
            working_names, target, None, float(fixed_aprad), "high",
            False, False, False, keep_wcs, telescope=TELESCOPE_KEY,
            posfile=Path(posfile).name, rejectionfilter=rejectionfilter)


def run_pp_groups(groups, target, fixed_aprad=0.0, rejectionfilter="pos",
                  keep_wcs=False):
    outputs = []
    for group in groups:
        image_dir = group["image_dir"]
        remove_stale_target_output(image_dir, target)
        run_pp_pipeline(image_dir, group["records"], target,
                        group["positions_file"],
                        fixed_aprad=fixed_aprad,
                        rejectionfilter=rejectionfilter,
                        keep_wcs=keep_wcs)
        photometry_path = ensure_canonical_target_photometry(image_dir,
                                                             target)
        outputs.append({
            "image_dir": image_dir,
            "filter": group["records"][0]["filter"],
            "working_names": [record["working_name"]
                              for record in group["records"]],
            "positions_file": group["positions_file"],
            "photometry_file": (str(photometry_path)
                                if photometry_path is not None else None),
        })
    return outputs


def discover_target_photometry_files(output_root, target):
    output_root = Path(output_root).expanduser().resolve()
    paths = []
    for candidate in target_photometry_filenames(target):
        paths.extend(path for path in output_root.rglob(candidate)
                     if path.is_file())
    return sorted(set(paths))


def write_combined_photometry_csv(output_root, target, output_path=None):
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
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / MANIFEST_NAME
    with path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def build_workflow(wht_dir, output_root, target, cutouts=None,
                   overwrite=False):
    manifest = prepare_pp_tree(wht_dir, output_root, target,
                               cutouts=cutouts, overwrite=overwrite)
    groups = write_group_positions(manifest["records"], target)
    manifest["groups"] = [{
        "image_dir": group["image_dir"],
        "positions_file": group["positions_file"],
        "working_names": [record["working_name"]
                          for record in group["records"]],
    } for group in groups]
    manifest_path = write_manifest(output_root, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest, groups


def run_workflow(wht_dir, output_root, target, cutouts=None,
                 overwrite=False, prepare_only=False, fixed_aprad=0.0,
                 rejectionfilter="pos", keep_wcs=False,
                 apply_heads=False):
    manifest, groups = build_workflow(wht_dir, output_root, target,
                                      cutouts=cutouts, overwrite=overwrite)
    manifest["keep_wcs"] = bool(keep_wcs)
    if apply_heads:
        manifest["applied_head_files"] = apply_scamp_heads(
            manifest["records"])
    if not prepare_only:
        manifest["pp_outputs"] = run_pp_groups(
            groups, target, fixed_aprad=fixed_aprad,
            rejectionfilter=rejectionfilter, keep_wcs=keep_wcs)
        try:
            combined_path, rows = write_combined_photometry_csv(
                output_root, target)
        except SubmissionError as exc:
            manifest["combined_photometry_error"] = str(exc)
        else:
            manifest["combined_photometry_csv"] = str(combined_path)
            manifest["combined_photometry_rows"] = len(rows)
        manifest_path = write_manifest(output_root, manifest)
        manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="prepare and run PP for WHT Prime CASU products")
    parser.add_argument("wht_dir",
                        help="CASU/WHT/Prime directory with cutouts.jsonl")
    parser.add_argument("-target", "--target", default="1998 QJ1",
                        help="target name to use in PP outputs")
    parser.add_argument("--output-root", default=None,
                        help="PP output root; defaults to <wht_dir>/PP")
    parser.add_argument("--cutout", action="append",
                        help="specific cutout FITS/PNG to process")
    parser.add_argument("--overwrite", action="store_true",
                        help="refresh copied FITS files in PP output dirs")
    parser.add_argument("--prepare-only", action="store_true",
                        help="copy/stamp FITS files but do not run PP")
    parser.add_argument("--keep-wcs", action="store_true",
                        help="skip PP registration and use existing WCS")
    parser.add_argument("--apply-heads", action="store_true",
                        help="apply existing SCAMP .head WCS files first")
    parser.add_argument("--fixed-aprad", type=float, default=0.0,
                        help="fixed PP aperture radius; 0 uses curve of growth")
    parser.add_argument("--reject", default="pos",
                        help="PP target rejection schema, e.g. pos or none")

    args = parser.parse_args(argv)
    output_root = (Path(args.wht_dir).expanduser().resolve() / "PP"
                   if args.output_root is None else args.output_root)
    manifest = run_workflow(
        args.wht_dir, output_root, args.target, cutouts=args.cutout,
        overwrite=args.overwrite, prepare_only=args.prepare_only,
        fixed_aprad=args.fixed_aprad, rejectionfilter=args.reject,
        keep_wcs=args.keep_wcs, apply_heads=args.apply_heads)

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
        "applied_head_files": manifest.get("applied_head_files"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
