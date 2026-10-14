#!/usr/bin/env python3

"""Organize a flat reduced-FITS directory and run PP by object and band."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np
from astropy.io import fits
from astropy.time import Time


REPO_ROOT = Path(__file__).resolve().parent
os.environ["PHOTPIPEDIR"] = str(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _pp_conf
import toolbox


MANIFEST_NAME = "reduced_data_workflow_manifest.json"
MANIFEST_VERSION = 1
FITS_SUFFIXES = {".fit", ".fits", ".fts"}
QUADRANTS = ("lower-left", "lower-right", "upper-left", "upper-right")


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("cannot JSON-encode %r" % type(value))


def target_directory_name(value):
    """Return the requested filesystem token for an object designation."""

    token = "_".join(str(value).strip().split()).replace("/", "+")
    if not token:
        raise ValueError("blank OBJNAME value")
    return token


def target_photometry_name(value):
    """Return PP's legacy target-photometry filename."""

    return "photometry_%s.dat" % str(value).translate(
        _pp_conf.target2filename)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_fits(input_dir):
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(str(input_dir))
    paths = sorted(path for path in input_dir.iterdir()
                   if path.is_file() and path.suffix.lower() in FITS_SUFFIXES)
    if not paths:
        raise FileNotFoundError("no FITS files in %s" % input_dir)
    return paths


def telescope_from_header(header):
    instrument = None
    for key in _pp_conf.instrument_keys:
        if key in header:
            instrument = header[key]
            break
    if instrument is None:
        raise KeyError("cannot identify telescope/instrument")
    if instrument in _pp_conf.instrument_identifiers:
        return _pp_conf.instrument_identifiers[instrument]
    if instrument in _pp_conf.telescope_parameters:
        return instrument
    raise KeyError("unknown telescope/instrument %s" % instrument)


def normalize_filter(value, obsparam):
    raw = str(value).strip()
    try:
        band = obsparam["filter_translations"][raw]
    except KeyError as exc:
        raise ValueError("unsupported filter %r" % raw) from exc
    if band in (None, ""):
        raise ValueError("filter %r has no PP band" % raw)
    return str(band)


def inspect_source(path):
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdulist:
        if len(hdulist) != 1 or hdulist[0].data is None:
            raise ValueError("%s is not a single-image FITS file" % path)
        header = hdulist[0].header
        if "OBJNAME" not in header:
            raise KeyError("%s lacks OBJNAME" % path)
        if "FILTER" not in header:
            raise KeyError("%s lacks FILTER" % path)
        target = str(header["OBJNAME"]).strip()
        target_dir = target_directory_name(target)
        telescope = telescope_from_header(header)
        obsparam = _pp_conf.telescope_parameters[telescope]
        band = normalize_filter(header["FILTER"], obsparam)
        binning = tuple(int(value)
                        for value in toolbox.get_binning(header, obsparam))
        shape = tuple(hdulist[0].data.shape)
        raw_filter = str(header["FILTER"]).strip()
        timesys = str(header.get("TIMESYS", "")).strip()
        date_obs = str(header.get("DATE-OBS", "")).strip()
    return {
        "source": str(path.resolve()),
        "source_name": path.name,
        "size_bytes": path.stat().st_size,
        "target": target,
        "target_dir": target_dir,
        "raw_filter": raw_filter,
        "band": band,
        "telescope": telescope,
        "shape": list(shape),
        "binning": list(binning),
        "timesys": timesys,
        "date_obs": date_obs,
    }


def check_output_root(output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("output directory is not empty: %s" %
                              output_dir)
    parent = output_dir.parent
    if not parent.is_dir():
        raise NotADirectoryError(str(parent))
    return output_dir


def preflight(input_dir, output_dir):
    output_dir = check_output_root(output_dir)
    paths = discover_fits(input_dir)
    records = [inspect_source(path) for path in paths]

    telescopes = {record["telescope"] for record in records}
    shapes = {tuple(record["shape"]) for record in records}
    binnings = {tuple(record["binning"]) for record in records}
    if len(telescopes) != 1:
        raise ValueError("mixed telescopes in input: %s" %
                         sorted(telescopes))
    if len(shapes) != 1:
        raise ValueError("mixed image shapes in input: %s" % sorted(shapes))
    if len(binnings) != 1:
        raise ValueError("mixed binning in input: %s" % sorted(binnings))

    targets_by_token = defaultdict(set)
    bands_by_target = defaultdict(set)
    for record in records:
        targets_by_token[record["target_dir"]].add(record["target"])
        bands_by_target[record["target"]].add(record["band"])
    collisions = {token: sorted(targets)
                  for token, targets in targets_by_token.items()
                  if len(targets) > 1}
    if collisions:
        raise ValueError("sanitized object-name collision: %s" % collisions)

    total_bytes = sum(record["size_bytes"] for record in records)
    free_bytes = shutil.disk_usage(output_dir.parent).free
    required_bytes = total_bytes * 2
    if free_bytes < required_bytes:
        raise OSError("insufficient free space: need at least %d bytes; "
                      "%d available" % (required_bytes, free_bytes))

    groups = {}
    destinations = set()
    for record in records:
        relative_dir = Path(record["target_dir"])
        if len(bands_by_target[record["target"]]) > 1:
            relative_dir /= record["band"]
        destination = output_dir / relative_dir / record["source_name"]
        if destination in destinations:
            raise ValueError("destination collision: %s" % destination)
        destinations.add(destination)
        record["relative_dir"] = str(relative_dir)
        record["destination"] = str(destination)
        group_id = str(relative_dir)
        group = groups.setdefault(group_id, {
            "group_id": group_id,
            "target": record["target"],
            "target_dir": record["target_dir"],
            "band": record["band"],
            "working_dir": str((output_dir / relative_dir).resolve()),
            "files": [],
            "status": "pending",
        })
        group["files"].append(record["source_name"])

    for record in records:
        record["source_sha256_before"] = sha256(record["source"])
    ordered_groups = [groups[key] for key in sorted(groups)]
    return {
        "input_dir": str(Path(input_dir).expanduser().resolve()),
        "output_dir": str(output_dir),
        "telescope": next(iter(telescopes)),
        "shape": list(next(iter(shapes))),
        "binning": list(next(iter(binnings))),
        "source_count": len(records),
        "object_count": len(bands_by_target),
        "group_count": len(ordered_groups),
        "input_bytes": total_bytes,
        "free_bytes_before": free_bytes,
        "required_bytes": required_bytes,
        "records": records,
        "groups": ordered_groups,
    }


def normalize_working_header(path, record):
    changes = {
        "filter_original": record["raw_filter"],
        "filter_normalized": record["band"],
        "target": record["target"],
        "time_converted": False,
    }
    with fits.open(path, mode="update", memmap=False,
                   ignore_missing_end=True) as hdulist:
        header = hdulist[0].header
        header["ORIGFILT"] = (record["raw_filter"],
                              "original reduced-data filter")
        header["FILTER"] = (record["band"], "PP normalized filter")
        header["OBJECT"] = (record["target"], "PP target name")
        if str(header.get("TIMESYS", "")).strip().upper() == "TAI":
            original_date = str(header["DATE-OBS"]).strip()
            original_scale = str(header["TIMESYS"]).strip()
            utc = Time(original_date, format="isot", scale="tai").utc
            utc.precision = 6
            header["DATEORIG"] = (original_date,
                                  "original DATE-OBS before UTC conversion")
            header["TSYSORIG"] = (original_scale,
                                  "original DATE-OBS time scale")
            header["DATE-OBS"] = (utc.isot, "UTC exposure start")
            header["TIMESYS"] = ("UTC", "DATE-OBS time scale")
            changes.update({
                "time_converted": True,
                "date_obs_original": original_date,
                "timesys_original": original_scale,
                "date_obs_utc": utc.isot,
            })
        hdulist.flush(output_verify="silentfix")
    return changes


def stage_records(preflight_result):
    for group in preflight_result["groups"]:
        Path(group["working_dir"]).mkdir(parents=True, exist_ok=True)
    for record in preflight_result["records"]:
        source = Path(record["source"])
        destination = Path(record["destination"])
        shutil.copy2(source, destination)
        record["header_changes"] = normalize_working_header(destination,
                                                              record)
        record["destination_sha256_staged"] = sha256(destination)


def quadrant_slices(shape, quadrant):
    if quadrant not in QUADRANTS:
        raise ValueError("unknown quadrant %r" % quadrant)
    if len(shape) != 2 or shape[0] % 2 or shape[1] % 2:
        raise ValueError("quadrant masks require an even two-dimensional shape")
    ny, nx = shape
    yslice = slice(0, ny // 2) if quadrant.startswith("lower") else slice(ny // 2, ny)
    xslice = slice(0, nx // 2) if quadrant.endswith("left") else slice(nx // 2, nx)
    return yslice, xslice


def create_quadrant_mask(path, shape, quadrant):
    path = Path(path)
    mask = np.ones(tuple(shape), dtype=np.float32)
    yslice, xslice = quadrant_slices(tuple(shape), quadrant)
    mask[yslice, xslice] = 0.0
    header = fits.Header()
    header["MASKTYPE"] = ("MAP_WEIGHT", "Source Extractor weight-map type")
    header["BADQUAD"] = (quadrant, "zero-weight detector quadrant")
    fits.PrimaryHDU(data=mask, header=header).writeto(path, overwrite=False)
    return {
        "path": str(path.resolve()),
        "kind": "generated_quadrant",
        "quadrant": quadrant,
        "shape": list(shape),
        "bad_pixels": int(np.count_nonzero(mask == 0)),
        "good_pixels": int(np.count_nonzero(mask > 0)),
        "sha256": sha256(path),
    }


def validate_weight_mask(path, shape):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdulist:
        data = hdulist[0].data
        if data is None or tuple(data.shape) != tuple(shape):
            raise ValueError("mask shape does not match input image shape")
        if not np.all(np.isfinite(data)) or np.any(data < 0):
            raise ValueError("mask weights must be finite and nonnegative")
    return path


def prepare_mask(output_dir, shape, mask_quad=None, mask_file=None):
    if mask_quad is None and mask_file is None:
        return None
    mask_dir = Path(output_dir) / ".masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if mask_quad is not None:
        name = "weight_%s_%dx%d.fits" % (
            mask_quad, int(shape[1]), int(shape[0]))
        return create_quadrant_mask(mask_dir / name, shape, mask_quad)

    source = validate_weight_mask(mask_file, shape)
    destination = mask_dir / source.name
    shutil.copy2(source, destination)
    validate_weight_mask(destination, shape)
    with fits.open(destination, memmap=False) as hdulist:
        data = np.asarray(hdulist[0].data)
    return {
        "path": str(destination.resolve()),
        "kind": "supplied",
        "source": str(source),
        "shape": list(shape),
        "bad_pixels": int(np.count_nonzero(data == 0)),
        "good_pixels": int(np.count_nonzero(data > 0)),
        "sha256": sha256(destination),
    }


def write_manifest(manifest):
    path = Path(manifest["output_dir"]) / MANIFEST_NAME
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                  default=json_default) + "\n")
    partial.replace(path)
    return path


def active_photometry_rows(path):
    if path is None or not Path(path).is_file():
        return 0
    with Path(path).open() as handle:
        return sum(1 for line in handle
                   if line.strip() and not line.lstrip().startswith("#"))


def resolve_target_photometry(working_dir, target):
    """Find target photometry even when Horizons canonicalizes its filename."""

    working_dir = Path(working_dir)
    exact = working_dir / target_photometry_name(target)
    if exact.is_file():
        return exact
    token = str(target).translate(_pp_conf.target2filename).lower()
    all_candidates = sorted(working_dir.glob("photometry_*.dat"))
    candidates = [path for path in all_candidates
                  if token in path.name.lower()]
    if candidates:
        return sorted(candidates,
                      key=lambda path: (-active_photometry_rows(path),
                                        path.name))[0]

    # Periodic comets can be canonicalized to a numbered asteroid identity
    # that shares no filename token with the requested designation (for
    # example, 433P -> 248370 (2005 QN173)). When PP emitted exactly one
    # non-control target file, it is unambiguous.
    non_control = [
        path for path in all_candidates
        if "control_star" not in path.name.lower()
    ]
    if len(non_control) == 1:
        return non_control[0]
    return None


def point_in_quadrant(x, y, shape, quadrant):
    ny, nx = shape
    lower = y <= ny / 2.0
    left = x <= nx / 2.0
    return {
        "lower-left": lower and left,
        "lower-right": lower and not left,
        "upper-left": not lower and left,
        "upper-right": not lower and not left,
    }[quadrant]


def validate_group_catalogs(group, shape, quadrant):
    catalogs = sorted(Path(group["working_dir"]).glob("*.ldac.csv"))
    detections = []
    rejected_sources = []
    source_count = 0
    for catalog_path in catalogs:
        with catalog_path.open(newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                if not row:
                    continue
                source_count += 1
                x = float(row["XWIN_IMAGE"])
                y = float(row["YWIN_IMAGE"])
                if point_in_quadrant(x, y, shape, quadrant):
                    finding = {
                        "catalog": catalog_path.name,
                        "row": row_number,
                        "x": x,
                        "y": y,
                    }
                    # Source Extractor can retain an edge-straddling,
                    # negative-flux bookkeeping row whose centroid lands
                    # fractionally inside a zero-weight pixel. PP rejects it
                    # from photometry (MAG_AUTO=99), so report it separately
                    # instead of treating it as a scientific mask leak.
                    try:
                        flux_auto = float(row["FLUX_AUTO"])
                        mag_auto = float(row["MAG_AUTO"])
                    except (KeyError, TypeError, ValueError):
                        usable = True
                    else:
                        usable = flux_auto > 0 and mag_auto < 90
                        finding.update({
                            "flux_auto": flux_auto,
                            "mag_auto": mag_auto,
                        })
                    if usable:
                        detections.append(finding)
                    else:
                        rejected_sources.append(finding)
    return {
        "catalog_count": len(catalogs),
        "source_count": source_count,
        "masked_detections": detections,
        "masked_detection_count": len(detections),
        "masked_rejected_sources": rejected_sources,
        "masked_rejected_source_count": len(rejected_sources),
    }


def run_group(group, manifest):
    working_dir = Path(group["working_dir"])
    if "command" in group:
        attempt = {key: group[key] for key in (
            "status", "returncode", "failure_reason", "stdout", "stderr",
            "photometry", "active_photometry_rows", "mask_validation")
                   if key in group}
        group.setdefault("attempts", []).append(attempt)
    attempt_number = len(group.get("attempts", [])) + 1
    suffix = "" if attempt_number == 1 else "_attempt_%d" % attempt_number
    stdout_path = working_dir / ("pp_run_stdout%s.txt" % suffix)
    stderr_path = working_dir / ("pp_run_stderr%s.txt" % suffix)
    for key in ("returncode", "failure_reason", "photometry",
                "active_photometry_rows", "mask_validation"):
        group.pop(key, None)
    command = [
        sys.executable, str(REPO_ROOT / "pp_run.py"),
        "-target", group["target"],
        "-filter", group["band"],
        "-telescope", manifest["telescope"],
    ]
    if group.get("keep_wcs"):
        command.append("-keep_wcs")
    if manifest.get("mask") is not None:
        command.extend(["-mask", manifest["mask"]["path"]])
    command.extend(sorted(group["files"]))
    group["command"] = command
    group["command_display"] = shlex.join(command)
    group["stdout"] = str(stdout_path)
    group["stderr"] = str(stderr_path)
    group["status"] = "running"
    write_manifest(manifest)

    env = os.environ.copy()
    env["PHOTPIPEDIR"] = str(REPO_ROOT)
    python_bin = str(Path(sys.executable).resolve().parent)
    if python_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = python_bin + os.pathsep + env.get("PATH", "")
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        completed = subprocess.run(command, cwd=str(working_dir), env=env,
                                   stdout=stdout, stderr=stderr, text=True)

    photometry_path = resolve_target_photometry(working_dir, group["target"])
    active_rows = active_photometry_rows(photometry_path)
    group["returncode"] = completed.returncode
    group["photometry"] = (None if photometry_path is None else
                           str(photometry_path))
    group["active_photometry_rows"] = active_rows
    mask_info = manifest.get("mask") or {}
    if mask_info.get("kind") == "generated_quadrant":
        group["mask_validation"] = validate_group_catalogs(
            group, tuple(manifest["shape"]), mask_info["quadrant"])
    masked_count = group.get("mask_validation", {}).get(
        "masked_detection_count", 0)
    if completed.returncode != 0:
        group["status"] = "failed"
        group["failure_reason"] = "pp_returncode_%d" % completed.returncode
    elif photometry_path is None or not photometry_path.is_file():
        group["status"] = "failed"
        group["failure_reason"] = "missing_target_photometry"
    elif active_rows == 0:
        group["status"] = "failed"
        group["failure_reason"] = "empty_target_photometry"
    elif masked_count:
        group["status"] = "failed"
        group["failure_reason"] = "detections_inside_masked_quadrant"
    else:
        group["status"] = "complete"
    write_manifest(manifest)
    return group


def reconcile_completed_group(group, manifest):
    """Recognize a completed prior run before an interrupted batch resumes."""

    if group.get("returncode") != 0:
        return False
    photometry_path = resolve_target_photometry(group["working_dir"],
                                                group["target"])
    active_rows = active_photometry_rows(photometry_path)
    if photometry_path is None or active_rows == 0:
        return False
    mask_info = manifest.get("mask") or {}
    if mask_info.get("kind") == "generated_quadrant":
        validation = validate_group_catalogs(
            group, tuple(manifest["shape"]), mask_info["quadrant"])
        if validation["masked_detection_count"]:
            return False
        group["mask_validation"] = validation
    group["photometry"] = str(photometry_path)
    group["active_photometry_rows"] = active_rows
    group["status"] = "complete"
    group["reconciled_existing_output"] = True
    group.pop("failure_reason", None)
    return True


def verify_sources_unchanged(manifest):
    unchanged = True
    for record in manifest["records"]:
        after = sha256(record["source"])
        record["source_sha256_after"] = after
        record["source_unchanged"] = (
            after == record["source_sha256_before"])
        unchanged = unchanged and record["source_unchanged"]
    manifest["sources_unchanged"] = unchanged
    return unchanged


def resume_workflow(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError("cannot resume without %s" % manifest_path)
    manifest = json.loads(manifest_path.read_text())
    input_dir = str(Path(args.input_dir).expanduser().resolve())
    if manifest.get("input_dir") != input_dir:
        raise ValueError("resume input does not match the workflow manifest")
    if manifest.get("output_dir") != str(output_dir):
        raise ValueError("resume output does not match the workflow manifest")
    if args.prepare_only or args.mask_quad is not None or args.mask_file is not None:
        raise ValueError(
            "--resume reuses the manifest mask and cannot be combined with "
            "--prepare-only, --mask-quad, or --mask-file")

    mask_info = manifest.get("mask")
    if mask_info is not None:
        validate_weight_mask(mask_info["path"], tuple(manifest["shape"]))
    manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
    manifest["workflow_status"] = "running"
    write_manifest(manifest)
    for group in manifest["groups"]:
        if group.get("status") != "complete":
            if not reconcile_completed_group(group, manifest):
                run_group(group, manifest)
            else:
                write_manifest(manifest)

    sources_unchanged = verify_sources_unchanged(manifest)
    groups_complete = all(group["status"] == "complete"
                          for group in manifest["groups"])
    manifest["workflow_status"] = (
        "complete" if sources_unchanged and groups_complete else "failed")
    manifest["free_bytes_after"] = shutil.disk_usage(output_dir.parent).free
    write_manifest(manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def run_workflow(args):
    if getattr(args, "resume", False):
        return resume_workflow(args)
    if args.mask_quad is not None and args.mask_file is not None:
        raise ValueError("--mask-quad and --mask-file are mutually exclusive")
    result = preflight(args.input_dir, args.output_dir)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        **result,
        "workflow_status": "staging",
        "prepare_only": bool(args.prepare_only),
        "mask": None,
    }
    Path(manifest["output_dir"]).mkdir(parents=True, exist_ok=True)
    write_manifest(manifest)
    stage_records(manifest)
    manifest["mask"] = prepare_mask(
        manifest["output_dir"], tuple(manifest["shape"]),
        mask_quad=args.mask_quad, mask_file=args.mask_file)
    manifest["workflow_status"] = "prepared"
    write_manifest(manifest)

    if not args.prepare_only:
        manifest["workflow_status"] = "running"
        write_manifest(manifest)
        for group in manifest["groups"]:
            run_group(group, manifest)

    sources_unchanged = verify_sources_unchanged(manifest)
    if args.prepare_only:
        manifest["workflow_status"] = (
            "prepared" if sources_unchanged else "failed")
    else:
        groups_complete = all(group["status"] == "complete"
                              for group in manifest["groups"])
        manifest["workflow_status"] = (
            "complete" if sources_unchanged and groups_complete else "failed")
    manifest["free_bytes_after"] = shutil.disk_usage(
        Path(manifest["output_dir"]).parent).free
    manifest_path = write_manifest(manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="flat directory of reduced FITS")
    parser.add_argument("--output-dir", required=True,
                        help="new or empty PP output directory")
    mask = parser.add_mutually_exclusive_group()
    mask.add_argument("--mask-quad", choices=QUADRANTS,
                      help="generate a zero-weight detector quadrant")
    mask.add_argument("--mask-file",
                      help="existing Source Extractor MAP_WEIGHT FITS")
    parser.add_argument("--prepare-only", action="store_true",
                        help="stage and validate without running PP")
    parser.add_argument("--resume", action="store_true",
                        help="rerun incomplete groups from an existing manifest")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_workflow(args)
    summary = {
        "workflow_status": result["workflow_status"],
        "source_count": result["source_count"],
        "object_count": result["object_count"],
        "group_count": result["group_count"],
        "complete_groups": sum(group["status"] == "complete"
                               for group in result["groups"]),
        "failed_groups": sum(group["status"] == "failed"
                             for group in result["groups"]),
        "manifest": str(Path(result["output_dir"]) / MANIFEST_NAME),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["workflow_status"] in ("prepared", "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
