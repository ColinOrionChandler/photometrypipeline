#!/usr/bin/env python3

"""Workflow helper for ARCSAT reduced image products.

The photometry command includes an opt-in saturated recovery mode for rare
cases like Vesta, where the true target is saturated and PP may distill nearby
non-target detections. The recovery output is explicitly labeled as a
model-based saturated-core estimate, not normal PP aperture photometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from contextlib import contextmanager

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.optimize import least_squares

from pptool_mpcsubmission import (
    PHOTOMETRY_COLUMNS,
    parse_photometry_file,
    resolve_fits_filename,
    target_to_filename,
)


TELESCOPE_BY_INSTRUME = {
    "BYUcam": "ARCSATBYUCAM",
    "dcam-spare": "ARCSATDCAMSPARE",
}
MANIFEST_NAME = "arcsat_workflow_manifest.json"
DEPTH_SUMMARY_NAME = "arcsat_depth_summary.csv"
BROADBAND_FILTERS = {"B", "V", "g", "r", "i", "sdss_r"}
FILTER_TRANSLATIONS = {
    "B": "B",
    "V": "V",
    "g": "g",
    "r": "r",
    "i": "i",
    "sdss_r": "r",
}
ARCSAT_SCAMP_CONTRAST_LIMIT = 1.5
SATURATED_RECOVERY_METHOD = "saturated_moffat_wing_fit"
SATURATED_RECOVERY_WARNING = (
    "target core is saturated; recovered_mag is a model-based saturated-wing "
    "estimate, not standard PP aperture photometry")
DEFAULT_SATURATION_ADU = 60000.0
SATURATED_FIT_RADIUS_PX = 35.0
SATURATED_MASKED_CORE_RADIUS_PX = 5.0
SATURATED_SKY_INNER_RADIUS_PX = 45.0
SATURATED_SKY_OUTER_RADIUS_PX = 60.0


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


def sanitize_filename_token(value):
    token = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value).strip())
    return token.strip("_") or "arcsat"


def reduced_data_dir(input_dir):
    root = Path(input_dir).expanduser().resolve()
    data_dir = root / "data"
    if not data_dir.exists():
        raise IOError("missing reduced data directory: %s" % data_dir)
    return data_dir


def iter_reduced_fits(input_dir):
    return sorted(path for path in reduced_data_dir(input_dir).glob("*.fits")
                  if path.is_file())


def read_header(path):
    return fits.getheader(str(path), ignore_missing_end=True)


def translated_filter(raw_filter):
    return FILTER_TRANSLATIONS.get(str(raw_filter).strip())


def telescope_key_for_header(header):
    instrume = str(header.get("INSTRUME", "")).strip()
    try:
        return TELESCOPE_BY_INSTRUME[instrume]
    except KeyError:
        raise KeyError("unsupported ARCSAT instrument %r" % instrume)


def header_has_wcs(header):
    return all(key in header for key in ("CTYPE1", "CTYPE2", "CRVAL1",
                                         "CRVAL2", "CRPIX1", "CRPIX2"))


def midtime_jd(header):
    if header.get("MIDTIMJD") not in (None, ""):
        return float(header["MIDTIMJD"])
    if header.get("JD") not in (None, ""):
        return float(header["JD"]) + float(header.get("EXPTIME", 0.0)) / 2.0 / 86400.0
    start = Time(str(header["DATE-OBS"]), format="isot", scale="utc")
    return float((start + float(header.get("EXPTIME", 0.0)) / 2.0 * u.second).jd)


def discover_photometry_images(input_dir, target, filter_name):
    selected = []
    for path in iter_reduced_fits(input_dir):
        header = read_header(path)
        if str(header.get("OBJECT", "")).strip().casefold() != target.casefold():
            continue
        if translated_filter(header.get("FILTER")) != filter_name:
            continue
        selected.append(path)
    if not selected:
        raise IOError("no %s/%s reduced FITS images found in %s" %
                      (target, filter_name, reduced_data_dir(input_dir)))
    return selected


def discover_depth_groups(input_dir):
    groups = {}
    for path in iter_reduced_fits(input_dir):
        header = read_header(path)
        filt = str(header.get("FILTER", "")).strip()
        if filt not in BROADBAND_FILTERS:
            continue
        band = translated_filter(filt)
        if band is None:
            continue
        obj = str(header.get("OBJECT", "")).strip() or "unknown"
        telescope = telescope_key_for_header(header)
        key = (sanitize_filename_token(obj), band, telescope)
        groups.setdefault(key, {
            "object": obj,
            "filter": band,
            "telescope": telescope,
            "paths": [],
        })["paths"].append(path)
    return sorted(groups.values(), key=lambda item: (
        item["object"].casefold(), item["filter"], item["telescope"]))


def prepare_output_dir(path, overwrite=False):
    path = Path(path).expanduser().resolve()
    if path.exists() and overwrite:
        shutil.rmtree(str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def working_name_for_source(source_path, sequence):
    return "%04d_%s.fits" % (
        int(sequence), sanitize_filename_token(Path(source_path).stem))


def copy_and_stamp(source_path, working_path, sequence, target=None):
    shutil.copy2(str(source_path), str(working_path))
    with fits.open(str(working_path), mode="update", memmap=False,
                   ignore_missing_end=True) as hdulist:
        header = hdulist[0].header
        telescope = telescope_key_for_header(header)
        if target is not None:
            header["OBJECT"] = (target, "PP target override")
        header["PPINSTRU"] = (telescope, "PP telescope override")
        header["TEL_KEYW"] = (telescope, "PP telescope keyword")
        header["ORIGFILE"] = (Path(source_path).name[:68],
                              "source reduced FITS filename")
        header["PPSEQ"] = (int(sequence), "ARCSAT workflow index")
        if "MIDTIMJD" not in header:
            header["MIDTIMJD"] = (midtime_jd(header), "PP observation midtime")
        return {
            "source": str(Path(source_path).resolve()),
            "working": str(Path(working_path).resolve()),
            "working_name": Path(working_path).name,
            "object": str(header.get("OBJECT", "")),
            "filter": translated_filter(header.get("FILTER")),
            "raw_filter": str(header.get("FILTER", "")),
            "telescope": telescope,
            "instrume": str(header.get("INSTRUME", "")),
            "date_obs": str(header.get("DATE-OBS", "")),
            "midtimjd": float(header["MIDTIMJD"]),
            "exptime": float(header.get("EXPTIME", 0.0)),
            "has_wcs": header_has_wcs(header),
        }


def prepare_work_tree(paths, output_dir, target=None, overwrite=False):
    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)
    records = []
    for idx, source_path in enumerate(sorted(paths), start=1):
        working_path = output_dir / working_name_for_source(source_path, idx)
        records.append(copy_and_stamp(source_path, working_path, idx,
                                      target=target))
    return output_dir, records


def remove_stale_target_outputs(image_dir, target):
    stem = target_to_filename(target)
    for path in Path(image_dir).glob("photometry_%s*" % stem):
        if path.is_file():
            path.unlink()


def run_pp_pipeline(image_dir, working_names, target, filter_name,
                    telescope, keep_wcs, posfile=None):
    configure_pp_environment()
    import pp_run

    remove_stale_target_outputs(image_dir, target)
    with working_directory(image_dir):
        pp_run.run_the_pipeline(
            sorted(working_names), target, filter_name, 0, "high",
            False, False, False, keep_wcs, telescope=telescope,
            posfile=Path(posfile).name if posfile else None,
            rejectionfilter="none" if posfile else "pos")


def run_registration(filenames, telescope, obsparam):
    import _pp_conf
    import pp_register

    old_as = _pp_conf.scamp_as_contrast_limit
    old_xy = _pp_conf.scamp_xy_contrast_limit
    _pp_conf.scamp_as_contrast_limit = ARCSAT_SCAMP_CONTRAST_LIMIT
    _pp_conf.scamp_xy_contrast_limit = ARCSAT_SCAMP_CONTRAST_LIMIT
    try:
        goodfits = []
        for _ in range(2):
            registration = pp_register.register(
                filenames, telescope, obsparam["source_snr"],
                obsparam["source_minarea"], obsparam["aprad_default"],
                None, obsparam, obsparam["source_tolerance"],
                nodeblending=False, display=True, diagnostics=False)
            goodfits = registration["goodfits"]
            if len(goodfits) == len(filenames):
                break
            if len(goodfits) > 0:
                filenames = goodfits
        return goodfits
    finally:
        _pp_conf.scamp_as_contrast_limit = old_as
        _pp_conf.scamp_xy_contrast_limit = old_xy


def ensure_frame_diagnostics(filenames):
    diagnostics_index = Path("diagnostics.html")
    if not diagnostics_index.exists():
        diagnostics_index.write_text(
            "<HTML>\n<BODY>\n<!-- Calibration -->\n</BODY>\n</HTML>\n")
    diagnostics_dir = Path(".diagnostics")
    diagnostics_dir.mkdir(exist_ok=True)
    for filename in filenames:
        frame_html = diagnostics_dir / ("%s.html" % filename)
        if frame_html.exists():
            continue
        frame_html.write_text(
            "<HTML>\n<BODY>\n<!-- Calibration -->\n</BODY>\n</HTML>\n")


def run_calibrated_pipeline(image_dir, records, filter_name, telescope,
                            keep_wcs, target=None, posfile=None,
                            distill=False):
    configure_pp_environment()
    import _pp_conf
    import pp_calibrate
    import pp_distill
    import pp_photometry
    import pp_prepare

    obsparam = _pp_conf.telescope_parameters[telescope]
    filenames = sorted(record["working_name"] for record in records)
    with working_directory(image_dir):
        Path(".diagnostics").mkdir(exist_ok=True)
        pp_prepare.prepare(filenames, obsparam, {}, diagnostics=False,
                           display=True, keep_wcs=keep_wcs)
        ensure_frame_diagnostics(filenames)
        if not keep_wcs:
            goodfits = run_registration(filenames, telescope, obsparam)
            if not goodfits:
                raise RuntimeError("registration failed for all frames")
            filenames = sorted(goodfits)
            ensure_frame_diagnostics(filenames)

        pp_photometry.photometry(
            filenames, 1.5, obsparam["source_minarea"], None, target,
            False, False, telescope, obsparam, display=True,
            diagnostics=False)
        calibration = pp_calibrate.calibrate(
            filenames, _pp_conf.minstars, filter_name, None, obsparam,
            display=True, diagnostics=False)
        distillate = None
        if distill:
            distillate = pp_distill.distill(
                calibration["catalogs"], target, [0, 0], None,
                Path(posfile).name if posfile else None,
                "none" if posfile else "pos", display=True,
                diagnostics=False)
    return {
        "filenames": filenames,
        "calibration": calibration,
        "distillate": distillate,
    }


def discover_target_photometry_files(output_root, target):
    output_root = Path(output_root).expanduser().resolve()
    name = "photometry_%s.dat" % target_to_filename(target)
    files = {path for path in output_root.rglob(name) if path.is_file()}
    has_exact_match = bool(files)
    target_token = target_to_filename(target).lower()
    for path in output_root.rglob("photometry_*.dat"):
        if not path.is_file():
            continue
        stem = path.stem.lower()
        if stem == "photometry_control_star":
            continue
        if target_token in stem or not has_exact_match:
            files.add(path)
    return sorted(files)


def parse_arcsat_photometry_file(filename):
    rows = []
    with Path(filename).open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                stripped = stripped[1:].strip()
            if not stripped or not re.match(r"^[A-Za-z0-9_.+-]+\s+\d", stripped):
                continue
            values = stripped.split()
            if len(values) < 20:
                continue
            padded = values + [""] * (len(PHOTOMETRY_COLUMNS) - len(values))
            rows.append(dict(zip(PHOTOMETRY_COLUMNS, padded)))
    return rows


def write_combined_photometry_csv(output_root, target):
    output_root = Path(output_root).expanduser().resolve()
    output_path = output_root / ("photometry_%s_combined.csv" %
                                 target_to_filename(target))
    rows = []
    for photometry_file in discover_target_photometry_files(output_root,
                                                            target):
        rows_for_file = parse_photometry_file(photometry_file,
                                              require_rows=False)
        if not rows_for_file:
            rows_for_file = parse_arcsat_photometry_file(photometry_file)
        for row in rows_for_file:
            combined = dict(row)
            try:
                source_fits = resolve_fits_filename(photometry_file,
                                                    row["catalog_token"])
            except Exception:
                source_fits = ""
            combined["source_photometry_file"] = str(photometry_file)
            combined["source_fits_file"] = str(source_fits)
            rows.append(combined)

    fieldnames = list(PHOTOMETRY_COLUMNS) + [
        "source_photometry_file",
        "source_fits_file",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path, rows


def float_value(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def predicted_target_radec(row):
    ra_deg = float_value(row.get("ra_deg"))
    dec_deg = float_value(row.get("dec_deg"))
    dra_arcsec = float_value(row.get("pred_minus_ra_arcsec"))
    ddec_arcsec = float_value(row.get("pred_minus_dec_arcsec"))
    if None in (ra_deg, dec_deg, dra_arcsec, ddec_arcsec):
        raise ValueError("missing source or predicted target coordinates")
    cos_dec = math.cos(math.radians(dec_deg))
    if abs(cos_dec) < 1e-8:
        raise ValueError("cannot project predicted RA near pole")
    return (
        ra_deg + dra_arcsec / 3600.0 / cos_dec,
        dec_deg + ddec_arcsec / 3600.0,
    )


def target_offset_arcsec(row):
    dra_arcsec = float_value(row.get("pred_minus_ra_arcsec"))
    ddec_arcsec = float_value(row.get("pred_minus_dec_arcsec"))
    if dra_arcsec is None or ddec_arcsec is None:
        return None
    return float(math.hypot(dra_arcsec, ddec_arcsec))


def saturation_threshold(header):
    value = float_value(header.get("SATURATE"))
    if value is None or not np.isfinite(value) or value <= 0:
        return DEFAULT_SATURATION_ADU
    return value


def radius_mask(data, x, y, inner_radius, outer_radius):
    yy, xx = np.indices(data.shape)
    rr = np.hypot(xx - x, yy - y)
    return (rr >= inner_radius) & (rr <= outer_radius)


def aperture_stats(data, x, y, radius):
    mask = radius_mask(data, x, y, 0, radius) & np.isfinite(data)
    values = data[mask]
    if len(values) == 0:
        raise ValueError("predicted target aperture has no finite pixels")
    return {
        "peak": float(np.nanmax(values)),
        "median": float(np.nanmedian(values)),
        "count": int(len(values)),
    }


def fit_saturated_moffat(data, x, y, saturation_adu):
    sky_mask = radius_mask(
        data, x, y, SATURATED_SKY_INNER_RADIUS_PX,
        SATURATED_SKY_OUTER_RADIUS_PX) & np.isfinite(data)
    if np.count_nonzero(sky_mask) < 20:
        raise ValueError("not enough sky pixels for saturated recovery")
    sky = float(np.nanmedian(data[sky_mask]))

    yy, xx = np.indices(data.shape)
    rr = np.hypot(xx - x, yy - y)
    fit_mask = (
        (rr >= SATURATED_MASKED_CORE_RADIUS_PX) &
        (rr <= SATURATED_FIT_RADIUS_PX) &
        np.isfinite(data) &
        (data < saturation_adu)
    )
    if np.count_nonzero(fit_mask) < 50:
        raise ValueError("not enough unsaturated wing pixels for fit")

    fit_x = xx[fit_mask] - x
    fit_y = yy[fit_mask] - y
    fit_z = data[fit_mask]

    def model(params):
        log_amp, log_alpha, beta, bg = params
        amp = np.exp(log_amp)
        alpha = np.exp(log_alpha)
        return bg + amp * (
            1.0 + (fit_x * fit_x + fit_y * fit_y) / (alpha * alpha)
        ) ** (-beta)

    def residual(params):
        weight = np.sqrt(np.maximum(fit_z - sky + 100.0, 100.0))
        return (model(params) - fit_z) / weight

    starts = [
        (np.log(2e5), np.log(4.0), 2.5, sky),
        (np.log(1e6), np.log(2.0), 2.0, sky),
        (np.log(1e5), np.log(8.0), 3.0, sky),
    ]
    bounds = (
        [np.log(1e3), np.log(0.5), 1.05, sky - 100.0],
        [np.log(1e9), np.log(50.0), 10.0, sky + 100.0],
    )
    best = None
    for start in starts:
        result = least_squares(residual, start, bounds=bounds, max_nfev=5000)
        if best is None or result.cost < best.cost:
            best = result

    if best is None or not best.success:
        raise ValueError("Moffat wing fit did not converge")

    log_amp, log_alpha, beta, bg = best.x
    amp = float(np.exp(log_amp))
    alpha = float(np.exp(log_alpha))
    beta = float(beta)
    if beta <= 1.0:
        raise ValueError("Moffat beta must exceed 1 for finite flux")
    flux = math.pi * amp * alpha * alpha / (beta - 1.0)
    if not np.isfinite(flux) or flux <= 0:
        raise ValueError("Moffat fit produced invalid flux")

    return {
        "recovered_flux": float(flux),
        "moffat_amp": amp,
        "moffat_alpha": alpha,
        "moffat_beta": beta,
        "moffat_background": float(bg),
        "fit_pixel_count": int(np.count_nonzero(fit_mask)),
        "sky_median_adu": sky,
    }


SATURATED_RECOVERY_COLUMNS = [
    "recovered_mag",
    "recovered_mag_method",
    "normal_pp_mag",
    "target_offset_arcsec",
    "predicted_x",
    "predicted_y",
    "predicted_peak_adu",
    "predicted_median_adu",
    "saturation_threshold_adu",
    "saturated_pixel_count",
    "fit_radius_px",
    "masked_core_radius_px",
    "fit_status",
    "warning",
    "recovered_flux",
    "moffat_amp",
    "moffat_alpha",
    "moffat_beta",
    "moffat_background",
    "fit_pixel_count",
    "sky_median_adu",
]


def empty_saturated_recovery_metadata(row):
    recovered = dict(row)
    for column in SATURATED_RECOVERY_COLUMNS:
        recovered[column] = ""
    recovered["recovered_mag_method"] = SATURATED_RECOVERY_METHOD
    recovered["normal_pp_mag"] = row.get("mag", "")
    recovered["fit_radius_px"] = SATURATED_FIT_RADIUS_PX
    recovered["masked_core_radius_px"] = SATURATED_MASKED_CORE_RADIUS_PX
    recovered["warning"] = SATURATED_RECOVERY_WARNING
    return recovered


def saturated_recovery_row(row):
    recovered = empty_saturated_recovery_metadata(row)
    recovered["target_offset_arcsec"] = target_offset_arcsec(row)
    try:
        fits_path = Path(row.get("source_fits_file", ""))
        if not fits_path.is_file():
            raise ValueError("source FITS file is unavailable")
        with fits.open(str(fits_path), ignore_missing_end=True) as hdulist:
            data = np.asarray(hdulist[0].data, dtype=float)
            header = hdulist[0].header
            wcs = WCS(header)

        pred_ra, pred_dec = predicted_target_radec(row)
        pred_x, pred_y = wcs.world_to_pixel_values(pred_ra, pred_dec)
        pred_x = float(np.asarray(pred_x))
        pred_y = float(np.asarray(pred_y))
        recovered["predicted_x"] = pred_x
        recovered["predicted_y"] = pred_y

        threshold = saturation_threshold(header)
        recovered["saturation_threshold_adu"] = threshold
        stats = aperture_stats(data, pred_x, pred_y,
                               SATURATED_MASKED_CORE_RADIUS_PX)
        recovered["predicted_peak_adu"] = stats["peak"]
        recovered["predicted_median_adu"] = stats["median"]
        core_mask = radius_mask(data, pred_x, pred_y, 0,
                                SATURATED_FIT_RADIUS_PX)
        recovered["saturated_pixel_count"] = int(np.count_nonzero(
            core_mask & np.isfinite(data) & (data >= threshold)))

        if recovered["saturated_pixel_count"] == 0:
            recovered["fit_status"] = "not_saturated"
            return recovered

        fit = fit_saturated_moffat(data, pred_x, pred_y, threshold)
        zeropoint = float_value(row.get("zeropoint"))
        exptime = float_value(row.get("exptime"))
        if zeropoint is None or exptime is None or exptime <= 0:
            raise ValueError("missing zeropoint or exposure time")
        recovered["recovered_mag"] = (
            zeropoint - 2.5 * math.log10(fit["recovered_flux"] / exptime))
        for key, value in fit.items():
            recovered[key] = value
        recovered["fit_status"] = "ok"
    except Exception as exc:
        recovered["fit_status"] = "failed"
        recovered["warning"] = "%s; recovery failed: %s" % (
            SATURATED_RECOVERY_WARNING, exc)
    return recovered


def write_saturated_recovery_csv(output_root, target, rows):
    output_root = Path(output_root).expanduser().resolve()
    output_path = output_root / ("photometry_%s_saturated_recovery.csv" %
                                 target_to_filename(target))
    recovery_rows = [saturated_recovery_row(row) for row in rows]
    fieldnames = list(PHOTOMETRY_COLUMNS) + [
        "source_photometry_file",
        "source_fits_file",
    ] + SATURATED_RECOVERY_COLUMNS
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(recovery_rows)
    return output_path, recovery_rows


def write_horizons_positions_file(image_dir, records, target):
    configure_pp_environment()
    from astroquery.jplhorizons import Horizons

    epochs = [record["midtimjd"] for record in records]
    obj = Horizons(id=target, id_type="smallbody", location="705",
                   epochs=epochs)
    eph = obj.ephemerides()
    path = Path(image_dir) / ("positions_%s.dat" % target_to_filename(target))
    ident = target_to_filename(target)
    with path.open("w") as handle:
        for record, row in zip(records, eph):
            handle.write("%s %.10f %.10f %.8f %s\n" % (
                record["working_name"], float(row["RA"]), float(row["DEC"]),
                float(record["midtimjd"]), ident))
    return path


def run_photometry(input_dir, target, filter_name, overwrite=False,
                   saturated_recovery=False):
    input_dir = Path(input_dir).expanduser().resolve()
    paths = discover_photometry_images(input_dir, target, filter_name)
    output_root = input_dir / "PP" / "photometry" / target_to_filename(target)
    work_dir = output_root / filter_name
    work_dir, records = prepare_work_tree(paths, work_dir, target=target,
                                          overwrite=overwrite)
    telescopes = sorted({record["telescope"] for record in records})
    if len(telescopes) != 1:
        raise RuntimeError("photometry run spans multiple ARCSAT instruments")
    keep_wcs = all(record["has_wcs"] for record in records)
    manifest = {
        "mode": "photometry",
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "work_dir": str(work_dir),
        "target": target,
        "filter": filter_name,
        "telescope": telescopes[0],
        "keep_wcs": keep_wcs,
        "saturated_recovery_requested": bool(saturated_recovery),
        "records": records,
    }

    try:
        result = run_calibrated_pipeline(
            work_dir, records, filter_name, telescopes[0], keep_wcs,
            target=target, distill=True)
        manifest["pp_status"] = "ok"
        manifest["pp_filenames"] = result["filenames"]
    except Exception as exc:
        manifest["pp_status"] = "error"
        manifest["pp_error"] = str(exc)
        write_manifest(output_root, manifest)
        raise

    combined_path, rows = write_combined_photometry_csv(output_root, target)
    manifest["combined_photometry_csv"] = str(combined_path)
    manifest["combined_photometry_rows"] = len(rows)

    if len(rows) == 0:
        try:
            positions_path = write_horizons_positions_file(work_dir, records,
                                                           target)
            result = run_calibrated_pipeline(
                work_dir, records, filter_name, telescopes[0], keep_wcs,
                target=target, posfile=positions_path, distill=True)
            combined_path, rows = write_combined_photometry_csv(output_root,
                                                                target)
            manifest["fallback_positions_file"] = str(positions_path)
            manifest["fallback_status"] = "ok"
            manifest["fallback_filenames"] = result["filenames"]
            manifest["combined_photometry_csv"] = str(combined_path)
            manifest["combined_photometry_rows"] = len(rows)
        except Exception as exc:
            manifest["fallback_status"] = "error"
            manifest["fallback_error"] = str(exc)

    if saturated_recovery:
        recovery_path, recovery_rows = write_saturated_recovery_csv(
            output_root, target, rows)
        manifest["saturated_recovery_csv"] = str(recovery_path)
        manifest["saturated_recovery_rows"] = len(recovery_rows)
        manifest["saturated_recovery_warning"] = SATURATED_RECOVERY_WARNING
        manifest["saturated_recovery_ok_rows"] = sum(
            1 for row in recovery_rows if row.get("fit_status") == "ok")

    write_manifest(output_root, manifest)
    return manifest


def run_depth_group(work_dir, records, filter_name, telescope, keep_wcs):
    return run_calibrated_pipeline(
        work_dir, records, filter_name, telescope, keep_wcs,
        distill=False)["calibration"]


def find_mag_column(columns, filter_name):
    candidates = [
        "%smag" % filter_name,
        "_%smag" % filter_name,
        "%sJohnsonmag" % filter_name,
        "_%sJohnsonmag" % filter_name,
    ]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    for column in columns:
        if column.endswith("mag") and not column.startswith("e_"):
            return column
    return None


def summarize_depth_catalog(record, catalog_obj, zeropoint):
    fields = list(catalog_obj.fields)
    mag_column = find_mag_column(fields, record["filter"])
    usedstars = zeropoint.get("zp_usedstars")
    if isinstance(usedstars, (list, tuple, np.ndarray)):
        usedstars = len(usedstars)
    finite_mags = []
    if mag_column is not None:
        finite_mags = [
            float(value) for value in np.asarray(catalog_obj[mag_column])
            if np.isfinite(value)
        ]
    return {
        "source_filename": Path(record["source"]).name,
        "working_name": record["working_name"],
        "object": record["object"],
        "filter": record["filter"],
        "telescope": record["telescope"],
        "instrume": record["instrume"],
        "date_obs": record["date_obs"],
        "midtimjd": record["midtimjd"],
        "exptime": record["exptime"],
        "source_count": int(catalog_obj.shape[0]),
        "zeropoint": zeropoint.get("zp"),
        "zeropoint_sig": zeropoint.get("zp_sig"),
        "zeropoint_nstars": zeropoint.get("zp_nstars"),
        "zeropoint_usedstars": usedstars,
        "mag_column": mag_column,
        "depth_mag_p90": (float(np.percentile(finite_mags, 90))
                          if finite_mags else None),
    }


def catalog_working_name(catalog_obj):
    name = Path(str(catalog_obj.catalogname)).name
    for suffix in (".ldac.db", ".ldac"):
        if name.endswith(suffix):
            return name[:-len(suffix)] + ".fits"
    return name


def summarize_depth_records(records, calibration):
    catalogs = calibration.get("catalogs", []) if calibration else []
    zeropoints = calibration.get("zeropoints", []) if calibration else []
    rows = []
    by_working = {record["working_name"]: record for record in records}
    for catalog_obj, zeropoint in zip(catalogs, zeropoints):
        name = catalog_working_name(catalog_obj)
        record = by_working.get(name)
        if record is None:
            continue
        rows.append(summarize_depth_catalog(record, catalog_obj, zeropoint))
    return rows


def write_depth_summary(path, rows):
    fieldnames = [
        "source_filename", "working_name", "object", "filter", "telescope",
        "instrume", "date_obs", "midtimjd", "exptime", "source_count",
        "zeropoint", "zeropoint_sig", "zeropoint_nstars",
        "zeropoint_usedstars", "mag_column", "depth_mag_p90",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return Path(path), rows


def run_depth(input_dir, overwrite=False):
    input_dir = Path(input_dir).expanduser().resolve()
    output_root = input_dir / "PP" / "depth"
    groups = discover_depth_groups(input_dir)
    all_rows = []
    manifest = {
        "mode": "depth",
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "groups": [],
    }
    for group in groups:
        group_dir = (output_root / group["object"] /
                     group["filter"] / group["telescope"])
        work_dir, records = prepare_work_tree(
            group["paths"], group_dir, overwrite=overwrite)
        keep_wcs = all(record["has_wcs"] for record in records)
        group_info = {
            "object": group["object"],
            "filter": group["filter"],
            "telescope": group["telescope"],
            "work_dir": str(work_dir),
            "keep_wcs": keep_wcs,
            "records": records,
        }
        try:
            calibration = run_depth_group(work_dir, records, group["filter"],
                                          group["telescope"], keep_wcs)
            rows = summarize_depth_records(records, calibration)
            group_info["status"] = "ok"
            group_info["summary_rows"] = len(rows)
            all_rows.extend(rows)
        except Exception as exc:
            group_info["status"] = "error"
            group_info["error"] = str(exc)
        manifest["groups"].append(group_info)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path, rows = write_depth_summary(
        output_root / DEPTH_SUMMARY_NAME, all_rows)
    manifest["depth_summary_csv"] = str(summary_path)
    manifest["depth_summary_rows"] = len(rows)
    write_manifest(output_root, manifest)
    return manifest


def write_manifest(output_root, manifest):
    path = Path(output_root) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="run PP workflows on ARCSAT reduced products")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    phot = subparsers.add_parser("photometry")
    phot.add_argument("--input", required=True)
    phot.add_argument("--target", required=True)
    phot.add_argument("--filter", required=True)
    phot.add_argument("--no-overwrite", action="store_true",
                      help="reuse existing PP work directory files")
    phot.add_argument(
        "--saturated-recovery", action="store_true",
        help=("write a rare/special-case saturated-wing recovery CSV; use "
              "only when the true target is saturated and PP matched nearby "
              "non-target detections"))

    depth = subparsers.add_parser("depth")
    depth.add_argument("--input", required=True)
    depth.add_argument("--no-overwrite", action="store_true",
                       help="reuse existing PP work directory files")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.mode == "photometry":
        manifest = run_photometry(args.input, args.target, args.filter,
                                  overwrite=not args.no_overwrite,
                                  saturated_recovery=args.saturated_recovery)
    elif args.mode == "depth":
        manifest = run_depth(args.input, overwrite=not args.no_overwrite)
    else:
        raise ValueError("unknown mode: %s" % args.mode)
    print(json.dumps({
        "mode": manifest["mode"],
        "output_root": manifest.get("output_root"),
        "combined_photometry_csv": manifest.get("combined_photometry_csv"),
        "combined_photometry_rows": manifest.get("combined_photometry_rows"),
        "saturated_recovery_csv": manifest.get("saturated_recovery_csv"),
        "saturated_recovery_rows": manifest.get("saturated_recovery_rows"),
        "depth_summary_csv": manifest.get("depth_summary_csv"),
        "depth_summary_rows": manifest.get("depth_summary_rows"),
        "manifest": str(Path(manifest.get("output_root", ".")) /
                        MANIFEST_NAME),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
