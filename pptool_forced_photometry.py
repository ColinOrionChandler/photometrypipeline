#!/usr/bin/env python3

"""Forced aperture photometry at fixed PP/click astrometric positions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image, ImageDraw

from pptool_mpcsubmission import parse_photometry_file, resolve_fits_filename
from pptool_pp_cutouts import _zscale_rgba, target_to_filename


DEFAULT_APERTURE_RADII = (1.5, 2.0, 2.5, 3.0, 4.0)
DEFAULT_SKY_INNER_RADIUS = 6.0
DEFAULT_SKY_OUTER_RADIUS = 10.0
DEFAULT_SNR_THRESHOLD = 5.0
DEFAULT_LIMIT_SIGMA = 3.0
DEFAULT_ZP_SIG = 0.03


def _float_or_none(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sigma_clipped_stats(values, sigma=3.0, maxiters=5):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None, 0
    clipped = values
    for _ in range(maxiters):
        median = np.median(clipped)
        std = np.std(clipped)
        if not np.isfinite(std) or std <= 0:
            break
        keep = np.abs(clipped - median) <= sigma * std
        if keep.sum() == clipped.size:
            break
        clipped = clipped[keep]
        if clipped.size == 0:
            return None, None, 0
    return float(np.median(clipped)), float(np.std(clipped)), int(clipped.size)


def aperture_photometry(data, x, y, aperture_radius,
                        sky_inner_radius=DEFAULT_SKY_INNER_RADIUS,
                        sky_outer_radius=DEFAULT_SKY_OUTER_RADIUS):
    """Measure circular-aperture flux at a fixed pixel position."""

    data = np.asarray(data, dtype=float)
    yy, xx = np.indices(data.shape)
    radius = np.hypot(xx - float(x), yy - float(y))
    aperture = radius <= float(aperture_radius)
    annulus = ((radius >= float(sky_inner_radius)) &
               (radius <= float(sky_outer_radius)))
    aperture_values = data[aperture & np.isfinite(data)]
    sky_values = data[annulus & np.isfinite(data)]
    sky, sky_std, n_sky = _sigma_clipped_stats(sky_values)
    if sky is None or n_sky == 0 or aperture_values.size == 0:
        return {
            "aperture_radius_px": float(aperture_radius),
            "sky_inner_radius_px": float(sky_inner_radius),
            "sky_outer_radius_px": float(sky_outer_radius),
            "usable": False,
            "reason": "insufficient finite aperture or sky pixels",
        }

    n_aperture = int(aperture_values.size)
    gross_flux = float(np.sum(aperture_values))
    flux = gross_flux - float(sky) * n_aperture
    sky_var = float(sky_std) ** 2
    flux_error = math.sqrt(max(abs(gross_flux) + n_aperture * sky_var *
                               (1.0 + n_aperture / max(n_sky, 1)), 0.0))
    snr = flux / flux_error if flux_error > 0 else None
    return {
        "aperture_radius_px": float(aperture_radius),
        "sky_inner_radius_px": float(sky_inner_radius),
        "sky_outer_radius_px": float(sky_outer_radius),
        "usable": True,
        "x": float(x),
        "y": float(y),
        "n_aperture": n_aperture,
        "n_sky": n_sky,
        "sky": float(sky),
        "sky_std": float(sky_std),
        "gross_flux": gross_flux,
        "flux": float(flux),
        "flux_error": float(flux_error),
        "snr": (float(snr) if snr is not None and math.isfinite(snr)
                else None),
    }


def flux_to_inst_mag(flux, exptime):
    if flux is None or flux <= 0 or exptime is None or exptime <= 0:
        return None
    return float(-2.5 * math.log10(float(flux) / float(exptime)))


def flux_mag_error(flux, flux_error):
    if flux is None or flux_error is None or flux <= 0 or flux_error <= 0:
        return None
    return float(1.0857362047581296 * float(flux_error) / float(flux))


def limiting_mag(zeropoint, flux_error, exptime, sigma=DEFAULT_LIMIT_SIGMA):
    limit_flux = float(sigma) * float(flux_error)
    inst = flux_to_inst_mag(limit_flux, exptime)
    if inst is None or zeropoint is None:
        return None
    return float(float(zeropoint) + inst)


def read_reference_catalog(path):
    """Read a simple calibration catalog CSV."""

    if path is None:
        return []
    rows = []
    with Path(path).expanduser().open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _catalog_mag(row, band):
    candidates = [
        "mag", "cat_mag", "%s_mag" % band, band,
        "%sMag" % band, "%smag" % band,
    ]
    for key in candidates:
        if key in row:
            value = _float_or_none(row.get(key))
            if value is not None:
                return value
    return None


def calibrate_zeropoint_from_catalog(image_path, band, aperture_radius,
                                     catalog_rows,
                                     sky_inner_radius=DEFAULT_SKY_INNER_RADIUS,
                                     sky_outer_radius=DEFAULT_SKY_OUTER_RADIUS,
                                     min_stars=3):
    """Fit zeropoint from reference-star positions and catalog magnitudes."""

    image_path = Path(image_path)
    if not catalog_rows:
        return {
            "method": "none",
            "usable": False,
            "reason": "no reference catalog supplied",
            "n_reference_stars": 0,
        }
    with fits.open(str(image_path), memmap=False,
                   ignore_missing_end=True) as hdulist:
        header = hdulist[0].header
        data = np.asarray(hdulist[0].data, dtype=float)
        wcs = WCS(header, relax=True)
        exptime = _float_or_none(header.get("EXPTIME")) or 1.0

    measurements = []
    for row in catalog_rows:
        cat_mag = _catalog_mag(row, band)
        if cat_mag is None:
            continue
        x = _float_or_none(row.get("x"))
        y = _float_or_none(row.get("y"))
        if x is None or y is None:
            ra = _float_or_none(row.get("ra") or row.get("ra_deg"))
            dec = _float_or_none(row.get("dec") or row.get("dec_deg"))
            if ra is None or dec is None:
                continue
            x, y = wcs.world_to_pixel_values(ra, dec)
        if not (0 <= x < data.shape[1] and 0 <= y < data.shape[0]):
            continue
        result = aperture_photometry(
            data, x, y, aperture_radius, sky_inner_radius, sky_outer_radius)
        inst_mag = flux_to_inst_mag(result.get("flux"), exptime)
        if inst_mag is None:
            continue
        measurements.append({
            "x": float(x),
            "y": float(y),
            "cat_mag": float(cat_mag),
            "inst_mag": inst_mag,
            "zeropoint": float(cat_mag - inst_mag),
            "snr": result.get("snr"),
        })

    if len(measurements) < int(min_stars):
        return {
            "method": "catalog_csv",
            "usable": False,
            "reason": "fewer than %d usable reference stars" % int(min_stars),
            "n_reference_stars": len(measurements),
            "measurements": measurements,
        }

    zps = np.array([item["zeropoint"] for item in measurements], dtype=float)
    median = float(np.median(zps))
    mad = float(np.median(np.abs(zps - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma > 0:
        keep = np.abs(zps - median) <= 3.0 * robust_sigma
    else:
        keep = np.isclose(zps, median)
    kept = [item for item, flag in zip(measurements, keep) if flag]
    rejected = [item for item, flag in zip(measurements, keep) if not flag]
    if len(kept) < int(min_stars):
        return {
            "method": "catalog_csv",
            "usable": False,
            "reason": "fewer than %d reference stars after clipping" %
                      int(min_stars),
            "n_reference_stars": len(measurements),
            "n_used": len(kept),
            "measurements": measurements,
            "rejected": rejected,
        }
    kept_zps = np.array([item["zeropoint"] for item in kept], dtype=float)
    return {
        "method": "catalog_csv",
        "usable": True,
        "zeropoint": float(np.median(kept_zps)),
        "zeropoint_sig": float(np.std(kept_zps) if len(kept_zps) > 1
                               else DEFAULT_ZP_SIG),
        "n_reference_stars": len(measurements),
        "n_used": len(kept),
        "n_rejected": len(rejected),
        "measurements": kept,
        "rejected": rejected,
    }


def _best_aperture(aperture_results):
    usable = [item for item in aperture_results if item.get("usable")]
    positive = [item for item in usable
                if item.get("flux") is not None and item["flux"] > 0]
    if positive:
        return max(positive, key=lambda item: item.get("snr") or -np.inf)
    return max(usable, key=lambda item: item.get("snr") or -np.inf) if usable else None


def _header_zeropoint(header):
    for key in ("MAGZP", "PHOT_C", "MAGZERO", "MAGZEROP"):
        value = _float_or_none(header.get(key))
        if value is not None:
            sig = (_float_or_none(header.get("MAGZPSIG")) or
                   _float_or_none(header.get("PHOT_CS")) or DEFAULT_ZP_SIG)
            return value, sig, key
    return None, None, None


def _annotated_png(data, x, y, aperture_radius, sky_inner_radius,
                   sky_outer_radius, path):
    image = _zscale_rgba(data)
    draw = ImageDraw.Draw(image)
    for radius, color in (
            (sky_outer_radius, (80, 180, 255, 255)),
            (sky_inner_radius, (80, 180, 255, 255)),
            (aperture_radius, (255, 80, 80, 255))):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     outline=color, width=1)
    draw.line((x - 5, y, x + 5, y), fill=(255, 255, 0, 255), width=1)
    draw.line((x, y - 5, x, y + 5), fill=(255, 255, 0, 255), width=1)
    image.save(path)


def measure_forced_row(photometry_file, row, aperture_radii=None,
                       sky_inner_radius=DEFAULT_SKY_INNER_RADIUS,
                       sky_outer_radius=DEFAULT_SKY_OUTER_RADIUS,
                       snr_threshold=DEFAULT_SNR_THRESHOLD,
                       reference_catalog_rows=None,
                       min_reference_stars=3,
                       output_dir=None):
    """Measure one PP photometry row at its fixed RA/Dec position."""

    aperture_radii = aperture_radii or DEFAULT_APERTURE_RADII
    source_fits = resolve_fits_filename(Path(photometry_file),
                                        row["catalog_token"])
    with fits.open(str(source_fits), memmap=False,
                   ignore_missing_end=True) as hdulist:
        header = hdulist[0].header.copy()
        data = np.asarray(hdulist[0].data, dtype=float)
        wcs = WCS(header, relax=True)

    ra = float(row["ra_deg"])
    dec = float(row["dec_deg"])
    x, y = wcs.world_to_pixel_values(ra, dec)
    exptime = (_float_or_none(row.get("exptime")) or
               _float_or_none(header.get("EXPTIME")) or 1.0)
    band = row.get("band") or header.get("FILTER")
    aperture_results = [
        aperture_photometry(data, x, y, radius, sky_inner_radius,
                            sky_outer_radius)
        for radius in aperture_radii
    ]
    best = _best_aperture(aperture_results)
    header_zp, header_zp_sig, header_zp_key = _header_zeropoint(header)
    catalog_calibration = calibrate_zeropoint_from_catalog(
        source_fits, band, best["aperture_radius_px"] if best else aperture_radii[0],
        reference_catalog_rows or [], sky_inner_radius, sky_outer_radius,
        min_stars=min_reference_stars)
    if catalog_calibration.get("usable"):
        zeropoint = catalog_calibration["zeropoint"]
        zeropoint_sig = catalog_calibration.get("zeropoint_sig", DEFAULT_ZP_SIG)
        calibration_method = catalog_calibration["method"]
    else:
        zeropoint = (_float_or_none(row.get("zeropoint")) or header_zp)
        zeropoint_sig = (_float_or_none(row.get("zeropoint_sig")) or
                         header_zp_sig or DEFAULT_ZP_SIG)
        calibration_method = ("pp_or_header_zp" if zeropoint is not None
                              else "none")

    flux = best.get("flux") if best else None
    flux_error = best.get("flux_error") if best else None
    inst_mag = flux_to_inst_mag(flux, exptime)
    inst_mag_sig = flux_mag_error(flux, flux_error)
    snr = best.get("snr") if best else None
    candidate_mag = (float(zeropoint + inst_mag)
                     if zeropoint is not None and inst_mag is not None
                     else None)
    candidate_mag_sig = (
        float(math.sqrt(inst_mag_sig ** 2 + zeropoint_sig ** 2))
        if inst_mag_sig is not None and zeropoint_sig is not None else None)
    accepted = bool(best and candidate_mag is not None and flux is not None and
                    flux > 0 and snr is not None and snr >= snr_threshold and
                    catalog_calibration.get("usable"))
    mag = candidate_mag if accepted else None
    mag_sig = candidate_mag_sig if accepted else None
    limit_mag = (limiting_mag(zeropoint, flux_error, exptime)
                 if zeropoint is not None and flux_error is not None else None)

    stem = Path(row["catalog_token"]).with_suffix("").name
    png_path = None
    if output_dir is not None and best is not None:
        png_path = Path(output_dir) / ("%s_forced_photometry.png" % stem)
        _annotated_png(data, x, y, best["aperture_radius_px"],
                       sky_inner_radius, sky_outer_radius, png_path)

    return {
        "catalog_token": row["catalog_token"],
        "source_fits": str(source_fits),
        "julian_date": _float_or_none(row.get("julian_date")),
        "ra_deg": ra,
        "dec_deg": dec,
        "x": float(x),
        "y": float(y),
        "filter": band,
        "exptime": float(exptime),
        "aperture_results": aperture_results,
        "best_aperture": best,
        "calibration": {
            "method": calibration_method,
            "zeropoint": zeropoint,
            "zeropoint_sig": zeropoint_sig,
            "header_zeropoint": header_zp,
            "header_zeropoint_sig": header_zp_sig,
            "header_zeropoint_key": header_zp_key,
            "catalog": catalog_calibration,
        },
        "inst_mag": inst_mag,
        "inst_mag_sig": inst_mag_sig,
        "candidate_mag": candidate_mag,
        "candidate_mag_sig": candidate_mag_sig,
        "mag": mag,
        "mag_sig": mag_sig,
        "limiting_mag_3sigma": limit_mag,
        "pp_mag": _float_or_none(row.get("mag")),
        "pp_mag_sig": _float_or_none(row.get("mag_sig")),
        "snr": snr,
        "accepted_photometry": accepted,
        "diagnostic_png": str(png_path) if png_path is not None else None,
    }


def build_recovery(photometry_file, target, output_dir=None,
                   aperture_radii=None, reference_catalog=None,
                   sky_inner_radius=DEFAULT_SKY_INNER_RADIUS,
                   sky_outer_radius=DEFAULT_SKY_OUTER_RADIUS,
                   snr_threshold=DEFAULT_SNR_THRESHOLD,
                   min_reference_stars=3):
    """Run forced photometry for all active rows in a PP photometry file."""

    photometry_file = Path(photometry_file).expanduser().resolve()
    output_dir = (photometry_file.parent / "photometry_recovery"
                  if output_dir is None
                  else Path(output_dir).expanduser().resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_photometry_file(photometry_file, require_rows=True)
    catalog_rows = read_reference_catalog(reference_catalog)
    results = []
    warnings = []
    for row in rows:
        try:
            results.append(measure_forced_row(
                photometry_file, row, aperture_radii=aperture_radii,
                sky_inner_radius=sky_inner_radius,
                sky_outer_radius=sky_outer_radius,
                snr_threshold=snr_threshold,
                reference_catalog_rows=catalog_rows,
                min_reference_stars=min_reference_stars,
                output_dir=output_dir))
        except Exception as exc:
            warnings.append("skipping %s: %s" %
                            (row.get("catalog_token"), exc))

    csv_path = output_dir / ("forced_photometry_%s.csv" %
                             target_to_filename(target))
    json_path = output_dir / ("forced_photometry_%s.json" %
                              target_to_filename(target))
    report_path = output_dir / ("forced_photometry_%s_report.md" %
                                target_to_filename(target))
    fieldnames = [
        "catalog_token", "julian_date", "filter", "ra_deg", "dec_deg",
        "x", "y", "snr", "flux", "flux_error", "aperture_radius_px",
        "zeropoint", "zeropoint_sig", "calibration_method",
        "candidate_mag", "candidate_mag_sig", "accepted_photometry",
        "mag", "mag_sig", "limiting_mag_3sigma", "pp_mag", "pp_mag_sig",
        "source_fits", "diagnostic_png",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            best = result.get("best_aperture") or {}
            cal = result.get("calibration") or {}
            writer.writerow({
                "catalog_token": result.get("catalog_token"),
                "julian_date": result.get("julian_date"),
                "filter": result.get("filter"),
                "ra_deg": result.get("ra_deg"),
                "dec_deg": result.get("dec_deg"),
                "x": result.get("x"),
                "y": result.get("y"),
                "snr": result.get("snr"),
                "flux": best.get("flux"),
                "flux_error": best.get("flux_error"),
                "aperture_radius_px": best.get("aperture_radius_px"),
                "zeropoint": cal.get("zeropoint"),
                "zeropoint_sig": cal.get("zeropoint_sig"),
                "calibration_method": cal.get("method"),
                "candidate_mag": result.get("candidate_mag"),
                "candidate_mag_sig": result.get("candidate_mag_sig"),
                "accepted_photometry": result.get("accepted_photometry"),
                "mag": result.get("mag"),
                "mag_sig": result.get("mag_sig"),
                "limiting_mag_3sigma": result.get("limiting_mag_3sigma"),
                "pp_mag": result.get("pp_mag"),
                "pp_mag_sig": result.get("pp_mag_sig"),
                "source_fits": result.get("source_fits"),
                "diagnostic_png": result.get("diagnostic_png"),
            })

    summary = {
        "target": target,
        "photometry_file": str(photometry_file),
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "report_path": str(report_path),
        "n_rows": len(rows),
        "n_measured": len(results),
        "n_accepted_photometry": sum(1 for item in results
                                     if item.get("accepted_photometry")),
        "snr_threshold": float(snr_threshold),
        "reference_catalog": (str(Path(reference_catalog).expanduser())
                              if reference_catalog else None),
        "warnings": warnings,
        "results": results,
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_report(report_path, summary)
    return summary


def _write_report(path, summary):
    lines = [
        "# Forced Photometry Recovery",
        "",
        "Target: `%s`" % summary["target"],
        "Rows measured: %d/%d" % (summary["n_measured"], summary["n_rows"]),
        "Accepted magnitudes: %d" % summary["n_accepted_photometry"],
        "",
        "| row | filter | SNR | PP mag | candidate mag | accepted mag | 3-sigma limit | accepted |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summary["results"]:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            item["catalog_token"],
            item.get("filter"),
            _format_report_float(item.get("snr"), 2),
            _format_report_float(item.get("pp_mag"), 4),
            _format_report_float(item.get("candidate_mag"), 4),
            _format_report_float(item.get("mag"), 4),
            _format_report_float(item.get("limiting_mag_3sigma"), 4),
            item.get("accepted_photometry"),
        ))
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend("- %s" % warning for warning in summary["warnings"])
    Path(path).write_text("\n".join(lines) + "\n")


def _format_report_float(value, digits):
    if value is None:
        return ""
    return ("%%.%df" % digits) % float(value)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="recover fixed-position forced photometry from PP rows")
    parser.add_argument("photometry_file")
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--reference-catalog",
                        help="CSV with x/y or ra/dec and catalog magnitudes")
    parser.add_argument("--aperture-radii", default=",".join(
        str(value) for value in DEFAULT_APERTURE_RADII))
    parser.add_argument("--sky-inner-radius", type=float,
                        default=DEFAULT_SKY_INNER_RADIUS)
    parser.add_argument("--sky-outer-radius", type=float,
                        default=DEFAULT_SKY_OUTER_RADIUS)
    parser.add_argument("--snr-threshold", type=float,
                        default=DEFAULT_SNR_THRESHOLD)
    parser.add_argument("--min-reference-stars", type=int, default=3)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    aperture_radii = [float(value) for value in
                      str(args.aperture_radii).split(",") if value.strip()]
    result = build_recovery(
        args.photometry_file, args.target, output_dir=args.output_dir,
        aperture_radii=aperture_radii,
        reference_catalog=args.reference_catalog,
        sky_inner_radius=args.sky_inner_radius,
        sky_outer_radius=args.sky_outer_radius,
        snr_threshold=args.snr_threshold,
        min_reference_stars=args.min_reference_stars)
    print(json.dumps({
        "output_dir": result["output_dir"],
        "csv_path": result["csv_path"],
        "json_path": result["json_path"],
        "report_path": result["report_path"],
        "n_rows": result["n_rows"],
        "n_measured": result["n_measured"],
        "n_accepted_photometry": result["n_accepted_photometry"],
        "warnings": result["warnings"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
