#!/usr/bin/env python3

"""Publish object-organized FITS/PNG cutouts and arrow overlays from PP."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from pptool_pp_cutouts import (
    DEFAULT_CUTOUT_SIZE_ARCSEC,
    _build_cutout,
    _pixel_scale_arcsec,
    _replace_celestial_wcs,
)


RESULTS_MANIFEST_NAME = "reduced_results_manifest.json"
HORIZONS_BATCH_SIZE = 10
DEFAULT_SITE_CODE = "705"
DEFAULT_INSTRUMENT_DIR = "ARCTIC"
ORIENTATION_MODES = ("discrete", "reproject")


_DISCRETE_ARRAY_TRANSFORMS = (
    ("identity", lambda array: array),
    ("rot90", lambda array: np.rot90(array, 1)),
    ("rot180", lambda array: np.rot90(array, 2)),
    ("rot270", lambda array: np.rot90(array, 3)),
    ("flip-lr", np.fliplr),
    ("flip-lr-rot90", lambda array: np.rot90(np.fliplr(array), 1)),
    ("flip-lr-rot180", lambda array: np.rot90(np.fliplr(array), 2)),
    ("flip-lr-rot270", lambda array: np.rot90(np.fliplr(array), 3)),
)


def _array_transform_affine(shape, transform):
    """Return output-to-input zero-based affine terms for an array transform."""

    yy, xx = np.indices(shape, dtype=np.int32)
    input_x = transform(xx)
    input_y = transform(yy)
    if min(input_x.shape) < 2:
        raise ValueError("discrete orientation requires both axes >= 2 pixels")
    offset = np.array([input_x[0, 0], input_y[0, 0]], dtype=float)
    matrix = np.column_stack((
        np.array([input_x[0, 1], input_y[0, 1]], dtype=float) - offset,
        np.array([input_x[1, 0], input_y[1, 0]], dtype=float) - offset,
    ))
    return matrix, offset


def _discrete_wcs(source_wcs, matrix, offset, output_shape):
    """Apply an exact quarter-turn/flip pixel transform to a linear WCS."""

    identity = (np.array_equal(matrix, np.eye(2)) and
                np.array_equal(offset, np.zeros(2)))
    distortion = any(getattr(source_wcs, name, None) is not None for name in
                     ("sip", "cpdis1", "cpdis2", "det2im1", "det2im2"))
    if distortion and not identity:
        raise ValueError(
            "cannot discretely rotate/flip a WCS with detector-plane "
            "distortions; use --orientation-mode reproject explicitly")

    output_wcs = source_wcs.deepcopy()
    # Convert p_in(0-based) = A p_out(0-based) + b to FITS 1-based terms.
    one = np.ones(2)
    fits_offset = offset + one - matrix @ one
    output_wcs.wcs.crpix = np.linalg.solve(
        matrix, source_wcs.wcs.crpix - fits_offset)
    output_wcs.wcs.cd = source_wcs.pixel_scale_matrix @ matrix
    output_wcs.pixel_shape = (int(output_shape[1]), int(output_shape[0]))
    return output_wcs


def orient_cutout_discrete(data, source_wcs):
    """Orient a cutout using only lossless quarter-turns and pixel flips."""

    candidates = []
    for name, transform in _DISCRETE_ARRAY_TRANSFORMS:
        matrix, offset = _array_transform_affine(data.shape, transform)
        linear = source_wcs.pixel_scale_matrix @ matrix
        x_norm = np.linalg.norm(linear[:, 0])
        y_norm = np.linalg.norm(linear[:, 1])
        if x_norm == 0 or y_norm == 0:
            continue
        # East-left means increasing RA points toward -x; north-up means
        # increasing Dec points toward +y. The stable ordering breaks ties in
        # favor of doing nothing, then rotations, then mirrored transforms.
        score = (-linear[0, 0] / x_norm +
                 linear[1, 1] / y_norm)
        candidates.append((score, name, transform, matrix, offset))
    if not candidates:
        raise ValueError("cannot determine a discrete cutout orientation")
    _, name, transform, matrix, offset = max(
        candidates, key=lambda candidate: candidate[0])
    output_data = np.ascontiguousarray(transform(data))
    output_wcs = _discrete_wcs(
        source_wcs, matrix, offset, output_data.shape)
    return output_data, output_wcs, name


def exposure_midpoint(header):
    """Return an exposure midpoint as UTC ``Time``."""

    if "MIDTIMJD" in header:
        return Time(float(header["MIDTIMJD"]), format="jd", scale="utc")
    if "DATE-OBS" not in header:
        raise KeyError("FITS header lacks DATE-OBS")
    start = Time(str(header["DATE-OBS"]).strip(), format="isot", scale="utc")
    seconds = float(header.get("EXPTIME", header.get("EXPOSURE", 0.0)))
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid exposure time %r" % seconds)
    return start + (seconds / 2.0) * u.s


def _chunks(values, size=HORIZONS_BATCH_SIZE):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def propagate_cached_ephemeris(base, base_jd, epoch_jd):
    """Linearly propagate a nearby cached PP ephemeris in the sky plane."""

    elapsed_hours = (float(epoch_jd) - float(base_jd)) * 24.0
    cos_dec = math.cos(math.radians(float(base["dec_deg"])))
    if abs(cos_dec) < 1e-8:
        raise ValueError("cannot propagate RA at a celestial pole")
    return {
        "ra_deg": (float(base["ra_deg"]) +
                   float(base["ra_rate_arcsec_hour"]) * elapsed_hours /
                   (3600.0 * cos_dec)),
        "dec_deg": (float(base["dec_deg"]) +
                    float(base["dec_rate_arcsec_hour"]) * elapsed_hours /
                    3600.0),
        "antisolar_deg": float(base["antisolar_deg"]),
        "antimotion_deg": float(base["antimotion_deg"]),
        "ephemeris_source": "propagated_from_cached_pp_ephemeris",
        "ephemeris_base_jd": float(base_jd),
    }


def query_horizons(target, epochs, site_code=DEFAULT_SITE_CODE):
    """Read target coordinates and tail vectors from PP's Horizons cache.

    PP queries every target epoch during distillation. Reusing those exact
    cached singleton requests avoids transmitting the observing manifest a
    second time. A missing cache entry fails closed instead of going online.
    """

    from astroquery.jplhorizons import Horizons
    from astroquery.jplhorizons import conf as horizons_conf
    from astroquery.query import AstroQuery

    results = []
    cached_base = None
    cached_base_jd = None
    for epoch in epochs:
        obj = Horizons(
            id=target,
            id_type="smallbody",
            location=str(site_code),
            epochs=float(epoch.jd),
        )
        payload = obj.ephemerides(get_query_payload=True)
        query = AstroQuery(
            "GET", horizons_conf.horizons_server, params=payload,
            timeout=obj.TIMEOUT)
        response = query.from_cache(obj.cache_location, cache_timeout=None)
        if response is None:
            if cached_base is None:
                raise FileNotFoundError(
                    "PP Horizons cache lacks %s at JD %.15f; refusing an "
                    "unapproved network lookup" % (target, epoch.jd))
            results.append(propagate_cached_ephemeris(
                cached_base, cached_base_jd, epoch.jd))
            continue
        obj._last_query = query
        table = obj._parse_result(response)
        if len(table) != 1:
            raise RuntimeError(
                "cached Horizons response has %d rows, expected 1" %
                len(table))
        row = table[0]
        cached_base = {
            "ra_deg": float(row["RA"]),
            "dec_deg": float(row["DEC"]),
            "antisolar_deg": float(row["sunTargetPA"]),
            "antimotion_deg": float(row["velocityPA"]),
            "ra_rate_arcsec_hour": float(row["RA_rate"]),
            "dec_rate_arcsec_hour": float(row["DEC_rate"]),
            "ephemeris_source": "cached_pp_horizons",
        }
        cached_base_jd = float(row["datetime_jd"])
        results.append({
            key: value for key, value in cached_base.items()
            if key not in ("ra_rate_arcsec_hour", "dec_rate_arcsec_hour")
        })
    return results


def coc_tools_make_arrows(antisolar, antimotion, output_path):
    """Draw an overlay with the established ``coc_tools`` implementation."""

    coc_tools_parent = Path.home() / "GitHub"
    if str(coc_tools_parent) not in sys.path:
        sys.path.insert(0, str(coc_tools_parent))
    from coc_tools.arrows import makeArrows

    makeArrows(antiSun=antisolar, antiMotion=antimotion,
               outfile=str(output_path))
    # coc_tools supports interactive use and leaves its figure open.
    import matplotlib.pyplot as plt
    plt.close("all")


def coc_tools_fits2png(output_fits, output_png):
    """Render a cutout with the established ``coc_tools`` FITS renderer."""

    coc_tools_parent = Path.home() / "GitHub"
    if str(coc_tools_parent) not in sys.path:
        sys.path.insert(0, str(coc_tools_parent))
    from coc_tools.fits.fits2png import fits2png

    output_fits = Path(output_fits)
    output_png = Path(output_png)
    fits2png([str(output_fits)])
    # Preserve fits2png's historical filename rule, including .fit/.fits.
    generated_png = Path(str(output_fits).split(".fit")[0] + ".png")
    if generated_png != output_png:
        shutil.move(str(generated_png), str(output_png))


def _result_stem(record, midpoint, source_path, size_arcsec):
    midpoint.precision = 6
    epoch = midpoint.utc.isot.replace("T", "_").replace(":", ".")
    size_token = ("%g" % float(size_arcsec)).replace(".", "p")
    return "%s_%s_%s_%s_%sarcsec_NuEl" % (
        record["target_dir"], epoch, source_path.stem, record["band"],
        size_token)


def _load_jobs(workflow_manifest):
    manifest_path = Path(workflow_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    required = {"records", "output_dir", "source_count"}
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError("workflow manifest lacks %s" % ", ".join(missing))

    jobs = []
    for record in manifest["records"]:
        source_path = Path(record["destination"]).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(str(source_path))
        with fits.open(source_path, memmap=False,
                       ignore_missing_end=True) as hdulist:
            midpoint = exposure_midpoint(hdulist[0].header)
        jobs.append({
            "record": record,
            "source_path": source_path,
            "midpoint": midpoint,
        })
    if len(jobs) != int(manifest["source_count"]):
        raise ValueError("manifest source count does not match its records")
    return manifest_path, manifest, jobs


def _check_results_root(results_dir, resume=False):
    results_dir = Path(results_dir).expanduser().resolve()
    if (not resume and results_dir.exists() and
            any(results_dir.iterdir())):
        raise FileExistsError("results directory is not empty: %s" %
                              results_dir)
    if not results_dir.parent.is_dir():
        raise NotADirectoryError(str(results_dir.parent))
    return results_dir


def publish_reduced_results(
        workflow_manifest, results_dir,
        site_code=DEFAULT_SITE_CODE,
        size_arcsec=DEFAULT_CUTOUT_SIZE_ARCSEC,
        query_func=query_horizons,
        arrow_func=coc_tools_make_arrows,
        png_func=coc_tools_fits2png,
        orientation_mode="discrete",
        resume=False):
    """Create FITS/PNG cutouts and matching arrow PDFs under each object."""

    if orientation_mode not in ORIENTATION_MODES:
        raise ValueError("orientation_mode must be one of %s" %
                         ", ".join(ORIENTATION_MODES))

    manifest_path, workflow, jobs = _load_jobs(workflow_manifest)
    results_dir = _check_results_root(results_dir, resume=resume)
    results_dir.mkdir(parents=True, exist_ok=True)

    output_manifest = manifest_path.parent / RESULTS_MANIFEST_NAME
    products = []
    if resume:
        if not output_manifest.is_file():
            raise FileNotFoundError(
                "cannot resume without %s" % output_manifest)
        previous = json.loads(output_manifest.read_text())
        if (previous.get("workflow_manifest") != str(manifest_path) or
                previous.get("results_dir") != str(results_dir)):
            raise ValueError("results resume paths do not match the manifest")
        if previous.get("orientation_mode") == orientation_mode:
            products = [product for product in previous.get("products", [])
                        if product.get("output_fits") and
                        Path(product["output_fits"]).is_file() and
                        Path(product["output_png"]).is_file() and
                        Path(product["output_arrows"]).is_file()]
    completed_sources = {product["source_fits"] for product in products}

    by_target = {}
    for job in jobs:
        record = job["record"]
        by_target.setdefault(record["target"], []).append(job)
        (results_dir / record["target_dir"] / DEFAULT_INSTRUMENT_DIR /
         "cutouts").mkdir(parents=True, exist_ok=True)

    failures = []
    for target in sorted(by_target):
        target_jobs = sorted(
            [job for job in by_target[target]
             if str(job["source_path"]) not in completed_sources],
            key=lambda job: (job["midpoint"].jd,
                             job["source_path"].name))
        for batch in _chunks(target_jobs):
            try:
                ephemerides = query_func(
                    target, [job["midpoint"] for job in batch],
                    site_code=site_code)
                if len(ephemerides) != len(batch):
                    raise RuntimeError("ephemeris row count mismatch")
            except Exception as exc:
                for job in batch:
                    failures.append({
                        "target": target,
                        "source_fits": str(job["source_path"]),
                        "stage": "horizons",
                        "error": str(exc),
                    })
                continue

            for job, eph in zip(batch, ephemerides):
                record = job["record"]
                source_path = job["source_path"]
                output_dir = (results_dir / record["target_dir"] /
                              DEFAULT_INSTRUMENT_DIR / "cutouts")
                stem = _result_stem(
                    record, job["midpoint"], source_path, size_arcsec)
                output_fits = output_dir / (stem + ".fits")
                output_png = output_dir / (stem + ".png")
                output_arrows = output_dir / (stem + "_arrows.pdf")
                try:
                    with fits.open(source_path, memmap=False,
                                   ignore_missing_end=True) as hdulist:
                        source_wcs = WCS(hdulist[0].header, relax=True)
                        pixel_scale = _pixel_scale_arcsec(source_wcs)
                    data, cutout_wcs, source_header, info = _build_cutout(
                        source_path, eph["ra_deg"], eph["dec_deg"],
                        size_arcsec,
                        output_pixel_scale_arcsec=(
                            pixel_scale
                            if orientation_mode == "reproject" else None))
                    if not info["inside"] or not info["overlap"]:
                        raise ValueError(
                            "predicted target is outside the source image")
                    if orientation_mode == "discrete":
                        data, cutout_wcs, discrete_transform = (
                            orient_cutout_discrete(data, cutout_wcs))
                        resampled = False
                    else:
                        discrete_transform = None
                        resampled = True
                    header = _replace_celestial_wcs(
                        source_header, cutout_wcs)
                    header["CUTSRC"] = (
                        source_path.name[:68], "source PP image")
                    header["CUTSIZE"] = (
                        float(size_arcsec), "cutout size arcsec")
                    header["CUTRA"] = (
                        float(eph["ra_deg"]), "predicted target RA deg")
                    header["CUTDEC"] = (
                        float(eph["dec_deg"]), "predicted target Dec deg")
                    header["CUTIN"] = (
                        info["inside"], "target inside source")
                    header["CUTOVER"] = (
                        info["overlap"], "cutout overlaps source")
                    header["CUTNAX"] = (
                        info["size_px"], "cutout axis pixels")
                    header["CUTSCALE"] = (
                        info["output_pixel_scale_arcsec"],
                        "output scale arcsec/pixel")
                    header["CUTORNT"] = (
                        "NUP-ELFT", "cutout orientation")
                    header["CUTMODE"] = (
                        "D4-PIX" if not resampled else "BILINEAR",
                        "orientation method")
                    header["CUTXFM"] = (
                        discrete_transform or "reproject",
                        "pixel transform")
                    header["CUTRSMP"] = (
                        resampled, "pixel values were resampled")
                    fits.PrimaryHDU(data=data, header=header).writeto(
                        output_fits, overwrite=True,
                        output_verify="silentfix")
                    png_func(output_fits, output_png)
                    arrow_func(eph["antisolar_deg"],
                               eph["antimotion_deg"], output_arrows)
                    if not all(path.is_file() for path in
                               (output_fits, output_png, output_arrows)):
                        raise RuntimeError(
                            "result writer did not create all three files")
                except Exception as exc:
                    output_fits.unlink(missing_ok=True)
                    output_png.unlink(missing_ok=True)
                    output_arrows.unlink(missing_ok=True)
                    failures.append({
                        "target": target,
                        "source_fits": str(source_path),
                        "stage": "cutout_or_arrows",
                        "error": str(exc),
                    })
                    continue
                products.append({
                    "target": target,
                    "target_dir": record["target_dir"],
                    "band": record["band"],
                    "source_fits": str(source_path),
                    "midpoint_utc": job["midpoint"].utc.isot,
                    "midpoint_jd": float(job["midpoint"].utc.jd),
                    **eph,
                    "output_fits": str(output_fits),
                    "output_png": str(output_png),
                    "output_arrows": str(output_arrows),
                    "size_arcsec": float(size_arcsec),
                    "orientation": (
                        "north approximately up, east approximately left"
                        if not resampled else "north up, east left"),
                    "orientation_mode": orientation_mode,
                    "discrete_transform": discrete_transform,
                    "resampled": resampled,
                })

    result = {
        "workflow_manifest": str(manifest_path),
        "workflow_output_dir": workflow["output_dir"],
        "results_dir": str(results_dir),
        "site_code": str(site_code),
        "instrument_dir": DEFAULT_INSTRUMENT_DIR,
        "source_count": len(jobs),
        "object_count": len(by_target),
        "product_count": len(products),
        "failure_count": len(failures),
        "orientation_mode": orientation_mode,
        "png_renderer": "coc_tools.fits.fits2png.fits2png",
        "products": products,
        "failures": failures,
        "status": "complete" if not failures else "partial",
    }
    output_manifest.write_text(json.dumps(result, indent=2,
                                          sort_keys=True) + "\n")
    result["manifest_path"] = str(output_manifest)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=("publish reduced-data FITS/PNG cutouts and "
                     "coc_tools arrows"))
    parser.add_argument("workflow_manifest")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--site-code", default=DEFAULT_SITE_CODE)
    parser.add_argument("--size-arcsec", type=float,
                        default=DEFAULT_CUTOUT_SIZE_ARCSEC)
    parser.add_argument(
        "--orientation-mode", choices=ORIENTATION_MODES, default="discrete",
        help=("discrete preserves detector pixels using only 90-degree "
              "rotations/flips; reproject explicitly enables bilinear "
              "north-up/east-left resampling"))
    parser.add_argument("--resume", action="store_true",
                        help="retry incomplete products from an existing run")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = publish_reduced_results(
        args.workflow_manifest, args.results_dir,
        site_code=args.site_code, size_arcsec=args.size_arcsec,
        orientation_mode=args.orientation_mode,
        resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
