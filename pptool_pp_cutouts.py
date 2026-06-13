#!/usr/bin/env python3

"""Build standardized cutouts from PP photometry result rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import math

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from PIL import Image

from pptool_mpcsubmission import (
    parse_photometry_file,
    resolve_fits_filename,
    target_to_filename,
)


DEFAULT_CUTOUT_SIZE_ARCSEC = 126.0
MANIFEST_NAME = "cutouts.jsonl"


def _float_or_none(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def row_is_astrometry_only(row):
    """Return True when a PP row has usable astrometry but no magnitude."""

    mag = _float_or_none(row.get("mag"))
    mag_sig = _float_or_none(row.get("mag_sig"))
    return mag is None or mag_sig is None or mag >= 90 or mag_sig >= 90


def _photometry_files(pp_root, target, photometry_file=None):
    pp_root = Path(pp_root).expanduser().resolve()
    if photometry_file is not None:
        if isinstance(photometry_file, (list, tuple)):
            return [Path(path).expanduser().resolve()
                    for path in photometry_file]
        return [Path(photometry_file).expanduser().resolve()]
    candidate = pp_root / ("photometry_%s.dat" % target_to_filename(target))
    if candidate.exists():
        return [candidate]
    matches = sorted(pp_root.rglob("photometry_%s.dat" %
                                   target_to_filename(target)))
    if not matches:
        raise IOError("no photometry_%s.dat found below %s" %
                      (target_to_filename(target), pp_root))
    return [path.resolve() for path in matches]


def _pixel_scale_arcsec(wcs):
    matrix = wcs.pixel_scale_matrix
    xscale = np.hypot(matrix[0, 0], matrix[1, 0]) * 3600.0
    yscale = np.hypot(matrix[0, 1], matrix[1, 1]) * 3600.0
    return float(np.mean([abs(xscale), abs(yscale)]))


def _cutout_size_pixels(wcs, size_arcsec):
    pixels = int(round(float(size_arcsec) / _pixel_scale_arcsec(wcs)))
    if pixels % 2 == 0:
        pixels += 1
    return max(pixels, 1)


def _centered_blank_wcs(source_wcs, target, size_px):
    header = source_wcs.to_header(relax=True)
    cd = getattr(source_wcs.wcs, "cd", None)
    wcs = WCS(header, relax=True)
    wcs.wcs.crpix = [size_px // 2 + 1, size_px // 2 + 1]
    wcs.wcs.crval = [target.ra.deg, target.dec.deg]
    if cd is not None and np.asarray(cd).shape == (2, 2):
        wcs.wcs.cd = cd
    else:
        pc = getattr(source_wcs.wcs, "pc", None)
        if pc is not None and np.asarray(pc).shape == (2, 2):
            wcs.wcs.pc = pc
            wcs.wcs.cdelt = source_wcs.wcs.cdelt
    return wcs


def _zscale_rgba(data):
    finite = np.isfinite(data)
    if not finite.any():
        scaled = np.zeros(data.shape, dtype=np.uint8)
    else:
        values = data[finite]
        try:
            vmin, vmax = ZScaleInterval().get_limits(values)
        except Exception:
            vmin, vmax = np.nanpercentile(values, [1, 99])
        if (not np.isfinite(vmin) or not np.isfinite(vmax) or
                vmax <= vmin):
            vmin, vmax = np.nanmin(values), np.nanmax(values)
        if vmax <= vmin:
            scaled = np.zeros(data.shape, dtype=np.uint8)
        else:
            norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
            norm[~finite] = 0
            scaled = (norm * 255).astype(np.uint8)

    alpha = np.full(scaled.shape, 255, dtype=np.uint8)
    return Image.fromarray(np.dstack([scaled, scaled, scaled, alpha]))


def _relative_to_root(path, root):
    path = Path(path).expanduser()
    root = Path(root).expanduser().resolve()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


def _output_stem(target, row):
    token = Path(row["catalog_token"]).with_suffix("").name
    jd = _float_or_none(row.get("julian_date")) or 0.0
    text = "|".join([
        row.get("catalog_token", ""),
        row.get("julian_date", ""),
        row.get("ra_deg", ""),
        row.get("dec_deg", ""),
    ])
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return "%s_%s_jd%.7f_%s" % (
        target_to_filename(target), token, jd, digest)


def _build_cutout(source_fits, ra_deg, dec_deg, size_arcsec):
    target = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg,
                      frame="icrs")
    with fits.open(str(source_fits), memmap=False,
                   ignore_missing_end=True) as hdulist:
        header = hdulist[0].header.copy()
        data = np.asarray(hdulist[0].data, dtype=np.float32)
        source_wcs = WCS(header, relax=True)

    x, y = source_wcs.world_to_pixel(target)
    size_px = _cutout_size_pixels(source_wcs, size_arcsec)
    try:
        cutout = Cutout2D(
            data, position=(x, y), size=(size_px, size_px),
            wcs=source_wcs, mode="partial", fill_value=np.nan, copy=True)
        cutout_data = cutout.data.astype(np.float32)
        cutout_wcs = cutout.wcs
        overlap = bool(np.isfinite(cutout_data).any())
    except NoOverlapError:
        cutout_data = np.full((size_px, size_px), np.nan, dtype=np.float32)
        cutout_wcs = _centered_blank_wcs(source_wcs, target, size_px)
        overlap = False

    inside = (0 <= float(x) < data.shape[1] and 0 <= float(y) < data.shape[0])
    return cutout_data, cutout_wcs, header, {
        "x": float(x),
        "y": float(y),
        "inside": bool(inside),
        "overlap": bool(overlap),
        "size_px": int(size_px),
    }


def build_pp_cutouts(pp_root, target, photometry_file=None, output_dir=None,
                     size_arcsec=DEFAULT_CUTOUT_SIZE_ARCSEC):
    """Create FITS/PNG cutouts centered on PP measured RA/Dec rows."""

    pp_root = Path(pp_root).expanduser().resolve()
    photometry_files = _photometry_files(pp_root, target, photometry_file)
    output_dir = (pp_root / "PP_cutouts" if output_dir is None
                  else Path(output_dir).expanduser().resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*_cutout.fits", "*_cutout.png", MANIFEST_NAME):
        for stale in output_dir.glob(pattern):
            stale.unlink()

    rows_by_file = [
        (path, parse_photometry_file(path, require_rows=False))
        for path in photometry_files
    ]
    products = []
    manifest_rows = []
    warnings = []

    for photometry_file, rows in rows_by_file:
        if not rows:
            warnings.append("skipping %s; no active PP rows" %
                            photometry_file)
            continue
        for row in rows:
            ra_deg = _float_or_none(row.get("ra_deg"))
            dec_deg = _float_or_none(row.get("dec_deg"))
            if ra_deg is None or dec_deg is None:
                warnings.append(
                    "skipping row in %s; missing source RA/Dec" %
                    photometry_file)
                continue
            try:
                source_fits = resolve_fits_filename(photometry_file,
                                                    row["catalog_token"])
            except Exception as exc:
                warnings.append("skipping row in %s; cannot resolve FITS: %s" %
                                (photometry_file, exc))
                continue

            try:
                data, cutout_wcs, source_header, info = _build_cutout(
                    source_fits, ra_deg, dec_deg, size_arcsec)
            except Exception as exc:
                warnings.append("skipping row in %s; cannot build cutout: %s" %
                                (photometry_file, exc))
                continue

            stem = _output_stem(target, row)
            output_fits = output_dir / ("%s_cutout.fits" % stem)
            output_png = output_dir / ("%s_cutout.png" % stem)
            header = source_header.copy()
            header.update(cutout_wcs.to_header(relax=True))
            header["CUTSRC"] = (source_fits.name[:68], "source PP image")
            header["CUTSIZE"] = (float(size_arcsec), "cutout size arcsec")
            header["CUTRA"] = (ra_deg, "PP measured source RA deg")
            header["CUTDEC"] = (dec_deg, "PP measured source Dec deg")
            header["CUTIN"] = (info["inside"], "target inside source")
            header["CUTOVER"] = (info["overlap"], "cutout overlaps source")
            header["ASTONLY"] = (row_is_astrometry_only(row),
                                 "PP row lacks usable photometry")
            fits.PrimaryHDU(data=data, header=header).writeto(
                output_fits, overwrite=True, output_verify="silentfix")
            _zscale_rgba(data).save(output_png)

            astrometry_only = row_is_astrometry_only(row)
            manifest_row = {
                "index": len(manifest_rows) + 1,
                "target": target,
                "catalog_token": row["catalog_token"],
                "source_photometry_file": str(photometry_file),
                "source_fits_file": str(source_fits),
                "output_fits": str(output_fits),
                "output_png": str(output_png),
                "relative_output_fits": _relative_to_root(
                    output_fits, pp_root),
                "relative_output_png": _relative_to_root(
                    output_png, pp_root),
                "julian_date": _float_or_none(row.get("julian_date")),
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "pred_minus_ra_arcsec": _float_or_none(
                    row.get("pred_minus_ra_arcsec")),
                "pred_minus_dec_arcsec": _float_or_none(
                    row.get("pred_minus_dec_arcsec")),
                "filter": row.get("band"),
                "mag": (None if astrometry_only else
                        _float_or_none(row.get("mag"))),
                "mag_sig": (None if astrometry_only else
                            _float_or_none(row.get("mag_sig"))),
                "astrometry_only": astrometry_only,
                "sextractor_flag": row.get("sextractor_flag"),
                "inside": info["inside"],
                "overlap": info["overlap"],
                "x": info["x"],
                "y": info["y"],
                "size_arcsec": float(size_arcsec),
                "size_px": info["size_px"],
            }
            manifest_rows.append(manifest_row)
            products.append({
                "fits": str(output_fits),
                "png": str(output_png),
                "astrometry_only": astrometry_only,
                "inside": info["inside"],
                "overlap": info["overlap"],
            })

    manifest_path = output_dir / MANIFEST_NAME
    with manifest_path.open("w") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "photometry_file": str(photometry_files[0]),
        "photometry_files": [str(path) for path in photometry_files],
        "n_rows": sum(len(rows) for _, rows in rows_by_file),
        "n_cutouts": len(products),
        "n_fits": len(products),
        "n_png": len(products),
        "n_astrometry_only": sum(1 for row in manifest_rows
                                 if row["astrometry_only"]),
        "n_inside": sum(1 for product in products if product["inside"]),
        "n_overlap": sum(1 for product in products if product["overlap"]),
        "warnings": warnings,
        "products": products,
    }


def record_pp_cutouts(manifest, pp_root, target, photometry_file=None,
                      output_dir=None,
                      size_arcsec=DEFAULT_CUTOUT_SIZE_ARCSEC):
    """Build cutouts and attach product metadata/warnings to a workflow manifest."""

    try:
        result = build_pp_cutouts(
            pp_root, target, photometry_file=photometry_file,
            output_dir=output_dir, size_arcsec=size_arcsec)
    except Exception as exc:
        manifest["pp_cutouts_error"] = str(exc)
        manifest.setdefault("warnings", []).append(
            "PP_cutouts generation failed: %s" % exc)
        return None

    manifest["pp_cutouts"] = result
    manifest.setdefault("warnings", []).extend(result.get("warnings", []))
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="create PP_cutouts from PP photometry results")
    parser.add_argument("pp_root", help="PP output root")
    parser.add_argument("--target", required=True, help="target name")
    parser.add_argument("--photometry-file", help="specific photometry file")
    parser.add_argument("--output-dir", help="cutout output directory")
    parser.add_argument("--size-arcsec", type=float,
                        default=DEFAULT_CUTOUT_SIZE_ARCSEC)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = build_pp_cutouts(
        args.pp_root, args.target, photometry_file=args.photometry_file,
        output_dir=args.output_dir, size_arcsec=args.size_arcsec)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
