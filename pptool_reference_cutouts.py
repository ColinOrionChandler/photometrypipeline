#!/usr/bin/env python3

"""Fetch deep DESI Legacy Surveys reference cutouts at reported positions.

A final independent check before submitting astrometry: pull a deep, coadded
Legacy Surveys cutout (grz) centered on each position we are about to report,
so a static source hiding at that location (galaxy, star, artifact) would be
obvious.  A genuine moving object is averaged out of the coadd and leaves the
spot empty.

Centers come from the same reported positions used for the MPC files: either a
PP photometry file or a supplied positions CSV (including SBSAR click/centroid
exports), so this lines up one-to-one with PP_cutouts_<source>/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from PIL import Image, ImageDraw

from pptool_mpcsubmission import parse_photometry_file, target_to_filename
from pptool_pp_cutouts import (
    _float_or_none,
    _output_stem,
    _sanitize_label,
    load_positions,
)


LEGACY_BASE_URL = "https://www.legacysurvey.org/viewer"
DEFAULT_LAYER = "ls-dr10"
DEFAULT_PIXSCALE = 0.262          # arcsec/pixel (DECam native)
DEFAULT_SIZE_ARCSEC = 126.0       # matches pptool_pp_cutouts default
DEFAULT_BANDS = "grz"
MANIFEST_NAME = "reference_cutouts.jsonl"
OUTPUT_BASENAME = "reference_cutouts"
_USER_AGENT = "photometrypipeline/pptool_reference_cutouts"


def size_pixels(size_arcsec, pixscale):
    pixels = int(round(float(size_arcsec) / float(pixscale)))
    return max(1, min(pixels, 3000))   # Legacy viewer hard cap


def legacy_cutout_url(ra_deg, dec_deg, kind, layer=DEFAULT_LAYER,
                      pixscale=DEFAULT_PIXSCALE,
                      size_arcsec=DEFAULT_SIZE_ARCSEC, bands=DEFAULT_BANDS):
    """Build a Legacy Surveys cutout URL. ``kind`` is 'jpg' or 'fits'."""

    params = {
        "ra": "%.7f" % float(ra_deg),
        "dec": "%.7f" % float(dec_deg),
        "layer": layer,
        "pixscale": "%.4f" % float(pixscale),
        "size": size_pixels(size_arcsec, pixscale),
    }
    if kind == "fits":
        params["bands"] = bands
    return "%s/cutout.%s?%s" % (LEGACY_BASE_URL, kind, urlencode(params))


def _http_get(url, timeout=60):
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _coverage_from_fits(fits_bytes):
    """Return (has_coverage, finite_nonzero_fraction) for a Legacy FITS cutout."""

    try:
        from io import BytesIO
        with fits.open(BytesIO(fits_bytes), memmap=False) as hdulist:
            data = None
            for hdu in hdulist:
                if getattr(hdu, "data", None) is not None:
                    data = np.asarray(hdu.data, dtype=float)
                    break
        if data is None or data.size == 0:
            return False, 0.0
        finite = np.isfinite(data)
        nonzero = finite & (data != 0)
        frac = float(nonzero.sum()) / float(data.size)
        return frac > 0.01, frac
    except Exception:
        return False, 0.0


DEFAULT_BLINK_INTERVAL_MS = 500


def _zscale_gray(data):
    """Scale a 2D array to uint8 with ZScale, NaNs -> 0."""

    data = np.asarray(data, dtype=float)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=np.uint8)
    try:
        vmin, vmax = ZScaleInterval().get_limits(data[finite])
    except Exception:
        vmin, vmax = np.nanpercentile(data[finite], [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanmin(data[finite]), np.nanmax(data[finite])
    if vmax <= vmin:
        return np.zeros(data.shape, dtype=np.uint8)
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    norm[~finite] = 0
    return (norm * 255).astype(np.uint8)


def _band_plane(fits_path):
    """Return (2D first-band image, 2D celestial WCS) from a Legacy FITS cutout."""

    with fits.open(str(fits_path), memmap=False) as hdulist:
        hdu = next(h for h in hdulist if getattr(h, "data", None) is not None)
        data = np.asarray(hdu.data, dtype=float)
        wcs = WCS(hdu.header)
    if data.ndim == 3:           # grz cube -> first band (g)
        data = data[0]
    return data, wcs.celestial


def _reproject_onto(source_fits, target_wcs, shape):
    """Resample a source image onto target_wcs/shape using the source's own WCS.

    Uses the original full-frame WCS (correct for TAN-TPV), so it does not rely
    on any cutout WCS.
    """

    from scipy.ndimage import map_coordinates

    with fits.open(str(source_fits), memmap=False) as hdulist:
        src = np.asarray(hdulist[0].data, dtype=float)
        src_wcs = WCS(hdulist[0].header).celestial
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra, dec = target_wcs.all_pix2world(xx, yy, 0)
    sx, sy = src_wcs.all_world2pix(ra, dec, 0)
    out = map_coordinates(src, [sy.ravel(), sx.ravel()], order=1,
                          mode="constant", cval=np.nan)
    return out.reshape(shape)


def _label(img_uint8, text):
    """Return an RGB PIL image (north-up) with a corner label and center mark."""

    arr = np.flipud(img_uint8)                       # FITS bottom-origin -> top
    im = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(im)
    h, w = arr.shape
    cx, cy = w // 2, h // 2
    draw.line([(cx - 9, cy), (cx - 3, cy)], fill=(0, 255, 255), width=1)
    draw.line([(cx + 3, cy), (cx + 9, cy)], fill=(0, 255, 255), width=1)
    draw.line([(cx, cy - 9), (cx, cy - 3)], fill=(0, 255, 255), width=1)
    draw.line([(cx, cy + 3), (cx, cy + 9)], fill=(0, 255, 255), width=1)
    draw.rectangle([2, 2, 8 * len(text) + 6, 16], fill=(0, 0, 0))
    draw.text((5, 4), text, fill=(255, 255, 0))
    return im


def build_comparison_products(source_fits, ref_fits, out_png, out_gif,
                              our_label="DECam (ours)", ref_label="LS ref",
                              interval_ms=DEFAULT_BLINK_INTERVAL_MS):
    """Make a side-by-side PNG and a blink GIF of our cutout vs the reference.

    Our frame is reprojected onto the reference WCS grid so static sources
    register and only a moving object differs between blink frames.
    """

    ref_img, ref_wcs = _band_plane(ref_fits)
    our_img = _reproject_onto(source_fits, ref_wcs, ref_img.shape)
    our_u = _zscale_gray(our_img)
    ref_u = _zscale_gray(ref_img)

    our_rgb = _label(our_u, our_label)
    ref_rgb = _label(ref_u, ref_label)

    gap = 6
    canvas = Image.new("RGB", (our_rgb.width + ref_rgb.width + gap,
                               max(our_rgb.height, ref_rgb.height)),
                       (20, 20, 20))
    canvas.paste(our_rgb, (0, 0))
    canvas.paste(ref_rgb, (our_rgb.width + gap, 0))
    canvas.save(str(out_png))

    our_rgb.save(str(out_gif), save_all=True, append_images=[ref_rgb],
                 duration=int(interval_ms), loop=0, disposal=2)
    return {"side_by_side_png": str(out_png), "blink_gif": str(out_gif)}


def _resolve_source_fits(spec, pp_root):
    fits_file = spec.get("fits_file")
    if fits_file:
        path = Path(str(fits_file)).expanduser()
        if path.exists():
            return path.resolve()
    token = str(spec.get("catalog_token") or "")
    for cand in (Path(token), pp_root / token, pp_root / (token + ".fits"),
                 pp_root / Path(token).name,
                 pp_root / (Path(token).stem + ".fits")):
        if cand and Path(cand).expanduser().exists():
            return Path(cand).expanduser().resolve()
    return None


def _specs(pp_root, target, positions, photometry_file, position_column):
    """Build the list of center positions to fetch."""

    if positions is not None:
        specs, _ = load_positions(positions, position_column=position_column)
        return specs
    pp_root = Path(pp_root).expanduser().resolve()
    if photometry_file is not None:
        files = [Path(photometry_file).expanduser().resolve()]
    else:
        candidate = pp_root / ("photometry_%s.dat" % target_to_filename(target))
        files = ([candidate] if candidate.exists()
                 else sorted(pp_root.rglob("photometry_%s.dat" %
                                           target_to_filename(target))))
    specs = []
    for path in files:
        for row in parse_photometry_file(path, require_rows=False):
            ra = _float_or_none(row.get("ra_deg"))
            dec = _float_or_none(row.get("dec_deg"))
            if ra is None or dec is None:
                continue
            specs.append({"ra_deg": ra, "dec_deg": dec,
                          "catalog_token": row.get("catalog_token"),
                          "julian_date": _float_or_none(row.get("julian_date"))})
    return specs


def build_reference_cutouts(pp_root, target, positions=None,
                            photometry_file=None, output_dir=None,
                            position_source=None, position_column="auto",
                            layer=DEFAULT_LAYER, size_arcsec=DEFAULT_SIZE_ARCSEC,
                            pixscale=DEFAULT_PIXSCALE, bands=DEFAULT_BANDS,
                            make_comparisons=False,
                            blink_interval_ms=DEFAULT_BLINK_INTERVAL_MS,
                            fetcher=_http_get):
    """Download Legacy Surveys reference cutouts for each reported position."""

    pp_root = Path(pp_root).expanduser().resolve()
    if output_dir is None:
        name = OUTPUT_BASENAME
        if position_source:
            name = "%s_%s" % (OUTPUT_BASENAME, _sanitize_label(position_source))
        output_dir = pp_root / name
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*_ref.fits", "*_ref.jpg", "*_compare.png", "*_blink.gif",
                    MANIFEST_NAME):
        for stale in output_dir.glob(pattern):
            stale.unlink()

    specs = _specs(pp_root, target, positions, photometry_file, position_column)
    manifest_rows = []
    warnings = []
    for spec in specs:
        ra = _float_or_none(spec.get("ra_deg"))
        dec = _float_or_none(spec.get("dec_deg"))
        if ra is None or dec is None:
            continue
        stem = _output_stem(target, spec)
        jpg_url = legacy_cutout_url(ra, dec, "jpg", layer, pixscale,
                                    size_arcsec, bands)
        fits_url = legacy_cutout_url(ra, dec, "fits", layer, pixscale,
                                     size_arcsec, bands)
        out_jpg = output_dir / ("%s_ref.jpg" % stem)
        out_fits = output_dir / ("%s_ref.fits" % stem)
        has_coverage = None
        finite_fraction = None
        errors = []
        try:
            out_jpg.write_bytes(fetcher(jpg_url))
        except Exception as exc:
            errors.append("jpg: %s" % exc)
            out_jpg = None
        try:
            fits_bytes = fetcher(fits_url)
            out_fits.write_bytes(fits_bytes)
            has_coverage, finite_fraction = _coverage_from_fits(fits_bytes)
        except Exception as exc:
            errors.append("fits: %s" % exc)
            out_fits = None
        if errors:
            warnings.append("%s: %s" % (stem, "; ".join(errors)))
        if has_coverage is False:
            warnings.append("%s: no Legacy Surveys coverage at %.5f %.5f" %
                            (stem, ra, dec))
        out_compare = None
        out_blink = None
        if make_comparisons and out_fits is not None:
            source_fits = _resolve_source_fits(spec, pp_root)
            if source_fits is None:
                warnings.append("%s: no source FITS for comparison/blink" % stem)
            else:
                try:
                    products = build_comparison_products(
                        source_fits, out_fits,
                        output_dir / ("%s_compare.png" % stem),
                        output_dir / ("%s_blink.gif" % stem),
                        our_label=target, ref_label=layer,
                        interval_ms=blink_interval_ms)
                    out_compare = products["side_by_side_png"]
                    out_blink = products["blink_gif"]
                except Exception as exc:
                    warnings.append("%s: comparison build failed: %s" %
                                    (stem, exc))
        manifest_rows.append({
            "index": len(manifest_rows) + 1,
            "target": target,
            "position_source": position_source,
            "layer": layer,
            "ra_deg": ra,
            "dec_deg": dec,
            "julian_date": _float_or_none(spec.get("julian_date")),
            "catalog_token": spec.get("catalog_token"),
            "size_arcsec": float(size_arcsec),
            "pixscale": float(pixscale),
            "jpg_url": jpg_url,
            "fits_url": fits_url,
            "output_jpg": str(out_jpg) if out_jpg else None,
            "output_fits": str(out_fits) if out_fits else None,
            "output_compare_png": out_compare,
            "output_blink_gif": out_blink,
            "has_coverage": has_coverage,
            "finite_nonzero_fraction": finite_fraction,
        })

    manifest_path = output_dir / MANIFEST_NAME
    with manifest_path.open("w") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "layer": layer,
        "position_source": position_source,
        "n_positions": len(specs),
        "n_cutouts": len(manifest_rows),
        "n_no_coverage": sum(1 for r in manifest_rows
                             if r["has_coverage"] is False),
        "warnings": warnings,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="fetch deep Legacy Surveys reference cutouts at reported "
                    "positions (static-source sanity check before MPC files)")
    parser.add_argument("pp_root", help="PP output root")
    parser.add_argument("--target", required=True, help="target name")
    parser.add_argument("--positions", help="positions CSV (PP or SBSAR click "
                                            "export); default uses PP photometry")
    parser.add_argument("--photometry-file", help="specific PP photometry file")
    parser.add_argument("--position-source",
                        help="provenance label -> reference_cutouts_<label> dir")
    parser.add_argument("--position-column", default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--layer", default=DEFAULT_LAYER,
                        help="Legacy Surveys layer (e.g. ls-dr10)")
    parser.add_argument("--size-arcsec", type=float,
                        default=DEFAULT_SIZE_ARCSEC)
    parser.add_argument("--pixscale", type=float, default=DEFAULT_PIXSCALE)
    parser.add_argument("--bands", default=DEFAULT_BANDS)
    parser.add_argument("--comparisons", action="store_true",
                        help="also build side-by-side PNG + blink GIF of our "
                             "cutout vs the reference (needs source fits_file)")
    parser.add_argument("--blink-interval-ms", type=int,
                        default=DEFAULT_BLINK_INTERVAL_MS)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = build_reference_cutouts(
        args.pp_root, args.target, positions=args.positions,
        photometry_file=args.photometry_file, output_dir=args.output_dir,
        position_source=args.position_source,
        position_column=args.position_column, layer=args.layer,
        size_arcsec=args.size_arcsec, pixscale=args.pixscale, bands=args.bands,
        make_comparisons=args.comparisons,
        blink_interval_ms=args.blink_interval_ms)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
