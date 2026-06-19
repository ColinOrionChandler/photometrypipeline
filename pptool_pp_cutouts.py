#!/usr/bin/env python3

"""Build standardized cutouts from PP photometry result rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import math
import re

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
DEFAULT_OUTPUT_BASENAME = "PP_cutouts"

# Right ascension / declination column pairs understood when ingesting an
# externally supplied positions table (e.g. the CSV emitted by the SBSAR
# click_astrometry_orbits tool).  Order encodes the "auto" preference: the
# refined centroid is used when present, otherwise the raw click, otherwise a
# plain ra_deg/dec_deg pair.
POSITION_COLUMN_PAIRS = {
    "centroid": ("centroid_ra_deg", "centroid_dec_deg"),
    "click": ("click_ra_deg", "click_dec_deg"),
    "plain": ("ra_deg", "dec_deg"),
}
POSITION_COLUMN_AUTO_ORDER = ("centroid", "click", "plain")


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


def _sanitize_label(label):
    """Reduce a free-form provenance label to a safe directory suffix."""

    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(label).strip())
    cleaned = cleaned.strip("_.")
    return cleaned or "custom"


def default_output_dir(pp_root, position_source=None):
    """Return the default cutout directory, distinct per position source.

    Cutouts centered on different position provenances (PP-measured rows, an
    SBSAR click export, raw ephemeris, ...) must not share a directory because
    each run wipes stale products from its own directory.  A ``position_source``
    label yields ``PP_cutouts_<label>`` so those product sets stay separate.
    """

    pp_root = Path(pp_root).expanduser().resolve()
    if position_source:
        return pp_root / ("%s_%s" % (DEFAULT_OUTPUT_BASENAME,
                                     _sanitize_label(position_source)))
    return pp_root / DEFAULT_OUTPUT_BASENAME


def _resolve_position_column(fieldnames, position_column):
    """Pick the (ra, dec) column pair from an ingested positions table."""

    available = set(fieldnames or ())
    if position_column and position_column != "auto":
        if position_column not in POSITION_COLUMN_PAIRS:
            raise ValueError("unknown position column %r (choose from %s)" %
                             (position_column,
                              ", ".join(sorted(POSITION_COLUMN_PAIRS))))
        ra_col, dec_col = POSITION_COLUMN_PAIRS[position_column]
        if ra_col not in available or dec_col not in available:
            raise ValueError("positions table lacks %s/%s columns" %
                             (ra_col, dec_col))
        return position_column, ra_col, dec_col
    for name in POSITION_COLUMN_AUTO_ORDER:
        ra_col, dec_col = POSITION_COLUMN_PAIRS[name]
        if ra_col in available and dec_col in available:
            return name, ra_col, dec_col
    raise ValueError("positions table has no recognised RA/Dec columns; "
                     "expected one of %s" %
                     ", ".join("%s/%s" % pair
                               for pair in POSITION_COLUMN_PAIRS.values()))


def _position_julian_date(row):
    for key in ("julian_date", "jd"):
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    for key in ("click_mjd_utc", "centroid_mjd_utc", "mjd", "mjd_utc"):
        value = _float_or_none(row.get(key))
        if value is not None:
            return value + 2400000.5
    return None


def load_positions(positions, position_column="auto"):
    """Normalise supplied positions into cutout center specifications.

    ``positions`` is either a path to a CSV file or an iterable of mappings.
    Each returned spec carries the RA/Dec to center on plus enough provenance
    (a FITS path or a same-stem token) to locate the source image.  SBSAR
    ``click_astrometry_orbits`` exports are recognised directly.
    """

    if isinstance(positions, (str, Path)):
        path = Path(positions).expanduser().resolve()
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
    else:
        rows = [dict(row) for row in positions]
        fieldnames = sorted({key for row in rows for key in row})

    _, ra_col, dec_col = _resolve_position_column(fieldnames, position_column)

    specs = []
    for row in rows:
        ra_deg = _float_or_none(row.get(ra_col))
        dec_deg = _float_or_none(row.get(dec_col))
        if ra_deg is None or dec_deg is None:
            continue
        token = ""
        for key in ("catalog_token", "fits_file", "image", "stem", "filename"):
            value = row.get(key)
            if value:
                token = str(value)
                break
        specs.append({
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "catalog_token": token,
            "fits_file": row.get("fits_file") or None,
            "julian_date": _position_julian_date(row),
            "mag": _float_or_none(row.get("mag")),
            "mag_sig": _float_or_none(row.get("mag_sig")),
            "sextractor_flag": row.get("sextractor_flag"),
        })
    return specs, ra_col


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
    token_value = str(row.get("catalog_token") or "position")
    token = Path(token_value).with_suffix("").name or "position"
    jd = _float_or_none(row.get("julian_date")) or 0.0
    text = "|".join(str(row.get(key, "")) for key in
                    ("catalog_token", "julian_date", "ra_deg", "dec_deg"))
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


def _resolve_spec_fits(spec, search_dirs):
    """Locate the source FITS for a cutout center specification."""

    fits_file = spec.get("fits_file")
    if fits_file:
        path = Path(str(fits_file)).expanduser()
        if path.exists():
            return path.resolve()
    token = str(spec.get("catalog_token") or "")
    if token:
        token_path = Path(token).expanduser()
        if token_path.exists():
            return token_path.resolve()
    bases = []
    resolver_base = spec.get("resolver_base")
    if resolver_base:
        bases.append(Path(resolver_base))
    for directory in search_dirs:
        bases.append(Path(directory) / "_cutout_positions.dat")
    for base in bases:
        try:
            return resolve_fits_filename(base, token)
        except Exception:
            continue
    raise FileNotFoundError("cannot resolve source FITS for %r" %
                            (spec.get("fits_file") or token))


def _specs_from_photometry(pp_root, target, photometry_file):
    """Yield (specs, photometry_files, n_rows, warnings) for PP result rows."""

    photometry_files = _photometry_files(pp_root, target, photometry_file)
    specs = []
    warnings = []
    n_rows = 0
    for path in photometry_files:
        rows = parse_photometry_file(path, require_rows=False)
        if not rows:
            warnings.append("skipping %s; no active PP rows" % path)
            continue
        n_rows += len(rows)
        for row in rows:
            ra_deg = _float_or_none(row.get("ra_deg"))
            dec_deg = _float_or_none(row.get("dec_deg"))
            if ra_deg is None or dec_deg is None:
                warnings.append(
                    "skipping row in %s; missing source RA/Dec" % path)
                continue
            astrometry_only = row_is_astrometry_only(row)
            specs.append({
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "catalog_token": row.get("catalog_token"),
                "fits_file": None,
                "resolver_base": path,
                "julian_date": _float_or_none(row.get("julian_date")),
                "mag": (None if astrometry_only else
                        _float_or_none(row.get("mag"))),
                "mag_sig": (None if astrometry_only else
                            _float_or_none(row.get("mag_sig"))),
                "astrometry_only": astrometry_only,
                "sextractor_flag": row.get("sextractor_flag"),
                "band": row.get("band"),
                "pred_minus_ra_arcsec": _float_or_none(
                    row.get("pred_minus_ra_arcsec")),
                "pred_minus_dec_arcsec": _float_or_none(
                    row.get("pred_minus_dec_arcsec")),
                "source_photometry_file": str(path),
            })
    return specs, photometry_files, n_rows, warnings


def build_pp_cutouts(pp_root, target, photometry_file=None, output_dir=None,
                     size_arcsec=DEFAULT_CUTOUT_SIZE_ARCSEC,
                     position_source=None, positions=None,
                     position_column="auto", fits_dir=None):
    """Create FITS/PNG cutouts centered on reported RA/Dec positions.

    By default each active PP photometry row is centered (the PP-measured
    source position).  When ``positions`` is supplied (a CSV path or iterable of
    mappings, e.g. an SBSAR ``click_astrometry_orbits`` export) cutouts are
    centered on those reported positions instead.  ``position_source`` is a
    provenance label that, absent an explicit ``output_dir``, selects a distinct
    ``PP_cutouts_<label>`` directory so click-canvas and final-report cutout
    sets never clobber one another.
    """

    pp_root = Path(pp_root).expanduser().resolve()
    output_dir = (default_output_dir(pp_root, position_source)
                  if output_dir is None
                  else Path(output_dir).expanduser().resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*_cutout.fits", "*_cutout.png", MANIFEST_NAME):
        for stale in output_dir.glob(pattern):
            stale.unlink()

    search_dirs = []
    if fits_dir is not None:
        search_dirs.append(Path(fits_dir).expanduser().resolve())
    search_dirs.append(pp_root)

    products = []
    manifest_rows = []
    warnings = []

    positions_path = None
    photometry_files = []
    if positions is not None:
        if isinstance(positions, (str, Path)):
            positions_path = str(Path(positions).expanduser().resolve())
        specs, _ = load_positions(positions, position_column=position_column)
        for spec in specs:
            spec.setdefault("resolver_base", positions_path)
            spec["astrometry_only"] = row_is_astrometry_only(spec)
            spec.setdefault("band", None)
            spec.setdefault("pred_minus_ra_arcsec", None)
            spec.setdefault("pred_minus_dec_arcsec", None)
            spec.setdefault("source_photometry_file", positions_path)
        n_rows = len(specs)
    else:
        specs, photometry_files, n_rows, photometry_warnings = \
            _specs_from_photometry(pp_root, target, photometry_file)
        warnings.extend(photometry_warnings)

    for spec in specs:
        ra_deg = spec["ra_deg"]
        dec_deg = spec["dec_deg"]
        try:
            source_fits = _resolve_spec_fits(spec, search_dirs)
        except Exception as exc:
            warnings.append("skipping %s; cannot resolve FITS: %s" %
                            (spec.get("catalog_token") or spec.get("fits_file"),
                             exc))
            continue
        try:
            data, cutout_wcs, source_header, info = _build_cutout(
                source_fits, ra_deg, dec_deg, size_arcsec)
        except Exception as exc:
            warnings.append("skipping %s; cannot build cutout: %s" %
                            (source_fits, exc))
            continue

        stem = _output_stem(target, spec)
        output_fits = output_dir / ("%s_cutout.fits" % stem)
        output_png = output_dir / ("%s_cutout.png" % stem)
        header = source_header.copy()
        header.update(cutout_wcs.to_header(relax=True))
        header["CUTSRC"] = (source_fits.name[:68], "source PP image")
        header["CUTSIZE"] = (float(size_arcsec), "cutout size arcsec")
        header["CUTRA"] = (ra_deg, "reported source RA deg")
        header["CUTDEC"] = (dec_deg, "reported source Dec deg")
        header["CUTIN"] = (info["inside"], "target inside source")
        header["CUTOVER"] = (info["overlap"], "cutout overlaps source")
        header["ASTONLY"] = (spec["astrometry_only"],
                             "row lacks usable photometry")
        if position_source:
            header["CUTPSRC"] = (str(position_source)[:68],
                                 "cutout position provenance")
        fits.PrimaryHDU(data=data, header=header).writeto(
            output_fits, overwrite=True, output_verify="silentfix")
        _zscale_rgba(data).save(output_png)

        manifest_row = {
            "index": len(manifest_rows) + 1,
            "target": target,
            "position_source": position_source,
            "catalog_token": spec.get("catalog_token"),
            "source_photometry_file": spec.get("source_photometry_file"),
            "source_fits_file": str(source_fits),
            "output_fits": str(output_fits),
            "output_png": str(output_png),
            "relative_output_fits": _relative_to_root(output_fits, pp_root),
            "relative_output_png": _relative_to_root(output_png, pp_root),
            "julian_date": _float_or_none(spec.get("julian_date")),
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "pred_minus_ra_arcsec": _float_or_none(
                spec.get("pred_minus_ra_arcsec")),
            "pred_minus_dec_arcsec": _float_or_none(
                spec.get("pred_minus_dec_arcsec")),
            "filter": spec.get("band"),
            "mag": (None if spec["astrometry_only"] else
                    _float_or_none(spec.get("mag"))),
            "mag_sig": (None if spec["astrometry_only"] else
                        _float_or_none(spec.get("mag_sig"))),
            "astrometry_only": spec["astrometry_only"],
            "sextractor_flag": spec.get("sextractor_flag"),
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
            "astrometry_only": spec["astrometry_only"],
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
        "position_source": position_source,
        "positions_file": positions_path,
        "photometry_file": (str(photometry_files[0])
                            if photometry_files else positions_path),
        "photometry_files": [str(path) for path in photometry_files],
        "n_rows": n_rows,
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
                      size_arcsec=DEFAULT_CUTOUT_SIZE_ARCSEC,
                      position_source=None, positions=None,
                      position_column="auto", fits_dir=None):
    """Build cutouts and attach product metadata/warnings to a workflow manifest."""

    try:
        result = build_pp_cutouts(
            pp_root, target, photometry_file=photometry_file,
            output_dir=output_dir, size_arcsec=size_arcsec,
            position_source=position_source, positions=positions,
            position_column=position_column, fits_dir=fits_dir)
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
    parser.add_argument("--position-source",
                        help="provenance label; selects a distinct "
                             "PP_cutouts_<label> directory when --output-dir "
                             "is not given (e.g. ephemeris, sbsar_click)")
    parser.add_argument("--positions",
                        help="CSV of reported positions to center on instead "
                             "of PP rows (accepts SBSAR click/centroid exports)")
    parser.add_argument("--position-column", default="auto",
                        choices=sorted(POSITION_COLUMN_PAIRS) + ["auto"],
                        help="which RA/Dec columns to use from --positions")
    parser.add_argument("--fits-dir",
                        help="directory to search for source FITS when "
                             "--positions rows lack an explicit fits_file")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = build_pp_cutouts(
        args.pp_root, args.target, photometry_file=args.photometry_file,
        output_dir=args.output_dir, size_arcsec=args.size_arcsec,
        position_source=args.position_source, positions=args.positions,
        position_column=args.position_column, fits_dir=args.fits_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
