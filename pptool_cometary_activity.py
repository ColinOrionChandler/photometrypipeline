#!/usr/bin/env python3

"""Measure cometary morphology in PP images and prepare MPC form values.

The utility is deliberately review-only: it writes plots, measurements, and a
JSON object matching the MPC cometary-activity form, but it contains no network
submission path.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Iterable
import warnings

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy import ndimage, optimize
from scipy.spatial import cKDTree


FORM_FIELDS = (
    "desig", "observer", "email", "obsdate", "obstime", "obscode",
    "aperture", "is_comet", "coma_appearance", "coma_diam",
    "coma_diam_units", "tail_appearance", "tail_length",
    "tail_length_units", "tail_pa", "tail_pa2", "additional_comments",
)
DEFAULT_OUTPUT_PREFIX = "cometary_activity"
DEFAULT_STAMP_SIZE = 201
DEFAULT_N_PSF_STARS = 20
DEFAULT_BLOCK_GAP_MIN = 30.0


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("cannot JSON-encode %r" % type(value))


def robust_location_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0, 1.0
    location = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - location)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return location, scale


def finite_float(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def frame_id(path_or_name):
    match = re.search(r"\.(\d{4})\.", Path(path_or_name).name)
    return match.group(1) if match else Path(path_or_name).stem


def target_token(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")


def circular_delta_deg(first, second):
    result = (np.asarray(first, dtype=float) - np.asarray(second, dtype=float) +
              180.0) % 360.0 - 180.0
    return float(result) if result.ndim == 0 else result


def circular_weighted_mean_deg(angles, weights=None):
    angles = np.asarray(angles, dtype=float)
    if weights is None:
        weights = np.ones_like(angles)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(angles) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return None
    radians = np.deg2rad(angles[good])
    vector = np.sum(weights[good] * np.exp(1j * radians))
    if abs(vector) == 0:
        return None
    return float(np.rad2deg(np.angle(vector)) % 360.0)


@dataclass
class Position:
    filename: str
    jd: float
    ra_deg: float
    dec_deg: float
    source: str = "unknown"
    selected: bool = False
    reason: str = ""


@dataclass
class CatalogSource:
    x: float
    y: float
    ra_deg: float
    dec_deg: float
    flux: float
    flux_max: float
    fwhm_px: float
    a_px: float
    b_px: float
    theta_deg: float
    flags: int


@dataclass
class FrameData:
    position: Position
    path: Path
    header: fits.Header
    data: np.ndarray
    wcs: WCS
    pixel_scale_arcsec: float
    exptime: float
    catalog: list[CatalogSource]
    psf: np.ndarray | None = None
    psf_stars: list[np.ndarray] = field(default_factory=list)
    psf_fwhm_arcsec: float | None = None
    pretrail_psf: np.ndarray | None = None
    pretrail_psf_stars: list[np.ndarray] = field(default_factory=list)
    psf_fit: dict | None = None
    motion_east_arcsec_hour: float = 0.0
    motion_north_arcsec_hour: float = 0.0
    motion_speed_arcsec_hour: float = 0.0
    motion_pa_deg: float = 0.0
    trail_arcsec: float = 0.0
    block_index: int = -1


def load_positions(path):
    path = Path(path).expanduser().resolve()
    positions = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                filename = (row.get("filename") or row.get("fits_file") or
                            row.get("image"))
                jd = finite_float(row.get("julian_date") or row.get("jd"))
                ra = finite_float(row.get("ra_deg") or row.get("source_ra") or
                                  row.get("centroid_ra_deg"))
                dec = finite_float(row.get("dec_deg") or
                                   row.get("source_dec") or
                                   row.get("centroid_dec_deg"))
                if filename and jd is not None and ra is not None and dec is not None:
                    positions.append(Position(
                        filename=Path(filename).name, jd=jd, ra_deg=ra,
                        dec_deg=dec,
                        source=str(row.get("position_source", "csv")),
                        selected=truthy(row.get("selected_detection", False)),
                    ))
    else:
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = line.split()
            if len(values) < 4:
                continue
            positions.append(Position(
                filename=Path(values[0]).name, ra_deg=float(values[1]),
                dec_deg=float(values[2]), jd=float(values[3]), source="table",
            ))
    if not positions:
        raise ValueError("no usable positions in %s" % path)
    return positions


def apply_selection(positions, path):
    if path is None:
        return positions, {}
    path = Path(path).expanduser().resolve()
    selection = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = Path(row.get("filename") or row.get("frame") or "").name
            if not key:
                continue
            selection[key] = {
                "selected": truthy(row.get("included_selected_orbit", False)),
                "reason": str(row.get("reason", "")),
            }
    for position in positions:
        record = selection.get(position.filename)
        if record is None:
            record = selection.get(frame_id(position.filename))
        if record:
            position.selected = record["selected"]
            position.reason = record["reason"]
    return positions, selection


def load_catalog(path):
    path = Path(path)
    if not path.is_file():
        return []
    sources = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            values = {
                "x": finite_float(row.get("XWIN_IMAGE")),
                "y": finite_float(row.get("YWIN_IMAGE")),
                "ra": finite_float(row.get("ra_deg") or
                                   row.get("ra_deg_sextractor")),
                "dec": finite_float(row.get("dec_deg") or
                                    row.get("dec_deg_sextractor")),
                "flux": finite_float(row.get("FLUX_AUTO")),
                "flux_max": finite_float(row.get("FLUX_MAX")),
                "fwhm": finite_float(row.get("FWHM_IMAGE")),
                "a": finite_float(row.get("A_IMAGE")),
                "b": finite_float(row.get("B_IMAGE")),
                "theta": finite_float(row.get("THETA_IMAGE"), 0.0),
            }
            if any(values[key] is None for key in
                   ("x", "y", "ra", "dec", "flux", "flux_max", "fwhm",
                    "a", "b")):
                continue
            try:
                flags = int(float(row.get("FLAGS", 0)))
            except (TypeError, ValueError):
                flags = 99
            sources.append(CatalogSource(
                x=values["x"] - 1.0, y=values["y"] - 1.0,
                ra_deg=values["ra"], dec_deg=values["dec"],
                flux=values["flux"], flux_max=values["flux_max"],
                fwhm_px=values["fwhm"], a_px=values["a"],
                b_px=values["b"], theta_deg=values["theta"], flags=flags,
            ))
    return sources


def catalog_path_for_fits(path):
    path = Path(path)
    candidates = [
        path.with_suffix(".ldac.csv"),
        path.parent / (path.stem + ".ldac.csv"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def wcs_pixel_scale_arcsec(wcs):
    scales = [abs(value.to_value("deg"))
              for value in wcs.proj_plane_pixel_scales()]
    return float(np.mean(scales) * 3600.0)


def load_frames(pp_root, positions):
    pp_root = Path(pp_root).expanduser().resolve()
    frames = []
    for position in positions:
        path = pp_root / position.filename
        if not path.is_file():
            raise FileNotFoundError("missing source FITS: %s" % path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            header = fits.getheader(path)
            data = np.asarray(fits.getdata(path), dtype=np.float32)
            wcs = WCS(header, relax=True)
        frames.append(FrameData(
            position=position, path=path, header=header, data=data, wcs=wcs,
            pixel_scale_arcsec=wcs_pixel_scale_arcsec(wcs),
            exptime=float(header.get("EXPTIME", 1.0)),
            catalog=load_catalog(catalog_path_for_fits(path)),
        ))
    return frames


def group_frames(frames, gap_minutes=DEFAULT_BLOCK_GAP_MIN):
    frames = sorted(frames, key=lambda item: item.position.jd)
    blocks = []
    for frame in frames:
        if (not blocks or
                (frame.position.jd - blocks[-1][-1].position.jd) * 1440.0 >
                gap_minutes):
            blocks.append([])
        frame.block_index = len(blocks) - 1
        blocks[-1].append(frame)
    return blocks


def ephemeris_mapping(path):
    return {row.filename: row for row in load_positions(path)}


def assign_motion(blocks, ephemeris):
    for block in blocks:
        rows = []
        for frame in block:
            row = ephemeris.get(frame.position.filename, frame.position)
            rows.append(row)
        for index, frame in enumerate(block):
            if len(block) == 1:
                first = last = 0
            elif index == 0:
                first, last = 0, 1
            elif index == len(block) - 1:
                first, last = len(block) - 2, len(block) - 1
            else:
                first, last = index - 1, index + 1
            if first == last:
                east_rate = north_rate = 0.0
            else:
                dt_hour = (rows[last].jd - rows[first].jd) * 24.0
                dec = rows[index].dec_deg
                east_rate = ((rows[last].ra_deg - rows[first].ra_deg) *
                             math.cos(math.radians(dec)) * 3600.0 / dt_hour)
                north_rate = ((rows[last].dec_deg - rows[first].dec_deg) *
                              3600.0 / dt_hour)
            frame.motion_east_arcsec_hour = float(east_rate)
            frame.motion_north_arcsec_hour = float(north_rate)
            frame.motion_speed_arcsec_hour = float(
                math.hypot(east_rate, north_rate))
            frame.motion_pa_deg = float(
                math.degrees(math.atan2(east_rate, north_rate)) % 360.0)
            frame.trail_arcsec = float(
                frame.motion_speed_arcsec_hour * frame.exptime / 3600.0)


def canonical_wcs(ra_deg, dec_deg, size, pixel_scale_arcsec):
    wcs = WCS(naxis=2)
    center = (float(size) + 1.0) / 2.0
    wcs.wcs.crpix = [center, center]
    wcs.wcs.cdelt = [-pixel_scale_arcsec / 3600.0,
                     pixel_scale_arcsec / 3600.0]
    wcs.wcs.crval = [float(ra_deg), float(dec_deg)]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def resample_to_wcs(data, source_wcs, output_wcs, shape, order=1):
    yy, xx = np.indices(shape, dtype=float)
    world = output_wcs.all_pix2world(
        np.column_stack((xx.ravel(), yy.ravel())), 0)
    source = source_wcs.all_world2pix(world, 0)
    values = ndimage.map_coordinates(
        np.asarray(data, dtype=float),
        [source[:, 1], source[:, 0]], order=order,
        mode="constant", cval=np.nan, prefilter=False)
    return values.reshape(shape)


def annulus_mask(shape, r_inner, r_outer, center=None):
    yy, xx = np.indices(shape, dtype=float)
    if center is None:
        center = ((shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0)
    radius = np.hypot(xx - center[0], yy - center[1])
    return (radius >= r_inner) & (radius < r_outer)


def subtract_local_sky(stamp):
    half = min(stamp.shape) / 2.0
    mask = annulus_mask(stamp.shape, 0.72 * half, 0.95 * half)
    sky, sigma = robust_location_scale(stamp[mask])
    return np.asarray(stamp, dtype=float) - sky, sky, sigma


def choose_psf_sources(frame, n_stars=DEFAULT_N_PSF_STARS):
    sources = [source for source in frame.catalog
               if source.flags == 0 and source.flux > 0 and source.fwhm_px > 0
               and source.b_px > 0 and source.a_px / source.b_px < 1.4]
    if len(sources) < 5:
        return []
    fwhm = np.asarray([source.fwhm_px for source in sources])
    flux_max = np.asarray([source.flux_max for source in sources])
    flux = np.asarray([source.flux for source in sources])
    fwhm_med, fwhm_sig = robust_location_scale(fwhm)
    fwhm_sig = max(fwhm_sig, 0.15 * fwhm_med)
    low_flux, high_flux = np.nanpercentile(flux, [35, 97])
    high_peak = np.nanpercentile(flux_max, 97)
    target_x, target_y = frame.wcs.all_world2pix(
        [[frame.position.ra_deg, frame.position.dec_deg]], 0)[0]
    all_xy = np.asarray([(source.x, source.y) for source in frame.catalog])
    tree = cKDTree(all_xy) if len(all_xy) else None
    chosen = []
    for source in sources:
        if not (abs(source.fwhm_px - fwhm_med) <= 3.0 * fwhm_sig):
            continue
        if not (low_flux <= source.flux <= high_flux and
                source.flux_max <= high_peak):
            continue
        if min(source.x, source.y,
               frame.data.shape[1] - source.x,
               frame.data.shape[0] - source.y) < 70:
            continue
        if math.hypot(source.x - target_x, source.y - target_y) < 60:
            continue
        if tree is not None:
            distances, _ = tree.query([source.x, source.y], k=min(2, len(all_xy)))
            neighbor = distances[-1] if np.ndim(distances) else np.inf
            if neighbor < max(3.0 * fwhm_med, 18.0):
                continue
        chosen.append(source)
    chosen.sort(key=lambda source: (
        math.hypot(source.x - target_x, source.y - target_y),
        -source.flux))
    if len(chosen) < 5:
        relaxed = [source for source in sources
                   if min(source.x, source.y,
                          frame.data.shape[1] - source.x,
                          frame.data.shape[0] - source.y) >= 70
                   and math.hypot(source.x - target_x,
                                  source.y - target_y) >= 50]
        relaxed.sort(key=lambda source: -source.flux)
        chosen = relaxed
    return chosen[:n_stars]


def build_empirical_psf(frame, output_scale_arcsec, size,
                        n_stars=DEFAULT_N_PSF_STARS):
    stars = choose_psf_sources(frame, n_stars=n_stars)
    if len(stars) < 5:
        raise ValueError("only %d usable PSF stars" % len(stars))
    stamps = []
    yy, xx = np.indices((size, size), dtype=float)
    radius = np.hypot(xx - (size - 1) / 2.0, yy - (size - 1) / 2.0)
    normalization_radius = min(0.28 * size, 5.0 / output_scale_arcsec)
    for source in stars:
        output_wcs = canonical_wcs(
            source.ra_deg, source.dec_deg, size, output_scale_arcsec)
        stamp = resample_to_wcs(
            frame.data, frame.wcs, output_wcs, (size, size), order=1)
        stamp, _, _ = subtract_local_sky(stamp)
        flux = np.nansum(stamp[radius <= normalization_radius])
        if np.isfinite(flux) and flux > 0 and np.isfinite(stamp).mean() > 0.95:
            stamps.append(stamp / flux)
    if len(stamps) < 5:
        raise ValueError("only %d valid normalized PSF stamps" % len(stamps))
    array = np.asarray(stamps)
    psf = np.nanmedian(array, axis=0)
    psf = np.where(np.isfinite(psf), psf, 0.0)
    psf = np.maximum(psf, 0.0)
    total = psf.sum()
    if total <= 0:
        raise ValueError("empirical PSF has non-positive flux")
    psf /= total
    frame.psf = psf
    frame.psf_stars = [np.maximum(stamp, 0.0) /
                       max(np.nansum(np.maximum(stamp, 0.0)), 1e-12)
                       for stamp in stamps]
    source_fwhm = np.median([source.fwhm_px for source in stars])
    frame.psf_fwhm_arcsec = float(source_fwhm * frame.pixel_scale_arcsec)
    return psf, frame.psf_stars


def gaussian_match_sigma(source_fwhm_arcsec, common_fwhm_arcsec,
                         pixel_scale_arcsec):
    source_sigma = max(float(source_fwhm_arcsec), 0.0) / 2.354820045
    common_sigma = max(float(common_fwhm_arcsec), source_sigma) / 2.354820045
    return math.sqrt(max(common_sigma ** 2 - source_sigma ** 2, 0.0)) / \
        pixel_scale_arcsec


def apply_motion_trail(psf, east_rate, north_rate, exptime,
                       pixel_scale_arcsec, samples=None):
    trail_east = float(east_rate) * float(exptime) / 3600.0
    trail_north = float(north_rate) * float(exptime) / 3600.0
    dx = -trail_east / pixel_scale_arcsec
    dy = trail_north / pixel_scale_arcsec
    length = math.hypot(dx, dy)
    if samples is None:
        samples = max(3, int(math.ceil(length * 3.0)) + 1)
    positions = np.linspace(-0.5, 0.5, samples)
    trailed = np.zeros_like(psf, dtype=float)
    for value in positions:
        trailed += ndimage.shift(psf, (value * dy, value * dx), order=1,
                                 mode="constant", cval=0.0,
                                 prefilter=False)
    trailed /= float(samples)
    total = np.nansum(trailed)
    return trailed / total if total > 0 else trailed


def elliptical_moffat_psf(shape, fwhm_major_px, axis_ratio, beta,
                          pa_deg=0.0, dx_px=0.0, dy_px=0.0):
    """Return a normalized elliptical Moffat PSF in a north-up stamp."""
    if np.isscalar(shape):
        shape = (int(shape), int(shape))
    yy, xx = np.indices(shape, dtype=float)
    center_x = (shape[1] - 1) / 2.0 + float(dx_px)
    center_y = (shape[0] - 1) / 2.0 + float(dy_px)
    delta_x = xx - center_x
    delta_y = yy - center_y
    radians = math.radians(float(pa_deg))
    major_x = -math.sin(radians)
    major_y = math.cos(radians)
    along = delta_x * major_x + delta_y * major_y
    across = -delta_x * major_y + delta_y * major_x
    beta = max(float(beta), 1.01)
    major_alpha = float(fwhm_major_px) / (
        2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    minor_alpha = major_alpha * float(axis_ratio)
    radius2 = (along / max(major_alpha, 1e-6)) ** 2 + \
        (across / max(minor_alpha, 1e-6)) ** 2
    model = (1.0 + radius2) ** (-beta)
    total = float(np.sum(model))
    return model / total if total > 0 else model


def apply_line_trail(psf, length_px, pa_deg, samples=None):
    """Convolve a north-up PSF with a centered line at celestial PA."""
    length_px = max(float(length_px), 0.0)
    if length_px <= 1e-6:
        return np.asarray(psf, dtype=float).copy()
    radians = math.radians(float(pa_deg))
    dx = -length_px * math.sin(radians)
    dy = length_px * math.cos(radians)
    if samples is None:
        samples = max(3, int(math.ceil(length_px * 3.0)) + 1)
    result = np.zeros_like(psf, dtype=float)
    for fraction in np.linspace(-0.5, 0.5, int(samples)):
        result += ndimage.shift(
            psf, (fraction * dy, fraction * dx), order=1,
            mode="constant", cval=0.0, prefilter=False)
    result /= float(samples)
    total = float(np.sum(result))
    return result / total if total > 0 else result


def _central_psf_crop(image, fit_size):
    fit_size = min(int(fit_size), image.shape[0], image.shape[1])
    if fit_size % 2 == 0:
        fit_size -= 1
    center_y = (image.shape[0] - 1) // 2
    center_x = (image.shape[1] - 1) // 2
    half = fit_size // 2
    return np.asarray(
        image[center_y - half:center_y + half + 1,
              center_x - half:center_x + half + 1], dtype=float)


def _initial_pretrail_parameters(observed, trail_length_px, trail_pa_deg):
    yy, xx = np.indices(observed.shape, dtype=float)
    center = (observed.shape[0] - 1) / 2.0
    radius = np.hypot(xx - center, yy - center)
    outer = radius >= 0.82 * radius.max()
    background = float(np.nanmedian(observed[outer]))
    weight = np.maximum(np.where(np.isfinite(observed),
                                 observed - background, 0.0), 0.0)
    weight *= radius <= 0.65 * radius.max()
    if weight.sum() <= 0:
        return np.asarray([8.0, 0.9, 3.5, 0.0, 0.0, 0.0])
    weight /= weight.sum()
    mean_x = float(np.sum(weight * xx))
    mean_y = float(np.sum(weight * yy))
    delta_x = xx - mean_x
    delta_y = yy - mean_y
    covariance = np.asarray([
        [np.sum(weight * delta_x ** 2),
         np.sum(weight * delta_x * delta_y)],
        [np.sum(weight * delta_x * delta_y),
         np.sum(weight * delta_y ** 2)],
    ])
    radians = math.radians(float(trail_pa_deg))
    direction = np.asarray([-math.sin(radians), math.cos(radians)])
    covariance -= (float(trail_length_px) ** 2 / 12.0) * \
        np.outer(direction, direction)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 0.5)
    major = int(np.argmax(values))
    major_vector = vectors[:, major]
    major_sigma = math.sqrt(float(values[major]))
    minor_sigma = math.sqrt(float(values[1 - major]))
    pa = math.degrees(math.atan2(-major_vector[0], major_vector[1])) % 180.0
    return np.asarray([
        np.clip(2.354820045 * major_sigma, 1.5, 25.0),
        np.clip(minor_sigma / major_sigma, 0.35, 1.0),
        3.5,
        pa,
        np.clip(mean_x - center, -2.0, 2.0),
        np.clip(mean_y - center, -2.0, 2.0),
    ])


def fit_pretrail_moffat(observed_psf, pixel_scale_arcsec,
                        trail_length_arcsec, trail_pa_deg,
                        fit_radius_arcsec=7.0, initial=None,
                        max_nfev=300):
    """Fit the untrailed PSF whose known line convolution matches stars."""
    fit_size = max(
        31, 2 * int(math.ceil(fit_radius_arcsec /
                              float(pixel_scale_arcsec))) + 1)
    observed = _central_psf_crop(observed_psf, fit_size)
    yy, xx = np.indices(observed.shape, dtype=float)
    center = (observed.shape[0] - 1) / 2.0
    radius = np.hypot(xx - center, yy - center)
    fit_mask = radius <= fit_radius_arcsec / float(pixel_scale_arcsec)
    background_mask = (radius >= 0.78 * radius.max()) & np.isfinite(observed)
    background = float(np.nanmedian(observed[background_mask]))
    data = np.maximum(np.where(np.isfinite(observed),
                               observed - background, 0.0), 0.0)
    normalization = float(np.sum(data[fit_mask]))
    if normalization <= 0:
        raise ValueError("trailed stellar PSF has non-positive fitted flux")
    data /= normalization
    trail_length_px = float(trail_length_arcsec) / float(pixel_scale_arcsec)
    x0 = (_initial_pretrail_parameters(
        data, trail_length_px, trail_pa_deg)
          if initial is None else np.asarray(initial, dtype=float))
    lower = np.asarray([1.0, 0.30, 1.25, -180.0, -3.0, -3.0])
    upper = np.asarray([40.0, 1.00, 12.0, 360.0, 3.0, 3.0])
    x0 = np.minimum(np.maximum(x0, lower + 1e-6), upper - 1e-6)
    floor = max(float(np.nanmax(data)) * 0.02, 1e-12)

    def model_for(parameters):
        base = elliptical_moffat_psf(
            data.shape, parameters[0], parameters[1], parameters[2],
            parameters[3], parameters[4], parameters[5])
        model = apply_line_trail(
            base, trail_length_px, trail_pa_deg)
        model /= max(float(np.sum(model[fit_mask])), 1e-12)
        return model

    def residual(parameters):
        model = model_for(parameters)
        weight = 1.0 / np.sqrt(np.maximum(data, floor))
        return ((model - data) * weight)[fit_mask]

    result = optimize.least_squares(
        residual, x0=x0, bounds=(lower, upper), loss="soft_l1",
        x_scale="jac", max_nfev=int(max_nfev))
    model = model_for(result.x)
    peak = max(float(np.nanmax(data[fit_mask])), 1e-12)
    rms_fraction = float(np.sqrt(
        np.mean((model[fit_mask] - data[fit_mask]) ** 2)) / peak)
    parameters = result.x
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "rms_peak_fraction": rms_fraction,
        "fwhm_major_px": float(parameters[0]),
        "fwhm_minor_px": float(parameters[0] * parameters[1]),
        "axis_ratio": float(parameters[1]),
        "beta": float(parameters[2]),
        "pa_deg": float(parameters[3] % 180.0),
        "dx_px": float(parameters[4]),
        "dy_px": float(parameters[5]),
        "trail_length_arcsec": float(trail_length_arcsec),
        "trail_pa_deg": float(trail_pa_deg % 180.0),
        "fit_data": data,
        "fit_model": model,
        "fit_mask": fit_mask,
        "parameters": parameters,
    }


def build_pretrail_psf(frame, output_scale_arcsec, bootstrap=8):
    """Recover the target-tracked PSF from field stars trailed by tracking."""
    fit = fit_pretrail_moffat(
        frame.psf, output_scale_arcsec, frame.trail_arcsec,
        frame.motion_pa_deg)
    if not fit["success"] or fit["rms_peak_fraction"] > 0.25:
        raise ValueError(
            "pretrail PSF fit failed QA: success=%s RMS/peak=%.3f" %
            (fit["success"], fit["rms_peak_fraction"]))
    frame.pretrail_psf = elliptical_moffat_psf(
        frame.psf.shape, fit["fwhm_major_px"], fit["axis_ratio"],
        fit["beta"], fit["pa_deg"], fit["dx_px"], fit["dy_px"])
    bootstrap_models = []
    rng = np.random.default_rng(
        int(re.sub(r"\D", "", frame_id(frame.path)) or "0") + 20260716)
    stars = np.asarray(frame.psf_stars)
    for _ in range(max(int(bootstrap), 0)):
        indices = rng.integers(0, len(stars), len(stars))
        sample = np.nanmedian(stars[indices], axis=0)
        sample = np.maximum(np.where(np.isfinite(sample), sample, 0.0), 0.0)
        total = float(np.sum(sample))
        if total <= 0:
            continue
        sample /= total
        try:
            sample_fit = fit_pretrail_moffat(
                sample, output_scale_arcsec, frame.trail_arcsec,
                frame.motion_pa_deg, initial=fit["parameters"],
                max_nfev=120)
        except ValueError:
            continue
        if sample_fit["success"] and sample_fit["rms_peak_fraction"] <= 0.35:
            bootstrap_models.append(elliptical_moffat_psf(
                frame.psf.shape, sample_fit["fwhm_major_px"],
                sample_fit["axis_ratio"], sample_fit["beta"],
                sample_fit["pa_deg"], sample_fit["dx_px"],
                sample_fit["dy_px"]))
    if len(bootstrap_models) < 3:
        bootstrap_models = [frame.pretrail_psf.copy() for _ in range(3)]
    frame.pretrail_psf_stars = bootstrap_models
    frame.psf_fit = {
        key: value for key, value in fit.items()
        if key not in {"fit_data", "fit_model", "fit_mask", "parameters"}
    }
    frame.psf_fit.update({
        "fwhm_major_arcsec": fit["fwhm_major_px"] * output_scale_arcsec,
        "fwhm_minor_arcsec": fit["fwhm_minor_px"] * output_scale_arcsec,
        "bootstrap_models": len(bootstrap_models),
        "tracking_mode": "target",
        "_fit_data": fit["fit_data"],
        "_fit_model": fit["fit_model"],
        "_fit_mask": fit["fit_mask"],
    })
    return frame.pretrail_psf, frame.pretrail_psf_stars, fit


def matched_catalog_scale(science, reference):
    left = [source for source in science.catalog
            if source.flags == 0 and source.flux > 0]
    right = [source for source in reference.catalog
             if source.flags == 0 and source.flux > 0]
    if len(left) < 5 or len(right) < 5:
        return science.exptime / max(reference.exptime, 1e-9), 0
    dec0 = np.median([source.dec_deg for source in left])
    cosdec = math.cos(math.radians(dec0))
    right_xy = np.asarray([(source.ra_deg * cosdec * 3600.0,
                            source.dec_deg * 3600.0) for source in right])
    tree = cKDTree(right_xy)
    ratios = []
    for source in left:
        point = (source.ra_deg * cosdec * 3600.0,
                 source.dec_deg * 3600.0)
        distance, index = tree.query(point)
        if distance <= 1.0:
            ratios.append(source.flux / right[index].flux)
    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) < 5:
        return science.exptime / max(reference.exptime, 1e-9), len(ratios)
    median, sigma = robust_location_scale(np.log(ratios))
    keep = np.abs(np.log(ratios) - median) <= max(3.0 * sigma, 0.05)
    return float(np.median(ratios[keep])), int(np.sum(keep))


def target_source_info(frame, output_wcs, common_fwhm_px):
    center = np.asarray([(output_wcs.array_shape[1] - 1) / 2.0,
                         (output_wcs.array_shape[0] - 1) / 2.0])
    source_rows = []
    for source in frame.catalog:
        x, y = output_wcs.all_world2pix(
            [[source.ra_deg, source.dec_deg]], 0)[0]
        separation = float(math.hypot(x - center[0], y - center[1]))
        source_rows.append((separation, source, float(x), float(y)))
    source_rows.sort(key=lambda item: item[0])
    nearest = source_rows[0] if source_rows else None
    central = [item for item in source_rows
               if item[0] <= 2.0 * common_fwhm_px]
    expected_target = bool(nearest and nearest[0] <= common_fwhm_px)
    other_count = len(central) - (1 if expected_target else 0)
    flag = nearest[1].flags if expected_target else None
    return {
        "nearest_source_sep_px": nearest[0] if nearest else None,
        "nearest_source_flag": flag,
        "central_catalog_sources": len(central),
        "other_central_sources": max(other_count, 0),
        "blended": bool((flag is not None and flag > 1) or other_count > 0),
        "sources": source_rows,
    }


def source_mask_for_stamp(frame, output_wcs, shape, common_fwhm_px):
    yy, xx = np.indices(shape, dtype=float)
    center_x = (shape[1] - 1) / 2.0
    center_y = (shape[0] - 1) / 2.0
    mask = np.zeros(shape, dtype=bool)
    for source in frame.catalog:
        x, y = output_wcs.all_world2pix(
            [[source.ra_deg, source.dec_deg]], 0)[0]
        if x < -10 or y < -10 or x > shape[1] + 10 or y > shape[0] + 10:
            continue
        separation = math.hypot(x - center_x, y - center_y)
        if separation <= 2.0 * common_fwhm_px:
            continue
        source_radius = max(1.8 * common_fwhm_px,
                            3.0 * source.a_px *
                            frame.pixel_scale_arcsec /
                            wcs_pixel_scale_arcsec(output_wcs))
        mask |= np.hypot(xx - x, yy - y) <= source_radius
    return mask


def median_template(stamps, masks=None, minimum=3):
    if not stamps:
        raise ValueError("no template stamps")
    array = np.asarray(stamps, dtype=float)
    if masks is not None:
        masks = np.asarray(masks, dtype=bool)
        array = np.where(masks, np.nan, array)
    valid = np.sum(np.isfinite(array), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmedian(array, axis=0)
    result[valid < minimum] = np.nan
    return result, valid


def prepare_frame_stamp(frame, block, output_scale_arcsec, size,
                        common_fwhm_arcsec, ephemeris):
    shape = (size, size)
    output_wcs = canonical_wcs(frame.position.ra_deg,
                               frame.position.dec_deg,
                               size, output_scale_arcsec)
    output_wcs.array_shape = shape
    science = resample_to_wcs(frame.data, frame.wcs, output_wcs, shape)
    science, _, direct_sigma = subtract_local_sky(science)
    match_sigma = gaussian_match_sigma(
        frame.psf_fwhm_arcsec, common_fwhm_arcsec, output_scale_arcsec)
    science_matched = ndimage.gaussian_filter(
        science, match_sigma, mode="constant", cval=np.nan) \
        if match_sigma > 0.05 else science.copy()

    template_stamps = []
    template_masks = []
    scales = []
    for neighbor in block:
        if neighbor is frame:
            continue
        stamp = resample_to_wcs(
            neighbor.data, neighbor.wcs, output_wcs, shape)
        stamp, _, _ = subtract_local_sky(stamp)
        scale, n_matches = matched_catalog_scale(frame, neighbor)
        scales.append({"frame": frame_id(neighbor.path), "scale": scale,
                       "matches": n_matches})
        sigma = gaussian_match_sigma(
            neighbor.psf_fwhm_arcsec, common_fwhm_arcsec,
            output_scale_arcsec)
        if sigma > 0.05:
            stamp = ndimage.gaussian_filter(
                stamp, sigma, mode="constant", cval=np.nan)
        stamp *= scale
        target = ephemeris.get(neighbor.position.filename,
                               neighbor.position)
        target_x, target_y = output_wcs.all_world2pix(
            [[target.ra_deg, target.dec_deg]], 0)[0]
        yy, xx = np.indices(shape, dtype=float)
        radius = max(1.0 * common_fwhm_arcsec / output_scale_arcsec,
                     0.75 * neighbor.trail_arcsec / output_scale_arcsec,
                     8.0)
        target_mask = np.hypot(xx - target_x, yy - target_y) <= radius
        template_stamps.append(stamp)
        template_masks.append(target_mask)

    minimum = 2
    template = np.zeros(shape, dtype=float)
    valid = np.zeros(shape, dtype=int)
    subtraction_used = False
    subtraction_reason = "fewer than three neighboring frames"
    if len(template_stamps) >= 3:
        template, valid = median_template(
            template_stamps, template_masks, minimum=minimum)
        cleaned = science_matched - template
        outer = annulus_mask(shape, 0.55 * size / 2.0, 0.92 * size / 2.0)
        central = annulus_mask(
            shape, 0.0,
            2.0 * common_fwhm_arcsec / output_scale_arcsec)
        _, residual_sigma = robust_location_scale(cleaned[outer])
        coverage = float(np.isfinite(template).mean())
        central_coverage = float(np.isfinite(template[central]).mean())
        ratio = residual_sigma / max(direct_sigma, 1e-12)
        if coverage >= 0.70 and central_coverage >= 0.95 and ratio <= 3.0:
            subtraction_used = True
            subtraction_reason = "passed"
        else:
            subtraction_reason = (
                "failed QA: coverage=%.3f central=%.3f "
                "residual/direct RMS=%.2f" %
                (coverage, central_coverage, ratio))
    if subtraction_used:
        working = science_matched - np.where(np.isfinite(template),
                                              template, 0.0)
    else:
        working = science_matched
    working, _, working_sigma = subtract_local_sky(working)

    common_fwhm_px = common_fwhm_arcsec / output_scale_arcsec
    catalog_mask = source_mask_for_stamp(
        frame, output_wcs, shape, common_fwhm_px)
    if subtraction_used:
        catalog_mask |= ~np.isfinite(template)
    target_info = target_source_info(frame, output_wcs, common_fwhm_px)
    return {
        "frame": frame,
        "wcs": output_wcs,
        "direct": science_matched,
        "template": template,
        "working": working,
        "catalog_mask": catalog_mask,
        "target_info": target_info,
        "sky_sigma": working_sigma,
        "direct_sky_sigma": direct_sigma,
        "subtraction_used": subtraction_used,
        "subtraction_reason": subtraction_reason,
        "template_scales": scales,
        "template_valid_fraction": float(np.mean(valid >= minimum))
        if len(template_stamps) >= 3 else 0.0,
        "common_fwhm_arcsec": common_fwhm_arcsec,
        "common_fwhm_px": common_fwhm_px,
        "psf_match_sigma_px": match_sigma,
    }


def radial_profile(image, mask, bin_width_px=2.0, max_radius_px=None):
    image = np.asarray(image, dtype=float)
    yy, xx = np.indices(image.shape, dtype=float)
    center_x = (image.shape[1] - 1) / 2.0
    center_y = (image.shape[0] - 1) / 2.0
    radius = np.hypot(xx - center_x, yy - center_y)
    if max_radius_px is None:
        max_radius_px = 0.46 * min(image.shape)
    edges = np.arange(0.0, max_radius_px + bin_width_px, bin_width_px)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full(len(centers), np.nan)
    scatter = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=int)
    base_mask = np.asarray(mask, dtype=bool) | ~np.isfinite(image)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        use = (~base_mask) & (radius >= lower) & (radius < upper)
        data = image[use]
        if data.size:
            values[index] = float(np.mean(data))
            _, scatter[index] = robust_location_scale(data)
            counts[index] = data.size
    return centers, values, scatter, counts, edges


def best_psf_registration(data, psf, mask, search_px=4):
    yy, xx = np.indices(data.shape, dtype=float)
    radius = np.hypot(xx - (data.shape[1] - 1) / 2.0,
                      yy - (data.shape[0] - 1) / 2.0)
    core = radius <= max(6.0, min(data.shape) * 0.08)
    valid = core & ~mask & np.isfinite(data)
    best = None
    for dy in range(-int(search_px), int(search_px) + 1):
        for dx in range(-int(search_px), int(search_px) + 1):
            shifted = ndimage.shift(psf, (dy, dx), order=1,
                                    mode="constant", cval=0.0,
                                    prefilter=False)
            denominator = float(np.sum(shifted[valid] ** 2))
            if denominator <= 0:
                continue
            amplitude = max(float(np.sum(data[valid] * shifted[valid]) /
                                  denominator), 0.0)
            residual = data[valid] - amplitude * shifted[valid]
            score = float(np.sum(residual ** 2))
            if best is None or score < best[0]:
                best = (score, dx, dy, amplitude, shifted)
    if best is None:
        return 0.0, 0.0, 0.0, psf.copy()
    return float(best[1]), float(best[2]), float(best[3]), best[4]


def contiguous_extent(significance, radii, threshold, minimum_bins,
                      start_radius, max_start_radius=None):
    good = (np.asarray(significance) >= threshold) & \
        (np.asarray(radii) >= start_radius) & np.isfinite(significance)
    runs = []
    start = None
    for index, value in enumerate(good):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(good) - 1):
            end = index if value and index == len(good) - 1 else index - 1
            if end - start + 1 >= minimum_bins:
                if (max_start_radius is None or
                        radii[start] <= max_start_radius):
                    runs.append((start, end))
            start = None
    if not runs:
        return None
    runs.sort(key=lambda run: (run[0], -(run[1] - run[0])))
    return runs[0]


def sky_geometry(wcs, shape):
    yy, xx = np.indices(shape, dtype=float)
    center = np.asarray([(shape[1] - 1) / 2.0,
                         (shape[0] - 1) / 2.0])
    world = wcs.all_pix2world(
        np.column_stack((xx.ravel(), yy.ravel())), 0)
    center_world = wcs.all_pix2world([center], 0)[0]
    east = circular_delta_deg(world[:, 0], center_world[0]) * \
        math.cos(math.radians(center_world[1])) * 3600.0
    north = (world[:, 1] - center_world[1]) * 3600.0
    radius = np.hypot(east, north).reshape(shape)
    pa = (np.degrees(np.arctan2(east, north)) % 360.0).reshape(shape)
    return radius, pa


def measure_tail(residual, mask, wcs, sky_sigma, core_radius_arcsec,
                 max_radius_arcsec=None, angle_bin_deg=10.0,
                 radial_bin_arcsec=0.75):
    residual = np.asarray(residual, dtype=float)
    radius, pa = sky_geometry(wcs, residual.shape)
    if max_radius_arcsec is None:
        max_radius_arcsec = float(np.nanpercentile(radius, 45))
    angle_edges = np.arange(0.0, 360.0 + angle_bin_deg, angle_bin_deg)
    angle_centers = (angle_edges[:-1] + angle_edges[1:]) / 2.0
    flux = np.full(len(angle_centers), np.nan)
    noise = np.full(len(angle_centers), np.nan)
    valid_base = (~mask) & np.isfinite(residual) & \
        (radius >= core_radius_arcsec) & (radius <= max_radius_arcsec)
    for index, (lower, upper) in enumerate(zip(angle_edges[:-1],
                                                angle_edges[1:])):
        use = valid_base & (pa >= lower) & (pa < upper)
        count = int(np.sum(use))
        if count:
            flux[index] = float(np.sum(residual[use]))
            noise[index] = float(sky_sigma * math.sqrt(count))
    baseline = float(np.nanmedian(flux)) if np.any(np.isfinite(flux)) else 0.0
    zscore = (flux - baseline) / np.where(noise > 0, noise, np.nan)
    if not np.any(np.isfinite(zscore)):
        return {"status": "unmeasurable", "appearance": "no tail",
                "pa_deg": None, "length_arcsec": None,
                "angle_centers_deg": angle_centers,
                "angle_significance": zscore}
    peak_index = int(np.nanargmax(zscore))
    peak_pa = float(angle_centers[peak_index])
    peak_z = float(zscore[peak_index])
    positive_peak = max(float(flux[peak_index] - baseline), 0.0)
    half_level = baseline + 0.5 * positive_peak
    broad_bins = [peak_index]
    for sign in (-1, 1):
        for offset in range(1, len(angle_centers) // 2 + 1):
            index = (peak_index + sign * offset) % len(angle_centers)
            if np.isfinite(flux[index]) and flux[index] >= half_level:
                broad_bins.append(index)
            else:
                break
    broad_bins = sorted(set(broad_bins))
    weights = np.maximum(flux[broad_bins] - baseline, 0.0)
    mean_pa = circular_weighted_mean_deg(angle_centers[broad_bins], weights)
    if mean_pa is None:
        mean_pa = peak_pa
    angular_offsets = np.asarray([
        abs(circular_delta_deg(angle_centers[index], mean_pa))
        for index in broad_bins])
    width = float(max(angle_bin_deg,
                      2.0 * (np.nanmax(angular_offsets) +
                             angle_bin_deg / 2.0)))
    half_width = max(angle_bin_deg, width / 2.0)

    radial_edges = np.arange(core_radius_arcsec,
                             max_radius_arcsec + radial_bin_arcsec,
                             radial_bin_arcsec)
    radial_centers = (radial_edges[:-1] + radial_edges[1:]) / 2.0
    radial_z = np.full(len(radial_centers), np.nan)
    wedge = np.abs(((pa - mean_pa + 180.0) % 360.0) - 180.0) <= half_width
    control = np.abs(((pa - (mean_pa + 180.0) + 180.0) % 360.0) -
                     180.0) <= half_width
    for index, (lower, upper) in enumerate(zip(radial_edges[:-1],
                                                radial_edges[1:])):
        use = (~mask) & np.isfinite(residual) & wedge & \
            (radius >= lower) & (radius < upper)
        opposite = (~mask) & np.isfinite(residual) & control & \
            (radius >= lower) & (radius < upper)
        if np.sum(use) >= 3:
            signal = float(np.mean(residual[use]))
            if np.sum(opposite) >= 3:
                signal -= float(np.mean(residual[opposite]))
            radial_z[index] = signal / max(
                sky_sigma / math.sqrt(np.sum(use)), 1e-12)
    significant_run = contiguous_extent(
        radial_z, radial_centers, 3.0, 2, core_radius_arcsec,
        max_start_radius=core_radius_arcsec + 3.0 * radial_bin_arcsec)
    marginal_run = contiguous_extent(
        radial_z, radial_centers, 1.0, 2, core_radius_arcsec,
        max_start_radius=core_radius_arcsec + 4.0 * radial_bin_arcsec)
    if width >= 140.0:
        status = "not_detected"
        last = None
    elif significant_run is not None and peak_z >= 5.0:
        status = "significant"
        last = significant_run[1]
    elif marginal_run is not None:
        status = "marginal"
        last = marginal_run[1]
    else:
        status = "not_detected"
        last = None
    length = float(radial_edges[last + 1]) if last is not None else None

    appearance = "no tail"
    if length is not None:
        appearance = "broad" if width >= 35.0 else "straight"
        halfway = core_radius_arcsec + 0.5 * (
            max(length, core_radius_arcsec) - core_radius_arcsec)
        inner = valid_base & (radius <= halfway)
        outer = valid_base & (radius > halfway) & (radius <= length)
        half_pas = []
        half_significance = []
        for region in (inner, outer):
            scores = []
            score_noise = []
            for lower, upper in zip(angle_edges[:-1], angle_edges[1:]):
                use = region & (pa >= lower) & (pa < upper)
                scores.append(float(np.sum(residual[use])) if np.any(use)
                              else np.nan)
                score_noise.append(float(sky_sigma * math.sqrt(np.sum(use)))
                                   if np.any(use) else np.nan)
            if np.any(np.isfinite(scores)):
                half_baseline = float(np.nanmedian(scores))
                index = int(np.nanargmax(scores))
                half_pas.append(float(angle_centers[index]))
                half_significance.append(
                    (scores[index] - half_baseline) /
                    max(score_noise[index], 1e-12))
        bend = (abs(circular_delta_deg(half_pas[0], half_pas[1]))
                if len(half_pas) == 2 else 0.0)
        if len(half_pas) == 2 and 20.0 <= bend <= 120.0 and \
                min(half_significance) >= 3.0 and \
                length >= 2.0 * core_radius_arcsec:
            appearance = "curved"
    return {
        "status": status,
        "appearance": appearance,
        "pa_deg": float(mean_pa) if length is not None else None,
        "pa_unc_deg": float(max(angle_bin_deg / 2.0, width / 4.0))
        if length is not None else None,
        "pa_lower_deg": float((mean_pa - width / 2.0) % 360.0)
        if appearance == "broad" else None,
        "pa_upper_deg": float((mean_pa + width / 2.0) % 360.0)
        if appearance == "broad" else None,
        "angular_width_deg": width if length is not None else None,
        "length_arcsec": length,
        "length_unc_arcsec": radial_bin_arcsec if length is not None else None,
        "peak_significance": peak_z,
        "angle_centers_deg": angle_centers,
        "angle_significance": zscore,
        "radial_centers_arcsec": radial_centers,
        "radial_significance": radial_z,
    }


def measure_stamp(stamp_record, psf, psf_stars, output_scale_arcsec,
                  motion_pa_deg, trail_arcsec, max_radius_arcsec=12.0,
                  motion_alignment_is_artifact=True):
    data = np.asarray(stamp_record["working"], dtype=float)
    mask = np.asarray(stamp_record["catalog_mask"], dtype=bool) | \
        ~np.isfinite(data)
    dx, dy, amplitude, fitted_psf = best_psf_registration(
        data, psf, mask, search_px=3)
    model = amplitude * fitted_psf
    residual = data - model
    bin_width_px = max(2.0, 0.5 / output_scale_arcsec)
    radii, target_profile, target_scatter, counts, edges = radial_profile(
        data / max(amplitude, 1e-12), mask,
        bin_width_px=bin_width_px,
        max_radius_px=max_radius_arcsec / output_scale_arcsec)
    star_profiles = []
    for star in psf_stars:
        _, profile, _, _, _ = radial_profile(
            star, np.zeros_like(mask), bin_width_px=bin_width_px,
            max_radius_px=edges[-1])
        if len(profile) < len(target_profile):
            profile = np.pad(profile, (0, len(target_profile) - len(profile)),
                             constant_values=np.nan)
        star_profiles.append(profile[:len(target_profile)])
    star_profiles = np.asarray(star_profiles)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        psf_profile = np.nanmedian(star_profiles, axis=0)
        psf_low = np.nanpercentile(star_profiles, 16, axis=0)
        psf_high = np.nanpercentile(star_profiles, 84, axis=0)
    mean_noise = (stamp_record["sky_sigma"] / max(amplitude, 1e-12)) / \
        np.sqrt(np.maximum(counts, 1))
    psf_noise = 0.5 * (psf_high - psf_low)
    total_noise = np.sqrt(mean_noise ** 2 + psf_noise ** 2)
    significance = (target_profile - psf_profile) / \
        np.where(total_noise > 0, total_noise, np.nan)
    start_radius_px = max(
        0.75 * stamp_record["common_fwhm_px"],
        trail_arcsec / output_scale_arcsec)
    extent = contiguous_extent(
        significance, radii, threshold=3.0, minimum_bins=2,
        start_radius=start_radius_px,
        max_start_radius=2.0 * stamp_record["common_fwhm_px"])
    coma_detected = extent is not None
    if coma_detected:
        outer_px = edges[extent[1] + 1]
        coma_diameter = float(2.0 * outer_px * output_scale_arcsec)
        bootstrap_diameters = []
        for profile in star_profiles:
            trial_sig = (target_profile - profile) / \
                np.where(total_noise > 0, total_noise, np.nan)
            trial_extent = contiguous_extent(
                trial_sig, radii, 3.0, 2, start_radius_px,
                max_start_radius=2.0 * stamp_record["common_fwhm_px"])
            bootstrap_diameters.append(
                2.0 * edges[trial_extent[1] + 1] * output_scale_arcsec
                if trial_extent else 0.0)
        low_diameter, high_diameter = np.nanpercentile(
            bootstrap_diameters, [16, 84])
        coma_unc = max(0.5 * (high_diameter - low_diameter),
                       2.0 * bin_width_px * output_scale_arcsec)
    else:
        coma_diameter = None
        coma_unc = None
        low_diameter = None
        high_diameter = None
    coma_upper_limit = float(2.0 * max(
        stamp_record["common_fwhm_arcsec"], trail_arcsec,
        bin_width_px * output_scale_arcsec))

    profile_max_sig = float(np.nanmax(significance[
        radii >= start_radius_px])) if np.any(
            np.isfinite(significance[radii >= start_radius_px])) else None
    yy, xx = np.indices(data.shape, dtype=float)
    pixel_radius = np.hypot(xx - (data.shape[1] - 1) / 2.0,
                            yy - (data.shape[0] - 1) / 2.0)
    core_radius = stamp_record["common_fwhm_px"]
    outer_radius = ((coma_diameter / 2.0 / output_scale_arcsec)
                    if coma_diameter else 2.0 * core_radius)
    positive = np.maximum(data, 0.0)
    core_flux = float(np.nansum(positive[(pixel_radius <= core_radius) &
                                         ~mask]))
    total_flux = float(np.nansum(positive[(pixel_radius <= outer_radius) &
                                          ~mask]))
    concentration = core_flux / total_flux if total_flux > 0 else None
    if not coma_detected:
        coma_appearance = "no coma"
    elif concentration is not None and concentration >= 0.70:
        coma_appearance = "very condensed"
    elif concentration is not None and concentration >= 0.50:
        coma_appearance = "condensed"
    elif concentration is not None and concentration >= 0.30:
        coma_appearance = "diffuse"
    else:
        coma_appearance = "very diffuse"

    tail = measure_tail(
        residual, mask, stamp_record["wcs"], stamp_record["sky_sigma"],
        core_radius_arcsec=max(stamp_record["common_fwhm_arcsec"],
                              trail_arcsec),
        max_radius_arcsec=max_radius_arcsec)
    tail["raw_status"] = tail["status"]
    if tail.get("pa_deg") is not None:
        tail["motion_alignment_deg"] = abs(circular_delta_deg(
            tail["pa_deg"], motion_pa_deg))
        trail_artifact_limit = (stamp_record["common_fwhm_arcsec"] +
                                2.0 * trail_arcsec)
        tail["motion_artifact_likely"] = bool(
            motion_alignment_is_artifact and
            tail["motion_alignment_deg"] <= 25.0 and
            tail.get("length_arcsec") is not None and
            tail["length_arcsec"] <= trail_artifact_limit)
        if tail["motion_artifact_likely"] and tail["status"] == "significant":
            tail["status"] = "marginal"
        if tail["appearance"] == "curved" and \
                tail.get("length_arcsec", 0.0) < \
                2.0 * stamp_record["common_fwhm_arcsec"]:
            tail["appearance"] = "straight"
    else:
        tail["motion_alignment_deg"] = None
        tail["motion_artifact_likely"] = False
    return {
        "centroid_dx_px": dx,
        "centroid_dy_px": dy,
        "psf_amplitude": amplitude,
        "centroid_at_search_limit": bool(abs(dx) >= 3 or abs(dy) >= 3),
        "model": model,
        "residual": residual,
        "radii_px": radii,
        "radii_arcsec": radii * output_scale_arcsec,
        "target_profile": target_profile,
        "psf_profile": psf_profile,
        "psf_profile_low": psf_low,
        "psf_profile_high": psf_high,
        "profile_significance": significance,
        "coma_detected": coma_detected,
        "coma_appearance": coma_appearance,
        "coma_diameter_arcsec": coma_diameter,
        "coma_diameter_unc_arcsec": coma_unc,
        "coma_diameter_low_arcsec": low_diameter,
        "coma_diameter_high_arcsec": high_diameter,
        "coma_upper_limit_arcsec": coma_upper_limit,
        "coma_peak_significance": profile_max_sig,
        "concentration": concentration,
        "tail": tail,
    }


def make_trailed_psf_products(frame, common_fwhm_arcsec,
                              output_scale_arcsec,
                              psf_mode="empirical-motion"):
    sigma = gaussian_match_sigma(
        frame.psf_fwhm_arcsec, common_fwhm_arcsec, output_scale_arcsec)
    if psf_mode == "pretrail-target-tracked":
        if frame.pretrail_psf is None or not frame.pretrail_psf_stars:
            raise ValueError("pretrail PSF products were not fitted")
        source_psf = frame.pretrail_psf
        source_stars = frame.pretrail_psf_stars
        target_east_rate = 0.0
        target_north_rate = 0.0
    else:
        source_psf = frame.psf
        source_stars = frame.psf_stars
        target_east_rate = frame.motion_east_arcsec_hour
        target_north_rate = frame.motion_north_arcsec_hour
    base = (ndimage.gaussian_filter(source_psf, sigma, mode="constant",
                                    cval=0.0)
            if sigma > 0.05 else source_psf.copy())
    trailed = apply_motion_trail(
        base, target_east_rate, target_north_rate, frame.exptime,
        output_scale_arcsec)
    stars = []
    for star in source_stars:
        matched = (ndimage.gaussian_filter(star, sigma, mode="constant",
                                           cval=0.0)
                   if sigma > 0.05 else star.copy())
        stars.append(apply_motion_trail(
            matched, target_east_rate, target_north_rate, frame.exptime,
            output_scale_arcsec))
    return trailed, stars


def measurement_public(record):
    measurement = record["measurement"]
    frame = record["frame"]
    tail = measurement["tail"]
    return {
        "frame": frame_id(frame.path),
        "filename": frame.path.name,
        "jd": frame.position.jd,
        "selected": frame.position.selected,
        "selection_reason": frame.position.reason,
        "block": frame.block_index + 1,
        "exptime_s": frame.exptime,
        "pixel_scale_arcsec": record["output_scale_arcsec"],
        "stellar_fwhm_arcsec": frame.psf_fwhm_arcsec,
        "common_fwhm_arcsec": record["stamp"]["common_fwhm_arcsec"],
        "motion_speed_arcsec_hour": frame.motion_speed_arcsec_hour,
        "motion_pa_deg": frame.motion_pa_deg,
        "trail_arcsec": frame.trail_arcsec,
        "target_psf_trail_arcsec": record.get(
            "target_psf_trail_arcsec", frame.trail_arcsec),
        "psf_star_count": len(frame.psf_stars),
        "psf_mode": record.get("psf_mode", "empirical-motion"),
        "pretrail_psf_fwhm_major_arcsec": (
            frame.psf_fit.get("fwhm_major_arcsec")
            if frame.psf_fit else None),
        "pretrail_psf_fwhm_minor_arcsec": (
            frame.psf_fit.get("fwhm_minor_arcsec")
            if frame.psf_fit else None),
        "pretrail_psf_pa_deg": (
            frame.psf_fit.get("pa_deg") if frame.psf_fit else None),
        "pretrail_psf_beta": (
            frame.psf_fit.get("beta") if frame.psf_fit else None),
        "pretrail_psf_fit_rms_peak_fraction": (
            frame.psf_fit.get("rms_peak_fraction")
            if frame.psf_fit else None),
        "sky_subtraction_used": record["stamp"]["subtraction_used"],
        "sky_subtraction_reason": record["stamp"]["subtraction_reason"],
        "template_valid_fraction": record["stamp"][
            "template_valid_fraction"],
        "blended": record["stamp"]["target_info"]["blended"],
        "nearest_source_flag": record["stamp"]["target_info"][
            "nearest_source_flag"],
        "coma_detected": measurement["coma_detected"],
        "coma_appearance": measurement["coma_appearance"],
        "coma_diameter_arcsec": measurement["coma_diameter_arcsec"],
        "coma_diameter_unc_arcsec": measurement["coma_diameter_unc_arcsec"],
        "coma_upper_limit_arcsec": measurement["coma_upper_limit_arcsec"],
        "coma_peak_significance": measurement["coma_peak_significance"],
        "tail_status": tail["status"],
        "tail_appearance": tail["appearance"],
        "tail_pa_deg": tail.get("pa_deg"),
        "tail_pa_unc_deg": tail.get("pa_unc_deg"),
        "tail_length_arcsec": tail.get("length_arcsec"),
        "tail_length_unc_arcsec": tail.get("length_unc_arcsec"),
        "tail_peak_significance": tail.get("peak_significance"),
        "tail_motion_alignment_deg": tail.get("motion_alignment_deg"),
        "tail_motion_artifact_likely": tail.get(
            "motion_artifact_likely", False),
    }


def render_diagnostic(record, output_path, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stamp = record["stamp"]
    measurement = record["measurement"]
    direct = stamp["direct"]
    template = stamp["template"]
    working = stamp["working"]
    residual = measurement["residual"]
    sigma = max(stamp["sky_sigma"], 1e-12)
    pretrail_mode = record.get("psf_mode") == "pretrail-target-tracked"
    residual_label = (
        "Pretrail-PSF residual" if pretrail_mode
        else "Trailed-PSF residual")
    profile_label = (
        "target-tracked pretrail PSF" if pretrail_mode
        else "trailed stellar PSF")
    extent = (np.array([-1, 1, -1, 1]) *
              (direct.shape[0] - 1) / 2.0 *
              record["output_scale_arcsec"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        (axes[0, 0], direct, "Direct / common PSF"),
        (axes[0, 1], template, "Stationary-sky template"),
        (axes[0, 2], working, "Sky-cleaned target"),
        (axes[1, 0], residual, residual_label),
    ]
    for axis, image, label in panels:
        finite = image[np.isfinite(image)]
        if finite.size:
            location, spread = robust_location_scale(finite)
            vmax = max(float(np.nanpercentile(finite, 99.5)),
                       location + 6.0 * spread)
            vmin = location - 2.0 * spread
        else:
            vmin, vmax = -1, 1
        axis.imshow(image, origin="lower", cmap="gray_r", vmin=vmin,
                    vmax=vmax, extent=extent)
        if label in {"Sky-cleaned target", residual_label}:
            levels = sigma * np.asarray([3.0, 5.0, 10.0])
            if np.nanmax(image) > levels[0]:
                axis.contour(image, levels=levels, colors=["cyan", "lime",
                                                           "yellow"],
                             linewidths=0.8, origin="lower", extent=extent)
        axis.axhline(0, color="tab:red", alpha=0.35, linewidth=0.6)
        axis.axvline(0, color="tab:red", alpha=0.35, linewidth=0.6)
        axis.set_title(label)
        axis.set_xlabel("East-left offset (arcsec)")
        axis.set_ylabel("North offset (arcsec)")
    radii = measurement["radii_arcsec"]
    axes[1, 1].plot(radii, measurement["target_profile"],
                    label="target", color="black")
    axes[1, 1].plot(radii, measurement["psf_profile"],
                    label=profile_label, color="tab:blue")
    axes[1, 1].fill_between(
        radii, measurement["psf_profile_low"],
        measurement["psf_profile_high"], color="tab:blue", alpha=0.2,
        label="stellar 16-84%")
    axes[1, 1].set_yscale("symlog", linthresh=1e-5)
    axes[1, 1].set_xlabel("Radius (arcsec)")
    axes[1, 1].set_ylabel("Normalized surface brightness")
    axes[1, 1].set_title("Radial profile")
    axes[1, 1].legend(fontsize=8)
    tail = measurement["tail"]
    axes[1, 2].plot(tail.get("angle_centers_deg", []),
                    tail.get("angle_significance", []), color="tab:purple")
    axes[1, 2].axhline(3.0, color="tab:red", linestyle="--", linewidth=0.8)
    if tail.get("pa_deg") is not None:
        axes[1, 2].axvline(tail["pa_deg"], color="black", linestyle=":",
                           label="feature PA")
    axes[1, 2].axvline(record["motion_pa_deg"], color="tab:orange",
                       linestyle="--", label="motion PA")
    if tail.get("expected_pa_deg") is not None:
        axes[1, 2].axvline(
            tail["expected_pa_deg"], color="tab:green", linestyle="-.",
            label=tail.get("expected_pa_label") or "expected tail PA")
    axes[1, 2].set_xlim(0, 360)
    axes[1, 2].set_xlabel("Position angle east of north (deg)")
    axes[1, 2].set_ylabel("Directional significance")
    axes[1, 2].set_title("Tail search")
    axes[1, 2].legend(fontsize=8)
    if title is None:
        title = "Cometary activity diagnostic"
    summary = (
        "%s | coma %s | tail %s | sky subtraction %s" %
        (title,
         ("%.1f±%.1f arcsec" %
          (measurement["coma_diameter_arcsec"],
           measurement["coma_diameter_unc_arcsec"]))
         if measurement["coma_detected"] else
         "not resolved (<%.1f arcsec)" %
         measurement["coma_upper_limit_arcsec"],
         ("%s %.1f arcsec PA %.0f" %
          (tail["status"], tail["length_arcsec"], tail["pa_deg"]))
         if tail.get("length_arcsec") is not None else tail["status"],
         "used" if stamp["subtraction_used"] else "fallback"))
    fig.suptitle(summary, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def render_psf_fit_diagnostic(frame, output_path, output_scale_arcsec):
    """Render the fitted star trail and recovered pretrail target PSF."""
    if not frame.psf_fit:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = frame.psf_fit["_fit_data"]
    model = frame.psf_fit["_fit_model"]
    pretrail = _central_psf_crop(frame.pretrail_psf, data.shape[0])
    pretrail /= max(float(np.sum(pretrail)), 1e-12)
    residual = data - model
    half = (data.shape[0] - 1) / 2.0 * output_scale_arcsec
    extent = [-half, half, -half, half]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [
        (data, "Empirical trailed stars"),
        (model, "Moffat + known tracking trail"),
        (pretrail, "Recovered pretrail PSF"),
        (residual, "Fit residual"),
    ]
    for axis, (image, title) in zip(axes, panels):
        if title == "Fit residual":
            limit = max(float(np.nanpercentile(np.abs(image), 99)), 1e-12)
            axis.imshow(image, origin="lower", cmap="coolwarm",
                        vmin=-limit, vmax=limit, extent=extent)
        else:
            vmax = max(float(np.nanpercentile(image, 99.8)), 1e-12)
            axis.imshow(image, origin="lower", cmap="gray_r",
                        vmin=0.0, vmax=vmax, extent=extent)
        axis.set_title(title)
        axis.set_xlabel("East-left offset (arcsec)")
        axis.set_ylabel("North offset (arcsec)")
    fig.suptitle(
        ("Frame %s: pretrail FWHM %.2f x %.2f arcsec, PA %.1f deg; "
         "stellar trail %.2f arcsec at axis PA %.1f deg; RMS/peak %.3f") %
        (frame_id(frame.path),
         frame.psf_fit["fwhm_major_arcsec"],
         frame.psf_fit["fwhm_minor_arcsec"],
         frame.psf_fit["pa_deg"],
         frame.psf_fit["trail_length_arcsec"],
         frame.psf_fit["trail_pa_deg"],
         frame.psf_fit["rms_peak_fraction"]),
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def annotate_expected_tail(measurement, expected_pa_deg=None,
                           expected_pa_unc_deg=None, label=None):
    tail = measurement["tail"]
    tail["expected_pa_deg"] = (
        float(expected_pa_deg) % 360.0
        if expected_pa_deg is not None else None)
    tail["expected_pa_unc_deg"] = (
        float(expected_pa_unc_deg)
        if expected_pa_unc_deg is not None else None)
    tail["expected_pa_label"] = label
    if tail.get("pa_deg") is None or expected_pa_deg is None:
        tail["expected_pa_separation_deg"] = None
        tail["physical_direction_consistent"] = False
        return
    separation = abs(circular_delta_deg(tail["pa_deg"], expected_pa_deg))
    tail["expected_pa_separation_deg"] = float(separation)
    tolerance = max(
        10.0, 3.0 * float(expected_pa_unc_deg or 0.0) +
        float(tail.get("pa_unc_deg") or 0.0))
    tail["physical_direction_consistent"] = bool(separation <= tolerance)
    if tail["physical_direction_consistent"]:
        tail["motion_artifact_likely"] = False


def combine_primary(records, primary_ids, output_scale_arcsec,
                    motion_alignment_is_artifact=True):
    chosen = [record for record in records
              if frame_id(record["frame"].path) in primary_ids]
    if not chosen:
        raise ValueError("no primary frames selected")
    shape = chosen[0]["stamp"]["working"].shape
    numerator = np.zeros(shape, dtype=float)
    denominator = np.zeros(shape, dtype=float)
    psf_sum = np.zeros(shape, dtype=float)
    psf_weight = 0.0
    star_profiles = []
    for record in chosen:
        frame = record["frame"]
        image = record["stamp"]["working"] / max(frame.exptime, 1e-12)
        sigma = record["stamp"]["sky_sigma"] / max(frame.exptime, 1e-12)
        weight = 1.0 / max(sigma ** 2, 1e-24)
        valid = np.isfinite(image) & ~record["stamp"]["catalog_mask"]
        numerator[valid] += image[valid] * weight
        denominator[valid] += weight
        psf_sum += record["trailed_psf"] * weight
        psf_weight += weight
        star_profiles.extend(record["trailed_psf_stars"])
    stack = np.full(shape, np.nan)
    good = denominator > 0
    stack[good] = numerator[good] / denominator[good]
    psf = psf_sum / psf_weight
    psf /= np.nansum(psf)
    combined_mask = denominator < 0.25 * np.nanmax(denominator)
    midpoint_jd = float(np.average(
        [record["frame"].position.jd for record in chosen],
        weights=[record["frame"].exptime for record in chosen]))
    center_ra = float(np.average(
        [record["frame"].position.ra_deg for record in chosen],
        weights=[record["frame"].exptime for record in chosen]))
    center_dec = float(np.average(
        [record["frame"].position.dec_deg for record in chosen],
        weights=[record["frame"].exptime for record in chosen]))
    output_wcs = canonical_wcs(center_ra, center_dec, shape[0],
                               output_scale_arcsec)
    output_wcs.array_shape = shape
    _, stack_sigma = robust_location_scale(stack[~combined_mask])
    common_fwhm = max(record["stamp"]["common_fwhm_arcsec"]
                      for record in chosen)
    motion_pa = circular_weighted_mean_deg(
        [record["frame"].motion_pa_deg for record in chosen],
        [record["frame"].exptime for record in chosen])
    predicted_trail = float(np.average(
        [record["frame"].trail_arcsec for record in chosen],
        weights=[record["frame"].exptime for record in chosen]))
    target_trail = float(np.average(
        [record.get("target_psf_trail_arcsec",
                    record["frame"].trail_arcsec) for record in chosen],
        weights=[record["frame"].exptime for record in chosen]))
    stamp_record = {
        "working": stack,
        "direct": stack,
        "template": np.zeros_like(stack),
        "catalog_mask": combined_mask,
        "wcs": output_wcs,
        "sky_sigma": stack_sigma,
        "common_fwhm_arcsec": common_fwhm,
        "common_fwhm_px": common_fwhm / output_scale_arcsec,
        "subtraction_used": all(record["stamp"]["subtraction_used"]
                                for record in chosen),
        "subtraction_reason": "combined primary-frame products",
        "target_info": {"blended": any(record["stamp"]["target_info"][
            "blended"] for record in chosen)},
    }
    measurement = measure_stamp(
        stamp_record, psf, star_profiles, output_scale_arcsec,
        motion_pa or 0.0, target_trail,
        motion_alignment_is_artifact=motion_alignment_is_artifact)
    return {
        "frame_ids": [frame_id(record["frame"].path) for record in chosen],
        "records": chosen,
        "stamp": stamp_record,
        "measurement": measurement,
        "psf": psf,
        "midpoint_jd": midpoint_jd,
        "center_ra_deg": center_ra,
        "center_dec_deg": center_dec,
        "total_exptime_s": float(sum(record["frame"].exptime
                                      for record in chosen)),
        "motion_pa_deg": motion_pa or 0.0,
        "trail_arcsec": predicted_trail,
        "target_psf_trail_arcsec": target_trail,
        "output_scale_arcsec": output_scale_arcsec,
    }


def weighted_midpoint_iso(records):
    jd = np.average([record["frame"].position.jd for record in records],
                    weights=[record["frame"].exptime for record in records])
    return Time(float(jd), format="jd", scale="utc").isot


def folder_date_warning(pp_root, observation_date):
    """Return a warning when a YYYYMMDDUT path token disagrees with FITS time."""
    matches = re.findall(r"(?<!\d)(\d{8})UT(?!\d)", str(pp_root))
    header_date = str(observation_date).replace("-", "")
    if matches and any(folder_date != header_date for folder_date in matches):
        return (
            "Directory name contains %sUT, but FITS/JD timing gives %s UTC; "
            "the payload uses the FITS/JD date." %
            (matches[-1], observation_date))
    return None


def build_comments(primary, frame_public, max_length=500):
    measurement = primary["measurement"]
    tail = measurement["tail"]
    frames = ",".join(primary["frame_ids"])
    if measurement["coma_detected"]:
        coma_text = "coma %.1f+/-%.1f arcsec" % (
            measurement["coma_diameter_arcsec"],
            measurement["coma_diameter_unc_arcsec"])
    else:
        coma_text = "coma unresolved; 3-sigma limit %.1f arcsec" % \
            measurement["coma_upper_limit_arcsec"]
    if tail.get("length_arcsec") is not None:
        tail_text = "%s tail candidate %.1f arcsec at PA %.0f deg (%s)" % (
            tail["appearance"], tail["length_arcsec"], tail["pa_deg"],
            tail["status"])
        if tail.get("physical_direction_consistent"):
            tail_text += (
                ", consistent with %s PA %.1f+/-%.1f deg" %
                (tail.get("expected_pa_label") or "expected direction",
                 tail["expected_pa_deg"],
                 tail.get("expected_pa_unc_deg") or 0.0))
    else:
        tail_text = "no measurable tail"
    support = sum(1 for row in frame_public
                  if row["selected"] and row["coma_detected"])
    psf_text = (
        "Target-tracked pretrail PSF fitted from motion-trailed stars"
        if primary.get("target_psf_trail_arcsec", 1.0) <= 1e-6 else
        "Motion-trailed stellar PSF used")
    text = ("Target-aligned stack of frames %s: %s; %s. %s; "
            "sky templates applied where QA passed; individual "
            "coma support %d/%d selected frames. Review diagnostics before "
            "submission." %
            (frames, coma_text, tail_text, psf_text, support,
             sum(1 for row in frame_public if row["selected"])))
    return text[:max_length]


def make_form_payload(primary, frame_public, designation, observer, email,
                      observatory_code, aperture_m):
    measurement = primary["measurement"]
    tail = measurement["tail"]
    isot = Time(primary["midpoint_jd"], format="jd", scale="utc").isot
    date, time = isot.split("T")
    time = Time(primary["midpoint_jd"], format="jd", scale="utc").to_datetime(
        timezone=timezone.utc).strftime("%H:%M:%S")
    support = sum(1 for row in frame_public
                  if row["selected"] and row["coma_detected"])
    selected_rows = [row for row in frame_public if row["selected"]]
    reliable_fraction = (sum(not row["blended"] for row in selected_rows) /
                         max(len(selected_rows), 1))
    fractional_uncertainty = (
        measurement["coma_diameter_unc_arcsec"] /
        max(measurement["coma_diameter_arcsec"], 1e-12)
        if measurement["coma_detected"] else np.inf)
    if measurement["coma_detected"] and support >= 3 and \
            (measurement["coma_peak_significance"] or 0) >= 5 and \
            fractional_uncertainty <= 0.5 and reliable_fraction >= 0.5:
        is_comet = "T"
    elif measurement["coma_detected"] or tail.get("length_arcsec") is not None:
        is_comet = "P"
    else:
        is_comet = "F"
    if measurement["coma_detected"]:
        coma_diam = "%.1f" % measurement["coma_diameter_arcsec"]
    else:
        coma_diam = ""
    if tail.get("length_arcsec") is None:
        tail_appearance = "no tail"
        tail_length = tail_pa = tail_pa2 = ""
    else:
        tail_appearance = tail["appearance"]
        tail_length = "%.1f" % tail["length_arcsec"]
        if tail_appearance == "broad":
            tail_pa = "%.0f" % tail["pa_lower_deg"]
            tail_pa2 = "%.0f" % tail["pa_upper_deg"]
        else:
            tail_pa = "%.0f" % tail["pa_deg"]
            tail_pa2 = ""
    payload = {
        "desig": str(designation),
        "observer": str(observer),
        "email": str(email),
        "obsdate": date,
        "obstime": time,
        "obscode": str(observatory_code),
        "aperture": "%g" % float(aperture_m),
        "is_comet": is_comet,
        "coma_appearance": measurement["coma_appearance"],
        "coma_diam": coma_diam,
        "coma_diam_units": "arc_sec",
        "tail_appearance": tail_appearance,
        "tail_length": tail_length,
        "tail_length_units": "arc_sec",
        "tail_pa": tail_pa,
        "tail_pa2": tail_pa2,
        "additional_comments": build_comments(primary, frame_public),
    }
    assert tuple(payload) == FORM_FIELDS
    assert len(payload["additional_comments"]) <= 500
    return payload


def write_primary_fits(primary, path):
    header = primary["stamp"]["wcs"].to_header(relax=True)
    header["MIDTIMJD"] = (primary["midpoint_jd"], "weighted midpoint JD UTC")
    header["EXPTIME"] = (primary["total_exptime_s"], "summed exposure, s")
    header["NCOMBINE"] = (len(primary["frame_ids"]), "number of frames")
    header["COMAFRMS"] = (",".join(primary["frame_ids"]),
                           "target-aligned source frames")
    fits.PrimaryHDU(
        data=np.asarray(primary["stamp"]["working"], dtype=np.float32),
        header=header).writeto(path, overwrite=True,
                              output_verify="silentfix")


def write_primary_png(primary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    image = primary["stamp"]["working"]
    finite = image[np.isfinite(image)]
    location, sigma = robust_location_scale(finite)
    vmax = max(float(np.nanpercentile(finite, 99.7)), location + 7 * sigma)
    extent = (np.array([-1, 1, -1, 1]) *
              (image.shape[0] - 1) / 2.0 * primary["output_scale_arcsec"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, origin="lower", cmap="gray_r",
              vmin=location - 2 * sigma, vmax=vmax, extent=extent)
    levels = sigma * np.asarray([3.0, 5.0, 10.0])
    if np.nanmax(image) > levels[0]:
        ax.contour(image, levels=levels, origin="lower", extent=extent,
                   colors=["cyan", "lime", "yellow"], linewidths=0.8)
    ax.set_xlabel("East-left offset (arcsec)")
    ax.set_ylabel("North offset (arcsec)")
    ax.set_title("RMM2026 primary target-aligned stack")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_frame_csv(rows, path):
    fields = list(rows[0]) if rows else []
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_review(path, payload, primary, frame_rows, warnings_list):
    measurement = primary["measurement"]
    tail = measurement["tail"]
    lines = [
        "# MPC cometary-activity review: %s" % payload["desig"], "",
        "This package is review-only. Nothing has been submitted to the MPC.",
        "", "## Recommended form values", "",
        "| Field | Value |", "|---|---|",
    ]
    for key in FORM_FIELDS:
        value = str(payload[key]).replace("|", "\\|")
        lines.append("| `%s` | %s |" % (key, value or "*(blank)*"))
    lines.extend(["", "## Primary evidence", "",
                  "- Frames: %s" % ", ".join(primary["frame_ids"]),
                  "- Weighted midpoint: %s UTC" %
                  Time(primary["midpoint_jd"], format="jd", scale="utc").isot,
                  "- Total exposure: %.0f s" % primary["total_exptime_s"],
                  "- Coma: %s" %
                  (("%.1f +/- %.1f arcsec (%s, peak %.1f sigma)" %
                    (measurement["coma_diameter_arcsec"],
                     measurement["coma_diameter_unc_arcsec"],
                     measurement["coma_appearance"],
                     measurement["coma_peak_significance"]))
                   if measurement["coma_detected"] else
                   "unresolved; limit %.1f arcsec" %
                   measurement["coma_upper_limit_arcsec"]),
                  "- Tail: %s" %
                  (("%s, %.1f +/- %.1f arcsec at PA %.0f +/- %.0f deg; %s" %
                    (tail["appearance"], tail["length_arcsec"],
                     tail["length_unc_arcsec"], tail["pa_deg"],
                     tail["pa_unc_deg"], tail["status"]))
                   if tail.get("length_arcsec") is not None else
                   tail["status"]), "",
                  "- Motion-artifact flag: %s (tail/motion separation %s deg)" %
                  (tail.get("motion_artifact_likely", False),
                   ("%.1f" % tail["motion_alignment_deg"]
                    if tail.get("motion_alignment_deg") is not None else
                    "n/a")),
                  "- Expected-direction agreement: %s" %
                  (("%.1f deg from %s PA %.1f +/- %.1f deg" %
                    (tail["expected_pa_separation_deg"],
                     tail.get("expected_pa_label") or "expected",
                     tail["expected_pa_deg"],
                     tail.get("expected_pa_unc_deg") or 0.0))
                   if tail.get("expected_pa_separation_deg") is not None
                   else "not supplied"), "",
                  "## PSF pre-analysis", "",
                  "- Mode: %s" %
                  (frame_rows[0].get("psf_mode", "empirical-motion")
                   if frame_rows else "unknown"),
                  ("- Telescope tracking: target; predicted field-star trail "
                   "removed before target comparison"
                   if frame_rows and frame_rows[0].get("psf_mode") ==
                   "pretrail-target-tracked" else
                   "- Telescope tracking: not modeled as target tracking"),
                  "- Median recovered pretrail FWHM: %s" %
                  (("%.2f x %.2f arcsec" %
                    (np.nanmedian([
                        row["pretrail_psf_fwhm_major_arcsec"]
                        for row in frame_rows
                        if row["pretrail_psf_fwhm_major_arcsec"] is not None]),
                     np.nanmedian([
                        row["pretrail_psf_fwhm_minor_arcsec"]
                        for row in frame_rows
                        if row["pretrail_psf_fwhm_minor_arcsec"] is not None])))
                   if any(row["pretrail_psf_fwhm_major_arcsec"] is not None
                          for row in frame_rows) else "n/a"), "",
                  "## Frame measurements", "",
                  "| Frame | Selected | Blend | Sky subtraction | Coma (arcsec) | Tail status | Tail (arcsec) | PA |",
                  "|---|---:|---:|---:|---:|---|---:|---:|",
                  ])
    for row in frame_rows:
        lines.append("| {frame} | {selected} | {blended} | {sky} | {coma} | "
                     "{tail_status} | {tail_length} | {tail_pa} |".format(
                         frame=row["frame"], selected=row["selected"],
                         blended=row["blended"],
                         sky=row["sky_subtraction_used"],
                         coma=("%.1f" % row["coma_diameter_arcsec"]
                               if row["coma_diameter_arcsec"] is not None
                               else "<%.1f" % row["coma_upper_limit_arcsec"]),
                         tail_status=row["tail_status"],
                         tail_length=("%.1f" % row["tail_length_arcsec"]
                                      if row["tail_length_arcsec"] is not None
                                      else ""),
                         tail_pa=("%.0f" % row["tail_pa_deg"]
                                  if row["tail_pa_deg"] is not None else "")))
    lines.extend(["", "## Warnings", ""])
    lines.extend("- %s" % warning for warning in warnings_list)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def serializable_measurement(measurement):
    ignored = {"model", "residual", "target_profile", "psf_profile",
               "psf_profile_low", "psf_profile_high",
               "profile_significance", "radii_px", "radii_arcsec"}
    result = {key: value for key, value in measurement.items()
              if key not in ignored}
    tail = dict(result["tail"])
    for key in ("angle_centers_deg", "angle_significance",
                "radial_centers_arcsec", "radial_significance"):
        if key in tail:
            tail[key] = np.asarray(tail[key]).tolist()
    result["tail"] = tail
    return result


def run_analysis(args):
    pp_root = Path(args.pp_root).expanduser().resolve()
    positions = load_positions(args.positions)
    positions, _ = apply_selection(positions, args.selection)
    frames = load_frames(pp_root, positions)
    blocks = group_frames(frames, gap_minutes=args.block_gap_min)
    ephemeris = ephemeris_mapping(args.ephemeris)
    assign_motion(blocks, ephemeris)
    output_scale = float(np.median([frame.pixel_scale_arcsec
                                    for frame in frames]))
    for frame in frames:
        build_empirical_psf(frame, output_scale, args.stamp_size,
                            n_stars=args.psf_stars)
        if args.psf_mode == "pretrail-target-tracked":
            build_pretrail_psf(
                frame, output_scale, bootstrap=args.psf_bootstrap)
    records = []
    for block in blocks:
        common_fwhm = max(frame.psf_fwhm_arcsec for frame in block)
        for frame in block:
            stamp = prepare_frame_stamp(
                frame, block, output_scale, args.stamp_size,
                common_fwhm, ephemeris)
            trailed_psf, trailed_stars = make_trailed_psf_products(
                frame, common_fwhm, output_scale, psf_mode=args.psf_mode)
            target_trail = (
                0.0 if args.psf_mode == "pretrail-target-tracked"
                else frame.trail_arcsec)
            measurement = measure_stamp(
                stamp, trailed_psf, trailed_stars, output_scale,
                frame.motion_pa_deg, target_trail,
                motion_alignment_is_artifact=(
                    args.psf_mode != "pretrail-target-tracked"))
            annotate_expected_tail(
                measurement, args.expected_tail_pa,
                args.expected_tail_pa_unc, args.expected_tail_label)
            records.append({
                "frame": frame, "stamp": stamp,
                "trailed_psf": trailed_psf,
                "trailed_psf_stars": trailed_stars,
                "measurement": measurement,
                "motion_pa_deg": frame.motion_pa_deg,
                "output_scale_arcsec": output_scale,
                "target_psf_trail_arcsec": target_trail,
                "psf_mode": args.psf_mode,
            })
    primary_ids = [value.strip() for value in args.primary_frames.split(",")
                   if value.strip()]
    primary = combine_primary(
        records, set(primary_ids), output_scale,
        motion_alignment_is_artifact=(
            args.psf_mode != "pretrail-target-tracked"))
    annotate_expected_tail(
        primary["measurement"], args.expected_tail_pa,
        args.expected_tail_pa_unc, args.expected_tail_label)
    frame_rows = [measurement_public(record) for record in records]
    payload = make_form_payload(
        primary, frame_rows, args.designation, args.observer, args.email,
        args.observatory_code, args.aperture_m)

    output_dir = (Path(args.output_dir).expanduser().resolve()
                  if args.output_dir else
                  pp_root / (
                      DEFAULT_OUTPUT_PREFIX + "_" +
                      target_token(args.designation) +
                      ("_pretrail_psf"
                       if args.psf_mode == "pretrail-target-tracked"
                       else "")))
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = output_dir / "diagnostics"
    diagnostic_dir.mkdir(exist_ok=True)
    psf_diagnostic_dir = output_dir / "psf_fits"
    if args.psf_mode == "pretrail-target-tracked":
        psf_diagnostic_dir.mkdir(exist_ok=True)
    for record in records:
        ident = frame_id(record["frame"].path)
        render_diagnostic(
            record, diagnostic_dir / ("frame_%s.png" % ident),
            title="Frame %s" % ident)
        if args.psf_mode == "pretrail-target-tracked":
            render_psf_fit_diagnostic(
                record["frame"],
                psf_diagnostic_dir / ("frame_%s_psf_fit.png" % ident),
                output_scale)
    render_diagnostic(
        {"stamp": primary["stamp"], "measurement": primary["measurement"],
         "output_scale_arcsec": output_scale,
         "motion_pa_deg": primary["motion_pa_deg"],
         "psf_mode": args.psf_mode},
        diagnostic_dir / "primary_stack_diagnostic.png",
        title="Primary stack %s" % ",".join(primary["frame_ids"]))
    write_primary_fits(primary, output_dir / "primary_stack.fits")
    write_primary_png(primary, output_dir / "primary_stack.png")
    write_frame_csv(frame_rows, output_dir / "frame_measurements.csv")
    (output_dir / "mpc_cometary_payload.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    warnings_list = []
    date_warning = folder_date_warning(pp_root, payload["obsdate"])
    if date_warning:
        warnings_list.append(date_warning)
    for row in frame_rows:
        if not row["sky_subtraction_used"]:
            warnings_list.append(
                "Frame %s used direct-image fallback: %s" %
                (row["frame"], next(record["stamp"]["subtraction_reason"]
                 for record in records
                 if frame_id(record["frame"].path) == row["frame"])))
    secondary = next((row for row in frame_rows if row["frame"] ==
                      args.secondary_frame), None)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "review_only": True,
        "network_submission_performed": False,
        "inputs": {
            "pp_root": str(pp_root),
            "positions": str(Path(args.positions).expanduser().resolve()),
            "ephemeris": str(Path(args.ephemeris).expanduser().resolve()),
            "selection": str(Path(args.selection).expanduser().resolve()),
            "designation": args.designation,
            "target": args.target,
            "psf_mode": args.psf_mode,
            "tracking_mode": (
                "target" if args.psf_mode == "pretrail-target-tracked"
                else "sidereal/unspecified"),
            "expected_tail_pa_deg": args.expected_tail_pa,
            "expected_tail_pa_unc_deg": args.expected_tail_pa_unc,
            "expected_tail_label": args.expected_tail_label,
        },
        "analysis": {
            "n_frames": len(records),
            "n_blocks": len(blocks),
            "output_pixel_scale_arcsec": output_scale,
            "primary_frames": primary["frame_ids"],
            "primary_midpoint_utc": Time(
                primary["midpoint_jd"], format="jd", scale="utc").isot,
            "primary_total_exptime_s": primary["total_exptime_s"],
            "primary_measurement": serializable_measurement(
                primary["measurement"]),
            "secondary_frame": secondary,
        },
        "frame_measurements": frame_rows,
        "mpc_form_payload": payload,
        "warnings": warnings_list,
        "artifacts": {
            "frame_measurements": str(output_dir /
                                      "frame_measurements.csv"),
            "primary_stack_fits": str(output_dir / "primary_stack.fits"),
            "primary_stack_png": str(output_dir / "primary_stack.png"),
            "diagnostics": str(diagnostic_dir),
            "psf_fit_diagnostics": (
                str(psf_diagnostic_dir)
                if args.psf_mode == "pretrail-target-tracked" else None),
            "payload": str(output_dir / "mpc_cometary_payload.json"),
            "review": str(output_dir / "mpc_cometary_review.md"),
        },
    }
    (output_dir / "cometary_activity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True,
                   default=json_default) + "\n", encoding="utf-8")
    write_review(output_dir / "mpc_cometary_review.md", payload, primary,
                 frame_rows, warnings_list)
    return {"output_dir": output_dir, "summary": summary,
            "payload": payload, "frame_rows": frame_rows,
            "primary": primary}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pp_root", help="PP directory containing FITS/LDAC files")
    parser.add_argument("--target", required=True,
                        help="PP target label used by local products")
    parser.add_argument("--designation", required=True,
                        help="object designation for the MPC form")
    parser.add_argument("--positions", required=True,
                        help="CSV/table with per-frame target centers")
    parser.add_argument("--ephemeris", required=True,
                        help="per-frame predicted positions used for motion")
    parser.add_argument("--selection", required=True,
                        help="CSV marking accepted detections")
    parser.add_argument("--observer", required=True,
                        help="properly formatted reporter name")
    parser.add_argument("--email", required=True)
    parser.add_argument("--observatory-code", required=True)
    parser.add_argument("--aperture-m", required=True, type=float)
    parser.add_argument("--primary-frames", default="0146,0147,0148,0149,0150,0151")
    parser.add_argument("--secondary-frame", default="0172")
    parser.add_argument("--output-dir")
    parser.add_argument("--stamp-size", type=int, default=DEFAULT_STAMP_SIZE)
    parser.add_argument("--psf-stars", type=int, default=DEFAULT_N_PSF_STARS)
    parser.add_argument(
        "--psf-mode",
        choices=("empirical-motion", "pretrail-target-tracked"),
        default="empirical-motion",
        help=("empirical-motion preserves the original sidereal-tracking "
              "assumption; pretrail-target-tracked fits an intrinsic Moffat "
              "PSF beneath the known field-star tracking trail"))
    parser.add_argument(
        "--psf-bootstrap", type=int, default=8,
        help="bootstrap pretrail PSF fits per frame")
    parser.add_argument(
        "--expected-tail-pa", type=float,
        help="optional physically expected tail PA east of north")
    parser.add_argument(
        "--expected-tail-pa-unc", type=float,
        help="uncertainty of --expected-tail-pa in degrees")
    parser.add_argument(
        "--expected-tail-label", default="expected anti-solar/anti-motion",
        help="label for the expected tail direction")
    parser.add_argument("--block-gap-min", type=float,
                        default=DEFAULT_BLOCK_GAP_MIN)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stamp_size < 81 or args.stamp_size % 2 == 0:
        raise ValueError("--stamp-size must be an odd integer >= 81")
    result = run_analysis(args)
    print("Wrote %s" % result["output_dir"])
    print("MPC payload prepared; no submission performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
