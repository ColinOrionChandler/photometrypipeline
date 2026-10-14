#!/usr/bin/env python3

"""Verify clicked ARC 3.5-m/ARCTIC pair stacks and run gated PP photometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from contextlib import contextmanager

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS


TELESCOPE_KEY = "ARC35ARCTIC"
OBSERVATORY_CODE = "705"
DEFAULT_FILTER = "CU VR"
MANIFEST_NAME = "arc35arctic_workflow_manifest.json"
VERIFICATION_JSON = "candidate_verification.json"
VERIFICATION_CSV = "candidate_verification.csv"
CHECKSUM_NAME = "original_checksums.json"


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("cannot JSON-encode %r" % type(value))


def target_token(value):
    return "_".join(str(value).strip().split())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def midpoint_jd(header):
    if header.get("MIDTIMJD") not in (None, ""):
        return float(header["MIDTIMJD"])
    start = Time(str(header["DATE-OBS"]), format="isot", scale="utc")
    return float((start + float(header.get("EXPTIME", 0.0)) * u.s / 2).jd)


def png_to_fits_pixel(row, shape):
    ny, nx = shape[-2:]
    width = float(row["width"])
    height = float(row["height"])
    x = float(row["x"]) * float(nx - 1) / float(width - 1)
    flipped = float(height - 1) - float(row["y"])
    y = flipped * float(ny - 1) / float(height - 1)
    return float(x), float(y)


def resolve_clicked_fits(click_csv, image_name):
    click_dir = Path(click_csv).expanduser().resolve().parent
    fits_name = Path(image_name).with_suffix(".fits").name
    candidates = (click_dir / fits_name, click_dir.parent / fits_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "cannot find %s beside %s or one directory above" %
        (fits_name, click_dir))


def read_click_seeds(click_csv):
    click_csv = Path(click_csv).expanduser().resolve()
    seeds = []
    with click_csv.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            if str(row.get("status", "")).strip().lower() != "clicked":
                continue
            stack_fits = resolve_clicked_fits(click_csv, row["image"])
            data = fits.getdata(stack_fits)
            header = fits.getheader(stack_fits)
            x, y = png_to_fits_pixel(row, data.shape)
            ra, dec = WCS(header, relax=True).all_pix2world([[x, y]], 0)[0]
            seeds.append({
                "index": index, "image": row["image"],
                "stack_fits": str(stack_fits), "click_x": float(row["x"]),
                "click_y": float(row["y"]), "fits_x": x, "fits_y": y,
                "ra_deg": float(ra), "dec_deg": float(dec),
            })
    if len(seeds) < 2:
        raise ValueError("need at least two clicked seed rows")
    return seeds


def input_names_from_stack(path):
    header = fits.getheader(path)
    keys = sorted(key for key in header if str(key).startswith("INFIL"))
    return [str(header[key]).strip() for key in keys if str(header[key]).strip()]


def resolve_input_name(stack_path, name):
    value = Path(name).expanduser()
    candidates = [value] if value.is_absolute() else []
    candidates.extend([Path(stack_path).parent / value.name,
                       Path(stack_path).parent.parent / value.name])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("cannot resolve stack input %s from %s" %
                            (name, stack_path))


def discover_originals(seeds):
    paths = set()
    for seed in seeds:
        stack_path = Path(seed["stack_fits"])
        names = input_names_from_stack(stack_path)
        if not names:
            raise ValueError("%s has no INFILnnn provenance" % stack_path)
        resolved = [resolve_input_name(stack_path, name) for name in names]
        seed["input_files"] = [str(path) for path in resolved]
        weights = np.asarray([
            float(fits.getheader(path).get("EXPTIME", 1.0)) for path in resolved
        ])
        times = np.asarray([midpoint_jd(fits.getheader(path)) for path in resolved])
        seed["midpoint_jd"] = float(np.average(times, weights=weights))
        paths.update(resolved)
    return sorted(paths, key=lambda path: midpoint_jd(fits.getheader(path)))


def group_adjacent(paths, pair_size):
    if pair_size <= 0:
        raise ValueError("pair size must be positive")
    if len(paths) % pair_size:
        raise ValueError("%d inputs cannot be divided into groups of %d" %
                         (len(paths), pair_size))
    return [paths[index:index + pair_size]
            for index in range(0, len(paths), pair_size)]


def write_checksums(output_root, click_csv, seeds, originals):
    paths = {Path(click_csv).resolve(), *originals}
    for seed in seeds:
        paths.add(Path(seed["stack_fits"]))
        png = Path(seed["stack_fits"]).with_suffix(".png")
        if png.exists():
            paths.add(png)
    payload = {str(path): sha256(path) for path in sorted(paths)}
    output = Path(output_root) / CHECKSUM_NAME
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output, payload


def run_stacker(stacker, originals, pair_size, pair_dir):
    pair_dir = Path(pair_dir)
    pair_dir.mkdir(parents=True, exist_ok=True)
    stacker = Path(stacker).expanduser().resolve()
    if not stacker.is_file():
        raise FileNotFoundError("missing stacker: %s" % stacker)
    command = [
        sys.executable, str(stacker), "--alignment-mode", "wcs-stars",
        "--combine", "mean", "--intermediate-stack-size", str(pair_size),
        "--output-fits", "stacked_all.fits", "--output-png",
        "stacked_all.png",
    ] + [str(path) for path in originals]
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(pair_dir / ".mplconfig")
    env["XDG_CACHE_HOME"] = str(pair_dir / ".cache")
    Path(env["MPLCONFIGDIR"]).mkdir(exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(exist_ok=True)
    completed = subprocess.run(command, cwd=str(pair_dir), env=env,
                               text=True, capture_output=True)
    (pair_dir / "stacker_stdout.txt").write_text(completed.stdout)
    (pair_dir / "stacker_stderr.txt").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError("stack_by_wcs failed with status %d" %
                           completed.returncode)
    return command


def stamp_pair(path, sources, target):
    headers = [fits.getheader(source) for source in sources]
    weights = np.asarray([float(header.get("EXPTIME", 1.0))
                          for header in headers])
    times = np.asarray([midpoint_jd(header) for header in headers])
    mid_jd = float(np.average(times, weights=weights))
    total_exptime = float(weights.sum())
    first = headers[0]
    with fits.open(path, mode="update", memmap=False) as hdus:
        header = hdus[0].header
        header["TELESCOP"] = (str(first.get("TELESCOP", "3.5m")),
                              "source telescope")
        header["OBSERVAT"] = (str(first.get("OBSERVAT", "APO")),
                              "source observatory")
        header["INSTRUME"] = ("arctic", "ARC 3.5-m ARCTIC")
        header["TEL_KEYW"] = (TELESCOPE_KEY, "PP telescope keyword")
        header["OBJNAME"] = (target, "target designation")
        header["OBJECT"] = (target, "target designation")
        header["FILTER"] = (str(first.get("FILTER", DEFAULT_FILTER)),
                            "source filter")
        header["DATE-OBS"] = (str(first["DATE-OBS"]),
                              "start of first constituent exposure")
        header["EXPTIME"] = (total_exptime, "cumulative pair exposure, s")
        header["MIDTIMJD"] = (mid_jd, "photon-weighted midpoint JD UTC")
        header["NCOMBINE"] = (len(sources), "number of pair inputs")
        for index, source in enumerate(sources, start=1):
            header["INFIL%03d" % index] = (Path(source).name,
                                            "constituent input")
        hdus.flush(output_verify="silentfix")
    return {"path": str(Path(path).resolve()),
            "sources": [str(Path(source).resolve()) for source in sources],
            "midpoint_jd": mid_jd,
            "midpoint_isot": Time(mid_jd, format="jd", scale="utc").isot,
            "exptime": total_exptime}


def collect_pair_records(groups, pair_dir, target):
    records = []
    for index, sources in enumerate(groups):
        end = index * len(sources) + len(sources) - 1
        start = index * len(sources)
        generated = Path(pair_dir) / ("stack_%dto%d.fits" % (start, end))
        if not generated.is_file():
            raise FileNotFoundError("stacker did not create %s" % generated)
        record = stamp_pair(generated, sources, target)
        record["index"] = index + 1
        record["png"] = str(generated.with_suffix(".png"))
        records.append(record)
    return records


def seed_position_at(seeds, jd):
    dec0 = float(np.mean([seed["dec_deg"] for seed in seeds]))
    cosdec = math.cos(math.radians(dec0))
    times = np.asarray([seed["midpoint_jd"] for seed in seeds])
    x = np.asarray([seed["ra_deg"] * cosdec for seed in seeds])
    y = np.asarray([seed["dec_deg"] for seed in seeds])
    px = np.polyfit(times - times.mean(), x, 1)
    py = np.polyfit(times - times.mean(), y, 1)
    dt = jd - times.mean()
    return float(np.polyval(px, dt) / cosdec), float(np.polyval(py, dt))


def robust_sky_sigma(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 1.0
    sky = float(np.median(finite))
    sigma = 1.4826 * float(np.median(np.abs(finite - sky)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(finite))
    return sky, sigma if np.isfinite(sigma) and sigma > 0 else 1.0


def matched_measure(data, x0, y0, search_radius=4.0, fwhm=4.0):
    image = np.asarray(data, dtype=float)
    yy, xx = np.indices(image.shape, dtype=float)
    annulus = (np.hypot(xx - x0, yy - y0) >= max(7.0, fwhm * 1.75)) & (
        np.hypot(xx - x0, yy - y0) <= max(12.0, fwhm * 3.0))
    sky, sigma = robust_sky_sigma(image[annulus])
    psf_sigma = float(fwhm) / 2.354820045
    radius = int(math.ceil(search_radius))
    best = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if math.hypot(dx, dy) > search_radius:
                continue
            cx, cy = x0 + dx, y0 + dy
            xmin = max(0, int(math.floor(cx - 4 * psf_sigma)))
            xmax = min(image.shape[1], int(math.ceil(cx + 4 * psf_sigma + 1)))
            ymin = max(0, int(math.floor(cy - 4 * psf_sigma)))
            ymax = min(image.shape[0], int(math.ceil(cy + 4 * psf_sigma + 1)))
            sx = xx[ymin:ymax, xmin:xmax]
            sy = yy[ymin:ymax, xmin:xmax]
            values = image[ymin:ymax, xmin:xmax] - sky
            weights = np.exp(-0.5 * ((sx - cx) ** 2 + (sy - cy) ** 2) /
                             psf_sigma ** 2)
            finite = np.isfinite(values)
            signal = float(np.sum(weights[finite] * values[finite]))
            norm2 = float(np.sum(weights[finite] ** 2))
            snr = signal / (sigma * math.sqrt(norm2)) if norm2 else -np.inf
            if best is None or snr > best["snr"]:
                amplitude = signal / norm2 if norm2 else 0.0
                best = {"peak_x": cx, "peak_y": cy, "snr": float(snr),
                        "matched_flux": float(amplitude),
                        "matched_flux_err": float(sigma / math.sqrt(norm2)),
                        "sky": sky, "sky_sigma": sigma}
    cx, cy = best["peak_x"], best["peak_y"]
    local = np.hypot(xx - cx, yy - cy) <= max(3.0, fwhm)
    weights = np.where(local & np.isfinite(image),
                       np.maximum(image - sky, 0.0), 0.0)
    if weights.sum() > 0:
        best["centroid_x"] = float(np.sum(weights * xx) / weights.sum())
        best["centroid_y"] = float(np.sum(weights * yy) / weights.sum())
        r2 = float(np.sum(weights * ((xx - best["centroid_x"]) ** 2 +
                                     (yy - best["centroid_y"]) ** 2)) /
                   weights.sum())
        best["fwhm_px"] = float(2.354820045 * math.sqrt(max(r2, 0.0) / 2.0))
    else:
        best.update({"centroid_x": cx, "centroid_y": cy, "fwhm_px": None})
    best.update({"seed_x": float(x0), "seed_y": float(y0),
                 "search_radius_px": float(search_radius),
                 "assumed_fwhm_px": float(fwhm)})
    return best


def pixel_scale_arcsec(wcs):
    scales = np.asarray([
        abs(scale.to_value(u.deg)) for scale in wcs.proj_plane_pixel_scales()
    ]) * 3600.0
    return float(np.mean(scales))


def constituent_evidence(record, ra, dec, fwhm):
    evidence = []
    for source in record["sources"]:
        data = fits.getdata(source)
        wcs = WCS(fits.getheader(source), relax=True)
        x, y = wcs.all_world2pix([[ra, dec]], 0)[0]
        measure = matched_measure(data, x, y, search_radius=2.0, fwhm=fwhm)
        evidence.append({"path": source, "seed_x": float(x), "seed_y": float(y),
                         "snr": measure["snr"],
                         "matched_flux": measure["matched_flux"]})
    return evidence


def fit_track(measurements, fwhm):
    dec0 = float(np.mean([row["dec_deg"] for row in measurements]))
    cosdec = math.cos(math.radians(dec0))
    jd = np.asarray([row["midpoint_jd"] for row in measurements])
    t = (jd - jd.mean()) * 24.0
    x = np.asarray([row["ra_deg"] * cosdec * 3600.0 for row in measurements])
    y = np.asarray([row["dec_deg"] * 3600.0 for row in measurements])
    sig = np.asarray([max(row["position_sigma_arcsec"], 0.05)
                      for row in measurements])
    weights = 1.0 / sig
    px = np.polyfit(t, x, 1, w=weights)
    py = np.polyfit(t, y, 1, w=weights)
    fx, fy = np.polyval(px, t), np.polyval(py, t)
    residual = np.hypot(x - fx, y - fy)
    chi2_line = float(np.sum(((x - fx) / sig) ** 2 + ((y - fy) / sig) ** 2))
    sx, sy = np.average(x, weights=weights ** 2), np.average(y, weights=weights ** 2)
    chi2_stationary = float(np.sum(((x - sx) / sig) ** 2 +
                                    ((y - sy) / sig) ** 2))
    return {
        "ra_rate_cosdec_arcsec_per_hour": float(px[0]),
        "dec_rate_arcsec_per_hour": float(py[0]),
        "speed_arcsec_per_hour": float(math.hypot(px[0], py[0])),
        "position_angle_deg": float(math.degrees(math.atan2(px[0], py[0])) % 360),
        "max_residual_arcsec": float(residual.max()),
        "chi2_linear": chi2_line, "chi2_stationary": chi2_stationary,
        "stationary_rejection_sigma": float(math.sqrt(max(
            chi2_stationary - chi2_line, 0.0))),
        "fwhm_arcsec": float(fwhm * np.mean([
            row["pixel_scale_arcsec"] for row in measurements])),
    }


def verify_candidate(pair_records, seeds, search_radius=4.0, fwhm=4.0,
                     joint_snr_min=5.0, pair_snr_min=3.0):
    measurements = []
    for record in pair_records:
        data = fits.getdata(record["path"])
        wcs = WCS(fits.getheader(record["path"]), relax=True)
        seed_ra, seed_dec = seed_position_at(seeds, record["midpoint_jd"])
        seed_x, seed_y = wcs.all_world2pix([[seed_ra, seed_dec]], 0)[0]
        measure = matched_measure(data, seed_x, seed_y, search_radius, fwhm)
        ra, dec = wcs.all_pix2world(
            [[measure["centroid_x"], measure["centroid_y"]]], 0)[0]
        scale = pixel_scale_arcsec(wcs)
        position_sigma = (fwhm * scale /
                          (2.354820045 * max(abs(measure["snr"]), 1.0)))
        row = dict(record)
        row.update(measure)
        row.update({"seed_ra_deg": seed_ra, "seed_dec_deg": seed_dec,
                    "ra_deg": float(ra), "dec_deg": float(dec),
                    "pixel_scale_arcsec": scale,
                    "position_sigma_arcsec": float(position_sigma)})
        row["constituents"] = constituent_evidence(row, ra, dec, fwhm)
        measurements.append(row)
    track = fit_track(measurements, fwhm)
    joint_snr = float(math.sqrt(sum(max(row["snr"], 0.0) ** 2
                                    for row in measurements)))
    checks = {
        "joint_snr": joint_snr >= joint_snr_min,
        "positive_flux_all_pairs": all(row["matched_flux"] > 0
                                         for row in measurements),
        "two_pair_detections": sum(row["snr"] >= pair_snr_min
                                    for row in measurements) >= 2,
        "linear_track_residual": (track["max_residual_arcsec"] <
                                   track["fwhm_arcsec"]),
        "nonstationary_3sigma": track["stationary_rejection_sigma"] >= 3.0,
        "constituent_evidence": all(
            constituent["snr"] > 0.5 for row in measurements
            for constituent in row["constituents"]),
        "psf_like": all(
            row["fwhm_px"] is not None and 0.5 * fwhm <= row["fwhm_px"] <= 2 * fwhm
            for row in measurements if row["snr"] >= pair_snr_min),
    }
    return {"verification_status": "verified" if all(checks.values()) else
            "not_verified", "checks": checks, "joint_snr": joint_snr,
            "thresholds": {"joint_snr_min": joint_snr_min,
                           "pair_snr_min": pair_snr_min,
                           "search_radius_px": search_radius,
                           "assumed_fwhm_px": fwhm},
            "track": track, "measurements": measurements}


def write_verification(output_root, result):
    output_root = Path(output_root)
    json_path = output_root / VERIFICATION_JSON
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                    default=json_default) + "\n")
    csv_path = output_root / VERIFICATION_CSV
    fields = ["index", "path", "midpoint_jd", "midpoint_isot", "exptime",
              "seed_ra_deg", "seed_dec_deg", "seed_x", "seed_y",
              "centroid_x", "centroid_y", "ra_deg", "dec_deg", "snr",
              "matched_flux", "matched_flux_err", "fwhm_px",
              "position_sigma_arcsec"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["measurements"]:
            writer.writerow({key: row.get(key) for key in fields})
    return json_path, csv_path


def render_verification(output_root, result):
    output_root = Path(output_root)
    artifact_dir = output_root / "verification_images"
    artifact_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    images = []
    for row in result["measurements"]:
        data = np.asarray(fits.getdata(row["path"]), dtype=float)
        x, y = row["centroid_x"], row["centroid_y"]
        half = 40
        xmin, xmax = max(0, int(x) - half), min(data.shape[1], int(x) + half)
        ymin, ymax = max(0, int(y) - half), min(data.shape[0], int(y) + half)
        cutout = data[ymin:ymax, xmin:xmax]
        finite = cutout[np.isfinite(cutout)]
        vmin, vmax = np.percentile(finite, [5, 99.5])
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(cutout, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
        ax.plot(row["seed_x"] - xmin, row["seed_y"] - ymin, marker="+",
                color="gold", markersize=12, markeredgewidth=1.5, label="seed")
        ax.add_patch(plt.Circle((x - xmin, y - ymin), 5, fill=False,
                                color="red", linewidth=1.3, label="fit"))
        ax.set_title("Pair %d  S/N %.2f" % (row["index"], row["snr"]))
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        path = artifact_dir / ("pair_%02d_annotated.png" % row["index"])
        fig.savefig(path, dpi=140)
        plt.close(fig)
        images.append(path)
    gif = output_root / "candidate_blink.gif"
    frames = [Image.open(path).convert("RGB") for path in images]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=700,
                   loop=0)
    for frame in frames:
        frame.close()
    return {"annotated_cutouts": [str(path) for path in images],
            "blink": str(gif)}


def snapshot_remote_context(output_root, target, pair_records):
    result = {"horizons": None, "sbdb": None, "errors": []}
    output_root = Path(output_root)
    try:
        from astroquery.jplhorizons import Horizons
        epochs = [record["midpoint_jd"] for record in pair_records]
        table = Horizons(id=target, id_type="smallbody", location=OBSERVATORY_CODE,
                         epochs=epochs).ephemerides(extra_precision=True,
                                                    cache=False)
        path = output_root / "horizons_ephemeris.ecsv"
        table.write(path, format="ascii.ecsv", overwrite=True)
        result["horizons"] = str(path)
    except Exception as exc:
        result["errors"].append("Horizons: %s" % exc)
    try:
        import requests
        response = requests.get("https://ssd-api.jpl.nasa.gov/sbdb.api",
                                params={"sstr": target, "full-prec": "true"},
                                timeout=30)
        response.raise_for_status()
        path = output_root / "sbdb_orbit.json"
        path.write_text(json.dumps(response.json(), indent=2,
                                   sort_keys=True) + "\n")
        result["sbdb"] = str(path)
    except Exception as exc:
        result["errors"].append("SBDB: %s" % exc)
    return result


@contextmanager
def working_directory(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def configure_pp_environment():
    root = Path(__file__).resolve().parent
    os.environ["PHOTPIPEDIR"] = str(root)
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/photometrypipeline_mpl")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def write_ades_photometry_input(photometry_path, snr_min=3.0):
    """Mask marginal PP magnitudes while retaining their astrometry rows."""

    photometry_path = Path(photometry_path)
    output = photometry_path.with_name(
        photometry_path.stem + "_ades" + photometry_path.suffix)
    lines = []
    for line in photometry_path.read_text().splitlines(True):
        if not line.strip() or line.lstrip().startswith("#"):
            lines.append(line)
            continue
        values = line.split()
        mag = float(values[2])
        mag_sig = float(values[3])
        snr = (1.0857362047581294 / mag_sig
               if mag < 90 and 0 < mag_sig < 90 else 0.0)
        if snr < float(snr_min):
            values[2] = "99.0000"
            values[3] = "99.0000"
            line = " " + " ".join(values) + "\n"
        lines.append(line)
    output.write_text("".join(lines))
    return output


def run_pp_products(output_root, target, verification):
    configure_pp_environment()
    import _pp_conf
    import pp_prepare
    import pp_register
    import pp_run
    import pptool_efosc2 as efosc2
    import pptool_mpcsubmission as mpcsub

    output_root = Path(output_root)
    filter_dir = output_root / "photometry" / "r"
    filter_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in verification["measurements"]:
        destination = filter_dir / ("arc35_pair_%04d.fits" % row["index"])
        shutil.copy2(row["path"], destination)
        with fits.open(destination, mode="update", memmap=False) as hdus:
            header = hdus[0].header
            header["FILTER"] = ("r", "PP normalized CU VR filter")
            header["TARGX"] = (row["centroid_x"], "target pixel x, origin zero")
            header["TARGY"] = (row["centroid_y"], "target pixel y, origin zero")
            hdus.flush(output_verify="silentfix")
        records.append({"working_path": str(destination),
                        "working_name": destination.name,
                        "centroid": {"centroid_x": row["centroid_x"],
                                     "centroid_y": row["centroid_y"]},
                        "midpoint_jd": row["midpoint_jd"]})
    obsparam = dict(_pp_conf.telescope_parameters[TELESCOPE_KEY])
    names = [record["working_name"] for record in records]
    with working_directory(filter_dir):
        pp_prepare.prepare(names, obsparam, {"OBJNAME": target}, keep_wcs=True,
                           diagnostics=True, display=True)
        registration = pp_register.register(
            names, TELESCOPE_KEY, obsparam["source_snr"],
            obsparam["source_minarea"], obsparam["aprad_default"], "GAIA",
            obsparam, obsparam["source_tolerance"], False,
            display=True, diagnostics=True)
    good = {Path(name).name for name in registration.get("goodfits", [])}
    if good != set(names):
        raise RuntimeError("Gaia registration succeeded for %d/%d pairs" %
                           (len(good), len(names)))
    for record in records:
        with fits.open(record["working_path"], mode="update", memmap=False) as hdus:
            header = hdus[0].header
            c = record["centroid"]
            ra, dec = WCS(header, relax=True).all_pix2world(
                [[c["centroid_x"], c["centroid_y"]]], 0)[0]
            header["TARGRA"] = (float(ra), "target RA through Gaia WCS")
            header["TARGDEC"] = (float(dec), "target Dec through Gaia WCS")
            header["TARGSOL"] = ("GAIA+PSF", "target solution provenance")
            hdus.flush(output_verify="silentfix")
            record.update({"refined_ra_deg": float(ra),
                           "refined_dec_deg": float(dec)})
    positions = filter_dir / ("positions_%s_r.dat" % target_token(target))
    with positions.open("w") as handle:
        for record in records:
            handle.write("%s %.10f %.10f %.8f %s\n" % (
                record["working_name"], record["refined_ra_deg"],
                record["refined_dec_deg"], record["midpoint_jd"],
                target_token(target)))
    with working_directory(filter_dir):
        pp_run.run_the_pipeline(
            names, target, "r", 0.0, "high", False, False, False, True,
            telescope=TELESCOPE_KEY, posfile=positions.name,
            rejectionfilter="none")
    filter_photometry = filter_dir / ("photometry_%s.dat" % target_token(target))
    if not filter_photometry.exists():
        raise RuntimeError("PP did not produce %s" % filter_photometry)
    combined = efosc2.combine_photometry(output_root, target, filter_photometry)[0]
    ades_input = write_ades_photometry_input(combined, snr_min=3.0)
    config = mpcsub.SubmissionConfig(
        target=target, review_only=True, observatory_code=OBSERVATORY_CODE,
        astcat="Gaia2", observers=["UNKNOWN"], output_dir=output_root)
    bundle = mpcsub.build_submission(ades_input, config)
    mpcsub.write_submission(bundle)
    review = {
        "review_only": True, "submission_ready": False,
        "psv": mpcsub.validate_ades_psv_text(bundle.ades_text),
        "xml": mpcsub.validate_ades_xml_text(bundle.ades_xml_text),
        "warnings": bundle.warnings,
        "submit_script": None,
    }
    review_path = output_root / ("mpc_%s_ADES_review.json" % target_token(target))
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    return {"positions": str(positions), "photometry": str(combined),
            "ades_photometry_input": str(ades_input),
            "ades_psv": str(bundle.ades_path), "ades_xml": str(bundle.ades_xml_path),
            "ades_review": str(review_path), "submit_script": None}


def run_workflow(args):
    click_csv = Path(args.click_points_csv).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        click_csv.parent.parent / "PP" / target_token(args.target))
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = read_click_seeds(click_csv)
    originals = discover_originals(seeds)
    groups = group_adjacent(originals, args.pair_size)
    checksum_path, checksums = write_checksums(
        output_root, click_csv, seeds, originals)
    pair_dir = output_root / "pairs"
    command = run_stacker(args.stacker, originals, args.pair_size, pair_dir)
    pair_records = collect_pair_records(groups, pair_dir, args.target)
    remote = ({"horizons": None, "sbdb": None, "errors": ["skipped"]}
              if args.skip_remote else
              snapshot_remote_context(output_root, args.target, pair_records))
    verification = verify_candidate(
        pair_records, seeds, search_radius=args.search_radius,
        fwhm=args.fwhm, joint_snr_min=args.joint_snr_min,
        pair_snr_min=args.pair_snr_min)
    verification_paths = write_verification(output_root, verification)
    images = render_verification(output_root, verification)
    manifest = {
        "target": args.target, "observatory_code": OBSERVATORY_CODE,
        "telescope_key": TELESCOPE_KEY, "click_csv": str(click_csv),
        "output_root": str(output_root), "originals": [str(p) for p in originals],
        "groups": [[str(p) for p in group] for group in groups],
        "pair_records": pair_records, "stacker_command": command,
        "checksums": str(checksum_path), "checksums_before": checksums,
        "remote_context": remote, "verification": verification,
        "verification_json": str(verification_paths[0]),
        "verification_csv": str(verification_paths[1]),
        "verification_images": images, "pp": None,
    }
    if verification["verification_status"] == "verified":
        try:
            manifest["pp"] = run_pp_products(output_root, args.target,
                                               verification)
            manifest["workflow_status"] = "complete"
        except Exception as exc:
            manifest["workflow_status"] = "pp_error"
            manifest["pp_error"] = str(exc)
            path = output_root / MANIFEST_NAME
            path.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                       default=json_default) + "\n")
            raise
    else:
        manifest["workflow_status"] = "not_verified"
    final_checksums = {path: sha256(path) for path in checksums}
    manifest["originals_unchanged"] = final_checksums == checksums
    manifest_path = output_root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                        default=json_default) + "\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("click_points_csv")
    parser.add_argument("--target", required=True)
    parser.add_argument("--pair-size", type=int, default=2)
    parser.add_argument("--stacker", required=True,
                        help="path to coc_tools/fits/stack_by_wcs.py")
    parser.add_argument("--output-dir")
    parser.add_argument("--search-radius", type=float, default=4.0)
    parser.add_argument("--fwhm", type=float, default=4.0)
    parser.add_argument("--joint-snr-min", type=float, default=5.0)
    parser.add_argument("--pair-snr-min", type=float, default=3.0)
    parser.add_argument("--skip-remote", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = run_workflow(args)
    print("Wrote %s" % manifest["manifest_path"])
    print("Verification: %s" % manifest["verification"]["verification_status"])
    if manifest["workflow_status"] == "not_verified":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
