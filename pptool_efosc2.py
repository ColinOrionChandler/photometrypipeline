#!/usr/bin/env python3

"""Run the NTT/EFOSC2 PP workflow for a clicked temporary object."""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u


TELESCOPE_KEY = "NTTEFOSC2"
OBSERVATORY_CODE = "809"
BASELINE_OBSERVATORY_CODE = "X05"
WORKING_PREFIX = "efosc2_r"
BACKUP_NAME = "originals_backup.tar"
MANIFEST_NAME = "efosc2_workflow_manifest.json"
FILTER_TRANSLATIONS = {"r#784": "r", "r": "r"}
DEFAULT_FIND_ORB = Path("/Users/colinchandler/GitHub/find_orb/fo")
DEFAULT_FIND_ORB_CWD = Path("/Users/colinchandler/GitHub/find_orb")


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("not JSON serializable: %r" % (value,))


@contextmanager
def working_directory(path):
    old = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


def repo_root():
    return Path(__file__).resolve().parent


def configure_pp_environment():
    root = repo_root()
    os.environ["PHOTPIPEDIR"] = str(root)
    bin_dir = str(Path(sys.executable).resolve().parent)
    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def target_token(target):
    return str(target).translate(str.maketrans(" /()", "____"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_filter(header):
    return str(header.get("ESO INS FILT1 NAME", header.get("FILTER", ""))).strip()


def normalize_filter(value):
    text = str(value).strip()
    if text not in FILTER_TRANSLATIONS:
        raise ValueError("unsupported EFOSC2 filter %r" % text)
    return FILTER_TRANSLATIONS[text]


def midpoint_jd(header):
    exposure = float(header["EXPTIME"])
    if "MJD-OBS" in header:
        return float(header["MJD-OBS"]) + 2400000.5 + exposure / 172800.0
    return Time(str(header["DATE-OBS"]), format="isot", scale="utc").jd + exposure / 172800.0


def read_clicks(click_csv):
    click_csv = Path(click_csv).expanduser().resolve()
    rows = []
    with click_csv.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            if str(row.get("status", "")).strip().lower() != "clicked":
                continue
            png = click_csv.parent / row["image"]
            fits_path = png.with_suffix(".fits")
            if not fits_path.exists():
                raise FileNotFoundError("no FITS counterpart for %s" % png.name)
            rows.append({
                "index": index,
                "png_path": str(png.resolve()),
                "fits_path": str(fits_path.resolve()),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "width": int(float(row["width"])),
                "height": int(float(row["height"])),
            })
    if not rows:
        raise ValueError("no clicked rows in %s" % click_csv)
    return rows


def png_to_fits_pixel(click, shape):
    """Convert zero-based, vertically flipped PNG pixels to FITS pixels."""

    ny, nx = shape[-2:]
    if click["width"] == nx and click["height"] == ny:
        x = float(click["x"])
        y = float(click["height"] - 1) - float(click["y"])
    else:
        x = float(click["x"]) * float(nx - 1) / float(click["width"] - 1)
        flipped_y = float(click["height"] - 1) - float(click["y"])
        y = flipped_y * float(ny - 1) / float(click["height"] - 1)
    return x, y


def robust_sky_sigma(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 1.0
    sky = float(np.median(finite))
    sigma = 1.4826 * float(np.median(np.abs(finite - sky)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(finite))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return sky, sigma


def centroid_from_pixel(data, x0, y0, search_radius=8.0,
                        aperture_radius=3.0, min_snr=1.5):
    """Centroid the nearest positive local peak around a clicked pixel."""

    image = np.asarray(data, dtype=float)
    radius = int(np.ceil(search_radius))
    xmin = max(0, int(np.floor(x0)) - radius)
    xmax = min(image.shape[1] - 1, int(np.floor(x0)) + radius)
    ymin = max(0, int(np.floor(y0)) - radius)
    ymax = min(image.shape[0] - 1, int(np.floor(y0)) + radius)
    patch = image[ymin:ymax + 1, xmin:xmax + 1]
    sky, sigma = robust_sky_sigma(patch)
    yy, xx = np.indices(patch.shape, dtype=float)
    ax = xx + xmin
    ay = yy + ymin
    distance = np.hypot(ax - x0, ay - y0)
    positive = (np.isfinite(patch) & (distance <= search_radius) &
                ((patch - sky) >= min_snr * sigma))
    if not positive.any():
        raise ValueError("no positive centroid source within %.1f pixels" % search_radius)
    indices = np.argwhere(positive)
    distances = distance[positive]
    fluxes = (patch - sky)[positive]
    order = np.lexsort((-fluxes, distances))
    peak_y, peak_x = indices[order[0]]
    px = float(peak_x + xmin)
    py = float(peak_y + ymin)
    aperture = np.hypot(ax - px, ay - py) <= aperture_radius
    weights = np.where(aperture & np.isfinite(patch), patch - sky, 0.0)
    weights = np.maximum(weights, 0.0)
    if not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError("centroid aperture has no positive flux")
    cx = float((weights * ax).sum() / weights.sum())
    cy = float((weights * ay).sum() / weights.sum())
    return {
        "click_x": float(x0), "click_y": float(y0),
        "peak_x": px, "peak_y": py,
        "centroid_x": cx, "centroid_y": cy,
        "peak_snr": float((patch[peak_y, peak_x] - sky) / sigma),
        "sky": sky, "sigma": sigma,
        "search_radius_px": float(search_radius),
        "aperture_radius_px": float(aperture_radius),
    }


def original_paths(click_csv, clicks, initial_positions=None):
    paths = [Path(click_csv).expanduser().resolve()]
    for click in clicks:
        paths.extend([Path(click["fits_path"]), Path(click["png_path"])])
    if initial_positions is not None:
        paths.append(Path(initial_positions).expanduser().resolve())
    return sorted(set(path for path in paths if path.exists()))


def create_originals_backup(pp_dir, paths, refresh=False):
    pp_dir = Path(pp_dir)
    backup = pp_dir / BACKUP_NAME
    checksums = {str(path): sha256(path) for path in paths}
    if not backup.exists() or refresh:
        if backup.exists():
            backup.unlink()
        with tarfile.open(str(backup), "w") as archive:
            for path in paths:
                archive.add(str(path), arcname=path.name)
        created = True
    else:
        created = False
    return {"path": str(backup), "created": created, "checksums": checksums}


def flatten_working_copy(source, destination, target, click,
                         search_radius=8.0, aperture_radius=3.0,
                         min_snr=1.5):
    """Write the primary science HDU only and record click provenance."""

    source = Path(source)
    destination = Path(destination)
    with fits.open(str(source), memmap=False, ignore_missing_end=True) as hdus:
        data = np.asarray(hdus[0].data, dtype=np.float32).copy()
        header = hdus[0].header.copy()
        extension_count = len(hdus)
    bad = ~np.isfinite(data)
    if bad.any():
        finite = data[~bad]
        data[bad] = float(np.median(finite)) if finite.size else 0.0
    filt_native = native_filter(header)
    filt = normalize_filter(filt_native)
    click_x, click_y = png_to_fits_pixel(click, data.shape)
    centroid = centroid_from_pixel(data, click_x, click_y, search_radius,
                                   aperture_radius, min_snr)
    native_ra, native_dec = WCS(header, relax=True).all_pix2world(
        [[centroid["centroid_x"], centroid["centroid_y"]]], 0)[0]
    mid_jd = midpoint_jd(header)
    header["FILTER"] = (filt, "PP normalized filter")
    header["OBJECT"] = (target, "temporary target name")
    header["MIDTIMJD"] = (mid_jd, "exposure midpoint JD UTC")
    header["TEL_KEYW"] = (TELESCOPE_KEY, "PP telescope keyword")
    header["TELESCOP"] = ("ESO 3.58-m NTT", "normalized telescope")
    header["INSTRUME"] = ("EFOSC", "EFOSC2 instrument identifier")
    header["ORIGFILE"] = (source.name[:68], "original EFOSC2 filename")
    header["CLICKX"] = (float(click["x"]), "zero-based PNG click x")
    header["CLICKY"] = (float(click["y"]), "zero-based PNG click y")
    header["CLICKFX"] = (click_x, "zero-based FITS click x")
    header["CLICKFY"] = (click_y, "zero-based FITS click y after y flip")
    header["TARGX"] = (centroid["centroid_x"], "centroid x, origin zero")
    header["TARGY"] = (centroid["centroid_y"], "centroid y, origin zero")
    header["TARGRA"] = (float(native_ra), "target RA through native WCS")
    header["TARGDEC"] = (float(native_dec), "target Dec through native WCS")
    header["CENTROID"] = (True, "target centroided from PNG click")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data, header=header).writeto(
        str(destination), overwrite=True, output_verify="silentfix")
    return {
        "source_path": str(source), "working_path": str(destination),
        "working_name": destination.name, "source_hdu_count": extension_count,
        "working_hdu_count": 1, "filter": filt, "native_filter": filt_native,
        "midpoint_jd": float(mid_jd), "midpoint_isot": Time(mid_jd, format="jd").isot,
        "centroid": centroid, "native_ra_deg": float(native_ra),
        "native_dec_deg": float(native_dec), "nonfinite_replaced": int(bad.sum()),
    }


def prepare_working_images(click_csv, pp_dir, target, search_radius=8.0,
                           aperture_radius=3.0, min_snr=1.5):
    clicks = read_clicks(click_csv)
    records = []
    filter_dir = Path(pp_dir) / "r"
    for seq, click in enumerate(clicks, start=1):
        source = Path(click["fits_path"])
        # PP's distillation table has a fixed-width filename column.  Keep the
        # working name short and retain the full source name in ORIGFILE.
        destination = filter_dir / ("%s_%04d.fits" %
                                    (WORKING_PREFIX, seq))
        record = flatten_working_copy(
            source, destination, target, click, search_radius,
            aperture_radius, min_snr)
        record["sequence"] = seq
        record["click"] = click
        records.append(record)
    return records


def register_working_images(records):
    """Prepare and Gaia-register all images before target coordinates exist."""

    configure_pp_environment()
    import _pp_conf
    import pp_prepare
    import pp_register
    obsparam = dict(_pp_conf.telescope_parameters[TELESCOPE_KEY])
    names = [record["working_name"] for record in records]
    filter_dir = Path(records[0]["working_path"]).parent
    with working_directory(filter_dir):
        pp_prepare.prepare(names, obsparam, {"OBJECT": "McLeod"},
                           keep_wcs=True, diagnostics=True, display=True)
        registration = pp_register.register(
            names, TELESCOPE_KEY, obsparam["source_snr"],
            obsparam["source_minarea"], obsparam["aprad_default"], "GAIA",
            obsparam, obsparam["source_tolerance"], False,
            display=True, diagnostics=True)
    good = {Path(name).name for name in registration.get("goodfits", [])}
    if good != set(names):
        raise RuntimeError("Gaia registration succeeded for %d/%d EFOSC2 frames" %
                           (len(good), len(names)))
    return registration


def apply_refined_wcs(records):
    """Convert stored centroid pixels through the post-SCAMP WCS."""

    for record in records:
        path = Path(record["working_path"])
        with fits.open(str(path), mode="update", memmap=False,
                       ignore_missing_end=True) as hdus:
            header = hdus[0].header
            c = record["centroid"]
            ra, dec = WCS(header, relax=True).all_pix2world(
                [[c["centroid_x"], c["centroid_y"]]], 0)[0]
            header["TARGRA"] = (float(ra), "target RA through refined WCS")
            header["TARGDEC"] = (float(dec), "target Dec through refined WCS")
            header["TARGSOL"] = ("GAIA+CENTROID", "target solution provenance")
            hdus.flush(output_verify="silentfix")
            record["refined_ra_deg"] = float(ra)
            record["refined_dec_deg"] = float(dec)
            record["registration_catalog"] = str(header.get("REGCAT", ""))
            record["wcs_refined"] = bool(header.get("REGCAT"))
        native = SkyCoord(record["native_ra_deg"] * u.deg,
                          record["native_dec_deg"] * u.deg)
        refined = SkyCoord(record["refined_ra_deg"] * u.deg,
                           record["refined_dec_deg"] * u.deg)
        record["wcs_shift_arcsec"] = float(native.separation(refined).arcsec)
    return records


def write_positions(records, target):
    path = Path(records[0]["working_path"]).parent / (
        "positions_%s_r.dat" % target_token(target))
    with path.open("w") as handle:
        for record in records:
            handle.write("%s %.10f %.10f %.8f %s\n" % (
                record["working_name"], record["refined_ra_deg"],
                record["refined_dec_deg"], record["midpoint_jd"],
                target_token(target)))
    return path


def run_pp(records, target, positions, fixed_aprad=0.0):
    configure_pp_environment()
    import pp_run
    filter_dir = Path(records[0]["working_path"]).parent
    names = sorted(record["working_name"] for record in records)
    with working_directory(filter_dir):
        pp_run.run_the_pipeline(
            names, target, "r", float(fixed_aprad), "high", False, False,
            False, True, telescope=TELESCOPE_KEY,
            posfile=Path(positions).name, rejectionfilter="none")
    path = filter_dir / ("photometry_%s.dat" % target_token(target))
    if not path.exists():
        raise RuntimeError("PP did not produce %s" % path)
    return path


def combine_photometry(pp_dir, target, filter_photometry):
    """Copy r output to the root while making FITS tokens root-relative."""

    lines = []
    for line in Path(filter_photometry).read_text().splitlines(True):
        if line.strip() and not line.lstrip().startswith("#"):
            values = line.split()
            values[0] = str(Path("r") / values[0])
            line = " " + " ".join(values) + "\n"
        lines.append(line)
    text = "".join(lines)
    outputs = []
    for name in ("photometry_%s.dat" % target_token(target),
                 "photometry_%s_all_filters.dat" % target_token(target)):
        path = Path(pp_dir) / name
        path.write_text(text)
        outputs.append(path)
    return outputs


def read_photometry_qa(path):
    """Extract source-match and calibration QA from active PP rows."""

    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = line.split()
        dra = float(values[6])
        ddec = float(values[7])
        mag = float(values[2])
        mag_sig = float(values[3])
        rows.append({
            "catalog_token": values[0],
            "julian_date": float(values[1]),
            "source_ra_deg": float(values[4]),
            "source_dec_deg": float(values[5]),
            "predicted_minus_source_ra_arcsec": dra,
            "predicted_minus_source_dec_arcsec": ddec,
            "source_match_separation_arcsec": float(np.hypot(dra, ddec)),
            "mag": None if mag >= 90 else mag,
            "mag_sig": None if mag_sig >= 90 else mag_sig,
            "zeropoint": float(values[11]),
            "zeropoint_sig": float(values[12]),
            "photometry_catalog": values[15],
            "band": values[16],
            "source_extractor_flag": int(values[17]),
            "fwhm_arcsec": float(values[20]),
            "calibrated_photometry": mag < 90 and mag_sig < 90,
        })
    return rows


def split_initial_positions(path, baseline_path=None):
    """Parse arbitrary Unicode whitespace and enforce the McLeod 3/4 split."""

    path = Path(path).expanduser().resolve()
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = line.split()
        if len(values) < 3:
            raise ValueError("invalid initial position line %d" % line_no)
        time = Time(values[0].rstrip("Z"), format="isot", scale="utc")
        rows.append({"obs_time": time.isot, "jd": float(time.jd),
                     "ra_deg": float(values[1]), "dec_deg": float(values[2]),
                     "source_line": line_no})
    if len(rows) != 7:
        raise ValueError("expected seven initial positions, found %d" % len(rows))
    baseline, removed = rows[:3], rows[3:]
    baseline_path = Path(baseline_path or path.with_name("initial_positions_baseline.txt"))
    with baseline_path.open("w") as handle:
        for row in baseline:
            handle.write("%sZ %.9f %.9f\n" %
                         (row["obs_time"], row["ra_deg"], row["dec_deg"]))
    return baseline, removed, baseline_path


def observations_from_bundle(bundle):
    return [{"obs_time": obs.obs_time.isot, "jd": float(obs.julian_date),
             "ra_deg": float(obs.ra_deg), "dec_deg": float(obs.dec_deg),
             "rms_ra": obs.ra_tot_sig, "rms_dec": obs.dec_tot_sig,
             "astcat": "Gaia2"} for obs in bundle.observations]


def write_position_comparison(removed, final, path, tolerance_seconds=2.0):
    path = Path(path)
    rows = []
    unmatched = list(final)
    for old in removed:
        match = min(unmatched, key=lambda item: abs(item["jd"] - old["jd"]))
        delta_seconds = abs(match["jd"] - old["jd"]) * 86400.0
        if delta_seconds > tolerance_seconds:
            raise ValueError("no PP observation within %.1f seconds" % tolerance_seconds)
        unmatched.remove(match)
        mean_dec = np.deg2rad((old["dec_deg"] + match["dec_deg"]) / 2.0)
        dra = (match["ra_deg"] - old["ra_deg"]) * np.cos(mean_dec) * 3600.0
        ddec = (match["dec_deg"] - old["dec_deg"]) * 3600.0
        rows.append({"obs_time": match["obs_time"],
                     "old_ra_deg": old["ra_deg"], "old_dec_deg": old["dec_deg"],
                     "new_ra_deg": match["ra_deg"], "new_dec_deg": match["dec_deg"],
                     "delta_ra_cosdec_arcsec": float(dra),
                     "delta_dec_arcsec": float(ddec),
                     "separation_arcsec": float(np.hypot(dra, ddec)),
                     "timestamp_delta_seconds": float(delta_seconds)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def format_find_orb_psv(target, baseline, final=None):
    rows = []
    for obs in baseline:
        rows.append((obs, BASELINE_OBSERVATORY_CODE))
    for obs in final or []:
        rows.append((obs, OBSERVATORY_CODE))
    lines = ["# version=2022", "trkSub|mode|stn|obsTime|ra|dec|rmsRA|rmsDec|astCat"]
    for obs, station in rows:
        lines.append("%s|CCD|%s|%sZ|%.9f|%.9f|%s|%s|%s" % (
            target, station, obs["obs_time"], obs["ra_deg"], obs["dec_deg"],
            "" if obs.get("rms_ra") is None else "%.4f" % obs["rms_ra"],
            "" if obs.get("rms_dec") is None else "%.4f" % obs["rms_dec"],
            obs.get("astcat", "Gaia2")))
    return "\n".join(lines) + "\n"


def parse_find_orb_stdout(text):
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    object_match = re.search(r"^\s*\d+:\s*(?P<object>[^;\n]+)", clean,
                             flags=re.MULTILINE)
    orbit_match = re.search(
        r";\s*a=(?P<a>[-+0-9.]+),\s*e=(?P<e>[-+0-9.]+),\s*"
        r"i=(?P<i>[-+0-9.]+)\s+(?:(?P<used>\d+)\s*/\s*)?"
        r"(?P<total>\d+)\s+obs;\s*(?P<arc>[^\n]+)", clean)
    if object_match is None or orbit_match is None:
        return None
    return {
        "object": object_match.group("object").strip(),
        "semimajor_axis_au": float(orbit_match.group("a")),
        "eccentricity": float(orbit_match.group("e")),
        "inclination_deg": float(orbit_match.group("i")),
        "observations": int(orbit_match.group("used") or
                            orbit_match.group("total")),
        "total_observations": int(orbit_match.group("total")),
        "arc": orbit_match.group("arc").strip(),
    }


def run_find_orb_case(executable, input_text, case_dir):
    """Run Find_Orb in a short private temporary workspace and retain it."""

    executable = Path(executable).expanduser().resolve()
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / "input.psv"
    input_path.write_text(input_text)
    with tempfile.TemporaryDirectory(prefix="mcleod_fo_", dir="/private/tmp") as temp:
        scratch = Path(temp)
        scratch_input = scratch / "input.psv"
        shutil.copy2(input_path, scratch_input)
        home = scratch / "home"
        output = scratch / "output"
        config = scratch / "config"
        for directory in (home, output, config):
            directory.mkdir()
        residuals = scratch / "residuals.txt"
        env = os.environ.copy()
        env["HOME"] = str(home)
        command = [str(executable), str(scratch_input),
                   "-x%s/" % config, "-O%s/" % output,
                   "-b%s" % residuals, "-q"]
        proc = subprocess.run(
            command, cwd=str(DEFAULT_FIND_ORB_CWD), env=env,
            capture_output=True, text=True, timeout=300)
        (case_dir / "fo_stdout.txt").write_text(proc.stdout)
        (case_dir / "fo_stderr.txt").write_text(proc.stderr)
        for path in scratch.iterdir():
            if path.name in {"input.psv", "home"}:
                continue
            destination = case_dir / path.name
            if path.is_dir():
                shutil.copytree(path, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(path, destination)
    elements_json = case_dir / "output" / "elements.json"
    parsed = parse_find_orb_stdout(proc.stdout)
    if parsed is None and elements_json.exists():
        payload = json.loads(elements_json.read_text())
        objects = payload.get("objects", {})
        if objects:
            name, object_data = next(iter(objects.items()))
            elements = object_data.get("elements", {})
            observations = object_data.get("observations", {})
            earliest = str(observations.get("earliest iso", "")).split("T")[0]
            latest = str(observations.get("latest iso", "")).split("T")[0]
            parsed = {
                "object": name,
                "semimajor_axis_au": float(elements["a"]),
                "eccentricity": float(elements["e"]),
                "inclination_deg": float(elements["i"]),
                "observations": int(observations.get("used", 0)),
                "total_observations": int(observations.get("count", 0)),
                "arc": "%s to %s" % (earliest, latest),
                "rms_residual_arcsec": float(elements.get(
                    "rms_residual", np.nan)),
                "weighted_rms_residual": float(elements.get(
                    "weighted_rms_residual", np.nan)),
            }
    if parsed is not None:
        (case_dir / "orbit_summary.json").write_text(
            json.dumps(parsed, indent=2, sort_keys=True) + "\n")
        (case_dir / "orbit_summary.md").write_text(
            "# Find_Orb Summary\n\n"
            "- Object: {object}\n"
            "- Observations: {observations}/{total_observations}\n"
            "- Arc: {arc}\n"
            "- Semimajor axis: {semimajor_axis_au:.3f} au\n"
            "- Eccentricity: {eccentricity:.3f}\n"
            "- Inclination: {inclination_deg:.3f} deg\n".format(**parsed))
    result = {"returncode": proc.returncode, "parsed": parsed,
              "case_dir": str(case_dir),
              "elements_json": (str(elements_json)
                                if elements_json.exists() else None),
              "residuals": (str(case_dir / "residuals.txt")
                            if (case_dir / "residuals.txt").exists() else None),
              "files": sorted(path.name for path in case_dir.iterdir())}
    (case_dir / "run_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def write_orbit_comparison(root, baseline_result, augmented_result):
    root = Path(root)
    comparison = {"baseline": baseline_result, "augmented": augmented_result}
    json_path = root / "orbit_comparison.json"
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    lines = ["# McLeod Find_Orb comparison", "",
             "| Fit | Return code | Used/total observations | a (au) | e | i (deg) |",
             "|---|---:|---:|---:|---:|---:|"]
    for label, result in (("Baseline", baseline_result), ("Augmented", augmented_result)):
        parsed = result.get("parsed") or {}
        lines.append("| %s | %s | %s/%s | %s | %s | %s |" % (
            label, result.get("returncode"), parsed.get("observations", ""),
            parsed.get("total_observations", ""), parsed.get("semimajor_axis_au", ""),
            parsed.get("eccentricity", ""), parsed.get("inclination_deg", "")))
    md_path = root / "orbit_comparison.md"
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def build_review_submission(pp_dir, target, photometry_path):
    import pptool_mpcsubmission as mpcsub
    config = mpcsub.SubmissionConfig(
        target=target, trk_sub=target, review_only=True,
        observatory_code=OBSERVATORY_CODE, astcat="Gaia2",
        observers=["UNKNOWN"], prog=None, output_dir=Path(pp_dir))
    bundle = mpcsub.build_submission(Path(photometry_path), config)
    mpcsub.write_submission(bundle)
    psv_validation = mpcsub.validate_ades_psv_text(bundle.ades_text)
    xml_validation = mpcsub.validate_ades_xml_text(bundle.ades_xml_text)
    validation = {"review_only": True, "submission_ready": False,
                  "psv": psv_validation, "xml": xml_validation,
                  "warnings": bundle.warnings,
                  "missing_photometry_rows": sum(
                      obs.astrometry_only for obs in bundle.observations)}
    validation_path = Path(pp_dir) / (
        "mpc_%s_ADES_review.json" % target_token(target))
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    return bundle, validation_path


def run_workflow(args):
    click_csv = Path(args.click_points_csv).expanduser().resolve()
    pp_dir = Path(args.pp_dir or click_csv.parent).expanduser().resolve()
    initial = Path(args.initial_positions).expanduser().resolve()
    pp_dir.mkdir(parents=True, exist_ok=True)
    clicks = read_clicks(click_csv)
    paths = original_paths(click_csv, clicks, initial)
    backup = create_originals_backup(pp_dir, paths, args.refresh_backup)
    records = prepare_working_images(
        click_csv, pp_dir, args.target, args.centroid_search_radius,
        args.centroid_aperture_radius, args.centroid_min_snr)
    registration = register_working_images(records)
    apply_refined_wcs(records)
    positions = write_positions(records, args.target)
    filter_photometry = run_pp(records, args.target, positions, args.fixed_aprad)
    combined = combine_photometry(pp_dir, args.target, filter_photometry)
    photometry_qa = read_photometry_qa(combined[0])
    bundle, validation_path = build_review_submission(pp_dir, args.target, combined[0])
    if len(bundle.observations) != 4:
        raise RuntimeError("expected four active PP observations, found %d" %
                           len(bundle.observations))
    baseline, removed, baseline_path = split_initial_positions(initial)
    final = observations_from_bundle(bundle)
    comparison_path = pp_dir / "preliminary_vs_pp_positions.csv"
    comparison = write_position_comparison(removed, final, comparison_path)
    orbit_root = pp_dir / "orbit_solver_runs" / args.target
    baseline_text = format_find_orb_psv(args.target, baseline)
    augmented_text = format_find_orb_psv(args.target, baseline, final)
    if sum(1 for line in augmented_text.splitlines() if line and not line.startswith("#") and not line.startswith("trkSub|")) != 7:
        raise RuntimeError("augmented orbit input does not contain seven observations")
    baseline_run = run_find_orb_case(args.find_orb, baseline_text,
                                     orbit_root / "baseline")
    augmented_run = run_find_orb_case(args.find_orb, augmented_text,
                                      orbit_root / "augmented")
    for label, result in (("baseline", baseline_run),
                          ("augmented", augmented_run)):
        if (result["returncode"] != 0 or result["parsed"] is None or
                not result["elements_json"] or not result["residuals"]):
            raise RuntimeError(
                "Find_Orb %s run did not produce parseable elements and "
                "residuals" % label)
    orbit_comparison = write_orbit_comparison(orbit_root, baseline_run, augmented_run)
    final_checksums = {str(path): sha256(path) for path in paths}
    if final_checksums != backup["checksums"]:
        raise RuntimeError("one or more original input files changed")
    manifest = {
        "target": args.target, "telescope_key": TELESCOPE_KEY,
        "observatory_code": OBSERVATORY_CODE, "pp_dir": str(pp_dir),
        "backup": backup, "original_checksums_after": final_checksums,
        "originals_unchanged": True, "records": records,
        "registration": registration, "positions_file": str(positions),
        "filter_photometry": str(filter_photometry),
        "combined_photometry": [str(path) for path in combined],
        "source_match_qa": photometry_qa,
        "calibration_qa": [{
            "catalog_token": row["catalog_token"],
            "photometry_catalog": row["photometry_catalog"],
            "band": row["band"], "zeropoint": row["zeropoint"],
            "zeropoint_sig": row["zeropoint_sig"],
            "mag": row["mag"], "mag_sig": row["mag_sig"],
            "calibrated_photometry": row["calibrated_photometry"],
        } for row in photometry_qa],
        "active_observations": len(bundle.observations),
        "ades": {"psv": str(bundle.ades_path), "xml": str(bundle.ades_xml_path),
                 "summary": str(bundle.summary_path), "review": str(validation_path),
                 "submission_ready": False, "submit_script": None,
                 "missing_photometry_rows": sum(obs.astrometry_only for obs in bundle.observations)},
        "initial_positions": {"source": str(initial), "baseline": str(baseline_path),
                              "baseline_count": len(baseline), "removed_count": len(removed)},
        "preliminary_position_comparison": {"path": str(comparison_path),
                                            "rows": comparison},
        "orbit": {"baseline": baseline_run, "augmented": augmented_run,
                  "comparison_json": str(orbit_comparison[0]),
                  "comparison_markdown": str(orbit_comparison[1])},
        "acceptance": {"four_gaia_wcs": all(r["wcs_refined"] for r in records),
                       "four_active_rows": len(bundle.observations) == 4,
                       "four_click_centered_matches": (
                           len(photometry_qa) == 4 and all(
                               row["source_match_separation_arcsec"] < 2.0
                               for row in photometry_qa)),
                       "four_calibrated_r_magnitudes": (
                           len(photometry_qa) == 4 and all(
                               row["calibrated_photometry"] and
                               row["band"] == "r" for row in photometry_qa)),
                       "seven_augmented_rows": True,
                       "two_parseable_orbits": True,
                       "no_live_submission": bundle.submit_script_path is None},
    }
    manifest_path = pp_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(
        manifest, indent=2, sort_keys=True, default=json_default) + "\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("click_points_csv")
    parser.add_argument("--initial-positions", required=True)
    parser.add_argument("--pp-dir")
    parser.add_argument("--target", default="McLeod")
    parser.add_argument("--fixed-aprad", type=float, default=0.0)
    parser.add_argument("--centroid-search-radius", type=float, default=8.0)
    parser.add_argument("--centroid-aperture-radius", type=float, default=3.0)
    parser.add_argument("--centroid-min-snr", type=float, default=1.5)
    parser.add_argument("--find-orb", type=Path, default=DEFAULT_FIND_ORB)
    parser.add_argument("--refresh-backup", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = run_workflow(args)
    print("Wrote %s" % manifest["manifest_path"])
    print("PP observations: %d" % manifest["active_observations"])
    print("ADES review-only PSV: %s" % manifest["ades"]["psv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
