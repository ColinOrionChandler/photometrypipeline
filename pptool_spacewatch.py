#!/usr/bin/env python3

"""Workflow helper for Spacewatch PDS/CATCH image products."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from contextlib import contextmanager
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time


TELESCOPE_KEY = "SPACEWATCH09"
MANIFEST_NAME = "spacewatch_workflow_manifest.json"
WORKING_PREFIX = "spacewatch"
TARGET_TRANSLATION = str.maketrans(" /()", "____")


@contextmanager
def working_directory(path):
    """Temporarily run PP in the image directory."""

    old_cwd = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old_cwd)


def repo_root():
    """Return the photometrypipeline checkout that contains this script."""

    return Path(__file__).resolve().parent


def configure_pp_environment():
    """Set the environment needed before importing PP modules."""

    root = repo_root()
    os.environ["PHOTPIPEDIR"] = str(root)
    bin_dir = str(Path(sys.executable).resolve().parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in path_parts:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_download_rows(downloads_jsonl):
    """Read AstroChipmunk CATCH download rows."""

    rows = []
    with Path(downloads_jsonl).expanduser().open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cutout_key(path_or_name):
    """Return the FITS cutout filename key for a CATCH PNG/FITS path."""

    name = Path(path_or_name).name
    if name.lower().endswith(".png"):
        return name[:-4] + ".fits"
    return name


def select_download_rows(downloads_jsonl, cutouts=None):
    """Select CATCH rows matching explicit cutouts, or all rows if omitted."""

    rows = load_download_rows(downloads_jsonl)
    if not cutouts:
        return sorted(rows, key=lambda row: row.get("obs_time", ""))

    requested = [cutout_key(path) for path in cutouts]
    rows_by_name = {}
    for row in rows:
        for key in (row.get("filename"), row.get("local_path")):
            if key:
                rows_by_name[Path(key).name] = row

    selected = []
    missing = []
    for name in requested:
        row = rows_by_name.get(name)
        if row is None:
            missing.append(name)
        else:
            selected.append(row)

    if missing:
        raise KeyError("cutout(s) not found in %s: %s" %
                       (downloads_jsonl, ", ".join(missing)))
    return sorted(selected, key=lambda row: row.get("obs_time", ""))


def normalize_filter(value):
    """Map Spacewatch archive filter labels onto PP bands."""

    text = str(value or "").strip()
    canonical = text.lower().replace(" ", "").replace("_", "")
    mapping = {
        "schottog-515": "r",
        "schottog515": "r",
        "og-515": "r",
        "og515": "r",
        "r": "r",
    }
    return mapping.get(canonical, text)


def date_from_row(row):
    """Return the UTC date component for one CATCH row."""

    text = row.get("obs_time") or row.get("metadata", {}).get("date")
    if not text:
        raise KeyError("download row has no obs_time/date")
    return text[:10]


def archive_url(row):
    """Return the full-frame archive URL for a CATCH row."""

    url = row.get("metadata", {}).get("archive_url")
    if not url:
        raise KeyError("download row has no metadata.archive_url")
    return url


def raw_full_image_name(row):
    """Return the archive full-frame FITS filename."""

    parsed = urlparse(archive_url(row))
    name = Path(parsed.path).name
    if not name:
        raise ValueError("cannot derive filename from %s" % archive_url(row))
    return name


def download_file(url, destination, refresh=False):
    """Download a source file, reusing a complete local copy by default."""

    destination = Path(destination)
    if destination.exists() and not refresh:
        return {"path": str(destination), "downloaded": False}

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    request = Request(url, headers={"User-Agent": "photometrypipeline/spacewatch"})
    with urlopen(request) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    partial.replace(destination)
    return {"path": str(destination), "downloaded": True}


def science_hdu_index(hdulist):
    """Locate the first image-bearing HDU in a FITS file."""

    for idx, hdu in enumerate(hdulist):
        data = getattr(hdu, "data", None)
        if data is not None and len(data.shape) >= 2:
            return idx
    raise ValueError("no image-bearing HDU found")


def strip_extension_only_cards(header):
    """Remove cards that should not be copied into a primary image HDU."""

    cleaned = header.copy()
    remove_exact = {
        "XTENSION", "EXTNAME", "PCOUNT", "GCOUNT", "CHECKSUM", "DATASUM",
        "THEAP", "TFIELDS", "BZERO", "BSCALE", "BLANK",
    }
    for key in list(cleaned.keys()):
        if key in remove_exact or key.startswith("Z"):
            try:
                cleaned.remove(key, remove_all=True, ignore_missing=True)
            except TypeError:
                cleaned.remove(key)
    return cleaned


def repair_nonfinite_pixels(data):
    """Replace NaN/Inf pixels with the finite median sky level."""

    image = np.asarray(data, dtype=np.float32).copy()
    bad = ~np.isfinite(image)
    n_bad = int(np.count_nonzero(bad))
    if n_bad == 0:
        return image, n_bad, None

    good = image[~bad]
    fill_value = float(np.median(good)) if good.size else 0.0
    image[bad] = fill_value
    return image, n_bad, fill_value


def normalize_spacewatch_wcs(header):
    """Use explicit TPV CTYPE values when Spacewatch PV terms are present."""

    has_pv = any(key.startswith("PV") and "_" in key for key in header)
    if not has_pv:
        return
    if header.get("CTYPE1") == "RA---TAN":
        header["CTYPE1"] = ("RA---TPV", "PP: Spacewatch PV distortion WCS")
    if header.get("CTYPE2") == "DEC--TAN":
        header["CTYPE2"] = ("DEC--TPV", "PP: Spacewatch PV distortion WCS")


def midtime_values(header):
    """Return MJD start, JD midtime, and ISO start time from FITS headers."""

    date_obs = str(header["DATE-OBS"]).replace(" ", "T")
    exptime = float(header.get("EXPTIME", 0.0) or 0.0)
    start = Time(date_obs, format="isot", scale="utc")
    mid = start + (exptime / 2.0) * u.second
    return float(start.mjd), float(mid.jd), start.utc.isot


def convert_to_pp_fits(raw_path, working_path, original_index, target, row):
    """Create a PP-ready Spacewatch FITS while preserving archive WCS."""

    with fits.open(str(raw_path), memmap=False,
                   ignore_missing_end=True) as hdulist:
        hdulist.verify("fix")
        hdu_index = science_hdu_index(hdulist)
        image_hdu = hdulist[hdu_index]
        header = strip_extension_only_cards(image_hdu.header)
        image, n_bad, fill_value = repair_nonfinite_pixels(image_hdu.data)
        normalize_spacewatch_wcs(header)

        mjd_obs, mid_jd, date_obs = midtime_values(header)
        raw_filter = row.get("filter_name") or header.get("FILTER")

        header["PPINSTRU"] = (TELESCOPE_KEY, "PP telescope override")
        header["FILTER"] = (raw_filter, "PP archive filter label")
        header["OBJECT"] = (target, "PP target name")
        header["DATE-OBS"] = (date_obs, "exposure start UTC")
        header["MJD-OBS"] = (mjd_obs, "exposure start MJD")
        header["MIDTIMJD"] = (mid_jd, "exposure midtime JD")
        header["ORIGFILE"] = (Path(raw_path).name[:68],
                              "source full-frame FITS filename")
        header["ORIGEXT"] = (hdu_index, "source FITS extension")
        header["PPSEQ"] = (original_index, "Spacewatch workflow index")
        header["CATCHRA"] = (float(row["ra_deg"]), "CATCH target RA deg")
        header["CATCHDEC"] = (float(row["dec_deg"]), "CATCH target Dec deg")
        header["NANNPIX"] = (n_bad, "non-finite pixels replaced for PP")
        if fill_value is not None:
            header["NANFILL"] = (fill_value, "replacement value for NaNs")

        fits.PrimaryHDU(data=image, header=header).writeto(
            str(working_path), overwrite=True, output_verify="silentfix")

    with fits.open(str(working_path), ignore_missing_end=True) as hdulist:
        header = hdulist[0].header
        return {
            "raw": str(raw_path),
            "working": str(working_path),
            "working_name": working_path.name,
            "source_hdu": hdu_index,
            "image_index": original_index - 1,
            "filter": header.get("FILTER"),
            "band": normalize_filter(header.get("FILTER")),
            "mjd_obs": float(header["MJD-OBS"]),
            "midtimjd": float(header["MIDTIMJD"]),
            "date_obs": header.get("DATE-OBS"),
            "exptime": float(header["EXPTIME"]),
            "magzp": header.get("MAGZP"),
            "nan_pixels_replaced": int(header.get("NANNPIX", 0)),
            "nan_fill": header.get("NANFILL"),
            "target_position": {
                "ra_deg": float(row["ra_deg"]),
                "dec_deg": float(row["dec_deg"]),
            },
            "download_row": {
                "filename": row.get("filename"),
                "image_id": row.get("image_id"),
                "archive_url": archive_url(row),
                "cutout_url": row.get("url"),
                "obs_time": row.get("obs_time"),
            },
        }


def positions_filename(target):
    """Return the PP positions filename for a target."""

    return "positions_%s.dat" % target.translate(TARGET_TRANSLATION)


def target_photometry_filename(target):
    """Return PP's photometry filename for a target name."""

    return "photometry_%s.dat" % target.translate(TARGET_TRANSLATION)


def write_positions_file(image_dir, records, target):
    """Write PP -positions input for one Spacewatch date/band group."""

    path = Path(image_dir) / positions_filename(target)
    ident = target.translate(TARGET_TRANSLATION)
    with path.open("w") as handle:
        for record in records:
            pos = record["target_position"]
            handle.write("%s %.10f %.10f %.8f %s\n" %
                         (record["working_name"], pos["ra_deg"],
                          pos["dec_deg"], record["midtimjd"], ident))
    return path


def remove_stale_target_output(image_dir, target):
    """Remove stale PP photometry output before a fresh run."""

    photometry_path = Path(image_dir) / target_photometry_filename(target)
    if photometry_path.exists():
        photometry_path.unlink()


def run_pp_pipeline(image_dir, records, target, posfile,
                    fixed_aprad=0.0, rejectionfilter="pos"):
    """Run PP on one date/band group with kept Spacewatch WCS."""

    configure_pp_environment()
    import pp_run

    working_names = [record["working_name"] for record in records]
    with working_directory(image_dir):
        pp_run.run_the_pipeline(
            working_names, target, None, float(fixed_aprad), "high",
            False, False, False, True, telescope=TELESCOPE_KEY,
            posfile=Path(posfile).name, rejectionfilter=rejectionfilter)


def group_key(row):
    """Return the PP/date/band grouping for one CATCH row."""

    return date_from_row(row), normalize_filter(row.get("filter_name"))


def prepare_records(spacewatch_dir, pp_root, rows, target,
                    refresh_downloads=False):
    """Download full frames and write PP-ready FITS/position files."""

    records_by_group = {}
    group_counts = {}
    for row in rows:
        date, band = group_key(row)
        image_dir = Path(pp_root) / date / band
        image_dir.mkdir(parents=True, exist_ok=True)

        group_counts[(date, band)] = group_counts.get((date, band), 0) + 1
        group_index = group_counts[(date, band)]
        raw_path = image_dir / raw_full_image_name(row)
        download = download_file(
            archive_url(row), raw_path, refresh=refresh_downloads)
        working_path = image_dir / ("%s_%s_%04d.fits" %
                                    (WORKING_PREFIX, band, group_index))
        record = convert_to_pp_fits(
            raw_path, working_path, group_index, target, row)
        record["download"] = download
        record["image_dir"] = str(image_dir)
        record["spacewatch_dir"] = str(spacewatch_dir)
        records_by_group.setdefault((date, band, image_dir), []).append(record)

    outputs = []
    for (date, band, image_dir), records in sorted(records_by_group.items()):
        positions_path = write_positions_file(image_dir, records, target)
        outputs.append({
            "date": date,
            "band": band,
            "image_dir": str(image_dir),
            "positions_file": str(positions_path),
            "records": records,
        })
    return outputs


def write_manifest(pp_root, manifest):
    """Write the Spacewatch workflow manifest JSON."""

    path = Path(pp_root) / MANIFEST_NAME
    with path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def build_workflow(spacewatch_dir, target, cutouts=None, output_root=None,
                   refresh_downloads=False):
    """Prepare Spacewatch full-frame products for PP."""

    spacewatch_dir = Path(spacewatch_dir).expanduser().resolve()
    if not spacewatch_dir.is_dir():
        raise NotADirectoryError(str(spacewatch_dir))

    downloads_jsonl = spacewatch_dir / "downloads.jsonl"
    rows = select_download_rows(downloads_jsonl, cutouts=cutouts)
    pp_root = (Path(output_root).expanduser().resolve()
               if output_root is not None else spacewatch_dir / "PP")
    groups = prepare_records(spacewatch_dir, pp_root, rows, target,
                             refresh_downloads=refresh_downloads)

    manifest = {
        "workflow": "spacewatch",
        "target": target,
        "telescope": TELESCOPE_KEY,
        "spacewatch_dir": str(spacewatch_dir),
        "downloads_jsonl": str(downloads_jsonl),
        "pp_root": str(pp_root),
        "selected_cutouts": [str(path) for path in (cutouts or [])],
        "groups": groups,
        "header_magzp_policy": "ignored_by_default",
    }
    manifest_path = write_manifest(pp_root, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def run_workflow(manifest, fixed_aprad=0.0, rejectionfilter="pos"):
    """Run PP for every prepared Spacewatch date/band group."""

    outputs = []
    for group in manifest["groups"]:
        image_dir = Path(group["image_dir"])
        remove_stale_target_output(image_dir, manifest["target"])
        run_pp_pipeline(
            image_dir, group["records"], manifest["target"],
            group["positions_file"], fixed_aprad=fixed_aprad,
            rejectionfilter=rejectionfilter)
        photometry_path = image_dir / target_photometry_filename(
            manifest["target"])
        group_output = {
            "date": group["date"],
            "band": group["band"],
            "image_dir": str(image_dir),
            "positions_file": group["positions_file"],
            "photometry_file": str(photometry_path),
            "working_names": [record["working_name"]
                              for record in group["records"]],
        }
        outputs.append(group_output)
    manifest["pp_outputs"] = outputs
    from pptool_pp_cutouts import record_pp_cutouts
    record_pp_cutouts(manifest, manifest["pp_root"], manifest["target"])
    manifest_path = write_manifest(manifest["pp_root"], manifest)
    manifest["manifest_path"] = str(manifest_path)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="prepare and run PP for Spacewatch PDS/CATCH products")
    parser.add_argument("spacewatch_dir",
                        help="CATCH Spacewatch directory with downloads.jsonl")
    parser.add_argument("-target", default="2006 TN147",
                        help="target name to use in PP outputs")
    parser.add_argument("--cutout", action="append", default=[],
                        help="specific CATCH cutout PNG/FITS to include")
    parser.add_argument("--output-root",
                        help="PP output root; defaults to spacewatch_dir/PP")
    parser.add_argument("--refresh-downloads", action="store_true",
                        help="re-download existing full-frame FITS files")
    parser.add_argument("--fixed-aprad", type=float, default=0.0,
                        help="fixed PP aperture radius; 0 uses curve of growth")
    parser.add_argument("--reject", default="pos",
                        help="PP target rejection schema, e.g. pos or none")
    parser.add_argument("--prepare-only", action="store_true",
                        help="write PP-ready FITS/positions but do not run PP")

    args = parser.parse_args(argv)

    manifest = build_workflow(
        args.spacewatch_dir, args.target, cutouts=args.cutout,
        output_root=args.output_root,
        refresh_downloads=args.refresh_downloads)

    if not args.prepare_only:
        run_workflow(manifest, fixed_aprad=args.fixed_aprad,
                     rejectionfilter=args.reject)

    print(json.dumps({
        "manifest_path": manifest["manifest_path"],
        "pp_root": manifest["pp_root"],
        "n_records": sum(len(group["records"])
                         for group in manifest["groups"]),
        "groups": [{
            "date": group["date"],
            "band": group["band"],
            "image_dir": group["image_dir"],
            "positions_file": group["positions_file"],
        } for group in manifest["groups"]],
        "pp_outputs": manifest.get("pp_outputs", []),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
