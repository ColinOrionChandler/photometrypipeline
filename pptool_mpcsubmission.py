#!/usr/bin/env python3

"""Build ADES PSV and MPC 80-column submission files from PP photometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time


DEFAULT_ASTCAT = "Gaia2"
DEFAULT_CONTACT = "coc123@uw.edu"
ALWAYS_ADES_CONTACTS = ("murtagh@uw.edu",)
DEFAULT_AC2_CONTACTS = "coc123@uw.edu, murtagh@uw.edu"
DEFAULT_MEASURER = "C. O. Chandler, J. Murtagh"
DEFAULT_OBSERVATORY_CODE = "W84"
DEFAULT_ACK_SUFFIX = "Small Body Search and Rescue"
DEFAULT_SUBMITTER = "C. O. Chandler"
DEFAULT_PROGRAM_CODE_CONTACT = "C. O. Chandler"
DEFAULT_SBSAR_PROGRAM_CODES_CSV = (
    Path.home() / "GitHub" / "SBSAR" / "sbsar_program_codes.csv"
)
DEFAULT_PROGRAM_CODES_URL = (
    "https://www.minorplanetcenter.net/static/downloadable-files/"
    "program_codes.json"
)
DEFAULT_PROGRAM_CODES_CACHE = (
    Path(__file__).resolve().parent / ".cache" / "mpc_program_codes.json"
)
DEFAULT_MPC_XML_LIVE_ENDPOINT = "https://minorplanetcenter.net/submit_xml"
DEFAULT_MPC_XML_OBJ_TYPE = "tno"
DEFAULT_ADES_SCHEMA_URL = (
    "https://raw.githubusercontent.com/IAU-ADES/ADES-Master/master/"
    "xsd/submit.xsd"
)
DEFAULT_ADES_CATALOG_VALUES_URL = (
    "https://www.minorplanetcenter.net/iau/info/astCat_photCat.json"
)
DEFAULT_ADES_SCHEMA_CACHE = (
    Path(__file__).resolve().parent / ".cache" / "ades_submit.xsd"
)
DEFAULT_ADES_CATALOG_VALUES_CACHE = (
    Path(__file__).resolve().parent / ".cache" / "astCat_photCat.json"
)
ADES_VERSION = "2022"
VALID_ADES_CATALOGS_FALLBACK = {
    "Gaia_Int", "PS1_DR2", "PS1_DR1", "ATLAS2", "Gaia3", "Gaia3E",
    "Gaia2", "Gaia1", "Gaia2016", "URAT1", "UCAC5", "UCAC4", "PPMXL",
    "NOMAD", "2MASS", "UBSC",
}
DEPRECATED_ADES_CATALOGS_FALLBACK = {
    "UCAC3", "UCAC2", "UCAC1", "USNOB1", "USNOA2", "USNOSA2",
    "USNOA1", "USNOSA1", "Tyc2", "Tyc1", "Hip2", "Hip1", "ACT",
    "GSCACT", "GSC2.3", "GSC2.2", "GSC1.2", "GSC1.1", "GSC1.0",
    "GSC", "SDSS8", "SDSS7", "CMC15", "CMC14", "SSTRC4", "SSTRC1",
    "MPOSC3", "PPM", "AC", "SAO1984", "SAO", "AGK3", "FK4",
    "ACRS", "LickGas", "Ida93", "Perth70", "COSMOS", "Yale",
    "ZZCAT", "IHW", "GZ", "UNK",
}
ADES_CATALOG_ALIASES = {
    "PANSTARRS": "PS1_DR1",
    "PAN-STARRS": "PS1_DR1",
    "PANSTARRS1": "PS1_DR1",
    "PAN-STARRS1": "PS1_DR1",
    "SDSS-R9": "SDSS8",
    "SDSS_R9": "SDSS8",
}
NON_CATALOG_PHOTCAT_VALUES = {
    "forced_photometry", "manual_zp", "manual-zp", "header_zp",
    "pp_or_header_zp", "PHOT_C", "MAGZP", "SDSS-R13", "SDSS_R13",
}
NON_SURVEY_OBSNOTE = "Z"
NON_SURVEY_OBSNOTE_CODES = {"F51", "F52", "G96", "I41", "703"}
ORBIT_SOLVER_INPUT_MANIFEST_SUFFIX = "orbit_solver_inputs.json"
ORBIT_SOLVER_PARENT_DIRS = ("orbit_solver_runs", "find_orb_runs")
ORBIT_SOLVER_DIR_SPECS = {
    "mpfit_runs": {
        "solver": "MPFit",
        "slug": "mpfit",
        "success_suffixes": (".fit", ".abg", ".aei", ".xv"),
    },
    "bandk_runs": {
        "solver": "BandK2000",
        "slug": "bandk2000",
        "success_suffixes": (".abg", ".aei", ".fit"),
    },
    "bandk2000_runs": {
        "solver": "BandK2000",
        "slug": "bandk2000",
        "success_suffixes": (".abg", ".aei", ".fit"),
    },
    "bandk_2000_runs": {
        "solver": "BandK2000",
        "slug": "bandk2000",
        "success_suffixes": (".abg", ".aei", ".fit"),
    },
}
ADES_XML_FIELD_ORDER = (
    "permID",
    "provID",
    "artSat",
    "trkSub",
    "obsSubID",
    "mode",
    "stn",
    "sys",
    "ctr",
    "pos1",
    "pos2",
    "pos3",
    "vel1",
    "vel2",
    "vel3",
    "posCov11",
    "posCov12",
    "posCov13",
    "posCov22",
    "posCov23",
    "posCov33",
    "obsTime",
    "rmsTime",
    "ra",
    "dec",
    "rmsRA",
    "rmsDec",
    "rmsCorr",
    "astCat",
    "mag",
    "rmsMag",
    "band",
    "fltr",
    "photCat",
    "photAp",
    "logSNR",
    "seeing",
    "exp",
    "rmsFit",
    "nStars",
    "disc",
    "uncTime",
    "notes",
    "remarks",
)
ADES_XML_CONTEXT_ORDER = (
    "observatory",
    "submitter",
    "observers",
    "measurers",
    "telescope",
    "software",
    "coinvestigators",
    "collaborators",
    "fundingSource",
    "comment",
)

PHOTOMETRY_COLUMNS = (
    "catalog_token",
    "julian_date",
    "mag",
    "mag_sig",
    "ra_deg",
    "dec_deg",
    "pred_minus_ra_arcsec",
    "pred_minus_dec_arcsec",
    "manual_ra_offset_arcsec",
    "manual_dec_offset_arcsec",
    "exptime",
    "zeropoint",
    "zeropoint_sig",
    "inst_mag",
    "inst_mag_sig",
    "phot_cat",
    "band",
    "sextractor_flag",
    "instrument",
    "photometry_method",
    "fwhm",
    "mag_src_sig",
    "mag_cal_sig",
    "mag_tot_sig",
    "ra_src_sig",
    "dec_src_sig",
    "pos_src_sig",
    "ra_ast_sig",
    "dec_ast_sig",
    "pos_ast_sig",
    "ra_tot_sig",
    "dec_tot_sig",
    "pos_tot_sig",
)

CENTURY_CODES = {
    18: "I",
    19: "J",
    20: "K",
    21: "L",
}

PACKED_SEQUENCE_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class FitsMetadata:
    filename: Path
    observer_header: str | None = None
    observers: list[str] = field(default_factory=list)
    propid: str | None = None
    proposer: str | None = None
    telescope: str | None = None
    instrument: str | None = None
    telescope_keyword: str | None = None


@dataclass
class PhotometryObservation:
    source_file: Path
    photometry_file: Path
    catalog_token: str
    julian_date: float
    mag: float | None
    mag_sig: float | None
    ra_deg: float
    dec_deg: float
    exptime: float
    phot_cat: str
    photometry_method: str
    band: str
    instrument: str
    fwhm: float | None
    ra_src_sig: float | None
    dec_src_sig: float | None
    pos_src_sig: float | None
    ra_tot_sig: float | None
    dec_tot_sig: float | None
    pos_tot_sig: float | None
    metadata: FitsMetadata

    @property
    def obs_time(self) -> Time:
        return Time(self.julian_date, format="jd", scale="utc")

    @property
    def astrometry_only(self) -> bool:
        return self.mag is None or self.mag_sig is None

    @property
    def photometric_calibration_source(self) -> str:
        return self.phot_cat


@dataclass
class SubmissionConfig:
    target: str
    observatory_code: str = DEFAULT_OBSERVATORY_CODE
    astcat: str = DEFAULT_ASTCAT
    photcat: str | None = None
    measurer: str = DEFAULT_MEASURER
    submitter: str = DEFAULT_SUBMITTER
    contact: str = DEFAULT_CONTACT
    observers: list[str] | None = None
    prog: str | None = None
    program_code_contact: str = DEFAULT_PROGRAM_CODE_CONTACT
    program_codes_cache: Path | None = None
    program_codes_sbsar_csv: Path | None = None
    refresh_program_codes: bool = False
    non_survey_measurer: bool = False
    output_dir: Path | None = None
    strict: bool = False


@dataclass
class SubmissionBundle:
    observations: list[PhotometryObservation]
    warnings: list[str]
    ades_text: str
    ades_xml_text: str
    obs80_text: str
    summary_text: str
    submit_script_text: str
    ades_path: Path
    ades_xml_path: Path
    obs80_path: Path
    summary_path: Path
    submit_script_path: Path
    orbit_solver_input_snapshots: list[dict[str, object]] = field(
        default_factory=list)
    orbit_solver_input_manifest_path: Path | None = None


class SubmissionError(RuntimeError):
    """Raised when a submission cannot be built safely."""


def _program_code_cache_path(config: SubmissionConfig | None = None) -> Path:
    if config is not None and config.program_codes_cache is not None:
        return config.program_codes_cache.expanduser()
    return DEFAULT_PROGRAM_CODES_CACHE


def read_sbsar_program_code_entries(
        path: Path | None = None) -> dict[str, dict[str, str]]:
    """Read local SBSAR site-code program-code rows if available."""

    csv_path = (path or DEFAULT_SBSAR_PROGRAM_CODES_CSV).expanduser()
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        code_map: dict[str, dict[str, str]] = {}
        for row in reader:
            site_code = (row.get("site_code") or "").strip().upper()
            program_code = (row.get("program_code") or "").strip()
            base62_code = (row.get("base62_code") or "").strip()
            if site_code:
                code_map[site_code] = {
                    "program_code": program_code,
                    "base62_code": base62_code,
                }
    return code_map


def read_sbsar_program_code_map(path: Path | None = None) -> dict[str, str]:
    """Read the local SBSAR site-code to MPC1992 program-code CSV if available."""

    return {
        site_code: values["program_code"]
        for site_code, values in read_sbsar_program_code_entries(path).items()
    }


def _load_program_code_cache(cache_path: Path) -> list[dict[str, object]]:
    if not cache_path.exists():
        return []
    data = json.loads(cache_path.read_text())
    if isinstance(data, dict) and isinstance(data.get("program_codes"), list):
        return data["program_codes"]
    if isinstance(data, list):
        return data
    return []


def fetch_program_codes(cache_path: Path | None = None,
                        url: str = DEFAULT_PROGRAM_CODES_URL,
                        refresh: bool = False) -> list[dict[str, object]]:
    """Return MPC program-code rows, caching the JSON source locally."""

    cache = (cache_path or DEFAULT_PROGRAM_CODES_CACHE).expanduser()
    if cache.exists() and not refresh:
        rows = _load_program_code_cache(cache)
        if rows:
            return rows

    with urlopen(url, timeout=30) as response:
        rows = json.load(response)
    if not isinstance(rows, list):
        raise SubmissionError("unexpected MPC program-code JSON structure")

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "source_url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "program_codes": rows,
    }, indent=2, sort_keys=True) + "\n")
    return rows


def lookup_mpc_program_code(observatory_code: str,
                            contact_name: str = DEFAULT_PROGRAM_CODE_CONTACT,
                            cache_path: Path | None = None,
                            refresh: bool = False) -> str | None:
    """Look up one MPC program code by observatory code and contact name."""

    site_code = str(observatory_code).strip().upper()
    contact = str(contact_name).strip().casefold()
    rows = fetch_program_codes(cache_path=cache_path, refresh=refresh)
    for row in rows:
        row_site = str(row.get("observatory_code", "")).strip().upper()
        row_contact = str(row.get("contact_name", "")).strip().casefold()
        if row_site == site_code and row_contact == contact:
            code = str(row.get("program_code", "")).strip()
            return code or None
    return None


def lookup_mpc_program_code_base62(
        observatory_code: str,
        program_code: str | None = None,
        contact_name: str = DEFAULT_PROGRAM_CODE_CONTACT,
        cache_path: Path | None = None,
        refresh: bool = False) -> str | None:
    """Look up an ADES base62 program code from MPC program-code rows."""

    site_code = str(observatory_code).strip().upper()
    code = str(program_code or "").strip()
    contact = str(contact_name).strip().casefold()
    rows = fetch_program_codes(cache_path=cache_path, refresh=refresh)
    for row in rows:
        row_site = str(row.get("observatory_code", "")).strip().upper()
        if row_site != site_code:
            continue
        base62 = str(row.get("program_code_base62", "")).strip()
        row_code = str(row.get("program_code", "")).strip()
        row_contact = str(row.get("contact_name", "")).strip().casefold()
        if code and code in {row_code, base62}:
            return base62 or None
        if not code and row_contact == contact:
            return base62 or None
    return None


def resolve_program_code(config: SubmissionConfig,
                         warnings: list[str] | None = None) -> str | None:
    """Resolve the program code for a submission, preserving explicit input."""

    if config.prog:
        return config.prog

    warn = warnings.append if warnings is not None else (lambda message: None)
    site_code = str(config.observatory_code).strip().upper()
    sbsar_map = read_sbsar_program_code_map(config.program_codes_sbsar_csv)
    if site_code in sbsar_map:
        code = sbsar_map[site_code].strip()
        if code:
            return code
        warn("local SBSAR program-code row for %s is blank" % site_code)

    try:
        code = lookup_mpc_program_code(
            site_code,
            contact_name=config.program_code_contact,
            cache_path=_program_code_cache_path(config),
            refresh=config.refresh_program_codes)
    except (OSError, URLError, TimeoutError, json.JSONDecodeError,
            SubmissionError) as exc:
        warn("could not refresh MPC program-code cache: %s" % exc)
        try:
            code = lookup_mpc_program_code(
                site_code,
                contact_name=config.program_code_contact,
                cache_path=_program_code_cache_path(config),
                refresh=False)
        except Exception:
            code = None

    if code:
        return code
    warn("no program code found for observatory %s and contact %s" %
         (site_code, config.program_code_contact))
    return None


def resolve_ades_program_code(config: SubmissionConfig,
                              warnings: list[str] | None = None) -> str | None:
    """Return the ADES base62 prog value while preserving 80-column prog."""

    if not config.prog:
        return None
    warn = warnings.append if warnings is not None else (lambda message: None)
    site_code = str(config.observatory_code).strip().upper()
    code = str(config.prog).strip()
    sbsar_entry = read_sbsar_program_code_entries(
        config.program_codes_sbsar_csv).get(site_code, {})
    if code == sbsar_entry.get("base62_code"):
        return code
    if code == sbsar_entry.get("program_code") and sbsar_entry.get("base62_code"):
        return sbsar_entry["base62_code"]

    try:
        base62 = lookup_mpc_program_code_base62(
            site_code,
            program_code=code,
            contact_name=config.program_code_contact,
            cache_path=_program_code_cache_path(config),
            refresh=config.refresh_program_codes,
        )
    except (OSError, URLError, TimeoutError, json.JSONDecodeError,
            SubmissionError) as exc:
        warn("could not refresh MPC program-code cache for ADES prog: %s" %
             exc)
        base62 = None
    if base62:
        return base62
    if len(code) == 1:
        warn("could not resolve ADES base62 program code for %s/%s; "
             "leaving PSV prog unchanged" % (site_code, code))
    return code


def target_to_filename(target: str) -> str:
    """Return the PP target-file stem used in photometry_<target>.dat."""

    return target.strip().translate(str.maketrans(" /()", "____"))


def normalize_target(target: str) -> str:
    """Normalize a target spelling for ADES display."""

    compact = target.strip().replace("_", " ")
    return re.sub(r"\s+", " ", compact)


def pack_minor_planet_provisional_designation(designation: str) -> str:
    """Pack a minor-planet provisional designation for MPC 80-column output."""

    normalized = normalize_target(designation).upper()
    if re.fullmatch(r"[IJKL]\d{2}[A-HJ-Y][0-9A-Z]{2}[A-Z]", normalized):
        return normalized

    match = re.fullmatch(r"(\d{4})\s*([A-HJ-Y])([A-Z])(\d*)", normalized)
    if match is None:
        raise SubmissionError(
            "cannot pack target designation %r for MPC 80-column output" %
            designation)

    year = int(match.group(1))
    half_month = match.group(2)
    second_letter = match.group(3)
    sequence = int(match.group(4) or "0")
    century = year // 100

    try:
        century_code = CENTURY_CODES[century]
    except KeyError as exc:
        raise SubmissionError("unsupported provisional designation year %d" %
                              year) from exc

    if sequence >= len(PACKED_SEQUENCE_DIGITS) * 10:
        raise SubmissionError("sequence number %d is too large to pack" %
                              sequence)

    packed_sequence = (PACKED_SEQUENCE_DIGITS[sequence // 10] +
                       str(sequence % 10))
    return "%s%02d%s%s%s" % (century_code, year % 100, half_month,
                             packed_sequence, second_letter)


def discover_photometry_files(input_path: Path, target: str) -> list[Path]:
    """Find target photometry files below an input directory or file path."""

    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise SubmissionError("input path does not exist: %s" % input_path)
    if not input_path.is_dir():
        raise SubmissionError("input path is neither a file nor a directory: "
                              "%s" % input_path)

    target_stem = target_to_filename(target)
    files = sorted(input_path.rglob("photometry_%s.dat" % target_stem))
    if files:
        return files

    raise SubmissionError("no photometry_%s.dat files found below %s" %
                          (target_stem, input_path))


def _float_or_none(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _photometry_float_or_none(value: str) -> float | None:
    result = _float_or_none(value)
    if result is None or result >= 90:
        return None
    return result


def parse_photometry_file(filename: Path,
                          require_rows: bool = True) -> list[dict[str, str]]:
    """Read active PP photometry rows from a photometry_<target>.dat file."""

    rows = []
    with filename.open("r") as inf:
        for line in inf:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = line.split()
            if len(values) < 20:
                raise SubmissionError(
                    "photometry row in %s has only %d columns" %
                    (filename, len(values)))
            padded = values + [""] * (len(PHOTOMETRY_COLUMNS) - len(values))
            rows.append(dict(zip(PHOTOMETRY_COLUMNS, padded)))
    if not rows and require_rows:
        raise SubmissionError("no observations found in %s" % filename)
    return rows


def resolve_fits_filename(photometry_file: Path, catalog_token: str) -> Path:
    """Find the FITS image that produced one PP photometry row."""

    photometry_dir = photometry_file.parent
    token_path = Path(catalog_token)
    candidates = []
    if token_path.suffix.lower() != ".ldac":
        candidates.append(token_path)
    candidates.extend([
        token_path.with_suffix(".fits") if token_path.suffix else
        Path(catalog_token + ".fits"),
        Path(catalog_token.replace(".ldac", ".fits")),
    ])
    if token_path.suffix.lower() != ".ldac":
        candidates.append(photometry_dir / catalog_token)
    candidates.extend([
        photometry_dir / (catalog_token + ".fits"),
        photometry_dir / catalog_token.replace(".ldac", ".fits"),
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    fits_suffixes = (".fits", ".fit", ".fts")
    token_prefixes = [
        catalog_token,
        catalog_token.replace(".ldac", ""),
        Path(catalog_token).name,
        Path(catalog_token).name.replace(".ldac", ""),
    ]
    for prefix in dict.fromkeys(token_prefixes):
        if not prefix:
            continue
        prefixed = sorted(
            path for path in photometry_dir.iterdir()
            if path.suffix.lower() in fits_suffixes and
            path.name.startswith(prefix))
        if len(prefixed) == 1:
            return prefixed[0].resolve()

    fits_candidates = sorted(
        path for path in photometry_dir.iterdir()
        if path.suffix.lower() in fits_suffixes)
    if len(fits_candidates) == 1:
        return fits_candidates[0].resolve()

    raise SubmissionError("cannot identify FITS file for %s row %s" %
                          (photometry_file, catalog_token))


def _first_header_with_metadata(hdulist: fits.HDUList) -> fits.Header:
    metadata_keys = {"OBSERVER", "TELESCOP", "INSTRUME", "TEL_KEYW",
                     "PROPID", "PROPOSER", "DATE-OBS", "MIDTIMJD"}
    for hdu in hdulist:
        if metadata_keys & set(hdu.header.keys()):
            return hdu.header
    return hdulist[0].header


def split_observer_names(observer_header: str | None) -> list[str]:
    """Split a FITS OBSERVER value into ADES-style individual names."""

    if observer_header is None:
        return []
    parts = re.split(r"\s*,\s*|\s*;\s*", observer_header.strip())
    return [part for part in parts if part]


def _format_observer_name(name: str) -> str:
    """Return MPC observer style for names written as first/middle/last."""

    parts = name.strip().split()
    if len(parts) < 2:
        return name.strip()
    if parts[-1].casefold() in {"team", "collaboration", "survey"}:
        return name.strip()
    if all(len(part.rstrip(".")) == 1 for part in parts[:-1]):
        return " ".join(parts)
    initials = ["%s." % part[0] for part in parts[:-1] if part]
    return " ".join(initials + [parts[-1]])


def read_fits_metadata(filename: Path) -> FitsMetadata:
    """Extract submission metadata from a FITS image header."""

    with fits.open(filename, ignore_missing_end=True, memmap=False,
                   verify="silentfix") as hdulist:
        header = _first_header_with_metadata(hdulist)
        observer_header = header.get("OBSERVER") or header.get("OBSERVERS")
        return FitsMetadata(
            filename=filename,
            observer_header=observer_header,
            observers=split_observer_names(observer_header),
            propid=header.get("PROPID"),
            proposer=header.get("PROPOSER"),
            telescope=header.get("TELESCOP"),
            instrument=header.get("INSTRUME"),
            telescope_keyword=header.get("TEL_KEYW"),
        )


def make_observation(photometry_file: Path,
                     row: dict[str, str]) -> PhotometryObservation:
    """Create a typed observation from one parsed PP photometry row."""

    fits_filename = resolve_fits_filename(photometry_file,
                                          row["catalog_token"])
    metadata = read_fits_metadata(fits_filename)
    return PhotometryObservation(
        source_file=fits_filename,
        photometry_file=photometry_file,
        catalog_token=row["catalog_token"],
        julian_date=float(row["julian_date"]),
        mag=_photometry_float_or_none(row["mag"]),
        mag_sig=_photometry_float_or_none(row["mag_sig"]),
        ra_deg=float(row["ra_deg"]),
        dec_deg=float(row["dec_deg"]),
        exptime=float(row["exptime"]),
        phot_cat=row["phot_cat"],
        photometry_method=row["photometry_method"],
        band=row["band"] if row["band"] != "-" else "C",
        instrument=row["instrument"],
        fwhm=_float_or_none(row["fwhm"]),
        ra_src_sig=_float_or_none(row["ra_src_sig"]),
        dec_src_sig=_float_or_none(row["dec_src_sig"]),
        pos_src_sig=_float_or_none(row["pos_src_sig"]),
        ra_tot_sig=_float_or_none(row["ra_tot_sig"]),
        dec_tot_sig=_float_or_none(row["dec_tot_sig"]),
        pos_tot_sig=_float_or_none(row["pos_tot_sig"]),
        metadata=metadata,
    )


def collect_observations(
        input_path: Path,
        config: SubmissionConfig,
        warnings: list[str] | None = None) -> list[PhotometryObservation]:
    """Collect all observations for a target below an input path."""

    observations = []
    for photometry_file in discover_photometry_files(input_path,
                                                     config.target):
        rows = parse_photometry_file(photometry_file, require_rows=False)
        if not rows:
            if warnings is not None:
                warnings.append("skipping %s; no active observations" %
                                photometry_file)
            continue
        for row in rows:
            observations.append(make_observation(photometry_file, row))
    return sorted(observations, key=lambda obs: obs.julian_date)


def _format_float(value: float | None, decimals: int) -> str:
    if value is None:
        return ""
    return ("%%.%df" % decimals % value).rstrip("0").rstrip(".")


def _format_time(time: Time) -> str:
    time.precision = 3
    return time.utc.isot + "Z"


def _format_80col_date(time: Time) -> str:
    dt = time.utc.to_datetime()
    day_fraction = (
        dt.day +
        dt.hour / 24.0 +
        dt.minute / 1440.0 +
        (dt.second + dt.microsecond / 1000000.0) / 86400.0
    )
    return "%04d %02d %08.5f " % (dt.year, dt.month, day_fraction)


def obs80_observation_note(config: SubmissionConfig) -> str:
    """Return the MPC1992 column-72 observation note for this submission."""

    site_code = config.observatory_code.upper()
    if config.non_survey_measurer and site_code in NON_SURVEY_OBSNOTE_CODES:
        return NON_SURVEY_OBSNOTE
    return " "


def _psv_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "/").strip()


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _read_cached_json(cache_path: Path) -> object | None:
    try:
        if cache_path.exists():
            return json.loads(cache_path.read_text())
    except Exception:
        return None
    return None


def load_ades_catalog_values(
        cache_path: Path | None = None,
        refresh: bool = False,
        url: str = DEFAULT_ADES_CATALOG_VALUES_URL) -> tuple[set[str], set[str]]:
    """Return current ADES astCat/photCat codes, with a bundled fallback."""

    cache = (cache_path or DEFAULT_ADES_CATALOG_VALUES_CACHE).expanduser()
    rows = None if refresh else _read_cached_json(cache)
    if rows is None:
        try:
            with urlopen(url, timeout=30) as response:
                rows = json.load(response)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        except Exception:
            rows = None

    if not isinstance(rows, list):
        return set(VALID_ADES_CATALOGS_FALLBACK), set(
            DEPRECATED_ADES_CATALOGS_FALLBACK)

    active = set()
    deprecated = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = str(row.get("Value") or "").strip()
        if not value:
            continue
        if str(row.get("Deprecated") or "").strip().lower() == "no":
            active.add(value)
        else:
            deprecated.add(value)
    return active or set(VALID_ADES_CATALOGS_FALLBACK), deprecated


def cache_ades_schema(cache_path: Path | None = None,
                      refresh: bool = False,
                      url: str = DEFAULT_ADES_SCHEMA_URL) -> Path:
    """Cache current submit.xsd and return its path."""

    cache = (cache_path or DEFAULT_ADES_SCHEMA_CACHE).expanduser()
    if cache.exists() and not refresh:
        return cache
    with urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text)
    return cache


def normalize_ades_catalog(value: str | None,
                           active: set[str] | None = None,
                           deprecated: set[str] | None = None) -> tuple[str, str | None]:
    """Return an ADES-safe catalog code and optional provenance warning."""

    text = str(value or "").strip()
    if not text:
        return "", None
    active = active or VALID_ADES_CATALOGS_FALLBACK
    deprecated = deprecated or DEPRECATED_ADES_CATALOGS_FALLBACK
    mapped = ADES_CATALOG_ALIASES.get(text, ADES_CATALOG_ALIASES.get(
        text.upper(), text))
    if mapped in active or mapped in deprecated:
        warning = None if mapped == text else "%s mapped to %s" % (
            text, mapped)
        return mapped, warning
    return "", "%s is not an ADES astCat/photCat catalog code" % text


def _photometry_provenance(obs: PhotometryObservation,
                           ades_photcat: str) -> list[str]:
    """Return compact ADES remarks describing non-catalog photometry provenance."""

    remarks = []
    method = (obs.photometry_method or "").strip()
    calibration = (obs.photometric_calibration_source or "").strip()
    if method and method != "APER":
        method_text = method.lower().replace("_", " ")
        if method_text == "aper forced":
            method_text = "forced aperture"
        remarks.append(method_text)
    if calibration and calibration != ades_photcat:
        if calibration == "forced_photometry":
            if obs.metadata.telescope_keyword == "CFHTCFH12K":
                remarks.append("photCal=SDSS9")
            else:
                remarks.append("photCal=manual/header")
        elif calibration in {"SDSS-R9", "SDSS_R9"}:
            remarks.append("photCal=SDSS9")
        elif calibration in {"manual_zp", "manual-zp", "header_zp",
                             "pp_or_header_zp", "PHOT_C", "MAGZP"}:
            remarks.append("photCal=%s" % calibration)
        else:
            remarks.append("PP photCat=%s" % calibration)
    return remarks


def resolve_ades_photcat(obs: PhotometryObservation,
                         config: SubmissionConfig,
                         active: set[str],
                         deprecated: set[str]) -> tuple[str, str | None]:
    """Resolve the ADES photCat field for one non-astrometry-only row."""

    if obs.astrometry_only:
        return "", None
    raw = config.photcat or obs.phot_cat
    if (raw == "forced_photometry" and
            obs.metadata.telescope_keyword == "CFHTCFH12K"):
        raw = "SDSS-R9"
    return normalize_ades_catalog(raw, active, deprecated)


def derive_observers(observations: list[PhotometryObservation],
                     config: SubmissionConfig) -> list[str]:
    """Return CLI-supplied observers or unique FITS-header observers."""

    if config.observers:
        return _unique_preserve_order(
            _format_observer_name(observer) for observer in config.observers)
    return _unique_preserve_order(
        _format_observer_name(observer)
        for obs in observations for observer in obs.metadata.observers)


def split_people(value: str) -> list[str]:
    """Split a comma- or semicolon-separated people field."""

    return [part.strip() for part in re.split(r"\s*,\s*|\s*;\s*", value)
            if part.strip()]


def ades_contact_list(contact: str) -> str:
    """Return ADES contact text with required SBSAR contacts included."""

    contacts = split_people(contact)
    seen = {item.casefold() for item in contacts}
    for required in ALWAYS_ADES_CONTACTS:
        if required.casefold() not in seen:
            contacts.append(required)
            seen.add(required.casefold())
    return ", ".join(contacts)


def derive_measurers(config: SubmissionConfig) -> list[str]:
    """Return individual measurer names from the config field."""

    return split_people(config.measurer)


def derive_telescope_context(observations: list[PhotometryObservation]) -> tuple[str, str, str]:
    """Derive a compact telescope context for ADES and 80-column headers."""

    instruments = {obs.metadata.instrument or obs.instrument
                   for obs in observations}
    telescopes = {obs.metadata.telescope for obs in observations
                  if obs.metadata.telescope}
    telescope_keywords = {obs.metadata.telescope_keyword
                          for obs in observations
                          if obs.metadata.telescope_keyword}
    if "DECam" in instruments or "DECam" in {
            obs.metadata.telescope_keyword for obs in observations}:
        return "4.0-m reflector", "4.0", "CCD"
    if "SDSS" in telescope_keywords:
        return "2.5-m SDSS telescope", "2.5", "CCD"
    if "WHTPFIP" in telescope_keywords or any(
            "William Herschel" in telescope for telescope in telescopes):
        return "4.2-m William Herschel Telescope", "4.2", "CCD"
    if "CFHTCFH12K" in telescope_keywords or any(
            "CFHT" in telescope for telescope in telescopes):
        return "CFHT 3.6m", "3.6", "CCD"
    if "SPACEWATCH09" in telescope_keywords or any(
            "Spacewatch 0.9-m" in telescope for telescope in telescopes):
        return "0.9-m f/3 reflector", "0.9", "CCD"
    if telescopes:
        design = sorted(telescopes)[0]
    else:
        design = "reflector"
    return design, "1.0", "CCD"


def validate_observations(observations: list[PhotometryObservation],
                          config: SubmissionConfig) -> list[str]:
    """Validate required values and return non-fatal warnings."""

    if not observations:
        raise SubmissionError("no observations collected")

    warnings = []
    seen = set()
    for obs in observations:
        if not math.isfinite(obs.julian_date):
            raise SubmissionError("invalid Julian date in %s" %
                                  obs.photometry_file)
        if not math.isfinite(obs.ra_deg) or not math.isfinite(obs.dec_deg):
            raise SubmissionError("invalid coordinates in %s" %
                                  obs.photometry_file)
        if obs.ra_tot_sig is None or obs.ra_tot_sig <= 0:
            if obs.pos_tot_sig is not None and obs.pos_tot_sig > 0:
                warnings.append("using pos_tot_sig for missing RA uncertainty "
                                "in %s" % obs.catalog_token)
                obs.ra_tot_sig = obs.pos_tot_sig
            elif obs.ra_src_sig is not None and obs.ra_src_sig > 0:
                warnings.append("using ra_src_sig for missing RA uncertainty "
                                "in %s" % obs.catalog_token)
                obs.ra_tot_sig = obs.ra_src_sig
            elif obs.pos_src_sig is not None and obs.pos_src_sig > 0:
                warnings.append("using pos_src_sig for missing RA uncertainty "
                                "in %s" % obs.catalog_token)
                obs.ra_tot_sig = obs.pos_src_sig
            else:
                raise SubmissionError("missing positive RA uncertainty for %s" %
                                      obs.catalog_token)
        if obs.dec_tot_sig is None or obs.dec_tot_sig <= 0:
            if obs.pos_tot_sig is not None and obs.pos_tot_sig > 0:
                warnings.append("using pos_tot_sig for missing Dec uncertainty "
                                "in %s" % obs.catalog_token)
                obs.dec_tot_sig = obs.pos_tot_sig
            elif obs.dec_src_sig is not None and obs.dec_src_sig > 0:
                warnings.append("using dec_src_sig for missing Dec uncertainty "
                                "in %s" % obs.catalog_token)
                obs.dec_tot_sig = obs.dec_src_sig
            elif obs.pos_src_sig is not None and obs.pos_src_sig > 0:
                warnings.append("using pos_src_sig for missing Dec uncertainty "
                                "in %s" % obs.catalog_token)
                obs.dec_tot_sig = obs.pos_src_sig
            else:
                raise SubmissionError(
                    "missing positive Dec uncertainty for %s" %
                    obs.catalog_token)

        key = (round(obs.julian_date, 8), round(obs.ra_deg, 8),
               round(obs.dec_deg, 8))
        if key in seen:
            warnings.append("possible duplicate observation at JD %.8f" %
                            obs.julian_date)
        seen.add(key)

    if config.prog and len(config.prog) != 1:
        warnings.append("program code %r will be included in ADES but left "
                        "blank in 80-column output" % config.prog)

    if not derive_observers(observations, config):
        warnings.append("no observers found in FITS headers; use --observer "
                        "to provide them")

    if warnings and config.strict:
        raise SubmissionError("; ".join(warnings))
    return warnings


def format_ades_psv(observations: list[PhotometryObservation],
                    config: SubmissionConfig) -> str:
    """Format ADES PSV text with a minimal submission context."""

    telescope_design, aperture, detector = derive_telescope_context(observations)
    observers = derive_observers(observations, config)
    measurers = derive_measurers(config)
    ades_prog = resolve_ades_program_code(config)
    fieldnames = [
        "permID",
        "provID",
        "trkSub",
        "mode",
        "stn",
        "obsTime",
        "ra",
        "dec",
        "rmsRA",
        "rmsDec",
        "astCat",
        "photCat",
        "mag",
        "rmsMag",
        "band",
        "exp",
        "seeing",
        "prog",
        "remarks",
    ]

    lines = [
        "# version=%s" % ADES_VERSION,
        "# observatory",
        "! mpcCode %s" % config.observatory_code,
        "# submitter",
        "! name %s" % config.submitter,
    ]
    if observers:
        lines.append("# observers")
        for observer in observers:
            lines.append("! name %s" % observer)
    lines.append("# measurers")
    for measurer in measurers:
        lines.append("! name %s" % measurer)
    lines.extend([
        "# telescope",
        "! design %s" % telescope_design,
        "! aperture %s" % aperture,
        "! detector %s" % detector,
        "# comment",
        "! line Contact: %s" % ades_contact_list(config.contact),
        "! line Small Body Search and Rescue (SBSAR) - http://sbsar.net",
        "! line COC Photometry Pipeline",
        "|".join(fieldnames),
    ])

    prov_id = normalize_target(config.target)
    active_catalogs, deprecated_catalogs = load_ades_catalog_values()
    for obs in observations:
        raw_photcat = config.photcat or obs.phot_cat
        photcat, photcat_warning = resolve_ades_photcat(
            obs, config, active_catalogs, deprecated_catalogs)
        remarks = [obs.source_file.name]
        if not obs.astrometry_only:
            remarks.extend(_photometry_provenance(obs, photcat))
        if photcat_warning and raw_photcat not in NON_CATALOG_PHOTCAT_VALUES:
            remarks.append(photcat_warning)
        if obs.metadata.propid:
            remarks.append("PROPID=%s" % obs.metadata.propid)
        values = {
            "permID": "",
            "provID": prov_id,
            "trkSub": "",
            "mode": "CCD",
            "stn": config.observatory_code,
            "obsTime": _format_time(obs.obs_time),
            "ra": _format_float(obs.ra_deg, 8),
            "dec": _format_float(obs.dec_deg, 8),
            "rmsRA": _format_float(obs.ra_tot_sig, 4),
            "rmsDec": _format_float(obs.dec_tot_sig, 4),
            "astCat": config.astcat,
            "photCat": "" if obs.astrometry_only else photcat,
            "mag": _format_float(obs.mag, 1),
            "rmsMag": _format_float(obs.mag_sig, 2),
            "band": "" if obs.astrometry_only else obs.band,
            "exp": _format_float(obs.exptime, 2),
            "seeing": _format_float(obs.fwhm, 2),
            "prog": ades_prog or "",
            "remarks": "; ".join(remarks),
        }
        lines.append("|".join(_psv_value(values[name]) for name in fieldnames))
    return "\n".join(lines) + "\n"


def _parse_psv_table(text: str) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames = []
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.lstrip().startswith("!"):
            continue
        if "|" not in line:
            continue
        values = [value.strip() for value in line.split("|")]
        if not fieldnames:
            fieldnames = values
            continue
        padded = values + [""] * (len(fieldnames) - len(values))
        rows.append(dict(zip(fieldnames, padded)))
    return fieldnames, rows


def _parse_psv_context(text: str) -> tuple[str, dict[str, list[tuple[str, str]]]]:
    version = ADES_VERSION
    context: dict[str, list[tuple[str, str]]] = {}
    section = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            directive = stripped[1:].strip()
            if directive.startswith("version="):
                version = directive.split("=", 1)[1].strip()
                section = "version"
            else:
                section = directive
                context.setdefault(section, [])
            continue
        if stripped.startswith("!") and section is not None:
            payload = stripped[1:].strip()
            parts = payload.split(None, 1)
            key = parts[0]
            value = parts[1].strip() if len(parts) > 1 else ""
            context.setdefault(section, []).append((key, value))
    return version, context


def format_ades_xml_from_psv(text: str) -> str:
    """Convert generated ADES PSV text to schema-compatible ADES XML."""

    version, context_sections = _parse_psv_context(text)
    _, rows = _parse_psv_table(text)
    root = ET.Element("ades", {"version": version})
    obs_block = ET.SubElement(root, "obsBlock")
    context = ET.SubElement(obs_block, "obsContext")
    for section in ADES_XML_CONTEXT_ORDER:
        entries = context_sections.get(section, [])
        if not entries:
            continue
        if section == "fundingSource":
            ET.SubElement(context, section).text = entries[0][1]
            continue
        section_element = ET.SubElement(context, section)
        for key, value in entries:
            ET.SubElement(section_element, key).text = value

    obs_data = ET.SubElement(obs_block, "obsData")
    for row in rows:
        optical = ET.SubElement(obs_data, "optical")
        for field_name in ADES_XML_FIELD_ORDER:
            value = row.get(field_name, "")
            if value == "":
                continue
            ET.SubElement(optical, field_name).text = value
    ET.indent(root, space="  ")
    return ET.tostring(
        root, encoding="utf-8", xml_declaration=True).decode("utf-8") + "\n"


def validate_ades_psv_text(
        text: str,
        catalog_values: tuple[set[str], set[str]] | None = None) -> dict[str, list[str]]:
    """Validate compact ADES PSV rules used by MPC submit.xsd."""

    errors = []
    warnings = []
    active_catalogs, deprecated_catalogs = (
        catalog_values if catalog_values is not None
        else load_ades_catalog_values())

    version_lines = [
        line.strip() for line in text.splitlines()
        if line.strip().startswith("# version=")
    ]
    if not version_lines:
        errors.append("missing # version=%s line" % ADES_VERSION)
    elif version_lines[0] != "# version=%s" % ADES_VERSION:
        errors.append("ADES version must be %s, found %s" %
                      (ADES_VERSION, version_lines[0].split("=", 1)[-1]))

    aperture_lines = [
        line.strip() for line in text.splitlines()
        if line.strip().startswith("! aperture")
    ]
    if not aperture_lines:
        errors.append("missing telescope aperture")
    else:
        try:
            aperture = float(aperture_lines[0].split(None, 2)[2])
        except Exception:
            aperture = None
        if aperture is None or aperture <= 0:
            errors.append("telescope aperture must be positive")

    fieldnames, rows = _parse_psv_table(text)
    if not fieldnames:
        errors.append("missing PSV field header")
        return {"errors": errors, "warnings": warnings}

    cat_pattern = re.compile(r"^[.A-Za-z0-9_]{0,8}$")
    for row_index, row in enumerate(rows, start=1):
        astcat = row.get("astCat", "")
        if not astcat:
            errors.append("row %d astCat is required" % row_index)
        elif not cat_pattern.match(astcat):
            errors.append("row %d astCat has invalid CatType syntax: %s" %
                          (row_index, astcat))
        elif astcat not in active_catalogs:
            if astcat in deprecated_catalogs:
                warnings.append("row %d astCat is deprecated: %s" %
                                (row_index, astcat))
            else:
                errors.append("row %d astCat is not a current MPC catalog: %s" %
                              (row_index, astcat))

        photcat = row.get("photCat", "")
        mag = row.get("mag", "")
        rms_mag = row.get("rmsMag", "")
        band = row.get("band", "")
        if photcat:
            if not cat_pattern.match(photcat):
                errors.append(
                    "row %d photCat has invalid CatType syntax: %s" %
                    (row_index, photcat))
            elif photcat not in active_catalogs:
                if photcat in deprecated_catalogs:
                    warnings.append("row %d photCat is deprecated: %s" %
                                    (row_index, photcat))
                else:
                    errors.append(
                        "row %d photCat is not a current MPC catalog: %s" %
                        (row_index, photcat))
        if mag or rms_mag or band or photcat:
            if not mag:
                errors.append("row %d has photometry fields but no mag" %
                              row_index)
            if not band:
                errors.append("row %d has photometry fields but no band" %
                              row_index)
        if rms_mag and not mag:
            errors.append("row %d has rmsMag without mag" % row_index)
        if mag:
            try:
                mag_value = float(mag)
            except ValueError:
                errors.append("row %d mag is not numeric: %s" %
                              (row_index, mag))
            else:
                if mag_value < -5 or mag_value > 35:
                    errors.append("row %d mag is outside ADES range: %s" %
                                  (row_index, mag))
        if (row.get("remarks") and "photCal=" in row["remarks"] and
                not photcat):
            warnings.append(
                "row %d cites non-catalog photometric provenance in remarks" %
                row_index)

    return {"errors": errors, "warnings": warnings}


def validate_ades_xml_text(
        text: str,
        schema_path: Path | None = None) -> dict[str, list[str]]:
    """Validate generated ADES XML against the current submit.xsd schema."""

    errors = []
    warnings = []
    if not text.lstrip().startswith("<"):
        return {
            "errors": ["ADES XML does not start with '<'"],
            "warnings": warnings,
        }
    try:
        from lxml import etree
    except Exception as exc:
        return {
            "errors": errors,
            "warnings": ["lxml unavailable; skipped ADES XML schema validation: %s" % exc],
        }

    try:
        schema_file = schema_path or cache_ades_schema()
        schema_text = Path(schema_file).expanduser().read_text()
    except Exception as exc:
        return {
            "errors": errors,
            "warnings": ["could not load ADES XML schema; skipped validation: %s" % exc],
        }
    if not schema_text.lstrip().startswith("<"):
        return {
            "errors": ["ADES XML schema file does not start with '<'"],
            "warnings": warnings,
        }

    try:
        schema = etree.XMLSchema(etree.XML(schema_text.encode("utf-8")))
        schema.assertValid(etree.XML(text.encode("utf-8")))
    except Exception as exc:
        for entry in getattr(exc, "error_log", []) or []:
            errors.append("XML schema validation failed: %s" % entry.message)
        if not errors:
            errors.append("XML schema validation failed: %s" % exc)
    return {"errors": errors, "warnings": warnings}


def format_80col_observation(obs: PhotometryObservation,
                             packed_target: str,
                             config: SubmissionConfig) -> str:
    """Format one MPC1992 80-column observation."""

    coord = SkyCoord(ra=obs.ra_deg * u.degree, dec=obs.dec_deg * u.degree,
                     frame="icrs")
    ra = coord.ra.to_string(unit=u.hour, sep=" ", precision=2, pad=True)
    dec = coord.dec.to_string(unit=u.degree, sep=" ", precision=1,
                              alwayssign=True, pad=True)
    date = _format_80col_date(obs.obs_time)
    band = (obs.band or "C")[:1]
    mag_band = "       " if obs.astrometry_only else "%5.1f %1s" % (
        obs.mag, band)
    program_code = config.prog if config.prog and len(config.prog) == 1 else " "
    observation_note = obs80_observation_note(config)

    line = (
        "%5s%-7s  C" % ("", packed_target) +
        date +
        "%s " % ra +
        "%s" % dec +
        "         " +
        mag_band +
        "%s%4s%s%3s" %
        (observation_note, "", program_code, config.observatory_code)
    )
    if len(line) != 80:
        raise SubmissionError("internal 80-column formatter produced %d "
                              "columns: %r" % (len(line), line))
    return line


def format_obs80(observations: list[PhotometryObservation],
                 config: SubmissionConfig) -> str:
    """Format MPC1992 80-column text, including traditional headers."""

    packed_target = pack_minor_planet_provisional_designation(config.target)
    observers = derive_observers(observations, config)
    measurers = derive_measurers(config)
    telescope_design, _, detector = derive_telescope_context(observations)
    bands = _unique_preserve_order(obs.band for obs in observations)

    lines = [
        "COD %s" % config.observatory_code,
        "CON %s [%s]" % (config.submitter, config.contact),
    ]
    if observers:
        lines.append("OBS %s" % ", ".join(observers))
    lines.extend([
        "MEA %s" % ", ".join(measurers),
        "TEL %s + %s" % (telescope_design, detector),
        "NET %s" % config.astcat,
        "BND %s" % ",".join(bands),
        "NUM %d" % len(observations),
        "ACK %s %s" % (normalize_target(config.target), DEFAULT_ACK_SUFFIX),
        "AC2 %s" % DEFAULT_AC2_CONTACTS,
        "",
    ])

    lines.extend(format_80col_observation(obs, packed_target, config)
                 for obs in observations)
    return "\n".join(lines) + "\n"


def output_paths(
        input_path: Path,
        config: SubmissionConfig) -> tuple[Path, Path, Path, Path]:
    """Return ADES PSV, ADES XML, 80-column, and summary paths."""

    if config.output_dir is not None:
        output_dir = config.output_dir.expanduser().resolve()
    elif input_path.is_dir():
        output_dir = input_path.expanduser().resolve()
    else:
        output_dir = input_path.expanduser().resolve().parent

    stem = "mpc_%s" % target_to_filename(config.target)
    return (
        output_dir / ("%s_ADES.psv" % stem),
        output_dir / ("%s_ADES.xml" % stem),
        output_dir / ("%s_80col.txt" % stem),
        output_dir / ("%s_summary.txt" % stem),
    )


def submit_script_path_for_ades_xml(ades_xml_path: Path) -> Path:
    """Return the shell-script sidecar path for one ADES XML file."""

    return ades_xml_path.with_name("submit_%s.sh" % ades_xml_path.stem)


def _strip_xml_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _xml_element_text(root: ET.Element, name: str) -> str | None:
    for element in root.iter():
        if _strip_xml_namespace(element.tag) == name:
            text = element.text.strip() if element.text else ""
            if text:
                return text
    return None


def read_generated_ades_xml_summary(ades_xml_text: str) -> tuple[str, str]:
    """Return target and observatory code from generated ADES XML text."""

    try:
        root = ET.fromstring(ades_xml_text)
    except ET.ParseError as exc:
        raise SubmissionError("could not parse generated ADES XML: %s" %
                              exc) from exc

    target = (
        _xml_element_text(root, "provID") or
        _xml_element_text(root, "permID") or
        _xml_element_text(root, "trkSub")
    )
    observatory_code = (
        _xml_element_text(root, "stn") or
        _xml_element_text(root, "mpcCode")
    )
    if not target:
        raise SubmissionError(
            "could not find provID, permID, or trkSub in generated ADES XML")
    if not observatory_code:
        raise SubmissionError(
            "could not find stn or mpcCode in generated ADES XML")
    return normalize_target(target), observatory_code.strip().upper()


def resolve_submit_script_prog(observatory_code: str,
                               config: SubmissionConfig) -> str:
    """Return the required base-62 MPC XML submission ``prog`` value."""

    site_code = str(observatory_code).strip().upper()
    configured_code = str(config.prog or "").strip()
    sbsar_entry = read_sbsar_program_code_entries(
        config.program_codes_sbsar_csv).get(site_code, {})
    if sbsar_entry.get("base62_code"):
        return sbsar_entry["base62_code"]

    try:
        prog = lookup_mpc_program_code_base62(
            site_code,
            program_code=configured_code or None,
            contact_name=config.program_code_contact,
            cache_path=_program_code_cache_path(config),
            refresh=config.refresh_program_codes,
        )
    except (OSError, URLError, TimeoutError, json.JSONDecodeError,
            SubmissionError) as exc:
        raise SubmissionError(
            "could not resolve MPC XML prog for observatory %s: %s" %
            (site_code, exc)) from exc
    if prog:
        return prog
    if configured_code and len(configured_code) > 1:
        return configured_code
    if configured_code:
        raise SubmissionError(
            "could not resolve base-62 MPC XML prog for observatory %s "
            "and program code %s" % (site_code, configured_code))
    raise SubmissionError(
        "could not resolve MPC XML prog for observatory %s and contact %s" %
        (site_code, config.program_code_contact))


def format_submit_script(ades_xml_text: str,
                         ades_xml_path: Path,
                         config: SubmissionConfig) -> str:
    """Return an executable shell script for live MPC ADES XML submission."""

    target, observatory_code = read_generated_ades_xml_summary(ades_xml_text)
    ack = "%s %s" % (target, DEFAULT_ACK_SUFFIX)
    prog = resolve_submit_script_prog(observatory_code, config)
    source_filename = ades_xml_path.name

    assignments = {
        "ENDPOINT": DEFAULT_MPC_XML_LIVE_ENDPOINT,
        "TARGET": target,
        "OBSERVATORY_CODE": observatory_code,
        "ACK": ack,
        "AC2": DEFAULT_AC2_CONTACTS,
        "OBJ_TYPE": DEFAULT_MPC_XML_OBJ_TYPE,
        "PROG": prog,
        "SOURCE_FILE": source_filename,
        "SOURCE_FORM": "source=<%s" % source_filename,
    }
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    lines.extend("%s=%s" % (key, shlex.quote(value))
                 for key, value in assignments.items())
    lines.extend([
        "NO_INTERACTION=0",
        "",
        "while (($#)); do",
        "  case \"$1\" in",
        "    --no-interaction)",
        "      NO_INTERACTION=1",
        "      ;;",
        "    *)",
        "      echo \"ERROR: unknown option: $1\" >&2",
        "      exit 2",
        "      ;;",
        "  esac",
        "  shift",
        "done",
        "",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        "cd \"$SCRIPT_DIR\"",
        "",
        "if [[ ! -f \"$SOURCE_FILE\" ]]; then",
        "  echo \"ERROR: ADES XML file not found: $SOURCE_FILE\" >&2",
        "  exit 1",
        "fi",
        "",
        "CURL_ARGS=(",
        "  curl \"$ENDPOINT\"",
        "  -F \"ack=$ACK\"",
        "  -F \"ac2=$AC2\"",
        "  -F \"obj_type=$OBJ_TYPE\"",
        "  -F \"prog=$PROG\"",
        "  -F \"$SOURCE_FORM\"",
        ")",
        "",
        "echo \"About to submit ADES XML to the MPC live endpoint.\"",
        "echo \"Target: $TARGET\"",
        "echo \"Observatory code: $OBSERVATORY_CODE\"",
        "echo \"ACK: $ACK\"",
        "echo \"AC2: $AC2\"",
        "echo \"Object type: $OBJ_TYPE\"",
        "echo \"prog: $PROG\"",
        "echo \"Endpoint: $ENDPOINT\"",
        "echo \"Source file: $SOURCE_FILE\"",
        "echo \"Source form: $SOURCE_FORM\"",
        "echo \"Curl command:\"",
        "printf '  '",
        "printf '%q ' \"${CURL_ARGS[@]}\"",
        "printf '\\n'",
        "",
        "if [[ \"$NO_INTERACTION\" != \"1\" ]]; then",
        "  read -r -p \"Type \\\"submit\\\" to submit to the live MPC endpoint: \" reply",
        "  if [[ \"$reply\" != \"submit\" ]]; then",
        "    echo \"Submission cancelled.\"",
        "    exit 1",
        "  fi",
        "fi",
        "",
        "\"${CURL_ARGS[@]}\"",
    ])
    return "\n".join(lines) + "\n"


def _slug_for_filename(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_")
    return slug.lower() or "case"


def _count_obs80_observations(path: Path) -> int:
    count = 0
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith((
                "COD ", "CON ", "OBS ", "MEA ", "TEL ", "NET ", "BND ",
                "NUM ", "ACK ", "AC2 ")):
            continue
        count += 1
    return count


def _solver_success_markers(
        case_dir: Path, success_suffixes: Iterable[str]) -> list[str]:
    markers = []
    for path in sorted(case_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(name.endswith(suffix) for suffix in success_suffixes):
            markers.append(path.name)
    return markers


def discover_orbit_solver_input_snapshots(
        output_dir: Path, target: str) -> tuple[list[dict[str, object]], Path]:
    """Find successful MPFit/BandK2000 obs80 inputs near submission outputs."""

    output_dir = output_dir.expanduser().resolve()
    safe_target = target_to_filename(target)
    search_roots = _unique_preserve_order([output_dir, output_dir.parent])
    records: list[dict[str, object]] = []
    seen_sources: set[Path] = set()
    used_destinations: set[str] = set()
    stem = "mpc_%s" % safe_target

    for search_root in search_roots:
        for parent_name in ORBIT_SOLVER_PARENT_DIRS:
            orbit_root = search_root / parent_name / safe_target
            if not orbit_root.exists():
                continue
            for solver_root in sorted(path for path in orbit_root.rglob("*")
                                      if path.is_dir()):
                spec = ORBIT_SOLVER_DIR_SPECS.get(solver_root.name.lower())
                if spec is None:
                    continue
                for obs80_path in sorted(solver_root.rglob("*.obs80")):
                    if obs80_path in seen_sources:
                        continue
                    case_dir = obs80_path.parent
                    markers = _solver_success_markers(
                        case_dir, spec["success_suffixes"])
                    if not markers:
                        continue
                    case_slug = _slug_for_filename(case_dir.name)
                    destination_name = "%s_%s_%s_input.obs80" % (
                        stem, spec["slug"], case_slug)
                    suffix = 2
                    while destination_name in used_destinations:
                        destination_name = "%s_%s_%s_%d_input.obs80" % (
                            stem, spec["slug"], case_slug, suffix)
                        suffix += 1
                    seen_sources.add(obs80_path)
                    used_destinations.add(destination_name)
                    records.append({
                        "solver": spec["solver"],
                        "case": case_dir.name,
                        "run_tree": parent_name,
                        "legacy_layout": parent_name == "find_orb_runs",
                        "source_path": str(obs80_path.resolve()),
                        "output_path": str(output_dir / destination_name),
                        "observations": _count_obs80_observations(obs80_path),
                        "success_markers": markers,
                    })

    manifest_path = output_dir / (
        "%s_%s" % (stem, ORBIT_SOLVER_INPUT_MANIFEST_SUFFIX))
    return records, manifest_path


def format_summary(input_path: Path,
                   observations: list[PhotometryObservation],
                   config: SubmissionConfig,
                   warnings: list[str],
                   ades_path: Path,
                   ades_xml_path: Path,
                   obs80_path: Path,
                   summary_path: Path,
                   submit_script_path: Path,
                   ades_prog_values: list[str] | None = None,
                   orbit_solver_input_snapshots:
                   list[dict[str, object]] | None = None) -> str:
    """Format a human-readable provenance and warning summary."""

    observers = derive_observers(observations, config)
    measurers = derive_measurers(config)
    photcats = _unique_preserve_order(
        config.photcat or obs.phot_cat for obs in observations)
    sources = _unique_preserve_order(str(obs.source_file)
                                     for obs in observations)
    ades_only = [
        "rmsRA",
        "rmsDec",
        "rmsMag",
        "photCat",
        "exp",
        "seeing",
        "remarks",
        "submitter/measurer context",
    ]

    lines = [
        "MPC submission sidecar summary",
        "Target: %s" % normalize_target(config.target),
        "Packed target: %s" %
        pack_minor_planet_provisional_designation(config.target),
        "Input: %s" % input_path.expanduser().resolve(),
        "Observations: %d" % len(observations),
        "Astrometry-only observations: %d" %
        sum(1 for obs in observations if obs.astrometry_only),
        "Observatory code: %s" % config.observatory_code,
        "Astrometric catalog: %s" % config.astcat,
        "Photometric catalog(s): %s" % ", ".join(photcats),
        "Submitter: %s" % config.submitter,
        "Measurer: %s" % ", ".join(measurers),
        "Contact: %s" % config.contact,
        "Program code: %s" % (config.prog if config.prog else "(none)"),
        "ADES prog (base62): %s" %
        (", ".join(ades_prog_values or []) if ades_prog_values else "(none)"),
        "Program-code lookup contact: %s" % config.program_code_contact,
        "Observers: %s" % (", ".join(observers) if observers else "(none)"),
        "Non-survey measurer/pipeline astrometry: %s" %
        ("yes" if config.non_survey_measurer else "no"),
        "ADES PSV output: %s" % ades_path,
        "ADES XML output: %s" % ades_xml_path,
        "80-column output: %s" % obs80_path,
        "Summary output: %s" % summary_path,
        "MPC XML submit script: %s" % submit_script_path,
        "ADES-only fields not represented in 80-column: %s" %
        ", ".join(ades_only),
        "Orbit-solver accepted inputs:",
    ]
    if orbit_solver_input_snapshots:
        for snapshot in orbit_solver_input_snapshots:
            lines.append(
                "  {solver} {case}: {output_path} ({observations} obs)".
                format(**snapshot))
    else:
        lines.append("  none found")
    lines.append("Source FITS files:")
    lines.extend("  %s" % source for source in sources)
    lines.append("Warnings:")
    if warnings:
        lines.extend("  %s" % warning for warning in warnings)
    else:
        lines.append("  none")
    return "\n".join(lines) + "\n"


def build_submission(input_path: Path,
                     config: SubmissionConfig) -> SubmissionBundle:
    """Build ADES, 80-column, and summary text for a PP output tree."""

    warnings = []
    config.prog = resolve_program_code(config, warnings)
    observations = collect_observations(input_path, config, warnings)
    warnings.extend(validate_observations(observations, config))
    ades_path, ades_xml_path, obs80_path, summary_path = output_paths(
        input_path, config)
    submit_script_path = submit_script_path_for_ades_xml(ades_xml_path)
    ades_text = format_ades_psv(observations, config)
    ades_xml_text = format_ades_xml_from_psv(ades_text)
    submit_script_text = format_submit_script(
        ades_xml_text, ades_xml_path, config)
    obs80_text = format_obs80(observations, config)
    _, ades_rows = _parse_psv_table(ades_text)
    ades_prog_values = _unique_preserve_order(
        row.get("prog", "") for row in ades_rows if row.get("prog", ""))
    orbit_solver_input_snapshots, orbit_solver_input_manifest_path = (
        discover_orbit_solver_input_snapshots(
            obs80_path.parent, config.target))
    summary_text = format_summary(input_path, observations, config, warnings,
                                  ades_path, ades_xml_path, obs80_path,
                                  summary_path, submit_script_path,
                                  ades_prog_values,
                                  orbit_solver_input_snapshots)
    return SubmissionBundle(
        observations=observations,
        warnings=warnings,
        ades_text=ades_text,
        ades_xml_text=ades_xml_text,
        obs80_text=obs80_text,
        summary_text=summary_text,
        submit_script_text=submit_script_text,
        ades_path=ades_path,
        ades_xml_path=ades_xml_path,
        obs80_path=obs80_path,
        summary_path=summary_path,
        submit_script_path=submit_script_path,
        orbit_solver_input_snapshots=orbit_solver_input_snapshots,
        orbit_solver_input_manifest_path=orbit_solver_input_manifest_path,
    )


def write_submission(bundle: SubmissionBundle) -> None:
    """Write all companion output files."""

    for path in (
            bundle.ades_path, bundle.ades_xml_path, bundle.obs80_path,
            bundle.summary_path, bundle.submit_script_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    bundle.ades_path.write_text(bundle.ades_text)
    bundle.ades_xml_path.write_text(bundle.ades_xml_text)
    bundle.obs80_path.write_text(bundle.obs80_text)
    bundle.summary_path.write_text(bundle.summary_text)
    bundle.submit_script_path.write_text(bundle.submit_script_text)
    bundle.submit_script_path.chmod(
        bundle.submit_script_path.stat().st_mode | 0o111)
    if bundle.orbit_solver_input_snapshots:
        for snapshot in bundle.orbit_solver_input_snapshots:
            source_path = Path(str(snapshot["source_path"]))
            output_path = Path(str(snapshot["output_path"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.resolve() != output_path.resolve():
                shutil.copy2(source_path, output_path)
        if bundle.orbit_solver_input_manifest_path is not None:
            bundle.orbit_solver_input_manifest_path.write_text(json.dumps({
                "description": (
                    "MPC 80-column inputs that successfully went through "
                    "local orbit solvers and were snapshotted with the "
                    "submission sidecars."),
                "inputs": bundle.orbit_solver_input_snapshots,
            }, indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="prepare ADES PSV and MPC 80-column submission files")
    parser.add_argument("input_path",
                        help="PP output directory or photometry file")
    parser.add_argument("--target", required=True,
                        help="target designation, e.g. '2016 CJ155'")
    parser.add_argument("--measurer", default=DEFAULT_MEASURER)
    parser.add_argument("--submitter", default=DEFAULT_SUBMITTER)
    parser.add_argument("--contact", default=DEFAULT_CONTACT)
    parser.add_argument("--observer", action="append", dest="observers",
                        help="observer name; may be supplied more than once")
    parser.add_argument("--observatory-code", default=DEFAULT_OBSERVATORY_CODE)
    parser.add_argument("--astcat", default=DEFAULT_ASTCAT)
    parser.add_argument("--photcat",
                        help="override photometric catalog for all rows")
    parser.add_argument("--prog", help="ADES program code; one character is "
                        "also placed in 80-column output")
    parser.add_argument("--program-code-contact",
                        default=DEFAULT_PROGRAM_CODE_CONTACT,
                        help="contact name used for automatic MPC program "
                             "code lookup when --prog is not supplied")
    parser.add_argument("--program-codes-cache", type=Path,
                        help="local MPC program-code JSON cache path")
    parser.add_argument("--program-codes-sbsar-csv", type=Path,
                        help="local SBSAR site/program-code CSV path")
    parser.add_argument("--refresh-program-codes", action="store_true",
                        help="refresh the local MPC program-code JSON cache")
    parser.add_argument("--non-survey", "--non-survey-measurer",
                        dest="non_survey_measurer", action="store_true",
                        help="set MPC1992 observation note Z in column 72 for "
                        "non-survey measurer/pipeline astrometry from F51, "
                        "F52, G96, I41, or 703")
    parser.add_argument("--output-dir", type=Path,
                        help="directory for all generated files")
    parser.add_argument("--strict", action="store_true",
                        help="turn warnings into hard errors")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and validate output text without writing")
    parser.add_argument("--validate-ades", action="store_true",
                        help="run compact current-definition ADES PSV "
                             "validation on the generated output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SubmissionConfig(
        target=args.target,
        observatory_code=args.observatory_code,
        astcat=args.astcat,
        photcat=args.photcat,
        measurer=args.measurer,
        submitter=args.submitter,
        contact=args.contact,
        observers=args.observers,
        prog=args.prog,
        program_code_contact=args.program_code_contact,
        program_codes_cache=args.program_codes_cache,
        program_codes_sbsar_csv=args.program_codes_sbsar_csv,
        refresh_program_codes=args.refresh_program_codes,
        non_survey_measurer=args.non_survey_measurer,
        output_dir=args.output_dir,
        strict=args.strict,
    )
    bundle = build_submission(Path(args.input_path), config)
    ades_validation = None
    ades_xml_validation = None
    if args.validate_ades:
        ades_validation = validate_ades_psv_text(bundle.ades_text)
        ades_xml_validation = validate_ades_xml_text(bundle.ades_xml_text)
        validation_errors = (
            ades_validation["errors"] + ades_xml_validation["errors"])
        if validation_errors and args.strict:
            raise SubmissionError("; ".join(validation_errors))
    if not args.dry_run:
        write_submission(bundle)

    action = "Would write" if args.dry_run else "Wrote"
    print("%s %d observations for %s" %
          (action, len(bundle.observations), normalize_target(config.target)))
    print("ADES PSV: %s" % bundle.ades_path)
    print("ADES XML: %s" % bundle.ades_xml_path)
    print("80-column: %s" % bundle.obs80_path)
    print("Summary: %s" % bundle.summary_path)
    print("Submit script: %s" % bundle.submit_script_path)
    if bundle.orbit_solver_input_snapshots:
        print("Orbit-solver accepted inputs: %d" %
              len(bundle.orbit_solver_input_snapshots))
        if bundle.orbit_solver_input_manifest_path is not None:
            print("Orbit-solver input manifest: %s" %
                  bundle.orbit_solver_input_manifest_path)
    if ades_validation is not None:
        print("ADES validation errors: %d" %
              len(ades_validation["errors"]))
        print("ADES validation warnings: %d" %
              len(ades_validation["warnings"]))
        for error in ades_validation["errors"]:
            print("ADES ERROR: %s" % error)
        for warning in ades_validation["warnings"]:
            print("ADES WARNING: %s" % warning)
    if ades_xml_validation is not None:
        print("ADES XML validation errors: %d" %
              len(ades_xml_validation["errors"]))
        print("ADES XML validation warnings: %d" %
              len(ades_xml_validation["warnings"]))
        for error in ades_xml_validation["errors"]:
            print("ADES XML ERROR: %s" % error)
        for warning in ades_xml_validation["warnings"]:
            print("ADES XML WARNING: %s" % warning)
    for warning in bundle.warnings:
        print("WARNING: %s" % warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
