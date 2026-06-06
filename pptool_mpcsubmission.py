#!/usr/bin/env python3

"""Build ADES PSV and MPC 80-column submission files from PP photometry."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time


DEFAULT_ASTCAT = "Gaia2"
DEFAULT_CONTACT = "coc123@uw.edu"
DEFAULT_AC2_CONTACTS = "coc123@uw.edu, murtagh@uw.edu"
DEFAULT_MEASURER = "C. O. Chandler, J. Murtagh"
DEFAULT_OBSERVATORY_CODE = "W84"
DEFAULT_ACK_SUFFIX = "Small-body Search and Rescue"
DEFAULT_SUBMITTER = "C. O. Chandler"
NON_SURVEY_OBSNOTE = "Z"
NON_SURVEY_OBSNOTE_CODES = {"F51", "F52", "G96", "I41", "703"}

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
    mag: float
    mag_sig: float
    ra_deg: float
    dec_deg: float
    exptime: float
    phot_cat: str
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
    non_survey_measurer: bool = False
    output_dir: Path | None = None
    strict: bool = False


@dataclass
class SubmissionBundle:
    observations: list[PhotometryObservation]
    warnings: list[str]
    ades_text: str
    obs80_text: str
    summary_text: str
    ades_path: Path
    obs80_path: Path
    summary_path: Path


class SubmissionError(RuntimeError):
    """Raised when a submission cannot be built safely."""


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
        mag=float(row["mag"]),
        mag_sig=float(row["mag_sig"]),
        ra_deg=float(row["ra_deg"]),
        dec_deg=float(row["dec_deg"]),
        exptime=float(row["exptime"]),
        phot_cat=row["phot_cat"],
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
    if "SPACEWATCH09" in telescope_keywords or any(
            "Spacewatch 0.9-m" in telescope for telescope in telescopes):
        return "0.9-m f/3 reflector", "0.9", "CCD"
    if telescopes:
        design = sorted(telescopes)[0]
    else:
        design = "reflector"
    return design, "0.0", "CCD"


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
        "# version=2017",
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
        "! line Contact: %s" % config.contact,
        "! line Generated by Photometry Pipeline pptool_mpcsubmission.py",
        "|".join(fieldnames),
    ])

    prov_id = normalize_target(config.target)
    for obs in observations:
        photcat = config.photcat or obs.phot_cat
        remarks = [obs.source_file.name]
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
            "photCat": photcat,
            "mag": _format_float(obs.mag, 4),
            "rmsMag": _format_float(obs.mag_sig, 4),
            "band": obs.band,
            "exp": _format_float(obs.exptime, 2),
            "seeing": _format_float(obs.fwhm, 2),
            "prog": config.prog or "",
            "remarks": "; ".join(remarks),
        }
        lines.append("|".join(_psv_value(values[name]) for name in fieldnames))
    return "\n".join(lines) + "\n"


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
    program_code = config.prog if config.prog and len(config.prog) == 1 else " "
    observation_note = obs80_observation_note(config)

    line = (
        "%5s%-7s  C" % ("", packed_target) +
        date +
        "%s " % ra +
        "%s" % dec +
        "         " +
        "%5.1f %1s" % (obs.mag, band) +
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


def output_paths(input_path: Path, config: SubmissionConfig) -> tuple[Path, Path, Path]:
    """Return ADES, 80-column, and summary paths for a target."""

    if config.output_dir is not None:
        output_dir = config.output_dir.expanduser().resolve()
    elif input_path.is_dir():
        output_dir = input_path.expanduser().resolve()
    else:
        output_dir = input_path.expanduser().resolve().parent

    stem = "mpc_%s" % target_to_filename(config.target)
    return (
        output_dir / ("%s_ADES.psv" % stem),
        output_dir / ("%s_80col.txt" % stem),
        output_dir / ("%s_summary.txt" % stem),
    )


def format_summary(input_path: Path,
                   observations: list[PhotometryObservation],
                   config: SubmissionConfig,
                   warnings: list[str],
                   ades_path: Path,
                   obs80_path: Path,
                   summary_path: Path) -> str:
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
        "Observatory code: %s" % config.observatory_code,
        "Astrometric catalog: %s" % config.astcat,
        "Photometric catalog(s): %s" % ", ".join(photcats),
        "Submitter: %s" % config.submitter,
        "Measurer: %s" % ", ".join(measurers),
        "Contact: %s" % config.contact,
        "Observers: %s" % (", ".join(observers) if observers else "(none)"),
        "Non-survey measurer/pipeline astrometry: %s" %
        ("yes" if config.non_survey_measurer else "no"),
        "ADES output: %s" % ades_path,
        "80-column output: %s" % obs80_path,
        "Summary output: %s" % summary_path,
        "ADES-only fields not represented in 80-column: %s" %
        ", ".join(ades_only),
        "Source FITS files:",
    ]
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
    observations = collect_observations(input_path, config, warnings)
    warnings.extend(validate_observations(observations, config))
    ades_path, obs80_path, summary_path = output_paths(input_path, config)
    ades_text = format_ades_psv(observations, config)
    obs80_text = format_obs80(observations, config)
    summary_text = format_summary(input_path, observations, config, warnings,
                                  ades_path, obs80_path, summary_path)
    return SubmissionBundle(
        observations=observations,
        warnings=warnings,
        ades_text=ades_text,
        obs80_text=obs80_text,
        summary_text=summary_text,
        ades_path=ades_path,
        obs80_path=obs80_path,
        summary_path=summary_path,
    )


def write_submission(bundle: SubmissionBundle) -> None:
    """Write all companion output files."""

    for path in (bundle.ades_path, bundle.obs80_path, bundle.summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    bundle.ades_path.write_text(bundle.ades_text)
    bundle.obs80_path.write_text(bundle.obs80_text)
    bundle.summary_path.write_text(bundle.summary_text)


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
        non_survey_measurer=args.non_survey_measurer,
        output_dir=args.output_dir,
        strict=args.strict,
    )
    bundle = build_submission(Path(args.input_path), config)
    if not args.dry_run:
        write_submission(bundle)

    action = "Would write" if args.dry_run else "Wrote"
    print("%s %d observations for %s" %
          (action, len(bundle.observations), normalize_target(config.target)))
    print("ADES: %s" % bundle.ades_path)
    print("80-column: %s" % bundle.obs80_path)
    print("Summary: %s" % bundle.summary_path)
    for warning in bundle.warnings:
        print("WARNING: %s" % warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
