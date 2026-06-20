#!/usr/bin/env python3

"""Export PP combined photometry and MPC photometry in Dima lightcurve format."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.request import Request, urlopen

from astropy.time import Time

try:
    from pptool_mpcsubmission import target_to_filename
except Exception:  # pragma: no cover - keeps this utility usable standalone.
    def target_to_filename(target: str) -> str:
        return target.strip().translate(str.maketrans(" /()", "____"))


DEFAULT_INPUT = (
    "/Users/colinchandler/Colinchandler Dropbox/Colin Chandler/School/"
    "Research/Objects/Lost Small Bodies/objects/Centaurs/1998_QJ1/CASU/"
    "WHT/Prime/PP/photometry_1998_QJ1_combined.csv"
)
DEFAULT_OUTPUT = (
    "/Users/colinchandler/Colinchandler Dropbox/Colin Chandler/School/"
    "Research/Objects/Lost Small Bodies/objects/Centaurs/1998_QJ1/CASU/"
    "WHT/Prime/lightcurve_dima/dima_format_obs.csv"
)
MPC_OBSERVATIONS_URL = "https://data.minorplanetcenter.net/api/get-obs"
DIMA_HEADERS = [
    "designation",
    "epoch",
    "band",
    "mag",
    "rmsmag",
    "flag",
    "topodist",
    "heliodist",
    "phase_angle",
]
MPC_OBSERVATION_CORE_COLUMNS = [
    "obstime",
    "stn",
    "mag",
    "rmsmag",
    "band",
    "permid",
    "provid",
    "trkmpc",
    "trkid",
    "obsid",
    "ref",
    "astcat",
    "photcat",
    "ra",
    "dec",
]
INSTRUMENT_SITE_FALLBACKS = {
    "WHTPFIP": "950",
}


@dataclass
class DimaRow:
    designation: str
    epoch: float
    epoch_text: str
    band: str
    mag: str
    rmsmag: str
    flag: str
    site_code: str
    source: str
    source_id: str = ""
    topodist: str = ""
    heliodist: str = ""
    phase_angle: str = ""

    def to_csv_row(self) -> dict[str, str]:
        return {
            "designation": self.designation,
            "epoch": self.epoch_text,
            "band": self.band,
            "mag": self.mag,
            "rmsmag": self.rmsmag,
            "flag": self.flag,
            "topodist": self.topodist,
            "heliodist": self.heliodist,
            "phase_angle": self.phase_angle,
        }


@dataclass(frozen=True)
class ExportResult:
    output_path: Path
    mpc_csv_path: Path | None
    mpc_json_path: Path | None
    pp_rows: int
    mpc_rows: int
    total_rows: int
    warnings: list[str]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def format_float(value: Any, digits: int = 15) -> str:
    result = finite_float(value)
    if result is None:
        return ""
    return ("%.*g" % (digits, result)).rstrip("0").rstrip(".")


def format_jd(value: float) -> str:
    return ("%.9f" % value).rstrip("0").rstrip(".")


def target_designation(target: str) -> str:
    return target_to_filename(target)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def infer_pp_site_code(rows: list[dict[str, str]], explicit_site_code: str | None) -> str:
    if explicit_site_code:
        return explicit_site_code.strip()

    row_site_codes = {
        clean_text(row.get("observatory_code") or row.get("site_code"))
        for row in rows
        if clean_text(row.get("observatory_code") or row.get("site_code"))
    }
    if len(row_site_codes) == 1:
        return next(iter(row_site_codes))

    instruments = {
        clean_text(row.get("instrument"))
        for row in rows
        if clean_text(row.get("instrument"))
    }
    if len(instruments) == 1:
        instrument = next(iter(instruments))
        try:
            import _pp_conf

            obsparam = _pp_conf.telescope_parameters.get(instrument, {})
            code = clean_text(obsparam.get("observatory_code"))
            if code:
                return code
        except Exception:
            pass
        if instrument in INSTRUMENT_SITE_FALLBACKS:
            return INSTRUMENT_SITE_FALLBACKS[instrument]

    return "950"


def pp_rows_to_dima(
    input_path: Path,
    *,
    target: str,
    pp_flag: str,
    pp_site_code: str | None = None,
) -> list[DimaRow]:
    rows = read_csv_rows(input_path)
    site_code = infer_pp_site_code(rows, pp_site_code)
    designation = target_designation(target)
    dima_rows: list[DimaRow] = []
    for index, row in enumerate(rows, start=1):
        epoch_text = clean_text(row.get("julian_date"))
        epoch = finite_float(epoch_text)
        band = clean_text(row.get("band"))
        mag = clean_text(row.get("mag"))
        if epoch is None:
            raise ValueError("PP row %d has invalid julian_date: %r" % (index, epoch_text))
        if not band or not mag:
            raise ValueError("PP row %d lacks band or mag" % index)
        dima_rows.append(
            DimaRow(
                designation=designation,
                epoch=epoch,
                epoch_text=epoch_text,
                band=band,
                mag=mag,
                rmsmag=clean_text(row.get("mag_sig")),
                flag=str(pp_flag),
                site_code=site_code,
                source="pp",
                source_id=clean_text(
                    row.get("source_fits_file") or row.get("catalog_token") or index
                ),
            )
        )
    return dima_rows


def fetch_mpc_observations(
    target: str,
    *,
    timeout: float = 60.0,
    user_agent: str = "photometrypipeline pptool_dima_export.py",
) -> tuple[Any, list[dict[str, Any]]]:
    payload = {"desigs": [target], "output_format": ["ADES_DF"]}
    request = Request(
        MPC_OBSERVATIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="GET",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        },
    )
    with urlopen(request, timeout=timeout) as response:
        response_payload = json.load(response)
    return response_payload, extract_mpc_ades_observations(response_payload)


def extract_mpc_ades_observations(payload: Any) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "ADES_DF" and isinstance(item, list):
                    observations.extend(row for row in item if isinstance(row, dict))
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return observations


def mpc_observations_to_dima_rows(
    observations: Iterable[dict[str, Any]],
    *,
    target: str,
    mpc_flag: str,
    site_code_fallback: str | None = None,
) -> tuple[list[DimaRow], list[str]]:
    designation = target_designation(target)
    rows: list[DimaRow] = []
    warnings: list[str] = []
    for index, observation in enumerate(observations, start=1):
        mag = clean_text(observation.get("mag"))
        band = clean_text(observation.get("band"))
        if not mag or not band:
            continue
        obstime = clean_text(observation.get("obstime"))
        try:
            epoch = float(Time(obstime, format="isot", scale="utc").jd)
        except Exception as exc:
            warnings.append("skipping MPC row %d with invalid obstime %r: %s" % (index, obstime, exc))
            continue
        site_code = clean_text(observation.get("stn")) or clean_text(site_code_fallback)
        if not site_code:
            warnings.append("MPC row %d has no site code; geometry will be blank" % index)
        rows.append(
            DimaRow(
                designation=designation,
                epoch=epoch,
                epoch_text=format_jd(epoch),
                band=band,
                mag=mag,
                rmsmag=clean_text(observation.get("rmsmag")),
                flag=str(mpc_flag),
                site_code=site_code,
                source="mpc",
                source_id=clean_text(observation.get("obsid") or observation.get("trkid") or index),
            )
        )
    return rows, warnings


def csv_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return str(value)


def save_mpc_observations(
    output_dir: Path,
    target: str,
    raw_payload: Any,
    observations: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "mpc_%s_observations" % target_designation(target)
    json_path = output_dir / ("%s_raw.json" % stem)
    csv_path = output_dir / ("%s.csv" % stem)
    json_path.write_text(
        json.dumps(raw_payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    extra_columns = sorted(
        {
            key
            for observation in observations
            for key in observation.keys()
            if key not in MPC_OBSERVATION_CORE_COLUMNS
        }
    )
    fieldnames = MPC_OBSERVATION_CORE_COLUMNS + extra_columns
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for observation in observations:
            writer.writerow({key: csv_scalar(observation.get(key)) for key in fieldnames})
    return csv_path, json_path


def query_horizons_ephemerides(
    target: str,
    site_code: str,
    epochs: list[float],
    id_type: str,
) -> Any:
    from astroquery.jplhorizons import Horizons

    return Horizons(
        id=target,
        location=site_code,
        epochs=[float(epoch) for epoch in epochs],
        id_type=id_type,
    ).ephemerides(quantities="19,20,24", cache=False)


HorizonsQuery = Callable[[str, str, list[float], str], Any]


def enrich_rows_with_horizons(
    rows: list[DimaRow],
    *,
    target: str,
    id_type: str,
    batch_size: int = 50,
    strict: bool = False,
    query_func: HorizonsQuery | None = None,
) -> list[str]:
    warnings: list[str] = []
    if query_func is None:
        query_func = query_horizons_ephemerides
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row.site_code:
            grouped.setdefault(row.site_code, []).append(index)
        else:
            warnings.append(
                "no site code for %s row %s at JD %s; geometry left blank"
                % (row.source, row.source_id, row.epoch_text)
            )

    for site_code, indexes in sorted(grouped.items()):
        sorted_indexes = sorted(indexes, key=lambda index: rows[index].epoch)
        for chunk_indexes in chunked(sorted_indexes, max(1, batch_size)):
            chunk_epochs = [rows[index].epoch for index in chunk_indexes]
            try:
                table = query_func(target, site_code, chunk_epochs, id_type)
            except Exception as exc:
                message = (
                    "Horizons query failed for site %s, JD %.8f-%.8f: %s"
                    % (site_code, min(chunk_epochs), max(chunk_epochs), str(exc).replace("\n", " "))
                )
                if strict:
                    raise RuntimeError(message) from exc
                warnings.append(message)
                continue

            table_length = len(table)
            if table_length < len(chunk_indexes):
                message = (
                    "Horizons returned %d row(s) for %d epoch(s) at site %s"
                    % (table_length, len(chunk_indexes), site_code)
                )
                if strict:
                    raise RuntimeError(message)
                warnings.append(message)

            for table_index, row_index in enumerate(chunk_indexes):
                if table_index >= table_length:
                    continue
                apply_horizons_row(rows[row_index], table, table_index, strict=strict)
    return warnings


def chunked(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def table_value(table: Any, column: str, index: int) -> Any:
    if column not in getattr(table, "colnames", []):
        raise KeyError(column)
    return table[column][index]


def apply_horizons_row(row: DimaRow, table: Any, index: int, *, strict: bool) -> None:
    try:
        row.topodist = format_float(table_value(table, "delta", index))
        row.heliodist = format_float(table_value(table, "r", index))
        row.phase_angle = format_float(table_value(table, "alpha", index))
    except Exception as exc:
        if strict:
            raise RuntimeError("cannot read Horizons geometry at row %d: %s" % (index, exc)) from exc


def write_dima_csv(path: Path, rows: list[DimaRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: (row.epoch, row.source, row.source_id))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIMA_HEADERS)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(row.to_csv_row())


def export_dima(
    *,
    input_path: Path,
    output_path: Path,
    target: str,
    include_mpc: bool,
    pp_flag: str,
    mpc_flag: str,
    pp_site_code: str | None,
    mpc_site_code_fallback: str | None,
    horizons_id_type: str,
    strict_horizons: bool,
    horizons_batch_size: int,
) -> ExportResult:
    pp_rows = pp_rows_to_dima(
        input_path,
        target=target,
        pp_flag=pp_flag,
        pp_site_code=pp_site_code,
    )
    rows = list(pp_rows)
    warnings: list[str] = []
    mpc_csv_path = None
    mpc_json_path = None
    mpc_count = 0

    if include_mpc:
        raw_payload, observations = fetch_mpc_observations(target)
        mpc_csv_path, mpc_json_path = save_mpc_observations(
            output_path.parent,
            target,
            raw_payload,
            observations,
        )
        mpc_rows, mpc_warnings = mpc_observations_to_dima_rows(
            observations,
            target=target,
            mpc_flag=mpc_flag,
            site_code_fallback=mpc_site_code_fallback,
        )
        rows.extend(mpc_rows)
        warnings.extend(mpc_warnings)
        mpc_count = len(mpc_rows)

    warnings.extend(
        enrich_rows_with_horizons(
            rows,
            target=target,
            id_type=horizons_id_type,
            batch_size=horizons_batch_size,
            strict=strict_horizons,
        )
    )
    write_dima_csv(output_path, rows)
    return ExportResult(
        output_path=output_path,
        mpc_csv_path=mpc_csv_path,
        mpc_json_path=mpc_json_path,
        pp_rows=len(pp_rows),
        mpc_rows=mpc_count,
        total_rows=len(rows),
        warnings=warnings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PP combined photometry to Dima lightcurve CSV format."
    )
    parser.add_argument("--target", default="1998 QJ1")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="PP combined photometry CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Dima-format output CSV.")
    mpc = parser.add_mutually_exclusive_group()
    mpc.add_argument("--include-mpc", dest="include_mpc", action="store_true", default=True)
    mpc.add_argument("--no-include-mpc", dest="include_mpc", action="store_false")
    parser.add_argument("--mpc-scope", choices=["all"], default="all")
    parser.add_argument("--pp-flag", default="1")
    parser.add_argument("--mpc-flag", default="0")
    parser.add_argument("--pp-site-code", default=None)
    parser.add_argument("--mpc-site-code-fallback", default=None)
    parser.add_argument("--horizons-id-type", default="smallbody")
    parser.add_argument("--horizons-batch-size", type=int, default=50)
    parser.add_argument("--strict-horizons", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = export_dima(
            input_path=Path(args.input).expanduser().resolve(),
            output_path=Path(args.output).expanduser().resolve(),
            target=args.target,
            include_mpc=args.include_mpc,
            pp_flag=args.pp_flag,
            mpc_flag=args.mpc_flag,
            pp_site_code=args.pp_site_code,
            mpc_site_code_fallback=args.mpc_site_code_fallback,
            horizons_id_type=args.horizons_id_type,
            strict_horizons=args.strict_horizons,
            horizons_batch_size=args.horizons_batch_size,
        )
    except Exception as exc:
        parser.exit(1, "error: %s\n" % exc)

    print("Wrote Dima CSV: %s" % result.output_path)
    if result.mpc_csv_path is not None:
        print("Wrote MPC observations CSV: %s" % result.mpc_csv_path)
    if result.mpc_json_path is not None:
        print("Wrote MPC observations JSON: %s" % result.mpc_json_path)
    print("PP rows: %d" % result.pp_rows)
    print("MPC photometry rows: %d" % result.mpc_rows)
    print("Total Dima rows: %d" % result.total_rows)
    for warning in result.warnings:
        print("warning: %s" % warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
