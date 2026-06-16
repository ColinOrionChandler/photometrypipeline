#!/usr/bin/env python3

"""Interactively submit an ADES XML file to the MPC XML endpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

from pptool_mpcsubmission import (
    DEFAULT_ACK_SUFFIX,
    DEFAULT_AC2_CONTACTS,
    DEFAULT_PROGRAM_CODE_CONTACT,
    DEFAULT_SBSAR_PROGRAM_CODES_CSV,
    DEFAULT_PROGRAM_CODES_CACHE,
    SubmissionConfig,
    SubmissionError,
    _program_code_cache_path,
    lookup_mpc_program_code_base62,
    normalize_target,
    read_sbsar_program_code_entries,
)


DEFAULT_TEST_ENDPOINT = "https://minorplanetcenter.net/submit_xml_test"
DEFAULT_LIVE_ENDPOINT = "https://minorplanetcenter.net/submit_xml"
DEFAULT_OBJ_TYPE = "tno"
VALID_OBJ_TYPES = (
    "unclassified",
    "neocp",
    "neo candidate",
    "neo",
    "new comet",
    "comet",
    "tno",
    "artsat",
)


@dataclass
class AdesXmlSummary:
    target: str
    observatory_code: str


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _iter_named(root: ET.Element, name: str):
    for element in root.iter():
        if _strip_namespace(element.tag) == name:
            yield element


def _element_text(root: ET.Element, name: str) -> str | None:
    for element in _iter_named(root, name):
        if element.text and element.text.strip():
            return element.text.strip()
    return None


def read_ades_xml_summary(ades_file: Path) -> AdesXmlSummary:
    """Return target and observatory code from an ADES XML file."""

    try:
        root = ET.parse(ades_file).getroot()
    except ET.ParseError as exc:
        raise SubmissionError("could not parse ADES XML %s: %s" %
                              (ades_file, exc)) from exc

    target = (
        _element_text(root, "provID") or
        _element_text(root, "permID") or
        _element_text(root, "trkSub")
    )
    observatory_code = (
        _element_text(root, "stn") or
        _element_text(root, "mpcCode")
    )
    if not target:
        raise SubmissionError("could not find provID, permID, or trkSub in %s"
                              % ades_file)
    if not observatory_code:
        raise SubmissionError("could not find stn or mpcCode in %s" %
                              ades_file)
    return AdesXmlSummary(
        target=normalize_target(target),
        observatory_code=observatory_code,
    )


def format_ack(target: str) -> str:
    """Return the ACK text shared with the MPC1992 sidecar."""

    return "%s %s" % (normalize_target(target), DEFAULT_ACK_SUFFIX)


def resolve_submit_prog(observatory_code: str,
                        explicit_prog: str | None = None,
                        program_code_contact: str = DEFAULT_PROGRAM_CODE_CONTACT,
                        program_codes_cache: Path | None = None,
                        program_codes_sbsar_csv: Path | None = None,
                        refresh_program_codes: bool = False) -> str:
    """Return the base-62 MPC XML form ``prog`` value."""

    if explicit_prog:
        return explicit_prog.strip()

    site_code = str(observatory_code).strip().upper()
    sbsar_entry = read_sbsar_program_code_entries(
        program_codes_sbsar_csv or DEFAULT_SBSAR_PROGRAM_CODES_CSV).get(
            site_code, {})
    if sbsar_entry.get("base62_code"):
        return sbsar_entry["base62_code"]

    config = SubmissionConfig(
        target="",
        observatory_code=site_code,
        program_code_contact=program_code_contact,
        program_codes_cache=program_codes_cache or DEFAULT_PROGRAM_CODES_CACHE,
        program_codes_sbsar_csv=program_codes_sbsar_csv,
        refresh_program_codes=refresh_program_codes,
    )
    prog = lookup_mpc_program_code_base62(
        site_code,
        contact_name=program_code_contact,
        cache_path=_program_code_cache_path(config),
        refresh=refresh_program_codes,
    )
    if not prog:
        raise SubmissionError(
            "could not resolve ADES form prog for observatory %s and contact %s"
            % (site_code, program_code_contact))
    return prog


def build_curl_args(endpoint: str,
                    ades_file: Path,
                    ack: str,
                    ac2: str,
                    obj_type: str,
                    prog: str) -> list[str]:
    """Return curl arguments for the MPC XML submission form."""

    return [
        "curl",
        endpoint,
        "-F",
        "ack=%s" % ack,
        "-F",
        "ac2=%s" % ac2,
        "-F",
        "obj_type=%s" % obj_type,
        "-F",
        "prog=%s" % prog,
        "-F",
        "source=<%s" % ades_file.name,
    ]


def display_curl_args(curl_args: list[str]) -> list[str]:
    """Return curl args safe to print without exposing local paths."""

    display_args = list(curl_args)
    for index, value in enumerate(display_args):
        if value.startswith("source=<") and "/" in value:
            display_args[index] = "source=<ADES_XML_FILE"
    return display_args


def normalize_obj_type(obj_type: str) -> str:
    """Return a valid MPC XML form obj_type value."""

    normalized = " ".join(str(obj_type).strip().lower().split())
    valid = {value.casefold(): value for value in VALID_OBJ_TYPES}
    if normalized.casefold() not in valid:
        raise SubmissionError(
            "invalid obj_type %r; valid options are: %s" %
            (obj_type, ", ".join(VALID_OBJ_TYPES)))
    return valid[normalized.casefold()]


def submit_ades_to_mpc(ades_file: Path,
                       endpoint: str,
                       obj_type: str = DEFAULT_OBJ_TYPE,
                       ac2: str = DEFAULT_AC2_CONTACTS,
                       ack: str | None = None,
                       target: str | None = None,
                       observatory_code: str | None = None,
                       prog: str | None = None,
                       program_code_contact: str = DEFAULT_PROGRAM_CODE_CONTACT,
                       program_codes_cache: Path | None = None,
                       program_codes_sbsar_csv: Path | None = None,
                       refresh_program_codes: bool = False,
                       assume_yes: bool = False,
                       dry_run: bool = False) -> subprocess.CompletedProcess:
    """Print, confirm, and execute the MPC ADES XML curl submission."""

    ades_path = ades_file.expanduser().resolve()
    summary = read_ades_xml_summary(ades_path)
    submit_target = normalize_target(target or summary.target)
    submit_obscode = observatory_code or summary.observatory_code
    submit_ack = ack or format_ack(submit_target)
    submit_obj_type = normalize_obj_type(obj_type)
    submit_prog = resolve_submit_prog(
        submit_obscode,
        explicit_prog=prog,
        program_code_contact=program_code_contact,
        program_codes_cache=program_codes_cache,
        program_codes_sbsar_csv=program_codes_sbsar_csv,
        refresh_program_codes=refresh_program_codes,
    )
    curl_args = build_curl_args(
        endpoint=endpoint,
        ades_file=ades_path,
        ack=submit_ack,
        ac2=ac2,
        obj_type=submit_obj_type,
        prog=submit_prog,
    )
    print("About to submit ADES XML to the MPC.")
    print("ADES file: %s (local path redacted)" % ades_path.name)
    print("Target: %s" % submit_target)
    print("Observatory code: %s" % submit_obscode)
    print("ACK: %s" % submit_ack)
    print("AC2: %s" % ac2)
    print("Object type: %s" % submit_obj_type)
    print("prog: %s" % submit_prog)
    print("Endpoint: %s" % endpoint)
    print("Curl command:")
    print(shlex.join(display_curl_args(curl_args)))

    if dry_run:
        print("Dry run: not submitting.")
        return subprocess.CompletedProcess(curl_args, 0, "", "")

    if not assume_yes:
        prompt = (
            'Are you sure you want to submit %s to the MPC? Type "submit" '
            "to submit. " % ades_path.name)
        if input(prompt).strip() != "submit":
            raise SubmissionError("submission cancelled")

    result = subprocess.run(
        curl_args,
        check=False,
        text=True,
        capture_output=True,
        cwd=ades_path.parent,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr,
              end="" if result.stderr.endswith("\n") else "\n")
    if result.returncode != 0:
        raise SubmissionError("curl exited with status %d" %
                              result.returncode)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="interactively submit an ADES XML file to the MPC")
    parser.add_argument("ades_file", type=Path,
                        help="schema-valid ADES XML file to submit")
    parser.add_argument("--endpoint", default=DEFAULT_TEST_ENDPOINT,
                        help="MPC XML endpoint; defaults to submit_xml_test")
    parser.add_argument("--live", action="store_true",
                        help="use the live submit_xml endpoint")
    parser.add_argument(
        "--obj-type",
        default=DEFAULT_OBJ_TYPE,
        help=("MPC obj_type form field; default: tno. Valid options: %s" %
              ", ".join(VALID_OBJ_TYPES)))
    parser.add_argument("--ac2", default=DEFAULT_AC2_CONTACTS,
                        help=("MPC ac2 form field; default: %s" %
                              DEFAULT_AC2_CONTACTS))
    parser.add_argument("--ack",
                        help="MPC ack form field; default is target plus "
                             "Small Body Search and Rescue")
    parser.add_argument("--target",
                        help="target designation for ACK; defaults to XML")
    parser.add_argument("--observatory-code",
                        help="MPC observatory code; defaults to XML")
    parser.add_argument("--prog",
                        help="explicit base-62 prog form field")
    parser.add_argument("--program-code-contact",
                        default=DEFAULT_PROGRAM_CODE_CONTACT,
                        help="contact lookup key for program code")
    parser.add_argument("--program-codes-cache", type=Path,
                        help="local MPC program-code JSON cache path")
    parser.add_argument("--program-codes-sbsar-csv", type=Path,
                        help="local SBSAR site/program-code CSV path")
    parser.add_argument("--refresh-program-codes", action="store_true",
                        help="refresh the local MPC program-code JSON cache")
    parser.add_argument("--yes", action="store_true",
                        help="skip the interactive confirmation")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command and metadata without posting")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    endpoint = DEFAULT_LIVE_ENDPOINT if args.live else args.endpoint
    try:
        submit_ades_to_mpc(
            ades_file=args.ades_file,
            endpoint=endpoint,
            obj_type=args.obj_type,
            ac2=args.ac2,
            ack=args.ack,
            target=args.target,
            observatory_code=args.observatory_code,
            prog=args.prog,
            program_code_contact=args.program_code_contact,
            program_codes_cache=args.program_codes_cache,
            program_codes_sbsar_csv=args.program_codes_sbsar_csv,
            refresh_program_codes=args.refresh_program_codes,
            assume_yes=args.yes,
            dry_run=args.dry_run,
        )
    except SubmissionError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
