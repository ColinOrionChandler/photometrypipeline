from pathlib import Path
import subprocess

import pytest

import pptool_mpcsubmission as mpcsub
import pptool_submit_ades_to_mpc as submit_ades


ADES_XML = """<?xml version='1.0' encoding='utf-8'?>
<ades version="2022">
  <obsBlock>
    <obsContext>
      <observatory>
        <mpcCode>568</mpcCode>
      </observatory>
    </obsContext>
    <obsData>
      <optical>
        <provID>2025 MH348</provID>
        <mode>CCD</mode>
        <stn>568</stn>
        <obsTime>2002-06-09T12:21:55.924Z</obsTime>
        <ra>1.0</ra>
        <dec>2.0</dec>
        <astCat>Gaia2</astCat>
      </optical>
    </obsData>
  </obsBlock>
</ades>
"""


def test_read_ades_xml_summary_and_ack(tmp_path):
    ades_file = tmp_path / "mpc_2025_MH348_ADES.xml"
    ades_file.write_text(ADES_XML)

    summary = submit_ades.read_ades_xml_summary(ades_file)

    assert summary.target == "2025 MH348"
    assert summary.observatory_code == "568"
    assert (
        submit_ades.format_ack(summary.target) ==
        "2025 MH348 Small Body Search and Rescue")


def test_resolve_submit_prog_prefers_sbsar_base62(tmp_path):
    csv_path = tmp_path / "sbsar_program_codes.csv"
    csv_path.write_text("site_code,program_code,base62_code\n568,c,18\n")

    assert submit_ades.resolve_submit_prog(
        "568",
        program_codes_sbsar_csv=csv_path,
        program_codes_cache=tmp_path / "missing.json",
    ) == "18"


def test_build_curl_args_contains_ack_ac2_prog_and_source(tmp_path):
    ades_file = tmp_path / "with spaces.xml"
    ades_file.write_text(ADES_XML)

    args = submit_ades.build_curl_args(
        endpoint=submit_ades.DEFAULT_TEST_ENDPOINT,
        ades_file=ades_file,
        ack="2025 MH348 Small Body Search and Rescue",
        ac2=mpcsub.DEFAULT_AC2_CONTACTS,
        obj_type="NEO",
        prog="18",
    )

    assert args[:2] == ["curl", submit_ades.DEFAULT_TEST_ENDPOINT]
    assert "ack=2025 MH348 Small Body Search and Rescue" in args
    assert "ac2=coc123@uw.edu, murtagh@uw.edu" in args
    assert "prog=18" in args
    assert "obj_type=NEO" in args
    assert "source=<%s" % ades_file.resolve() in args


def test_obj_type_defaults_and_normalizes_to_mpc_form_values():
    assert submit_ades.DEFAULT_OBJ_TYPE == "tno"
    assert submit_ades.normalize_obj_type("TNO") == "tno"
    assert submit_ades.normalize_obj_type("neo candidate") == "neo candidate"
    assert submit_ades.normalize_obj_type(" New   Comet ") == "new comet"
    with pytest.raises(mpcsub.SubmissionError, match="valid options"):
        submit_ades.normalize_obj_type("centaur")


def test_cli_help_lists_obj_types_and_ac2_default(capsys):
    with pytest.raises(SystemExit):
        submit_ades.parse_args(["--help"])
    output = capsys.readouterr().out

    assert "default: tno" in output
    for obj_type in submit_ades.VALID_OBJ_TYPES:
        assert obj_type in output
    assert "default: coc123@uw.edu" in output
    assert "murtagh@uw.edu" in output


def test_submit_ades_dry_run_prints_curl_and_skips_prompt(tmp_path, capsys):
    ades_file = tmp_path / "mpc_2025_MH348_ADES.xml"
    ades_file.write_text(ADES_XML)
    csv_path = tmp_path / "sbsar_program_codes.csv"
    csv_path.write_text("site_code,program_code,base62_code\n568,c,18\n")

    result = submit_ades.submit_ades_to_mpc(
        ades_file,
        endpoint=submit_ades.DEFAULT_TEST_ENDPOINT,
        program_codes_sbsar_csv=csv_path,
        dry_run=True,
    )
    output = capsys.readouterr().out

    assert result.returncode == 0
    assert "Curl command:" in output
    assert "ack=2025 MH348 Small Body Search and Rescue" in output
    assert "ac2=coc123@uw.edu, murtagh@uw.edu" in output
    assert "obj_type=tno" in output
    assert "prog=18" in output
    assert "Dry run: not submitting." in output


def test_submit_ades_requires_literal_submit(tmp_path, monkeypatch):
    ades_file = tmp_path / "mpc_2025_MH348_ADES.xml"
    ades_file.write_text(ADES_XML)
    csv_path = tmp_path / "sbsar_program_codes.csv"
    csv_path.write_text("site_code,program_code,base62_code\n568,c,18\n")
    monkeypatch.setattr("builtins.input", lambda prompt: "nope")

    with pytest.raises(mpcsub.SubmissionError, match="submission cancelled"):
        submit_ades.submit_ades_to_mpc(
            ades_file,
            endpoint=submit_ades.DEFAULT_TEST_ENDPOINT,
            program_codes_sbsar_csv=csv_path,
        )


def test_submit_ades_executes_curl_after_confirmation(tmp_path, monkeypatch):
    ades_file = tmp_path / "mpc_2025_MH348_ADES.xml"
    ades_file.write_text(ADES_XML)
    csv_path = tmp_path / "sbsar_program_codes.csv"
    csv_path.write_text("site_code,program_code,base62_code\n568,c,18\n")
    monkeypatch.setattr("builtins.input", lambda prompt: "submit")
    calls = []

    def fake_run(args, check, text, capture_output):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "accepted\n", "")

    monkeypatch.setattr(submit_ades.subprocess, "run", fake_run)

    result = submit_ades.submit_ades_to_mpc(
        ades_file,
        endpoint=submit_ades.DEFAULT_TEST_ENDPOINT,
        program_codes_sbsar_csv=csv_path,
    )

    assert result.returncode == 0
    assert len(calls) == 1
    assert "ack=2025 MH348 Small Body Search and Rescue" in calls[0]
    assert "ac2=coc123@uw.edu, murtagh@uw.edu" in calls[0]
    assert "prog=18" in calls[0]
