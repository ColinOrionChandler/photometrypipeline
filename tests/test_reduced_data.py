import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time


import _pp_conf
import pp_extract
import pp_run
import pptool_reduced_data as reduced


def write_arctic(path, target="2026 PD1", filt="SDSS r #1",
                 shape=(8, 8), date_obs="2026-08-16T07:05:12.411360"):
    header = fits.Header()
    header["INSTRUME"] = "arctic"
    header["TELESCOP"] = "3.5m"
    header["OBJNAME"] = target
    header["FILTER"] = filt
    header["CCDBIN1"] = 2
    header["CCDBIN2"] = 2
    header["DATE-OBS"] = date_obs
    header["TIMESYS"] = "TAI"
    header["EXPTIME"] = 120.0
    header["RA"] = "20:02:23.32"
    header["DEC"] = "-20:23:05.05"
    header["AIRMASS"] = 1.2
    fits.PrimaryHDU(data=np.ones(shape, dtype=np.float32),
                    header=header).writeto(path)
    return path


def workflow_args(input_dir, output_dir, prepare_only=True,
                  mask_quad="lower-left", mask_file=None):
    return argparse.Namespace(
        input_dir=str(input_dir), output_dir=str(output_dir),
        prepare_only=prepare_only, mask_quad=mask_quad,
        mask_file=mask_file, resume=False)


def test_target_directory_name_replaces_whitespace_and_slash():
    assert reduced.target_directory_name(" C/2020   A1 ") == "C+2020_A1"


def test_arctic_filter_aliases_include_canonical_and_numbered_g():
    translations = _pp_conf.telescope_parameters[
        "ARC35ARCTIC"]["filter_translations"]
    assert translations["SDSS g #1"] == "g"
    assert translations["CU VR"] == "r"
    assert translations["g"] == "g"
    assert translations["r"] == "r"


def test_prepare_tree_uses_band_subdirs_only_for_multiband_targets(tmp_path):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    write_arctic(input_dir / "single.fits", target="C/2020 A1", filt="CU VR")
    write_arctic(input_dir / "multi_r.fits", target="2026 PD1",
                 filt="SDSS r #1")
    write_arctic(input_dir / "multi_g.fits", target="2026 PD1",
                 filt="SDSS g #1")
    output_dir = tmp_path / "PP"

    result = reduced.run_workflow(workflow_args(input_dir, output_dir))

    assert result["workflow_status"] == "prepared"
    assert result["source_count"] == 3
    assert result["object_count"] == 2
    assert result["group_count"] == 3
    assert (output_dir / "C+2020_A1" / "single.fits").is_file()
    assert not (output_dir / "C+2020_A1" / "r").exists()
    assert (output_dir / "2026_PD1" / "r" / "multi_r.fits").is_file()
    assert (output_dir / "2026_PD1" / "g" / "multi_g.fits").is_file()
    assert not (output_dir / "C+2020_A1" / "single.fits").is_symlink()
    assert result["sources_unchanged"] is True


def test_working_header_normalizes_filter_target_and_tai_time(tmp_path):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    source = write_arctic(input_dir / "frame.fits", target="2026 PD1",
                          filt="CU VR")
    source_hash = reduced.sha256(source)
    output_dir = tmp_path / "PP"

    reduced.run_workflow(workflow_args(input_dir, output_dir))

    working = output_dir / "2026_PD1" / "frame.fits"
    header = fits.getheader(working)
    assert header["ORIGFILT"] == "CU VR"
    assert header["FILTER"] == "r"
    assert header["OBJECT"] == "2026 PD1"
    assert header["DATEORIG"] == "2026-08-16T07:05:12.411360"
    assert header["TSYSORIG"] == "TAI"
    assert header["TIMESYS"] == "UTC"
    expected = Time("2026-08-16T07:05:12.411360",
                    format="isot", scale="tai").utc
    actual = Time(header["DATE-OBS"], format="isot", scale="utc")
    assert abs((actual - expected).sec) < 1e-6
    assert reduced.sha256(source) == source_hash


def test_generated_quadrant_mask_has_binary_weight_geometry(tmp_path):
    mask_path = tmp_path / "mask.fits"
    info = reduced.create_quadrant_mask(mask_path, (8, 8), "lower-left")
    data = fits.getdata(mask_path)
    assert np.all(data[:4, :4] == 0)
    assert np.all(data[:4, 4:] == 1)
    assert np.all(data[4:, :] == 1)
    assert info["bad_pixels"] == 16
    assert info["good_pixels"] == 48


def test_mask_validation_separates_rejected_negative_flux_boundary_row(
        tmp_path):
    catalog = tmp_path / "image.ldac.csv"
    catalog.write_text(
        "XWIN_IMAGE,YWIN_IMAGE,FLUX_AUTO,MAG_AUTO\n"
        "900.0,1023.8,-100.0,99.0\n"
        "800.0,800.0,100.0,20.0\n")

    result = reduced.validate_group_catalogs(
        {"working_dir": str(tmp_path)}, (2048, 2048), "lower-left")

    assert result["masked_detection_count"] == 1
    assert result["masked_rejected_source_count"] == 1
    assert result["masked_rejected_sources"][0]["mag_auto"] == 99.0


def test_resolve_target_photometry_accepts_horizons_canonical_name(tmp_path):
    canonical = tmp_path / "photometry_85490_(1997_SE5).dat"
    canonical.write_text("active target row\n")
    (tmp_path / "photometry_Control_Star.dat").write_text(
        "control row\n")
    assert reduced.resolve_target_photometry(tmp_path, "1997 SE5") == canonical

    numbered = tmp_path / "photometry_248370_(2005_QN173).dat"
    canonical.unlink()
    numbered.write_text("active comet row\n")
    assert reduced.resolve_target_photometry(tmp_path, "433P") == numbered


def test_preflight_rejects_sanitized_target_collision(tmp_path):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    write_arctic(input_dir / "one.fits", target="A B")
    write_arctic(input_dir / "two.fits", target="A_B")
    with pytest.raises(ValueError, match="collision"):
        reduced.preflight(input_dir, tmp_path / "PP")


def test_mask_override_validates_shape_and_restores_global_config(tmp_path):
    image = write_arctic(tmp_path / "image.fits", shape=(8, 8))
    good_mask = tmp_path / "good_mask.fits"
    reduced.create_quadrant_mask(good_mask, (8, 8), "lower-left")
    original = _pp_conf.telescope_parameters["ARC35ARCTIC"]

    with pp_run.telescope_mask_override(
            [str(image)], telescope="ARC35ARCTIC",
            mask_file=str(good_mask)) as resolved:
        active = _pp_conf.telescope_parameters["ARC35ARCTIC"]
        assert active is not original
        assert active["mask_file"]["2,2"] == resolved
    assert _pp_conf.telescope_parameters["ARC35ARCTIC"] is original

    bad_mask = tmp_path / "bad_mask.fits"
    fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.float32)).writeto(bad_mask)
    with pytest.raises(ValueError, match="does not match"):
        pp_run.validate_mask_file(str(bad_mask), [str(image)])


def test_source_extractor_keeps_spaced_mask_path_as_one_argument(
        tmp_path, monkeypatch):
    image = write_arctic(tmp_path / "image.fits")
    mask_dir = tmp_path / "mask files"
    mask_dir.mkdir()
    mask_path = mask_dir / "weight map.fits"
    reduced.create_quadrant_mask(mask_path, (8, 8), "lower-left")
    captured = []

    class DummyProcess:
        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        captured.append(command)
        return DummyProcess()

    monkeypatch.setattr(pp_extract, "_sextractor_cmd", lambda: "sex")
    monkeypatch.setattr(pp_extract.subprocess, "Popen", fake_popen)
    parameters = {
        "aperture_diam": "8", "source_minarea": 3,
        "sex_snr": 1.5, "mask_file": str(mask_path),
        "obsparam": _pp_conf.telescope_parameters["ARC35ARCTIC"],
        "quiet": True,
    }
    assert pp_extract.extract_singleframe(
        (parameters, str(image))) is None
    weight_index = captured[0].index("-WEIGHT_IMAGE")
    assert captured[0][weight_index + 1] == str(mask_path)


def test_batch_continues_after_group_failure_and_records_manifest(
        tmp_path, monkeypatch):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    write_arctic(input_dir / "a.fits", target="A")
    write_arctic(input_dir / "b.fits", target="B")
    output_dir = tmp_path / "PP"
    calls = []

    def fake_run(command, cwd, env, stdout, stderr, text):
        calls.append(Path(cwd).name)
        if Path(cwd).name == "A":
            (Path(cwd) / "photometry_A.dat").write_text("a data row\n")
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(reduced.subprocess, "run", fake_run)
    result = reduced.run_workflow(workflow_args(
        input_dir, output_dir, prepare_only=False))

    assert calls == ["A", "B"]
    assert [group["status"] for group in result["groups"]] == [
        "complete", "failed"]
    assert result["workflow_status"] == "failed"
    manifest = json.loads((output_dir / reduced.MANIFEST_NAME).read_text())
    assert manifest["groups"][1]["failure_reason"] == "pp_returncode_2"
    assert manifest["sources_unchanged"] is True


def test_resume_reruns_only_incomplete_groups(tmp_path, monkeypatch):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    write_arctic(input_dir / "a.fits", target="A")
    write_arctic(input_dir / "b.fits", target="B")
    output_dir = tmp_path / "PP"

    def first_run(command, cwd, env, stdout, stderr, text):
        if Path(cwd).name == "A":
            (Path(cwd) / "photometry_A.dat").write_text("a data row\n")
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(reduced.subprocess, "run", first_run)
    first = reduced.run_workflow(workflow_args(
        input_dir, output_dir, prepare_only=False))
    assert first["workflow_status"] == "failed"
    first["groups"][1]["keep_wcs"] = True
    reduced.write_manifest(first)

    calls = []

    def resumed_run(command, cwd, env, stdout, stderr, text):
        calls.append((Path(cwd).name, command))
        (Path(cwd) / "photometry_B.dat").write_text("b data row\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(reduced.subprocess, "run", resumed_run)
    args = workflow_args(input_dir, output_dir, prepare_only=False,
                         mask_quad=None)
    args.resume = True
    resumed = reduced.run_workflow(args)

    assert [call[0] for call in calls] == ["B"]
    assert "-keep_wcs" in calls[0][1]
    assert resumed["workflow_status"] == "complete"
    assert resumed["resume_count"] == 1
    assert resumed["groups"][1]["attempts"][0]["status"] == "failed"
    assert Path(resumed["groups"][1]["stdout"]).name == (
        "pp_run_stdout_attempt_2.txt")


def test_existing_nonempty_output_is_rejected(tmp_path):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    write_arctic(input_dir / "frame.fits")
    output_dir = tmp_path / "PP"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("do not mix\n")
    with pytest.raises(FileExistsError, match="not empty"):
        reduced.preflight(input_dir, output_dir)
