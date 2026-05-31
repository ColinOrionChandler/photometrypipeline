from pathlib import Path

import pytest
from astropy.io import fits

import pptool_mpcsubmission as mpcsub


REAL_CJ155_DIR = Path(
    "/Users/orionadmin/Dropbox/PP_to_fold_into_my_GitHub/2016_CJ155")


PHOTOMETRY_TEXT = """# header
 c4d_test 2457398.7899512  21.3285 0.0877  144.17425351  +13.91718331 -0.25  2.36  0.00  0.00 47.00  29.4913 0.0342  -8.1628 0.0808 SDSS-R9 r   0 DECam      APER 0.87  0.0808  0.0342  0.0877  0.0121  0.0121  0.0171  0.0638  0.0816  0.1036  0.0650  0.0825  0.1050
"""


def write_test_fits(path):
    header = fits.Header()
    header["OBSERVER"] = "Beaudin, Moustakas, Blum"
    header["PROPID"] = "2014B-0404"
    header["PROPOSER"] = "Schlegel"
    header["TELESCOP"] = "CTIO 4.0-m telescope"
    header["INSTRUME"] = "DECam"
    header["TEL_KEYW"] = "DECam"
    fits.PrimaryHDU(header=header).writeto(path)


def write_spacewatch_test_fits(path):
    header = fits.Header()
    header["TELESCOP"] = "Spacewatch 0.9-m f/3 prime focus"
    header["INSTRUME"] = "Spacewatch Mosaic Camera"
    header["TEL_KEYW"] = "SPACEWATCH09"
    fits.PrimaryHDU(header=header).writeto(path)


def test_target_filename_and_packed_designation():
    assert mpcsub.target_to_filename("2016 CJ155") == "2016_CJ155"
    assert (
        mpcsub.pack_minor_planet_provisional_designation("2016 CJ155") ==
        "K16CF5J")
    assert (
        mpcsub.pack_minor_planet_provisional_designation("2016_CJ155") ==
        "K16CF5J")


def test_builds_ades_and_80col_from_synthetic_pp_output(tmp_path):
    photometry = tmp_path / "photometry_2016_CJ155.dat"
    photometry.write_text(PHOTOMETRY_TEXT)
    write_test_fits(tmp_path / "c4d_test.fits")

    config = mpcsub.SubmissionConfig(target="2016 CJ155",
                                     output_dir=tmp_path)
    bundle = mpcsub.build_submission(tmp_path, config)

    assert len(bundle.observations) == 1
    assert "provID|trkSub|mode|stn|obsTime" in bundle.ades_text
    assert "|2016 CJ155||CCD|W84|" in bundle.ades_text
    assert "|Gaia2|SDSS-R9|21.3285|0.0877|r|" in bundle.ades_text
    assert "! name Beaudin" in bundle.ades_text
    assert "! name C. O. Chandler" in bundle.ades_text
    assert "! name J. Murtagh" in bundle.ades_text
    assert "COD W84" in bundle.obs80_text
    assert "MEA C. O. Chandler, J. Murtagh" in bundle.obs80_text

    obs_lines = [
        line for line in bundle.obs80_text.splitlines()
        if line.startswith("     K16CF5J")
    ]
    assert len(obs_lines) == 1
    assert len(obs_lines[0]) == 80
    assert obs_lines[0].endswith("W84")


def test_observer_names_use_initials_for_first_last_names(tmp_path):
    photometry = tmp_path / "photometry_2016_CJ155.dat"
    photometry.write_text(PHOTOMETRY_TEXT)
    write_test_fits(tmp_path / "c4d_test.fits")

    config = mpcsub.SubmissionConfig(
        target="2016 CJ155",
        observers=["David James", "Liz Buckley-Geer",
                   "C. O. Chandler", "John Quincy Public"],
        output_dir=tmp_path,
    )
    bundle = mpcsub.build_submission(tmp_path, config)

    assert (
        "OBS D. James, L. Buckley-Geer, C. O. Chandler, J. Q. Public" in
        bundle.obs80_text)
    assert "! name D. James" in bundle.ades_text
    assert "! name L. Buckley-Geer" in bundle.ades_text
    assert "! name C. O. Chandler" in bundle.ades_text
    assert "! name J. Q. Public" in bundle.ades_text


def test_observer_override_replaces_fits_observers(tmp_path):
    photometry = tmp_path / "photometry_2016_CJ155.dat"
    photometry.write_text(PHOTOMETRY_TEXT)
    write_test_fits(tmp_path / "c4d_test.fits")

    config = mpcsub.SubmissionConfig(
        target="2016 CJ155",
        observers=["C. O. Chandler", "A. Collaborator"],
        output_dir=tmp_path,
    )
    bundle = mpcsub.build_submission(tmp_path, config)

    assert "OBS C. O. Chandler, A. Collaborator" in bundle.obs80_text
    assert "! name C. O. Chandler" in bundle.ades_text
    assert "! name A. Collaborator" in bundle.ades_text
    assert "! name Beaudin" not in bundle.ades_text


def test_spacewatch_submission_context_uses_09m_telescope(tmp_path):
    photometry = tmp_path / "photometry_2016_CJ155.dat"
    photometry.write_text(PHOTOMETRY_TEXT)
    write_spacewatch_test_fits(tmp_path / "c4d_test.fits")

    config = mpcsub.SubmissionConfig(target="2016 CJ155",
                                     observatory_code="691",
                                     output_dir=tmp_path)
    bundle = mpcsub.build_submission(tmp_path, config)

    assert "! aperture 0.9" in bundle.ades_text
    assert "TEL 0.9-m f/3 reflector + CCD" in bundle.obs80_text


def test_recursive_submission_skips_rejected_only_photometry_files(tmp_path):
    active_dir = tmp_path / "active"
    rejected_dir = tmp_path / "rejected"
    active_dir.mkdir()
    rejected_dir.mkdir()
    active_photometry = active_dir / "photometry_2016_CJ155.dat"
    rejected_photometry = rejected_dir / "photometry_2016_CJ155.dat"
    active_photometry.write_text(PHOTOMETRY_TEXT)
    rejected_photometry.write_text("# all rows were rejected by PP\n")
    write_test_fits(active_dir / "c4d_test.fits")

    config = mpcsub.SubmissionConfig(target="2016 CJ155",
                                     output_dir=tmp_path)
    bundle = mpcsub.build_submission(tmp_path, config)

    assert len(bundle.observations) == 1
    assert any("skipping" in warning and str(rejected_photometry) in warning
               for warning in bundle.warnings)


def test_resolves_truncated_catalog_token_by_unique_prefix(tmp_path):
    photometry = tmp_path / "photometry_2014_FU61.dat"
    row = PHOTOMETRY_TEXT.splitlines()[1].replace(
        "c4d_test", "OMEGA.2014-04-21T02:44:36.768_chip0")
    photometry.write_text("# header\n%s\n" % row)
    full_fits = tmp_path / "OMEGA.2014-04-21T02:44:36.768_chip031.new.fits"
    write_test_fits(full_fits)

    config = mpcsub.SubmissionConfig(target="2014 FU61", output_dir=tmp_path)
    bundle = mpcsub.build_submission(tmp_path, config)

    assert len(bundle.observations) == 1
    assert bundle.observations[0].source_file == full_fits.resolve()


@pytest.mark.skipif(not REAL_CJ155_DIR.exists(),
                    reason="local 2016_CJ155 fixture is unavailable")
def test_real_2016_cj155_outputs_both_formats(tmp_path):
    config = mpcsub.SubmissionConfig(target="2016 CJ155",
                                     output_dir=tmp_path)
    bundle = mpcsub.build_submission(REAL_CJ155_DIR, config)
    mpcsub.write_submission(bundle)

    assert len(bundle.observations) == 2
    ades_text = bundle.ades_path.read_text()
    obs80_text = bundle.obs80_path.read_text()
    assert "|2016 CJ155||CCD|W84|" in ades_text

    obs_lines = [
        line for line in obs80_text.splitlines()
        if line.startswith("     K16CF5J")
    ]
    assert len(obs_lines) == 2
    assert all(len(line) == 80 for line in obs_lines)
    assert all(line.endswith("W84") for line in obs_lines)
