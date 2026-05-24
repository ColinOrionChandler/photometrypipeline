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
    assert "COD W84" in bundle.obs80_text

    obs_lines = [
        line for line in bundle.obs80_text.splitlines()
        if line.startswith("     K16CF5J")
    ]
    assert len(obs_lines) == 1
    assert len(obs_lines[0]) == 80
    assert obs_lines[0].endswith("W84")


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
