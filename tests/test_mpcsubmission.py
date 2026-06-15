from pathlib import Path

import pytest
from astropy.io import fits

import pptool_mpcsubmission as mpcsub


REAL_CJ155_DIR = Path(
    "/Users/orionadmin/Dropbox/PP_to_fold_into_my_GitHub/2016_CJ155")


PHOTOMETRY_TEXT = """# header
 c4d_test 2457398.7899512  21.3285 0.0877  144.17425351  +13.91718331 -0.25  2.36  0.00  0.00 47.00  29.4913 0.0342  -8.1628 0.0808 SDSS-R9 r   0 DECam      APER 0.87  0.0808  0.0342  0.0877  0.0121  0.0121  0.0171  0.0638  0.0816  0.1036  0.0650  0.0825  0.1050
"""

PHOTOMETRY_TEXT_ASTROMETRY_ONLY = PHOTOMETRY_TEXT.replace(
    "21.3285 0.0877", "125.1690 99.0000", 1)


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


def write_sdss_test_fits(path):
    header = fits.Header()
    header["TELESCOP"] = "2.5-m SDSS telescope"
    header["INSTRUME"] = "SDSS imaging camera"
    header["TEL_KEYW"] = "SDSS"
    fits.PrimaryHDU(header=header).writeto(path)


def write_wht_test_fits(path):
    header = fits.Header()
    header["TELESCOP"] = "4.2-m William Herschel Telescope"
    header["INSTRUME"] = "WHT Prime Focus Imaging Platform"
    header["TEL_KEYW"] = "WHTPFIP"
    fits.PrimaryHDU(header=header).writeto(path)


def write_cfht_cfh12k_test_fits(path):
    header = fits.Header()
    header["TELESCOP"] = "CFHT 3.6m"
    header["INSTRUME"] = "CFH12K Mosaic"
    header["TEL_KEYW"] = "CFHTCFH12K"
    header["MAGZP"] = 26.169
    fits.PrimaryHDU(header=header).writeto(path)


def synthetic_obs80_line(tmp_path, *, observatory_code, non_survey_measurer):
    photometry = tmp_path / "photometry_2016_CJ155.dat"
    photometry.write_text(PHOTOMETRY_TEXT)
    write_test_fits(tmp_path / "c4d_test.fits")

    config = mpcsub.SubmissionConfig(
        target="2016 CJ155",
        observatory_code=observatory_code,
        non_survey_measurer=non_survey_measurer,
        output_dir=tmp_path,
    )
    bundle = mpcsub.build_submission(tmp_path, config)
    obs_lines = [
        line for line in bundle.obs80_text.splitlines()
        if line.startswith("     K16CF5J")
    ]
    assert len(obs_lines) == 1
    return obs_lines[0]


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
    assert "# version=2022" in bundle.ades_text
    assert bundle.ades_xml_text.startswith("<?xml")
    assert "<ades version=\"2022\">" in bundle.ades_xml_text
    assert "<provID>2016 CJ155</provID>" in bundle.ades_xml_text
    assert "<prog>" not in bundle.ades_xml_text
    assert bundle.ades_xml_path.name == "mpc_2016_CJ155_ADES.xml"
    assert "provID|trkSub|mode|stn|obsTime" in bundle.ades_text
    assert "|2016 CJ155||CCD|W84|" in bundle.ades_text
    assert "|Gaia2|SDSS8|21.3|0.09|r|" in bundle.ades_text
    assert "photCal=SDSS9" in bundle.ades_text
    assert "! name Beaudin" in bundle.ades_text
    assert "! name C. O. Chandler" in bundle.ades_text
    assert "! name J. Murtagh" in bundle.ades_text
    assert "COD W84" in bundle.obs80_text
    assert "MEA C. O. Chandler, J. Murtagh" in bundle.obs80_text
    mpcsub.write_submission(bundle)
    assert bundle.ades_path.exists()
    assert bundle.ades_xml_path.exists()
    assert bundle.obs80_path.exists()

    obs_lines = [
        line for line in bundle.obs80_text.splitlines()
        if line.startswith("     K16CF5J")
    ]
    assert len(obs_lines) == 1
    assert len(obs_lines[0]) == 80
    assert obs_lines[0].endswith("W84")


def test_ades_keeps_valid_photcat_catalog_code(tmp_path):
    photometry = tmp_path / "photometry_2016_CJ155.dat"
    photometry.write_text(PHOTOMETRY_TEXT)
    write_test_fits(tmp_path / "c4d_test.fits")

    config = mpcsub.SubmissionConfig(
        target="2016 CJ155", photcat="Gaia2", output_dir=tmp_path)
    bundle = mpcsub.build_submission(tmp_path, config)

    assert "|Gaia2|Gaia2|21.3|0.09|r|" in bundle.ades_text
    validation = mpcsub.validate_ades_psv_text(
        bundle.ades_text,
        catalog_values=({"Gaia2"}, {"SDSS8"}))
    assert validation["errors"] == []


def test_forced_photometry_uses_remarks_not_photcat(tmp_path):
    photometry = tmp_path / "photometry_2025_MH348.dat"
    photometry.write_text(
        PHOTOMETRY_TEXT.replace("c4d_test", "cfh12k_i")
        .replace("2016_CJ155", "2025_MH348")
        .replace("21.3285 0.0877", "23.7504 0.1358")
        .replace("SDSS-R9 r   0 DECam      APER",
                 "forced_photometry I 0 CFHTCFH12K APER_FORCED"))
    write_cfht_cfh12k_test_fits(tmp_path / "cfh12k_i.fits")

    config = mpcsub.SubmissionConfig(
        target="2025 MH348", observatory_code="568", prog="c",
        output_dir=tmp_path)
    bundle = mpcsub.build_submission(tmp_path, config)

    assert "! aperture 3.6" in bundle.ades_text
    assert "|Gaia2|SDSS8|23.8|0.14|I|" in bundle.ades_text
    assert "forced aperture" in bundle.ades_text
    assert "photCal=SDSS9" in bundle.ades_text
    assert "forced_photometry|23.8" not in bundle.ades_text
    validation = mpcsub.validate_ades_psv_text(
        bundle.ades_text,
        catalog_values=({"Gaia2"}, {"SDSS8"}))
    assert validation["errors"] == []


def test_compact_ades_validator_reports_current_schema_errors():
    text = (
        "# version=2017\n"
        "# telescope\n"
        "! aperture 0.0\n"
        "permID|provID|mode|stn|obsTime|ra|dec|astCat|photCat|mag|rmsMag|band\n"
        "|2025 MH348|CCD|568|2002-06-09T12:21:55Z|1|2|Gaia2|"
        "forced_photometry|23.8|0.14|I\n")

    result = mpcsub.validate_ades_psv_text(
        text, catalog_values=({"Gaia2"}, set()))

    assert any("version must be 2022" in error
               for error in result["errors"])
    assert any("aperture must be positive" in error
               for error in result["errors"])
    assert any("photCat has invalid CatType syntax" in error
               for error in result["errors"])


def test_program_code_lookup_prefers_sbsar_csv(tmp_path):
    csv_path = tmp_path / "sbsar_program_codes.csv"
    csv_path.write_text("site_code,program_code\n568,c\n")
    config = mpcsub.SubmissionConfig(
        target="2025 MH348",
        observatory_code="568",
        program_codes_sbsar_csv=csv_path,
        program_codes_cache=tmp_path / "missing_cache.json")
    warnings = []

    assert mpcsub.resolve_program_code(config, warnings) == "c"
    assert warnings == []


def test_program_code_lookup_uses_mpc_cache_for_punctuation_code(tmp_path):
    cache_path = tmp_path / "program_codes.json"
    cache_path.write_text(
        '{"program_codes": ['
        '{"observatory_code": "F51", "program_code": "&", '
        '"contact_name": "C. O. Chandler"}]}')
    config = mpcsub.SubmissionConfig(
        target="2016 CJ155",
        observatory_code="F51",
        program_codes_sbsar_csv=tmp_path / "missing.csv",
        program_codes_cache=cache_path)

    assert mpcsub.resolve_program_code(config, []) == "&"


def test_program_code_lookup_missing_warns_and_leaves_blank(tmp_path,
                                                            monkeypatch):
    def fail_fetch(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(mpcsub, "fetch_program_codes", fail_fetch)
    config = mpcsub.SubmissionConfig(
        target="2016 CJ155",
        observatory_code="ZZZ",
        program_codes_sbsar_csv=tmp_path / "missing.csv",
        program_codes_cache=tmp_path / "missing_cache.json")
    warnings = []

    assert mpcsub.resolve_program_code(config, warnings) is None
    assert any("no program code found" in warning for warning in warnings)


def test_submission_outputs_include_resolved_program_code(tmp_path):
    photometry = tmp_path / "photometry_2025_MH348.dat"
    photometry.write_text(PHOTOMETRY_TEXT.replace("2016_CJ155", "2025_MH348"))
    write_test_fits(tmp_path / "c4d_test.fits")
    csv_path = tmp_path / "sbsar_program_codes.csv"
    csv_path.write_text("site_code,program_code\n568,c\n")

    config = mpcsub.SubmissionConfig(
        target="2025 MH348",
        observatory_code="568",
        program_codes_sbsar_csv=csv_path,
        output_dir=tmp_path)
    bundle = mpcsub.build_submission(tmp_path, config)

    assert "|c|" in bundle.ades_text
    obs_lines = [
        line for line in bundle.obs80_text.splitlines()
        if line.startswith("     K25MY8H")
    ]
    assert len(obs_lines) == 1
    assert obs_lines[0][76] == "c"


def test_missing_photometry_row_is_astrometry_only_observation(tmp_path):
    photometry = tmp_path / "photometry_2016_CJ155.dat"
    photometry.write_text(PHOTOMETRY_TEXT_ASTROMETRY_ONLY)
    write_test_fits(tmp_path / "c4d_test.fits")

    config = mpcsub.SubmissionConfig(target="2016 CJ155",
                                     output_dir=tmp_path)
    bundle = mpcsub.build_submission(tmp_path, config)

    assert len(bundle.observations) == 1
    assert bundle.observations[0].astrometry_only
    assert bundle.observations[0].mag is None
    assert bundle.observations[0].mag_sig is None
    assert "Astrometry-only observations: 1" in bundle.summary_text

    ades_rows = [
        line for line in bundle.ades_text.splitlines()
        if line.startswith("|2016 CJ155||CCD|")
    ]
    assert len(ades_rows) == 1
    fields = ades_rows[0].split("|")
    assert fields[10] == "Gaia2"
    assert fields[11:15] == ["", "", "", ""]
    assert fields[15] == "47"

    obs_lines = [
        line for line in bundle.obs80_text.splitlines()
        if line.startswith("     K16CF5J")
    ]
    assert len(obs_lines) == 1
    assert len(obs_lines[0]) == 80
    assert obs_lines[0][65:71].strip() == ""
    assert obs_lines[0].endswith("W84")


def test_f51_non_survey_sets_obsnote_z_in_column_72(tmp_path):
    line = synthetic_obs80_line(
        tmp_path,
        observatory_code="F51",
        non_survey_measurer=True,
    )

    assert len(line) == 80
    assert line[70] == "r"
    assert line[71] == "Z"
    assert line[77:80] == "F51"


def test_f51_survey_leaves_obsnote_blank(tmp_path):
    line = synthetic_obs80_line(
        tmp_path,
        observatory_code="F51",
        non_survey_measurer=False,
    )

    assert len(line) == 80
    assert line[71] == " "
    assert line[77:80] == "F51"


def test_non_special_site_leaves_non_survey_obsnote_blank(tmp_path):
    line = synthetic_obs80_line(
        tmp_path,
        observatory_code="W84",
        non_survey_measurer=True,
    )

    assert len(line) == 80
    assert line[71] == " "
    assert line[77:80] == "W84"


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


def test_sdss_submission_context_uses_25m_telescope(tmp_path):
    photometry = tmp_path / "photometry_2016_CJ155.dat"
    photometry.write_text(PHOTOMETRY_TEXT.replace("DECam", "SDSS"))
    write_sdss_test_fits(tmp_path / "c4d_test.fits")

    config = mpcsub.SubmissionConfig(target="2016 CJ155",
                                     observatory_code="645",
                                     astcat="SDSS-R9",
                                     photcat="SDSS-R9",
                                     output_dir=tmp_path)
    bundle = mpcsub.build_submission(tmp_path, config)

    assert "! aperture 2.5" in bundle.ades_text
    assert "TEL 2.5-m SDSS telescope + CCD" in bundle.obs80_text
    assert "COD 645" in bundle.obs80_text
    assert "NET SDSS-R9" in bundle.obs80_text


def test_wht_submission_context_uses_42m_telescope(tmp_path):
    photometry = tmp_path / "photometry_1998_QJ1.dat"
    photometry.write_text(PHOTOMETRY_TEXT.replace("2016_CJ155", "1998_QJ1"))
    write_wht_test_fits(tmp_path / "c4d_test.fits")

    config = mpcsub.SubmissionConfig(target="1998 QJ1",
                                     observatory_code="950",
                                     output_dir=tmp_path)
    bundle = mpcsub.build_submission(tmp_path, config)

    assert "! aperture 4.2" in bundle.ades_text
    assert "TEL 4.2-m William Herschel Telescope + CCD" in bundle.obs80_text
    assert "COD 950" in bundle.obs80_text


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
    ades_xml_text = bundle.ades_xml_path.read_text()
    obs80_text = bundle.obs80_path.read_text()
    assert "|2016 CJ155||CCD|W84|" in ades_text
    assert ades_xml_text.startswith("<?xml")
    assert "<provID>2016 CJ155</provID>" in ades_xml_text

    obs_lines = [
        line for line in obs80_text.splitlines()
        if line.startswith("     K16CF5J")
    ]
    assert len(obs_lines) == 2
    assert all(len(line) == 80 for line in obs_lines)
    assert all(line.endswith("W84") for line in obs_lines)
