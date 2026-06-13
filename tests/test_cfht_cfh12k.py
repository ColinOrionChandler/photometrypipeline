import csv
import os
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["PHOTPIPEDIR"] = str(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _pp_conf
import pptool_cfht_cfh12k as cfh12k
import pptool_mpcsubmission as mpcsub


def base_cfh12k_header():
    header = fits.Header()
    header["INSTRUME"] = "CFH12K Mosaic"
    header["DETECTOR"] = "CFH12K"
    header["TELESCOP"] = "CFHT 3.6m"
    header["FILTER"] = "I"
    header["DATE-OBS"] = "2002-06-09"
    header["MJD-OBS"] = 52434.5114959
    header["MJDATE"] = 52434.5114959
    header["EXPTIME"] = 2.0
    header["AIRMASS"] = 1.1
    header["RA"] = "17:45:01.696"
    header["DEC"] = "+04:50:38.73"
    header["CCDSUM"] = "1 1"
    header["OBJECT"] = "IC 4665 J"
    header["OBSERVER"] = "QSO Team"
    header["PHOT_C"] = 26.169
    header["PHOT_CS"] = 0.019
    header["CRVAL1"] = 266.257066781
    header["CRVAL2"] = 4.844092925
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CDELT1"] = -0.206 / 3600.0
    header["CDELT2"] = 0.206 / 3600.0
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["RADESYS"] = "FK5"
    return header


def write_raw_cfh12k(path):
    hdus = [fits.PrimaryHDU(header=base_cfh12k_header())]
    for hdu_index in range(1, 13):
        header = base_cfh12k_header()
        detector_chip = hdu_index - 1
        header["EXTNAME"] = "chip%02d" % detector_chip
        header["IMAGEID"] = detector_chip
        data = np.full((3, 4), hdu_index, dtype=np.float32)
        hdus.append(fits.ImageHDU(data=data, header=header))
    fits.HDUList(hdus).writeto(path)


def test_cfh12k_telescope_config_is_registered():
    obsparam = _pp_conf.telescope_parameters["CFHTCFH12K"]

    assert obsparam["observatory_code"] == "568"
    assert obsparam["secpix"] == (0.206, 0.206)
    assert obsparam["filter_translations"]["I"] == "I"
    assert obsparam["filter_translations"]["Z"] == "z"
    assert _pp_conf.instrument_identifiers["CFH12K Mosaic"] == "CFHTCFH12K"
    assert _pp_conf.instrument_identifiers["CFH12K"] == "CFHTCFH12K"
    assert "DETECTOR" in _pp_conf.instrument_keys
    assert "CFHTCFH12K" in _pp_conf.implemented_telescopes


def test_parse_cutout_name_extracts_one_based_and_detector_chips():
    parsed = cfh12k.parse_cutout_name(
        "2025_MH348_2002-06-09_12.16.34.250000_641271p_"
        "chip10-chip09_126arcsec_NuEl.fits")

    assert parsed == {
        "image_id": "641271p",
        "chip_hdu": 10,
        "detector_chip": 9,
    }


def test_derive_click_position_uses_cutout_wcs(tmp_path):
    cutout_path = tmp_path / (
        "2025_MH348_2002-06-09_12.21.55.920000_641273p_"
        "chip10-chip09_126arcsec_NuEl.fits")
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 266.25
    header["CRVAL2"] = 4.75
    header["CRPIX1"] = 6.0
    header["CRPIX2"] = 6.0
    header["CDELT1"] = -1.0 / 3600.0
    header["CDELT2"] = 1.0 / 3600.0
    fits.PrimaryHDU(data=np.ones((11, 11), dtype=np.float32),
                    header=header).writeto(cutout_path)
    click = {
        "x": 5,
        "y": 5,
        "width": 11,
        "height": 11,
    }

    position = cfh12k.derive_click_position(cutout_path, click)

    assert np.isclose(position["ra_deg"], 266.25)
    assert np.isclose(position["dec_deg"], 4.75)
    assert position["fits_x"] == 5
    assert position["fits_y"] == 5


def test_refine_position_by_centroid_updates_full_chip_position(tmp_path):
    image_path = tmp_path / "chip.fits"
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 266.25
    header["CRVAL2"] = 4.75
    header["CRPIX1"] = 11.0
    header["CRPIX2"] = 11.0
    header["CDELT1"] = -1.0 / 3600.0
    header["CDELT2"] = 1.0 / 3600.0
    data = np.full((21, 21), 100.0, dtype=np.float32)
    data[11, 12] = 120.0
    data[11, 13] = 110.0
    fits.PrimaryHDU(data=data, header=header).writeto(image_path)

    result = cfh12k.refine_position_by_centroid(
        image_path, 266.25, 4.75, search_radius=4,
        aperture_radius=2, min_snr=0.5)

    assert result["used"] is True
    assert result["centroid_x"] > 11.5
    assert result["offset_from_input_px"] > 0
    with fits.open(image_path) as hdulist:
        assert hdulist[0].header["CENTROID"] is True
        assert np.isclose(hdulist[0].header["TARGRA"], result["ra_deg"])


def test_find_find_orb_executable_uses_local_project_pluto_build(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    exe = home / "Github" / "find_orb" / ".local" / "bin" / "fo"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("FIND_ORB_EXECUTABLE", raising=False)
    monkeypatch.setattr(cfh12k.shutil, "which", lambda name: None)

    assert cfh12k.find_find_orb_executable() == str(exe.resolve())


def test_parse_find_orb_stdout_strips_color_and_extracts_summary():
    text = (
        "Processing 1 objects\n"
        "1: 2025 MH348; \x1b[32ma=43.184,\x1b[0m "
        "e=0.551, i=44   6 obs\x1b[0m; 2002 June 9 (37.0 min)\n")

    parsed = cfh12k.parse_find_orb_stdout(text)

    assert parsed == {
        "object": "2025 MH348",
        "semimajor_axis_au": 43.184,
        "eccentricity": 0.551,
        "inclination_deg": 44.0,
        "observations": 6,
        "total_observations": 6,
        "arc": "2002 June 9 (37.0 min)",
    }


def test_parse_find_orb_stdout_extracts_used_and_total_observations():
    text = (
        "1: 2025 MH348; \x1b[32ma=29.950,\x1b[0m e=0.049, "
        "i=29 \x1b[30;47m 22 /  28 obs\x1b[0m; "
        "2002 June 9-2025 July 24\n")

    parsed = cfh12k.parse_find_orb_stdout(text)

    assert parsed["observations"] == 22
    assert parsed["total_observations"] == 28
    assert parsed["arc"] == "2002 June 9-2025 July 24"


def test_read_location_records_uses_cutout_chip_shortcut(tmp_path):
    image_dir = tmp_path / "641271p"
    fits_dir = image_dir / "FITS"
    fits_dir.mkdir(parents=True)
    raw_path = image_dir / "641271p.fits.fz"
    raw_path.write_text("placeholder")
    cutout_path = fits_dir / (
        "2025_MH348_2002-06-09_12.16.34.250000_641271p_"
        "chip10-chip09_126arcsec_NuEl.fits")
    cutout_path.write_text("placeholder")
    csv_path = image_dir / "2025_MH348_locations.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Filename", "UT", "JD", "OBJ_RA", "OBJ_Dec", "dRA", "dDec",
            "PsAng", "PsAMV", "Present_Flag", "ChipSliceNum", "Filter",
            "deltaMag", "X",
        ])
        writer.writerow([
            "/scratch/641271p.fits.fz", "2002-06-09 12:16:34.250000",
            "2452435.011507523", "266.283341669", "4.761027127",
            "-4.2", "0.2", "196", "281", "True", "10", "I",
            "None", "1.1",
        ])

    records = cfh12k.read_location_records(tmp_path, "2025 MH348")

    assert len(records) == 1
    assert records[0]["image_id"] == "641271p"
    assert records[0]["chip_hdu"] == 10
    assert records[0]["detector_chip"] == 9
    assert records[0]["filter"] == "I"
    assert records[0]["filter_dir_name"] == "I"
    assert records[0]["raw_path"] == str(raw_path.resolve())
    assert records[0]["cutout_path"] == str(cutout_path.resolve())


def test_extract_and_normalize_full_chip_sets_pp_headers(tmp_path):
    raw_path = tmp_path / "641271p.fits"
    working_path = tmp_path / "PP" / "2002-06-09" / "I" / (
        "cfh12k_i_0001_641271p_chip10.fits")
    write_raw_cfh12k(raw_path)
    location_record = {
        "raw_path": str(raw_path),
        "image_id": "641271p",
        "chip_hdu": 10,
        "detector_chip": 9,
        "source_filter": "I",
        "filter_dir_name": "I",
        "date": "2002-06-09",
        "midtimjd": 2452435.011507523,
        "target_ra_deg": 266.283341669,
        "target_dec_deg": 4.761027127,
        "cutout_path": None,
    }

    cfh12k.extract_full_chip(raw_path, working_path, 10)
    record = cfh12k.normalize_extracted_fits(
        working_path, location_record, "2025 MH348", 1)

    with fits.open(working_path) as hdulist:
        assert len(hdulist) == 1
        assert hdulist[0].data.shape == (3, 4)
        assert np.all(hdulist[0].data == 10)
        assert hdulist[0].header["PPINSTRU"] == "CFHTCFH12K"
        assert hdulist[0].header["TEL_KEYW"] == "CFHTCFH12K"
        assert hdulist[0].header["OBJECT"] == "2025 MH348"
        assert hdulist[0].header["FILTER"] == "I"
        assert hdulist[0].header["ORIGEXT"] == 10
        assert hdulist[0].header["ORIGEXTN"] == "chip09"
        assert hdulist[0].header["IMAGEID"] == 9
        assert np.isclose(hdulist[0].header["MAGZP"], 26.169)
        assert np.isclose(hdulist[0].header["MAGZPSIG"], 0.019)
        assert np.isclose(hdulist[0].header["MIDTIMJD"],
                          2452435.011507523)

    assert record["working_name"] == working_path.name
    assert record["filter"] == "I"
    assert record["source_hdu"] == 10
    assert record["source_extname"] == "chip09"


def test_write_positions_file_uses_location_positions(tmp_path):
    records = [{
        "working_name": "cfh12k_i_0001_641271p_chip10.fits",
        "midtimjd": 2452435.011507523,
        "target_ra_deg": 266.283341669,
        "target_dec_deg": 4.761027127,
    }]

    path = cfh12k.write_positions_file(tmp_path, records, "2025 MH348", "I")

    assert path.name == "positions_2025_MH348_I.dat"
    assert path.read_text().strip() == (
        "cfh12k_i_0001_641271p_chip10.fits 266.2833416690 "
        "4.7610271270 2452435.01150752 2025_MH348")


def test_combine_filter_photometry_rewrites_date_filter_paths(tmp_path):
    filter_dir = tmp_path / "2002-06-09" / "I"
    filter_dir.mkdir(parents=True)
    fits_path = filter_dir / "cfh12k_i_0001_641271p_chip10.fits"
    header = fits.Header()
    header["OBSERVER"] = "QSO Team"
    header["TELESCOP"] = "CFHT 3.6m"
    header["INSTRUME"] = "CFH12K Mosaic"
    header["TEL_KEYW"] = "CFHTCFH12K"
    fits.PrimaryHDU(header=header).writeto(fits_path)
    second_fits_path = filter_dir / "cfh12k_i_0002_641272p_chip10.fits"
    fits.PrimaryHDU(header=header).writeto(second_fits_path)
    photometry_path = filter_dir / "photometry_2025_MH348.dat"
    photometry_path.write_text(
        "# header\n"
        " cfh12k_i_0001_641271p_chip10.fits 2452435.0115075 "
        "22.1000 0.0500 266.28334167 +4.76102713 0.00 0.00 "
        "0.00 0.00 2.00 26.1690 0.0190 -4.3530 0.0499 "
        "manual_zp I 0 CFHTCFH12K APER 0.90 0.0499 0.0033 "
        "0.0500 0.0100 0.0100 0.0141 0.0200 0.0200 0.0283 "
        "0.0224 0.0224 0.0316\n"
        " cfh12k_i_0002_641272p_chip10.fits 2452435.0124560 "
        "125.1690 99.0000 266.28334167 +4.76102713 0.00 0.00 "
        "0.00 0.00 2.00 26.1690 0.0190 99.0000 99.0000 "
        "manual_zp I 0 CFHTCFH12K APER 0.90 99.0000 0.0033 "
        "99.0000 0.0100 0.0100 0.0141 0.0200 0.0200 0.0283 "
        "0.0224 0.0224 0.0316\n"
        "# footer\n")

    outputs = [{
        "relative_dir": "2002-06-09/I",
        "photometry_file": str(photometry_path),
    }]
    destinations = cfh12k.combine_filter_photometry(
        tmp_path, "2025 MH348", outputs)
    combined_path = Path(destinations["files"][0])

    assert combined_path.name == "photometry_2025_MH348.dat"
    assert " 2002-06-09/I/cfh12k_i_0001_641271p_chip10.fits " in (
        combined_path.read_text())
    assert " 2002-06-09/I/cfh12k_i_0002_641272p_chip10.fits " in (
        combined_path.read_text())

    config = mpcsub.SubmissionConfig(
        target="2025 MH348", observatory_code="568", output_dir=tmp_path)
    bundle = mpcsub.build_submission(combined_path, config)

    assert len(bundle.observations) == 2
    assert bundle.observations[0].source_file == fits_path.resolve()
    assert bundle.observations[1].astrometry_only
    assert "COD 568" in bundle.obs80_text


def test_combine_filter_photometry_forces_bad_match_to_astrometry_only(
        tmp_path):
    filter_dir = tmp_path / "2002-06-09" / "Z"
    filter_dir.mkdir(parents=True)
    fits_path = filter_dir / "cfh12k_z_0004_641278p_chip10.fits"
    header = fits.Header()
    header["OBSERVER"] = "QSO Team"
    header["TELESCOP"] = "CFHT 3.6m"
    header["INSTRUME"] = "CFH12K Mosaic"
    header["TEL_KEYW"] = "CFHTCFH12K"
    fits.PrimaryHDU(header=header).writeto(fits_path)
    photometry_path = filter_dir / "photometry_2025_MH348.dat"
    photometry_path.write_text(
        "# header\n"
        " cfh12k_z_0004_641278p_chip10.fits 2452435.0306800 "
        "17.3016 0.0284 266.28389404 +4.75916364 -3.06 6.92 "
        "0.00 0.00 2.00 31.4070 0.0280 -14.1051 0.0048 "
        "SDSS-R9 z 0 CFHTCFH12K APER 0.90 0.0048 0.0280 "
        "0.0284 0.0100 0.0100 0.0141 0.0200 0.0200 0.0283 "
        "0.0224 0.0224 0.0316\n"
        "# footer\n")

    outputs = [{
        "relative_dir": "2002-06-09/Z",
        "photometry_file": str(photometry_path),
        "target_positions": {
            "cfh12k_z_0004_641278p_chip10.fits": {
                "working_name": "cfh12k_z_0004_641278p_chip10.fits",
                "target_ra_deg": 266.2830442592,
                "target_dec_deg": 4.7610855897,
                "position_source": "click_points+centroid",
            },
        },
    }]
    result = cfh12k.combine_filter_photometry(
        tmp_path, "2025 MH348", outputs, max_match_residual_arcsec=2.0)
    combined_path = Path(result["files"][0])
    row = [
        line for line in combined_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ][0].split()

    assert len(result["forced_astrometry_only_rows"]) == 1
    assert row[2] == "99.0000"
    assert row[3] == "99.0000"
    assert row[4] == "266.28304426"
    assert row[5] == "+4.76108559"

    config = mpcsub.SubmissionConfig(
        target="2025 MH348", observatory_code="568", output_dir=tmp_path)
    bundle = mpcsub.build_submission(combined_path, config)

    assert len(bundle.observations) == 1
    assert bundle.observations[0].astrometry_only
    assert bundle.observations[0].ra_deg == 266.28304426
