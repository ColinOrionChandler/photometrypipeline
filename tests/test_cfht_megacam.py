import csv
import os
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["PHOTPIPEDIR"] = str(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _pp_conf
import pptool_cfht_megacam as cfht
import pptool_mpcsubmission as mpcsub


def base_cfht_header():
    header = fits.Header()
    header["INSTRUME"] = "MegaPrime"
    header["FILTER"] = "g.MP9401"
    header["DATE"] = "2003-03-24T11:24:43.22"
    header["DATE-OBS"] = "2003-03-24"
    header["MJD-OBS"] = 52722.4755002
    header["MJDATE"] = 52722.4755002
    header["EXPTIME"] = 140.057
    header["AIRMASS"] = 1.112
    header["RA"] = "12:46:46.09"
    header["DEC"] = "-5:33:28.0"
    header["CCDSUM"] = "1 1"
    header["OBJECT"] = "VWDAA14"
    header["OBSERVER"] = "QSO Team"
    header["PHOT_C"] = 26.453
    header["PHOT_CS"] = 0.0033
    header["CRVAL1"] = 191.0
    header["CRVAL2"] = -5.0
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CDELT1"] = -0.185 / 3600.0
    header["CDELT2"] = 0.185 / 3600.0
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    return header


def write_raw_megacam(path):
    hdus = [fits.PrimaryHDU(header=base_cfht_header())]
    for idx in range(1, 27):
        header = base_cfht_header()
        header["EXTNAME"] = "ccd%d" % (idx - 1)
        data = np.full((3, 4), idx, dtype=np.float32)
        hdus.append(fits.ImageHDU(data=data, header=header))
    fits.HDUList(hdus).writeto(path)


def test_cfht_telescope_config_is_registered():
    obsparam = _pp_conf.telescope_parameters["CFHTMEGAPRIME"]

    assert obsparam["observatory_code"] == "568"
    assert obsparam["filter_translations"]["g.MP9401"] == "g"
    assert obsparam["filter_translations"]["g"] == "g"
    assert _pp_conf.instrument_identifiers["MegaPrime"] == "CFHTMEGAPRIME"
    assert "CFHTMEGAPRIME" in _pp_conf.implemented_telescopes


def test_parse_click_image_name_extracts_image_and_chip():
    parsed = cfht.parse_click_image_name(
        "2013_SO107_2003-03-24_12.30.34.618000_696015p_"
        "chip26-ccd25_126arcsec_NuEl.png")

    assert parsed == {"image_id": "696015p", "chip_hdu": 26, "ccd": 25}


def test_read_clicked_rows_discovers_raw_and_cutout_paths(tmp_path):
    image_dir = tmp_path / "696015p"
    fits_dir = image_dir / "FITS"
    fits_dir.mkdir(parents=True)
    raw_path = image_dir / "696015p.fits.fz"
    cutout_path = fits_dir / (
        "2013_SO107_2003-03-24_12.30.34.618000_696015p_"
        "chip26-ccd25_126arcsec_NuEl.fits")
    raw_path.write_text("placeholder")
    cutout_path.write_text("placeholder")
    csv_path = tmp_path / "click_points.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "x", "y", "width", "height", "status"])
        writer.writerow([
            cutout_path.with_suffix(".png").name,
            164,
            212,
            370,
            370,
            "clicked",
        ])
        writer.writerow(["ignored_1p_chip1.png", 1, 2, 3, 4, "skipped"])

    rows = cfht.read_clicked_rows(csv_path)

    assert len(rows) == 1
    assert rows[0]["image_id"] == "696015p"
    assert rows[0]["chip_hdu"] == 26
    assert rows[0]["ccd"] == 25
    assert rows[0]["raw_path"] == str(raw_path.resolve())
    assert rows[0]["cutout_path"] == str(cutout_path.resolve())


def test_convert_to_pp_fits_extracts_requested_chip_and_sets_headers(tmp_path):
    raw_path = tmp_path / "696015p.fits.fz"
    working_path = tmp_path / "g" / "cfht_g_0001_696015p_chip26.fits"
    write_raw_megacam(raw_path)
    click = {
        "image_id": "696015p",
        "chip_hdu": 26,
        "ccd": 25,
        "x": 164.0,
        "y": 212.0,
        "width": 370.0,
        "height": 370.0,
    }

    record = cfht.convert_to_pp_fits(
        raw_path, working_path, click, "2013 SO107", 1)

    with fits.open(working_path) as hdulist:
        assert len(hdulist) == 1
        assert hdulist[0].data.shape == (3, 4)
        assert np.all(hdulist[0].data == 26)
        assert hdulist[0].header["PPINSTRU"] == "CFHTMEGAPRIME"
        assert hdulist[0].header["OBJECT"] == "2013 SO107"
        assert hdulist[0].header["FILTER"] == "g"
        assert hdulist[0].header["ORIGFILT"] == "g.MP9401"
        assert hdulist[0].header["ORIGEXT"] == 26
        assert hdulist[0].header["ORIGEXTN"] == "ccd25"
        assert np.isclose(hdulist[0].header["MAGZP"], 26.453)
        assert np.isclose(hdulist[0].header["MAGZPSIG"], 0.0033)
        assert hdulist[0].header["MIDTIMJD"] > 2400000.5

    assert record["working_name"] == working_path.name
    assert record["filter"] == "g"
    assert record["source_hdu"] == 26
    assert record["source_extname"] == "ccd25"


def test_derive_click_position_scales_display_pixels_to_cutout_shape(tmp_path):
    cutout_path = tmp_path / "cutout.fits"
    header = fits.Header()
    header["CRVAL1"] = 145.0
    header["CRVAL2"] = 38.0
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CDELT1"] = -0.25 / 3600.0
    header["CDELT2"] = 0.25 / 3600.0
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    data = np.ones((4, 8), dtype=np.float32)
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data=data, header=header)]).writeto(
        cutout_path)
    click = {"x": 2.0, "y": 1.0, "width": 4.0, "height": 2.0}

    position = cfht.derive_click_position(
        cutout_path, click, pixel_origin=0, flip_y=False)
    expected_ra, expected_dec = WCS(header).all_pix2world([[4.0, 2.0]], 0)[0]

    assert position["fits_x"] == 4.0
    assert position["fits_y"] == 2.0
    assert position["fits_width"] == 8
    assert position["fits_height"] == 4
    assert np.isclose(position["ra_deg"], expected_ra)
    assert np.isclose(position["dec_deg"], expected_dec)


def test_combine_filter_photometry_rewrites_nested_paths_for_submission(tmp_path):
    filter_dir = tmp_path / "g"
    filter_dir.mkdir()
    fits_path = filter_dir / "cfht_g_0001_695998p_chip26.fits"
    header = fits.Header()
    header["OBSERVER"] = "QSO Team"
    header["TELESCOP"] = "CFHT"
    header["INSTRUME"] = "MegaPrime"
    header["TEL_KEYW"] = "CFHTMEGAPRIME"
    fits.PrimaryHDU(header=header).writeto(fits_path)
    photometry_path = filter_dir / "photometry_2013_SO107.dat"
    photometry_path.write_text(
        "# header\n"
        " cfht_g_0001_695998p_chip26.fits 2452722.9763107 "
        "22.1000 0.0500 191.38770000 -5.75630000 0.00 0.00 "
        "0.00 0.00 140.00 26.4530 0.0033 -4.3530 0.0499 "
        "manual_zp g 0 CFHTMEGAPRIME APER 0.90 0.0499 0.0033 "
        "0.0500 0.0100 0.0100 0.0141 0.0200 0.0200 0.0283 "
        "0.0224 0.0224 0.0316\n"
        "# footer\n")

    outputs = [{"filter": "g", "photometry_file": str(photometry_path)}]
    destinations = cfht.combine_filter_photometry(
        tmp_path, "2013 SO107", outputs)
    combined_path = Path(destinations[0])

    assert combined_path.name == "photometry_2013_SO107.dat"
    assert " g/cfht_g_0001_695998p_chip26.fits " in combined_path.read_text()

    config = mpcsub.SubmissionConfig(
        target="2013 SO107", observatory_code="568", output_dir=tmp_path)
    bundle = mpcsub.build_submission(combined_path, config)

    assert len(bundle.observations) == 1
    assert bundle.observations[0].source_file == fits_path.resolve()
    assert "COD 568" in bundle.obs80_text
