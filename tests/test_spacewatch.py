import json
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
import pptool_spacewatch as sw


def test_spacewatch_telescope_config_is_registered():
    obsparam = _pp_conf.telescope_parameters["SPACEWATCH09"]

    assert obsparam["observatory_code"] == "691"
    assert obsparam["secpix"] == (1.0, 1.0)
    assert obsparam["filter_translations"]["Schott OG-515"] == "r"
    assert (_pp_conf.instrument_identifiers["Spacewatch Mosaic Camera"] ==
            "SPACEWATCH09")


def test_select_download_rows_accepts_png_cutout_names(tmp_path):
    downloads = tmp_path / "downloads.jsonl"
    rows = [
        {
            "filename": "target_a_cutout.fits",
            "obs_time": "2006-10-12 09:00:00.000",
        },
        {
            "filename": "target_b_cutout.fits",
            "obs_time": "2006-10-12 08:00:00.000",
        },
    ]
    downloads.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    selected = sw.select_download_rows(downloads, [tmp_path / "target_a_cutout.png"])

    assert [row["filename"] for row in selected] == ["target_a_cutout.fits"]


def test_convert_to_pp_fits_preserves_spacewatch_wcs_and_metadata(tmp_path):
    raw_path = tmp_path / "sw_raw.fits"
    working_path = tmp_path / "spacewatch_r_0001.fits"
    header = fits.Header()
    header["TELESCOP"] = "Spacewatch 0.9-m f/3 prime focus"
    header["INSTRUME"] = "Spacewatch Mosaic Camera"
    header["FILTER"] = "Schott OG-515"
    header["DATE-OBS"] = "2006-10-12T08:53:55.0"
    header["TIME-OBS"] = "08:53:55.0"
    header["EXPTIME"] = 120.0
    header["RA"] = "03:37:58"
    header["DEC"] = "+19:03:15"
    header["AIRMASS"] = 1.04
    header["MAGZP"] = 28.3
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 54.57
    header["CRVAL2"] = 19.19
    header["CRPIX1"] = 10.0
    header["CRPIX2"] = 10.0
    header["CD1_1"] = -0.0002777
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.0002777
    header["PV1_0"] = 0.0
    header["PV2_0"] = 0.0
    data = np.array([[1.0, np.nan], [3.0, 5.0]], dtype=np.float32)
    fits.PrimaryHDU(data=data, header=header).writeto(raw_path)
    row = {
        "filter_name": "Schott OG-515",
        "ra_deg": 55.17861,
        "dec_deg": 19.15939,
        "filename": "target_cutout.fits",
        "image_id": "urn:test",
        "obs_time": "2006-10-12 08:54:55.000",
        "url": "https://example.test/cutout.fits",
        "metadata": {
            "archive_url": "https://example.test/sw_raw.fits",
        },
    }

    record = sw.convert_to_pp_fits(raw_path, working_path, 1, "2006 TN147", row)

    with fits.open(working_path) as hdulist:
        header = hdulist[0].header
        assert np.isfinite(hdulist[0].data).all()
        assert header["PPINSTRU"] == "SPACEWATCH09"
        assert header["OBJECT"] == "2006 TN147"
        assert header["FILTER"] == "Schott OG-515"
        assert header["CTYPE1"] == "RA---TPV"
        assert header["CTYPE2"] == "DEC--TPV"
        assert header["NANNPIX"] == 1
        assert np.isclose(header["CATCHRA"], 55.17861)
        assert header["MIDTIMJD"] > 2400000.5
        assert "MAGZP" in header

    assert record["working_name"] == "spacewatch_r_0001.fits"
    assert record["band"] == "r"
    assert record["magzp"] == 28.3


def test_write_positions_file_uses_catch_position_and_midtime(tmp_path):
    records = [{
        "working_name": "spacewatch_r_0001.fits",
        "midtimjd": 2454020.87077546,
        "target_position": {
            "ra_deg": 55.17861234,
            "dec_deg": 19.15939876,
        },
    }]

    path = sw.write_positions_file(tmp_path, records, "2006 TN147")

    assert path.name == "positions_2006_TN147.dat"
    assert path.read_text().strip() == (
        "spacewatch_r_0001.fits 55.1786123400 19.1593987600 "
        "2454020.87077546 2006_TN147")
