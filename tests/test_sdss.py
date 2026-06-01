import os
from pathlib import Path
import sys

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["PHOTPIPEDIR"] = str(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _pp_conf
import pptool_sdss as sdss


PHOTOMETRY_TEXT = """# header
 sdss_r_0001 2452994.6986067  21.3285 0.0877  69.66644806  +22.02285211 -0.25  0.36  0.00  0.00 53.91  29.4913 0.0342  -8.1628 0.0808 SDSS-R9 r   0 SDSS       APER 0.87  0.0808  0.0342  0.0877  0.0121  0.0121  0.0171  0.0638  0.0816  0.1036  0.0650  0.0825  0.1050
"""


def write_sdss_raw(path, filt="r"):
    header = fits.Header()
    header["TELESCOP"] = "2.5m"
    header["FILTER"] = filt
    header["DATE-OBS"] = "2003-12-21"
    header["TAIHMS"] = "04:45:39.96"
    header["EXPTIME"] = "53.907456"
    header["RA"] = 69.5
    header["DEC"] = 22.0
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 69.5
    header["CRVAL2"] = 22.0
    header["CRPIX1"] = 2.0
    header["CRPIX2"] = 2.0
    header["CD1_1"] = -0.00011
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.00011
    fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.float32),
                    header=header).writeto(path)


def test_sdss_telescope_config_is_registered():
    obsparam = _pp_conf.telescope_parameters["SDSS"]

    assert obsparam["observatory_code"] == "645"
    assert obsparam["secpix"] == (0.396, 0.396)
    assert obsparam["date_keyword"] == "DATE-OBS|TAIHMS"
    assert obsparam["filter_translations"]["g"] == "g"
    assert obsparam["filter_translations"]["Z"] == "z"
    assert obsparam["photometry_catalogs"] == ["SDSS-R9"]
    assert _pp_conf.instrument_identifiers["SDSS"] == "SDSS"


def test_normalize_filter_handles_sdss_labels():
    assert sdss.normalize_filter("g") == "g"
    assert sdss.normalize_filter(" R ") == "r"
    assert sdss.normalize_filter("z") == "z"


def test_midtime_parsing_uses_date_taihms_and_exptime():
    header = fits.Header()
    header["DATE-OBS"] = "2003-12-21"
    header["TAIHMS"] = "04:45:39.96"
    header["EXPTIME"] = "53.907456"

    expected = (Time("2003-12-21T04:45:39.96", format="isot",
                     scale="utc") + (53.907456 / 2.0) * u.second).jd

    assert np.isclose(sdss.midtime_jd_from_header(header), expected)


def test_full_frame_selection_maps_original_cutouts(tmp_path):
    cutout_dir = tmp_path / "cutouts"
    cutout_dir.mkdir()
    cutout = cutout_dir / (
        "2003_WW218_20031221T044539.9888_r_exp54s_CADC_"
        "frame-r-004334-1-0182_chip0_126arcsec_cutout.fits")
    raw = tmp_path / (
        "2003_WW218_20031221T044539.9888_r_exp54s_SDSS_"
        "frame-r-004334-1-0182_image.fits")
    cutout.touch()
    raw.touch()

    records = sdss.selected_image_records(tmp_path, cutouts=[cutout])

    assert records == [{"cutout": cutout.resolve(), "raw": raw.resolve()}]


def test_prepare_pp_tree_copies_and_stamps_headers(tmp_path):
    cutout_dir = tmp_path / "cutouts"
    cutout_dir.mkdir()
    cutout = cutout_dir / (
        "2003_WW218_20031221T044539.9888_r_exp54s_CADC_"
        "frame-r-004334-1-0182_chip0_126arcsec_cutout.fits")
    raw = tmp_path / (
        "2003_WW218_20031221T044539.9888_r_exp54s_SDSS_"
        "frame-r-004334-1-0182_image.fits")
    fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.float32)).writeto(cutout)
    write_sdss_raw(raw, filt="r")

    manifest = sdss.prepare_pp_tree(
        tmp_path, tmp_path / "PP", "2003 WW218", cutouts=[cutout],
        overwrite=True)

    record = manifest["records"][0]
    working = Path(record["working"])
    assert working == (tmp_path / "PP" / "2003-12-21" / "r" / raw.name)
    assert record["filter"] == "r"
    assert np.isclose(record["midtimjd"], 2452994.6986067)

    with fits.open(working) as hdulist:
        header = hdulist[0].header
        assert header["PPINSTRU"] == "SDSS"
        assert header["TEL_KEYW"] == "SDSS"
        assert header["TELESCOP"] == "2.5-m SDSS telescope"
        assert header["INSTRUME"] == "SDSS imaging camera"
        assert header["OBJECT"] == "2003 WW218"
        assert header["FILTER"] == "r"
        assert header["ORIGFILE"].startswith("2003_WW218_20031221T044539")
        assert header["SDSSCUT"].startswith("2003_WW218_20031221T044539")
        assert header["PPSEQ"] == 1


def test_write_combined_photometry_csv_adds_source_provenance(tmp_path):
    image_dir = tmp_path / "PP" / "2003-12-21" / "r"
    image_dir.mkdir(parents=True)
    photometry = image_dir / "photometry__2003_WW218_.dat"
    photometry.write_text(PHOTOMETRY_TEXT)
    write_sdss_raw(image_dir / "sdss_r_0001.fits", filt="r")

    output_path, rows = sdss.write_combined_photometry_csv(
        tmp_path / "PP", "2003 WW218")

    assert output_path == tmp_path / "PP" / "photometry_2003_WW218_combined.csv"
    assert len(rows) == 1
    assert rows[0]["phot_cat"] == "SDSS-R9"
    assert rows[0]["band"] == "r"
    assert rows[0]["source_photometry_file"] == str(photometry.resolve())
    assert rows[0]["source_fits_file"] == str(
        (image_dir / "sdss_r_0001.fits").resolve())
    assert "source_fits_file" in output_path.read_text().splitlines()[0]
