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
import pptool_panstarrs1 as ps1


def test_panstarrs1_telescope_config_is_registered():
    obsparam = _pp_conf.telescope_parameters["PANSTARRS1"]

    assert obsparam["observatory_code"] == "F51"
    assert obsparam["secpix"] == (0.25, 0.25)
    assert obsparam["filter_translations"]["g.00000"] == "g"
    assert obsparam["filter_translations"]["r.00000"] == "r"
    assert _pp_conf.instrument_identifiers["GPC1"] == "PANSTARRS1"
    assert _pp_conf.instrument_identifiers["PS1"] == "PANSTARRS1"


def test_normalize_filter_handles_ps1_labels():
    assert ps1.normalize_filter("g.00000") == "g"
    assert ps1.normalize_filter("rp1") == "r"
    assert ps1.normalize_filter("i") == "i"


def test_convert_to_pp_fits_flattens_repairs_nans_and_sets_headers(tmp_path):
    raw_path = tmp_path / "raw_image.fits"
    working_path = tmp_path / "ps1_g_0001.fits"
    header = fits.Header()
    header["FPA.FILTER"] = "g.00000"
    header["MJD-OBS"] = 55972.0
    header["EXPTIME"] = 40.0
    header["CRVAL1"] = 145.0
    header["CRVAL2"] = 38.0
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CDELT1"] = -0.25 / 3600.0
    header["CDELT2"] = 0.25 / 3600.0
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["FPA.ZP"] = 24.5
    header["AIRMASS"] = 1.1
    header["CELL.XBIN"] = 1
    header["CELL.YBIN"] = 1
    data = np.array([[1.0, np.nan], [3.0, 5.0]], dtype=np.float32)
    fits.HDUList([fits.PrimaryHDU(),
                  fits.CompImageHDU(data=data, header=header)]).writeto(
        raw_path)

    record = ps1.convert_to_pp_fits(
        raw_path, working_path, 1, "2014_FU61", magzp_sig=0.03)

    with fits.open(working_path) as hdulist:
        assert len(hdulist) == 1
        assert np.isfinite(hdulist[0].data).all()
        assert hdulist[0].header["PPINSTRU"] == "PANSTARRS1"
        assert hdulist[0].header["FILTER"] == "g"
        assert hdulist[0].header["OBJECT"] == "2014_FU61"
        assert hdulist[0].header["NANNPIX"] == 1
        assert np.isclose(hdulist[0].header["NANFILL"], 3.0)
        assert np.isclose(hdulist[0].header["MAGZP"],
                          24.5 + 2.5 * np.log10(40.0))
        assert np.isclose(hdulist[0].header["MAGZPSIG"], 0.03)
        assert hdulist[0].header["MIDTIMJD"] > 2400000.5

    assert record["working_name"] == "ps1_g_0001.fits"
    assert record["filter"] == "g"
    assert record["nan_pixels_replaced"] == 1


def test_derive_cutout_position_uses_requested_pixel_origin(tmp_path):
    cutout_path = tmp_path / "target_gp1_cutout.fits"
    header = fits.Header()
    header["FPA.FILTER"] = "g.00000"
    header["MJD-OBS"] = 55972.0
    header["EXPTIME"] = 40.0
    header["CRVAL1"] = 145.0
    header["CRVAL2"] = 38.0
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CDELT1"] = -0.25 / 3600.0
    header["CDELT2"] = 0.25 / 3600.0
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    fits.PrimaryHDU(data=np.ones((5, 5), dtype=np.float32),
                    header=header).writeto(cutout_path)

    position = ps1.derive_cutout_position(
        cutout_path, xy=(1, 1), pixel_origin=1)

    assert position["filter"] == "g"
    assert np.isclose(position["ra_deg"], 145.0)
    assert np.isclose(position["dec_deg"], 38.0)
    assert position["midtimjd"] > 2400000.5


def test_write_positions_file_uses_target_position_records(tmp_path):
    records = [{
        "working_name": "ps1_g_0001.fits",
        "midtimjd": 2455972.5,
        "target_position": {
            "ra_deg": 145.123456789,
            "dec_deg": 38.123456789,
        },
    }]

    path = ps1.write_positions_file(tmp_path, records, "2014_FU61", filt="g")

    assert path.name == "positions_2014_FU61_g.dat"
    assert path.read_text().strip() == (
        "ps1_g_0001.fits 145.1234567890 38.1234567890 "
        "2455972.50000000 2014_FU61")


def test_combine_filter_photometry_writes_canonical_and_all_filters(tmp_path):
    g_path = tmp_path / "photometry_2014_FU61_g.dat"
    r_path = tmp_path / "photometry_2014_FU61_r.dat"
    g_path.write_text("# header\n g_row\n# footer\n")
    r_path.write_text("# header\n r_row\n# footer\n")
    outputs = [
        {"photometry_file": str(g_path)},
        {"photometry_file": str(r_path)},
    ]

    destinations = ps1.combine_filter_photometry(
        tmp_path, "2014_FU61", outputs)

    assert [Path(path).name for path in destinations] == [
        "photometry_2014_FU61.dat",
        "photometry_2014_FU61_all_filters.dat",
    ]
    assert (tmp_path / "photometry_2014_FU61.dat").read_text() == (
        "# header\n g_row\n r_row\n# footer\n")
    assert (tmp_path / "photometry_2014_FU61_all_filters.dat").read_text() == (
        "# header\n g_row\n r_row\n# footer\n")
