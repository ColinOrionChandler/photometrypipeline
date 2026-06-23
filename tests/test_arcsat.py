import os
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["PHOTPIPEDIR"] = str(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _pp_conf
import pptool_arcsat as arcsat


def test_arcsat_telescope_configs_are_registered():
    byucam = _pp_conf.telescope_parameters["ARCSATBYUCAM"]
    dcam = _pp_conf.telescope_parameters["ARCSATDCAMSPARE"]

    assert byucam["observatory_code"] == "705"
    assert byucam["secpix"] == (0.618, 0.618)
    assert byucam["object"] == "OBJECT"
    assert byucam["filter"] == "FILTER"
    assert byucam["date_keyword"] == "DATE-OBS"
    assert byucam["ra"] == "RA"
    assert byucam["dec"] == "DEC"
    assert byucam["radec_separator"] == " "
    assert byucam["binning"] == ("XBINNING", "YBINNING")
    assert dcam["observatory_code"] == "705"
    assert dcam["secpix"] == (0.667, 0.667)
    assert _pp_conf.instrument_identifiers["BYUcam"] == "ARCSATBYUCAM"
    assert _pp_conf.instrument_identifiers["dcam-spare"] == "ARCSATDCAMSPARE"


def test_arcsat_filter_translations_cover_broadband_and_narrowband():
    obsparam = _pp_conf.telescope_parameters["ARCSATBYUCAM"]

    assert obsparam["filter_translations"]["g"] == "g"
    assert obsparam["filter_translations"]["r"] == "r"
    assert obsparam["filter_translations"]["B"] == "B"
    assert obsparam["filter_translations"]["V"] == "V"
    assert obsparam["filter_translations"]["sdss_r"] == "r"
    assert obsparam["filter_translations"]["Halpha"] is None
    assert obsparam["filter_translations"]["OIII"] is None
    assert obsparam["filter_translations"]["OG570"] is None


def write_arcsat_fits(path, instrume="dcam-spare"):
    header = fits.Header()
    header["TELESCOP"] = "ARCSAT 0.5-m"
    header["INSTRUME"] = instrume
    header["OBSERVAT"] = "ARCSAT"
    header["OBJECT"] = "vesta"
    header["FILTER"] = "g"
    header["EXPTIME"] = 5.0
    header["DATE-OBS"] = "2025-05-16T04:00:38.770"
    header["TIME-OBS"] = "04:00:38"
    header["RA"] = "14 38 25.49"
    header["DEC"] = "-03 51 52.3"
    header["AIRMASS"] = 1.47
    header["XBINNING"] = 1
    header["YBINNING"] = 1
    fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.float32),
                    header=header).writeto(path)


def test_prepare_work_tree_copies_and_stamps_arcsat_headers(tmp_path):
    source = tmp_path / "vesta g raw name.fits"
    write_arcsat_fits(source)
    original_header = fits.getheader(source)

    work_dir, records = arcsat.prepare_work_tree(
        [source], tmp_path / "PP" / "photometry" / "vesta" / "g",
        target="vesta", overwrite=True)

    assert work_dir.exists()
    assert records[0]["working_name"] == "0001_vesta_g_raw_name.fits"
    working = work_dir / records[0]["working_name"]
    with fits.open(working) as hdulist:
        header = hdulist[0].header
        assert header["PPINSTRU"] == "ARCSATDCAMSPARE"
        assert header["TEL_KEYW"] == "ARCSATDCAMSPARE"
        assert header["OBJECT"] == "vesta"
        assert header["ORIGFILE"] == source.name
        assert "MIDTIMJD" in header

    assert "PPINSTRU" not in original_header
    assert "PPINSTRU" not in fits.getheader(source)


class FakeCatalog:
    def __init__(self):
        self.catalogname = "0001_test.fits"
        self.data = Table({
            "ra_deg": [1.0, 2.0, 3.0],
            "dec_deg": [4.0, 5.0, 6.0],
            "gmag": [20.0, 21.0, 22.0],
        })

    @property
    def fields(self):
        return self.data.columns

    @property
    def shape(self):
        return (len(self.data), len(self.data.columns))

    def __getitem__(self, key):
        return self.data[key]


def test_summarize_depth_catalog_uses_calibrated_percentile():
    record = {
        "source": "/tmp/source.fits",
        "working_name": "0001_test.fits",
        "object": "field",
        "filter": "g",
        "telescope": "ARCSATDCAMSPARE",
        "instrume": "dcam-spare",
        "date_obs": "2025-05-16T04:00:38.770",
        "midtimjd": 2460811.66714,
        "exptime": 5.0,
    }
    row = arcsat.summarize_depth_catalog(
        record, FakeCatalog(), {"zp": 24.1, "zp_sig": 0.03,
                                "zp_nstars": 20, "zp_usedstars": 18})

    assert row["source_filename"] == "source.fits"
    assert row["source_count"] == 3
    assert row["zeropoint"] == 24.1
    assert row["zeropoint_sig"] == 0.03
    assert row["mag_column"] == "gmag"
    assert np.isclose(row["depth_mag_p90"], 21.8)


def write_saturated_target_fits(path):
    shape = (120, 120)
    y, x = np.indices(shape)
    cx, cy = 60.0, 60.0
    background = 25.0
    alpha = 2.0
    beta = 2.5
    amp = 1.0e6
    rr2 = (x - cx) ** 2 + (y - cy) ** 2
    data = background + amp * (1.0 + rr2 / alpha ** 2) ** (-beta)
    data = np.minimum(data, 64000.0).astype(np.float32)

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [cx + 1, cy + 1]
    wcs.wcs.cdelt = np.array([-0.0001852778, 0.0001852778])
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["SATURATE"] = 60000.0
    header["EXPTIME"] = 5.0
    fits.PrimaryHDU(data=data, header=header).writeto(path)


def saturated_test_row(path):
    target_ra = 10.0
    target_dec = 20.0
    offset_arcsec = 10.0
    source_ra = (
        target_ra -
        offset_arcsec / 3600.0 / np.cos(np.radians(target_dec)))
    return {
        "catalog_token": path.stem,
        "julian_date": "2460811.0",
        "mag": "16.5",
        "mag_sig": "0.3",
        "ra_deg": str(source_ra),
        "dec_deg": str(target_dec),
        "pred_minus_ra_arcsec": str(offset_arcsec),
        "pred_minus_dec_arcsec": "0.0",
        "manual_ra_offset_arcsec": "0.0",
        "manual_dec_offset_arcsec": "0.0",
        "exptime": "5.0",
        "zeropoint": "23.0",
        "zeropoint_sig": "0.1",
        "inst_mag": "-6.5",
        "inst_mag_sig": "0.2",
        "phot_cat": "PANSTARRS_transformed",
        "band": "g",
        "sextractor_flag": "1",
        "instrument": "ARCSATDCAMSPARE",
        "photometry_method": "APER",
        "fwhm": "3.0",
        "source_photometry_file": str(path.with_suffix(".dat")),
        "source_fits_file": str(path),
    }


def test_saturated_recovery_detects_offset_saturated_target(tmp_path):
    fits_path = tmp_path / "vesta.fits"
    write_saturated_target_fits(fits_path)

    row = arcsat.saturated_recovery_row(saturated_test_row(fits_path))

    assert row["fit_status"] == "ok"
    assert row["recovered_mag_method"] == "saturated_moffat_wing_fit"
    assert float(row["target_offset_arcsec"]) > 9.0
    assert float(row["predicted_peak_adu"]) >= 60000.0
    assert int(row["saturated_pixel_count"]) > 0
    assert row["recovered_mag"] != ""
    assert "not standard PP aperture photometry" in row["warning"]


def test_saturated_recovery_csv_has_explicit_metadata(tmp_path):
    fits_path = tmp_path / "vesta.fits"
    write_saturated_target_fits(fits_path)

    output_path, rows = arcsat.write_saturated_recovery_csv(
        tmp_path, "vesta", [saturated_test_row(fits_path)])

    assert output_path.name == "photometry_vesta_saturated_recovery.csv"
    assert rows[0]["normal_pp_mag"] == "16.5"
    assert rows[0]["fit_radius_px"] == arcsat.SATURATED_FIT_RADIUS_PX
    assert rows[0]["masked_core_radius_px"] == (
        arcsat.SATURATED_MASKED_CORE_RADIUS_PX)
    with output_path.open() as handle:
        header = handle.readline()
    assert "recovered_mag_method" in header
    assert "warning" in header


def test_combined_photometry_does_not_write_recovery_by_default(tmp_path):
    photometry = tmp_path / "photometry_vesta.dat"
    photometry.write_text(
        "#frame 2460811.0 16.5 0.3 10 20 0 0 0 0 5 "
        "23 0.1 -6.5 0.2 PANSTARRS g 0 ARCSATDCAMSPARE APER\n")

    output_path, rows = arcsat.write_combined_photometry_csv(tmp_path, "vesta")

    assert output_path.exists()
    assert len(rows) == 1
    assert not (tmp_path / "photometry_vesta_saturated_recovery.csv").exists()


def test_saturated_recovery_failure_records_failed_status(tmp_path):
    row = saturated_test_row(tmp_path / "missing.fits")

    recovered = arcsat.saturated_recovery_row(row)

    assert recovered["fit_status"] == "failed"
    assert recovered["recovered_mag"] == ""
    assert "recovery failed" in recovered["warning"]
