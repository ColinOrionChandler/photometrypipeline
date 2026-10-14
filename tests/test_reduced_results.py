import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from pptool_reduced_results import (
    exposure_midpoint,
    orient_cutout_discrete,
    propagate_cached_ephemeris,
    publish_reduced_results,
)


def write_source(path, date_obs, band, target, target_dir):
    header = fits.Header()
    header["DATE-OBS"] = date_obs
    header["TIMESYS"] = "UTC"
    header["EXPTIME"] = 60.0
    header["FILTER"] = band
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    header["CRPIX1"] = 51.0
    header["CRPIX2"] = 51.0
    header["CD1_1"] = -1.0 / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0 / 3600.0
    fits.PrimaryHDU(
        data=np.arange(101 * 101, dtype=np.float32).reshape(101, 101),
        header=header).writeto(path)
    return {
        "target": target,
        "target_dir": target_dir,
        "band": band,
        "destination": str(path),
    }


def test_exposure_midpoint_uses_half_exposure():
    header = fits.Header()
    header["DATE-OBS"] = "2026-08-16T01:02:03.000000"
    header["EXPTIME"] = 60.0
    assert exposure_midpoint(header).isot == "2026-08-16T01:02:33.000"


def test_propagates_a_nearby_cached_ephemeris_without_network():
    base = {
        "ra_deg": 10.0, "dec_deg": 60.0,
        "ra_rate_arcsec_hour": 1.0,
        "dec_rate_arcsec_hour": -2.0,
        "antisolar_deg": 42.0, "antimotion_deg": 123.0,
    }
    result = propagate_cached_ephemeris(base, 2460000.0,
                                        2460000.0 + 1.0 / 24.0)
    assert result["ra_deg"] == pytest.approx(10.0 + 2.0 / 3600.0)
    assert result["dec_deg"] == pytest.approx(60.0 - 2.0 / 3600.0)
    assert result["antisolar_deg"] == 42.0
    assert result["ephemeris_source"].startswith("propagated")


def test_discrete_orientation_preserves_pixels_and_updates_wcs():
    data = np.arange(5 * 7, dtype=np.float64).reshape(5, 7)
    source_wcs = WCS(naxis=2)
    source_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    source_wcs.wcs.crval = [10.0, 20.0]
    source_wcs.wcs.crpix = [4.0, 3.0]
    scale = 1.0 / 3600.0
    source_wcs.wcs.cd = [[0.0, scale], [scale, 0.0]]

    oriented, output_wcs, transform = orient_cutout_discrete(
        data, source_wcs)

    assert transform != "identity"
    assert oriented.dtype == data.dtype
    assert np.array_equal(np.sort(oriented.ravel()), np.sort(data.ravel()))
    matrix = output_wcs.pixel_scale_matrix
    assert matrix[0, 0] < 0
    assert matrix[1, 1] > 0
    assert abs(matrix[0, 1]) < 1e-12
    assert abs(matrix[1, 0]) < 1e-12
    for output_y, output_x in ((0, 0), (2, 3),
                               (oriented.shape[0] - 1,
                                oriented.shape[1] - 1)):
        value = int(oriented[output_y, output_x])
        input_y, input_x = np.unravel_index(value, data.shape)
        source_world = source_wcs.all_pix2world(input_x, input_y, 0)
        output_world = output_wcs.all_pix2world(
            output_x, output_y, 0)
        assert np.allclose(source_world, output_world, atol=1e-10)


def test_publishes_fits_png_cutouts_and_arrows_with_band_names(tmp_path):
    pp_root = tmp_path / "PP"
    pp_root.mkdir()
    source_g = pp_root / "red_image.0001.fits"
    source_r = pp_root / "red_image.0002.fits"
    records = [
        write_source(source_g, "2026-08-16T01:02:03.000000", "g",
                     "P/2026 A1", "P+2026_A1"),
        write_source(source_r, "2026-08-16T01:04:03.000000", "r",
                     "P/2026 A1", "P+2026_A1"),
    ]
    workflow = pp_root / "reduced_data_workflow_manifest.json"
    workflow.write_text(json.dumps({
        "output_dir": str(pp_root),
        "source_count": 2,
        "records": records,
    }))

    def fake_query(target, epochs, site_code):
        assert target == "P/2026 A1"
        assert site_code == "705"
        return [{
            "ra_deg": 10.0,
            "dec_deg": 20.0,
            "antisolar_deg": 42.0,
            "antimotion_deg": 123.0,
        } for _ in epochs]

    def fake_arrows(antisolar, antimotion, output_path):
        assert (antisolar, antimotion) == (42.0, 123.0)
        Path(output_path).write_bytes(b"%PDF-test")

    def fake_png(output_fits, output_png):
        with fits.open(output_fits) as hdulist:
            assert hdulist[0].data.shape == (21, 21)
            assert hdulist[0].header["CUTORNT"] == "NUP-ELFT"
        Path(output_png).write_bytes(b"PNG-test")

    results_dir = tmp_path / "results"
    result = publish_reduced_results(
        workflow, results_dir, size_arcsec=20.0,
        query_func=fake_query, arrow_func=fake_arrows,
        png_func=fake_png)

    cutouts = results_dir / "P+2026_A1" / "ARCTIC" / "cutouts"
    assert result["status"] == "complete"
    assert result["product_count"] == 2
    assert result["orientation_mode"] == "discrete"
    assert len(list(cutouts.glob("*.fits"))) == 2
    assert len(list(cutouts.glob("*.png"))) == 2
    assert len(list(cutouts.glob("*_arrows.pdf"))) == 2
    assert {path.suffix for path in cutouts.iterdir()} == {
        ".fits", ".png", ".pdf"}
    assert any("_g_20arcsec_NuEl.png" in path.name
               for path in cutouts.glob("*.png"))
    assert any("_r_20arcsec_NuEl.png" in path.name
               for path in cutouts.glob("*.png"))
    assert result["png_renderer"] == "coc_tools.fits.fits2png.fits2png"
    assert all(not product["resampled"] for product in result["products"])
    assert all(Path(product["output_fits"]).is_file()
               for product in result["products"])
    with (fits.open(result["products"][0]["source_fits"]) as source_hdul,
          fits.open(result["products"][0]["output_fits"]) as cutout_hdul):
        assert cutout_hdul[0].header["CUTMODE"] == "D4-PIX"
        assert not cutout_hdul[0].header["CUTRSMP"]
        assert np.array_equal(
            cutout_hdul[0].data, source_hdul[0].data[40:61, 40:61])
    assert (pp_root / "reduced_results_manifest.json").is_file()


def test_continues_after_one_outside_image(tmp_path):
    pp_root = tmp_path / "PP"
    pp_root.mkdir()
    source = pp_root / "red_image.0001.fits"
    records = [write_source(
        source, "2026-08-16T01:02:03.000000", "r", "Target", "Target")]
    workflow = pp_root / "reduced_data_workflow_manifest.json"
    workflow.write_text(json.dumps({
        "output_dir": str(pp_root), "source_count": 1, "records": records}))

    def outside_query(target, epochs, site_code):
        return [{"ra_deg": 100.0, "dec_deg": -20.0,
                 "antisolar_deg": 1.0, "antimotion_deg": 2.0}]

    result = publish_reduced_results(
        workflow, tmp_path / "results", query_func=outside_query,
        arrow_func=lambda *args: None)

    assert result["status"] == "partial"
    assert result["product_count"] == 0
    assert result["failure_count"] == 1

    def inside_query(target, epochs, site_code):
        return [{"ra_deg": 10.0, "dec_deg": 20.0,
                 "antisolar_deg": 1.0, "antimotion_deg": 2.0}]

    def fake_arrows(antisolar, antimotion, output_path):
        Path(output_path).write_bytes(b"%PDF-test")

    def fake_png(output_fits, output_png):
        assert Path(output_fits).is_file()
        Path(output_png).write_bytes(b"PNG-test")

    resumed = publish_reduced_results(
        workflow, tmp_path / "results", query_func=inside_query,
        arrow_func=fake_arrows, png_func=fake_png, resume=True)
    assert resumed["status"] == "complete"
    assert resumed["product_count"] == 1
    assert Path(resumed["products"][0]["output_fits"]).is_file()
