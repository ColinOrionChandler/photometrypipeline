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
import pptool_efosc2 as efosc2
import pptool_mpcsubmission as mpcsub


def efosc2_header():
    header = fits.Header()
    header["TELESCOP"] = "ESO-NTT"
    header["INSTRUME"] = "EFOSC"
    header["OBJECT"] = "P2021R8"
    header["DATE-OBS"] = "2026-07-13T06:10:31.131"
    header["MJD-OBS"] = 61234.257304757
    header["EXPTIME"] = 300.012
    header["AIRMASS"] = 1.1
    header["RA"] = 298.625
    header["DEC"] = -22.016
    header["HIERARCH ESO INS FILT1 NAME"] = "r#784"
    header["HIERARCH ESO DET WIN1 BINX"] = 2
    header["HIERARCH ESO DET WIN1 BINY"] = 2
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 298.625
    header["CRVAL2"] = -22.016
    header["CRPIX1"] = 16.0
    header["CRPIX2"] = 16.0
    header["CD1_1"] = -0.2408 / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.2408 / 3600.0
    return header


def write_native(path):
    data = np.full((32, 32), 100.0, dtype=np.float32)
    data[20, 11] = 140.0
    data[20, 12] = 120.0
    fits.HDUList([
        fits.PrimaryHDU(data=data, header=efosc2_header()),
        fits.ImageHDU(data=np.ones_like(data)),
    ]).writeto(path)


def click_for(path):
    return {"fits_path": str(path), "png_path": str(path.with_suffix(".png")),
            "x": 11.0, "y": 11.0, "width": 32, "height": 32}


def test_efosc2_telescope_config_is_registered():
    obsparam = _pp_conf.telescope_parameters["NTTEFOSC2"]
    assert obsparam["observatory_code"] == "809"
    assert obsparam["secpix"] == (0.1204, 0.1204)
    assert obsparam["binning"] == ("ESO DET WIN1 BINX", "ESO DET WIN1 BINY")
    assert obsparam["filter_translations"]["r#784"] == "r"
    assert obsparam["astrometry_catalogs"] == ["GAIA"]
    assert obsparam["photometry_catalogs"][0] == "PANSTARRS"
    assert _pp_conf.instrument_identifiers["EFOSC"] == "NTTEFOSC2"
    assert "NTTEFOSC2" in _pp_conf.implemented_telescopes


def test_midpoint_and_png_vertical_inversion():
    header = efosc2_header()
    expected = header["MJD-OBS"] + 2400000.5 + header["EXPTIME"] / 172800.0
    assert np.isclose(efosc2.midpoint_jd(header), expected)
    assert efosc2.png_to_fits_pixel(
        {"x": 11, "y": 11, "width": 32, "height": 32}, (32, 32)) == (11, 20)


def test_primary_flattening_centroid_and_original_preservation(tmp_path):
    source = tmp_path / "frame.fits"
    png = tmp_path / "frame.png"
    write_native(source)
    png.write_bytes(b"png sentinel")
    original_hash = efosc2.sha256(source)
    output = tmp_path / "r" / "working.fits"

    record = efosc2.flatten_working_copy(
        source, output, "McLeod", click_for(source), min_snr=1.0)

    assert record["source_hdu_count"] == 2
    assert record["working_hdu_count"] == 1
    assert record["filter"] == "r"
    assert record["centroid"]["centroid_x"] > 11
    assert np.isclose(record["centroid"]["centroid_y"], 20)
    assert efosc2.sha256(source) == original_hash
    with fits.open(output) as hdus:
        assert len(hdus) == 1
        assert hdus[0].header["OBJECT"] == "McLeod"
        assert hdus[0].header["FILTER"] == "r"
        assert hdus[0].header["TEL_KEYW"] == "NTTEFOSC2"
        assert hdus[0].header["ESO DET WIN1 BINX"] == 2


def test_final_position_uses_refined_not_native_wcs(tmp_path):
    source = tmp_path / "frame.fits"
    source.with_suffix(".png").write_bytes(b"png")
    write_native(source)
    output = tmp_path / "r" / "working.fits"
    record = efosc2.flatten_working_copy(source, output, "McLeod",
                                         click_for(source), min_snr=1.0)
    with fits.open(output, mode="update") as hdus:
        hdus[0].header["CRVAL1"] += 0.001
        hdus[0].header["REGCAT"] = "GAIA"
    efosc2.apply_refined_wcs([record])
    assert record["wcs_refined"] is True
    assert record["wcs_shift_arcsec"] > 3.0
    assert abs(record["refined_ra_deg"] - record["native_ra_deg"]) > 0.0009


def test_backup_contains_originals_and_reuses_archive(tmp_path):
    fits_path = tmp_path / "frame.fits"
    png_path = tmp_path / "frame.png"
    csv_path = tmp_path / "click_points.csv"
    write_native(fits_path)
    png_path.write_bytes(b"png")
    csv_path.write_text("image,x,y,width,height,status\nframe.png,1,1,32,32,clicked\n")
    paths = [fits_path, png_path, csv_path]
    first = efosc2.create_originals_backup(tmp_path, paths)
    second = efosc2.create_originals_backup(tmp_path, paths)
    assert Path(first["path"]).exists()
    assert first["created"] is True
    assert second["created"] is False
    assert first["checksums"] == second["checksums"]


def test_initial_position_unicode_split_and_orbit_station_invariant(tmp_path):
    source = tmp_path / "initial_positions.txt"
    lines = []
    for index in range(7):
        lines.append("2026-07-%02dT06:00:00.000Z\u00a0298.%d\u2003-22.%d" %
                     (6 + index, index, index))
    source.write_text("\n".join(lines) + "\n")
    baseline, removed, baseline_path = efosc2.split_initial_positions(source)
    final = [{"obs_time": row["obs_time"], "jd": row["jd"],
              "ra_deg": row["ra_deg"], "dec_deg": row["dec_deg"],
              "rms_ra": 0.1, "rms_dec": 0.1, "astcat": "Gaia2"}
             for row in removed]
    augmented = efosc2.format_find_orb_psv("McLeod", baseline, final)
    data = [line for line in augmented.splitlines()
            if line and not line.startswith("#") and not line.startswith("trkSub|")]
    assert baseline_path.exists()
    assert len(baseline) == 3 and len(removed) == 4 and len(data) == 7
    assert sum("|X05|" in line for line in data) == 3
    assert sum("|809|" in line for line in data) == 4


def test_temporary_tracklet_review_only_ades_has_no_submit_script(tmp_path):
    photometry = tmp_path / "photometry_McLeod.dat"
    rows = []
    for index in range(4):
        token = "frame%d" % index
        rows.append(
            " %s 2461234.%d 20.1 0.1 298.6 -22.0 0.0 0.0 0.0 0.0 "
            "300 26.0 0.03 0.0 0.1 PANSTARRS r 0 EFOSC APER 1.0 "
            "0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1\n" %
            (token, index))
        header = fits.Header()
        header["TELESCOP"] = "ESO 3.58-m NTT"
        header["INSTRUME"] = "EFOSC"
        header["TEL_KEYW"] = "NTTEFOSC2"
        fits.PrimaryHDU(header=header).writeto(tmp_path / (token + ".fits"))
    photometry.write_text("# header\n" + "".join(rows))
    config = mpcsub.SubmissionConfig(
        target="McLeod", trk_sub="McLeod", review_only=True,
        observatory_code="809", observers=["UNKNOWN"], output_dir=tmp_path)
    bundle = mpcsub.build_submission(photometry, config)
    mpcsub.write_submission(bundle)
    data_rows = [line for line in bundle.ades_text.splitlines()
                 if "|CCD|809|" in line]
    assert len(data_rows) == 4
    assert all("|McLeod|CCD|809|" in line for line in data_rows)
    assert "provID" in bundle.ades_text and "|McLeod|CCD|809|" in bundle.ades_text
    assert bundle.obs80_path is None
    assert bundle.submit_script_path is None
    assert "Submission-ready: no (review only)" in bundle.summary_text
    assert not list(tmp_path.glob("submit_*.sh"))


def test_find_orb_parser_tolerates_multiline_ephemeris_warning():
    parsed = efosc2.parse_find_orb_stdout(
        "Processing 1 objects\n1: McLeod\nwarning text\n"
        "; a=3.077, e=0.212, i=2   7 obs\x1b[0m; 2026 July 6-13\n")
    assert parsed["object"] == "McLeod"
    assert parsed["observations"] == 7
    assert parsed["semimajor_axis_au"] == 3.077


def test_panstarrs_transformed_maps_to_current_ades_catalog():
    catalog, warning = mpcsub.normalize_ades_catalog("PANSTARRS_transformed")
    assert catalog == "PS1_DR1"
    assert "mapped" in warning


def test_photometry_qa_records_match_and_calibration(tmp_path):
    path = tmp_path / "photometry_McLeod.dat"
    path.write_text(
        "# header\n"
        " r/efosc2_r_0001 2461234.7 21.6 0.12 298.6 -22.0 0.06 -0.41 "
        "0 0 300.01 31.754 0.033 -10.1 0.11 PANSTARRS_transformed r "
        "1 NTTEFOSC2 APER 1.83 0.11\n")
    row = efosc2.read_photometry_qa(path)[0]
    assert np.isclose(row["source_match_separation_arcsec"],
                      np.hypot(0.06, -0.41))
    assert row["calibrated_photometry"] is True
    assert row["photometry_catalog"] == "PANSTARRS_transformed"
    assert row["zeropoint"] == 31.754
