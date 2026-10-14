import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["PHOTPIPEDIR"] = str(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pptool_cometary_activity as cometary


def gaussian_psf(size=101, sigma=4.0, x_offset=0.0, y_offset=0.0):
    yy, xx = np.indices((size, size), dtype=float)
    center = (size - 1) / 2.0
    image = np.exp(-0.5 * (((xx - center - x_offset) / sigma) ** 2 +
                           ((yy - center - y_offset) / sigma) ** 2))
    return image / image.sum()


def stamp_record(image, scale=0.25, fwhm_arcsec=2.35, sky_sigma=1.0,
                 wcs=None):
    size = image.shape[0]
    wcs = wcs or cometary.canonical_wcs(10.0, 20.0, size, scale)
    wcs.array_shape = image.shape
    return {
        "working": image,
        "direct": image,
        "template": np.zeros_like(image),
        "catalog_mask": np.zeros_like(image, dtype=bool),
        "wcs": wcs,
        "sky_sigma": sky_sigma,
        "common_fwhm_arcsec": fwhm_arcsec,
        "common_fwhm_px": fwhm_arcsec / scale,
        "subtraction_used": True,
        "subtraction_reason": "test",
        "target_info": {"blended": False},
    }


def test_motion_trailed_point_source_does_not_create_false_coma():
    scale = 0.25
    psf = gaussian_psf()
    trailed = cometary.apply_motion_trail(
        psf, east_rate=-30.0, north_rate=-8.0, exptime=240,
        pixel_scale_arcsec=scale)
    data = 50000.0 * trailed
    record = stamp_record(data, scale=scale, sky_sigma=2.0)

    result = cometary.measure_stamp(
        record, trailed, [trailed.copy() for _ in range(8)], scale,
        motion_pa_deg=255.0, trail_arcsec=2.0)

    assert result["coma_detected"] is False
    assert result["coma_diameter_arcsec"] is None


def test_pretrail_fit_recovers_psf_beneath_known_tracking_trail():
    scale = 0.25
    base = cometary.elliptical_moffat_psf(
        101, fwhm_major_px=10.0, axis_ratio=0.78, beta=3.5,
        pa_deg=142.0)
    observed = cometary.apply_line_trail(
        base, length_px=8.0, pa_deg=74.0)
    fit = cometary.fit_pretrail_moffat(
        observed, scale, trail_length_arcsec=2.0, trail_pa_deg=254.0)

    assert fit["success"] is True
    assert fit["rms_peak_fraction"] < 0.03
    assert abs(fit["fwhm_major_px"] - 10.0) < 1.0
    assert abs(fit["fwhm_minor_px"] - 7.8) < 1.0
    assert abs(cometary.circular_delta_deg(fit["pa_deg"], 142.0)) < 8.0


def test_target_tracking_uses_pretrail_psf_without_second_convolution(
        tmp_path):
    scale = 0.25
    wcs = cometary.canonical_wcs(10.0, 20.0, 101, scale)
    base = cometary.elliptical_moffat_psf(
        101, fwhm_major_px=9.0, axis_ratio=0.85, beta=3.5,
        pa_deg=135.0)
    observed = cometary.apply_line_trail(
        base, length_px=8.0, pa_deg=74.0)
    frame = cometary.FrameData(
        position=cometary.Position("frame.0001.fits", 2460000.5,
                                    10.0, 20.0),
        path=tmp_path / "frame.0001.fits", header=fits.Header(),
        data=np.zeros((101, 101)), wcs=wcs,
        pixel_scale_arcsec=scale, exptime=240.0, catalog=[],
        psf=observed, psf_stars=[observed],
        psf_fwhm_arcsec=2.5, pretrail_psf=base,
        pretrail_psf_stars=[base])
    frame.motion_east_arcsec_hour = -29.0
    frame.motion_north_arcsec_hour = -8.0

    target_psf, comparisons = cometary.make_trailed_psf_products(
        frame, common_fwhm_arcsec=2.5, output_scale_arcsec=scale,
        psf_mode="pretrail-target-tracked")

    assert np.allclose(target_psf, base)
    assert np.allclose(comparisons[0], base)
    assert not np.allclose(target_psf, observed)


def test_resolved_coma_produces_non_fwhm_diameter():
    scale = 0.25
    psf = gaussian_psf(sigma=3.5)
    coma = gaussian_psf(sigma=13.0)
    data = 40000.0 * psf + 18000.0 * coma
    record = stamp_record(data, scale=scale, fwhm_arcsec=2.06,
                          sky_sigma=0.5)

    result = cometary.measure_stamp(
        record, psf, [psf.copy() for _ in range(10)], scale,
        motion_pa_deg=250.0, trail_arcsec=0.0)

    assert result["coma_detected"] is True
    assert result["coma_diameter_arcsec"] > record["common_fwhm_arcsec"]
    assert result["coma_diameter_unc_arcsec"] >= 2 * scale


def synthetic_tail(pa_deg, width_deg=8.0, radius_arcsec=12.0,
                   scale=0.25, size=121, rotated_wcs=None):
    wcs = rotated_wcs or cometary.canonical_wcs(
        10.0, 20.0, size, scale)
    wcs.array_shape = (size, size)
    radius, pa = cometary.sky_geometry(wcs, (size, size))
    angle = np.abs(cometary.circular_delta_deg(pa, pa_deg))
    tail = np.exp(-0.5 * (angle / width_deg) ** 2)
    tail *= (radius >= 2.5) & (radius <= radius_arcsec)
    tail *= np.exp(-radius / 18.0)
    return tail, wcs


def test_straight_tail_pa_and_length_recovery():
    image, wcs = synthetic_tail(42.0)
    result = cometary.measure_tail(
        image, np.zeros_like(image, dtype=bool), wcs, sky_sigma=0.02,
        core_radius_arcsec=2.0, max_radius_arcsec=14.0,
        angle_bin_deg=5.0, radial_bin_arcsec=0.5)

    assert result["appearance"] == "straight"
    assert abs(cometary.circular_delta_deg(result["pa_deg"], 42.0)) <= 5.0
    assert abs(result["length_arcsec"] - 12.0) <= 1.0


def test_broad_tail_wraps_through_zero_degrees():
    image, wcs = synthetic_tail(355.0, width_deg=24.0)
    result = cometary.measure_tail(
        image, np.zeros_like(image, dtype=bool), wcs, sky_sigma=0.02,
        core_radius_arcsec=2.0, max_radius_arcsec=14.0,
        angle_bin_deg=5.0, radial_bin_arcsec=0.5)

    assert result["appearance"] == "broad"
    assert abs(cometary.circular_delta_deg(result["pa_deg"], 355.0)) <= 7.5
    assert result["pa_lower_deg"] > result["pa_upper_deg"]


def test_curved_tail_is_identified_from_radial_pa_change():
    size = 121
    scale = 0.25
    wcs = cometary.canonical_wcs(10.0, 20.0, size, scale)
    wcs.array_shape = (size, size)
    radius, pa = cometary.sky_geometry(wcs, (size, size))
    ridge_pa = 20.0 + 5.0 * radius
    angle = np.abs(cometary.circular_delta_deg(pa, ridge_pa))
    image = np.exp(-0.5 * (angle / 7.0) ** 2)
    image *= (radius >= 2.5) & (radius <= 12.0)
    image *= np.exp(-radius / 18.0)
    result = cometary.measure_tail(
        image, np.zeros_like(image, dtype=bool), wcs, sky_sigma=0.02,
        core_radius_arcsec=2.0, max_radius_arcsec=14.0,
        angle_bin_deg=5.0, radial_bin_arcsec=0.5)

    assert result["appearance"] == "curved"
    assert result["status"] == "significant"
    assert abs(result["length_arcsec"] - 12.0) <= 1.0


def test_tail_pa_uses_wcs_with_rotated_pixels():
    size = 121
    scale = 0.25 / 3600.0
    angle = math.radians(37.0)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(size + 1) / 2.0, (size + 1) / 2.0]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cd = np.asarray([
        [-scale * math.cos(angle), scale * math.sin(angle)],
        [scale * math.sin(angle), scale * math.cos(angle)],
    ])
    image, wcs = synthetic_tail(110.0, rotated_wcs=wcs)
    result = cometary.measure_tail(
        image, np.zeros_like(image, dtype=bool), wcs, sky_sigma=0.02,
        core_radius_arcsec=2.0, max_radius_arcsec=14.0,
        angle_bin_deg=5.0, radial_bin_arcsec=0.5)
    assert abs(cometary.circular_delta_deg(result["pa_deg"], 110.0)) <= 5.0


def test_masked_median_template_removes_moving_target_not_static_source():
    size = 31
    yy, xx = np.indices((size, size))
    static = 100.0 * np.exp(-0.5 * ((xx - 8) ** 2 + (yy - 8) ** 2) / 2.0)
    stamps, masks = [], []
    for target_x in (10, 16, 22, 28):
        target = 50.0 * np.exp(-0.5 * (
            (xx - target_x) ** 2 + (yy - 16) ** 2) / 2.0)
        stamps.append(static + target)
        masks.append(np.hypot(xx - target_x, yy - 16) <= 3)
    template, valid = cometary.median_template(stamps, masks, minimum=2)
    assert template[8, 8] > 90
    assert abs(template[16, 16]) < 1
    assert valid[8, 8] == 4


def test_missing_psf_stars_fails_cleanly(tmp_path):
    wcs = cometary.canonical_wcs(10.0, 20.0, 31, 0.25)
    frame = cometary.FrameData(
        position=cometary.Position("test.fits", 2460000.5, 10.0, 20.0),
        path=tmp_path / "test.fits", header=fits.Header(),
        data=np.zeros((31, 31), dtype=np.float32), wcs=wcs,
        pixel_scale_arcsec=0.25, exptime=30.0, catalog=[])
    with pytest.raises(ValueError, match="usable PSF stars"):
        cometary.build_empirical_psf(frame, 0.25, 31)


def test_crowded_target_is_flagged_as_blended(tmp_path):
    size = 41
    scale = 0.25
    wcs = cometary.canonical_wcs(10.0, 20.0, size, scale)
    wcs.array_shape = (size, size)
    center_world = wcs.all_pix2world([[(size - 1) / 2.0,
                                      (size - 1) / 2.0]], 0)[0]
    neighbor_world = wcs.all_pix2world([[(size - 1) / 2.0 + 4.0,
                                        (size - 1) / 2.0]], 0)[0]

    def source(world, flags=0):
        return cometary.CatalogSource(
            x=0.0, y=0.0, ra_deg=world[0], dec_deg=world[1],
            flux=1000.0, flux_max=100.0, fwhm_px=4.0,
            a_px=2.0, b_px=2.0, theta_deg=0.0, flags=flags)

    frame = cometary.FrameData(
        position=cometary.Position("test.fits", 2460000.5, 10.0, 20.0),
        path=tmp_path / "test.fits", header=fits.Header(),
        data=np.zeros((size, size), dtype=np.float32), wcs=wcs,
        pixel_scale_arcsec=scale, exptime=30.0,
        catalog=[source(center_world), source(neighbor_world)])
    info = cometary.target_source_info(frame, wcs, common_fwhm_px=8.0)

    assert info["central_catalog_sources"] == 2
    assert info["other_central_sources"] == 1
    assert info["blended"] is True


def test_failed_sky_subtraction_falls_back_to_direct_image(tmp_path):
    size = 41
    scale = 0.25
    wcs = cometary.canonical_wcs(10.0, 20.0, size, scale)
    wcs.array_shape = (size, size)
    rng = np.random.default_rng(7)
    frames = []
    for index in range(4):
        noise = 0.05 if index == 0 else 50.0
        frame = cometary.FrameData(
            position=cometary.Position(
                "frame.%04d.fits" % index,
                2460000.5 + index / 1440.0, 10.0, 20.0),
            path=tmp_path / ("frame.%04d.fits" % index),
            header=fits.Header(),
            data=rng.normal(0.0, noise, (size, size)), wcs=wcs,
            pixel_scale_arcsec=scale, exptime=30.0, catalog=[],
            psf_fwhm_arcsec=2.0)
        frames.append(frame)
    ephemeris = {frame.position.filename: frame.position for frame in frames}
    result = cometary.prepare_frame_stamp(
        frames[0], frames, scale, size, 2.0, ephemeris)

    assert result["subtraction_used"] is False
    assert result["subtraction_reason"].startswith("failed QA:")
    assert np.allclose(result["working"], result["direct"], atol=1e-10,
                       equal_nan=True)


def test_form_payload_has_exact_fields_and_comment_limit():
    measurement = {
        "coma_detected": True, "coma_appearance": "condensed",
        "coma_diameter_arcsec": 5.4, "coma_diameter_unc_arcsec": 0.5,
        "coma_peak_significance": 7.0,
        "tail": {"length_arcsec": 8.0, "appearance": "broad",
                 "pa_deg": 355.0, "pa_lower_deg": 340.0,
                 "pa_upper_deg": 10.0, "status": "marginal"},
    }
    primary = {
        "measurement": measurement, "midpoint_jd": 2461237.659913,
        "frame_ids": ["0146", "0147", "0148", "0149", "0150", "0151"],
    }
    rows = [{"selected": True, "coma_detected": True, "blended": False}
            for _ in range(6)]
    payload = cometary.make_form_payload(
        primary, rows, "RMM2026", "C. O. Chandler", "coc123@uw.edu",
        "705", 3.5)
    assert tuple(payload) == cometary.FORM_FIELDS
    assert payload["desig"] == "RMM2026"
    assert payload["obsdate"] == "2026-07-16"
    assert payload["obscode"] == "705"
    assert payload["coma_diam_units"] == "arc_sec"
    assert payload["tail_pa"] == "340"
    assert payload["tail_pa2"] == "10"
    assert len(payload["additional_comments"]) <= 500
    json.dumps(payload)


def test_no_tail_payload_leaves_tail_measurements_blank():
    primary = {
        "measurement": {
            "coma_detected": True, "coma_appearance": "condensed",
            "coma_diameter_arcsec": 5.0,
            "coma_diameter_unc_arcsec": 1.0,
            "coma_peak_significance": 6.0,
            "tail": {"length_arcsec": None, "appearance": "no tail",
                     "status": "not_detected"},
        },
        "midpoint_jd": 2461237.659913,
        "frame_ids": ["0146"],
    }
    rows = [{"selected": True, "coma_detected": True, "blended": False}]
    payload = cometary.make_form_payload(
        primary, rows, "RMM2026", "C. O. Chandler", "coc123@uw.edu",
        "705", 3.5)

    assert payload["tail_appearance"] == "no tail"
    assert payload["tail_length"] == ""
    assert payload["tail_pa"] == ""
    assert payload["tail_pa2"] == ""


def test_expected_physical_tail_direction_clears_motion_artifact_flag():
    measurement = {
        "tail": {
            "pa_deg": 275.0,
            "pa_unc_deg": 5.0,
            "motion_artifact_likely": True,
        },
    }
    cometary.annotate_expected_tail(
        measurement, expected_pa_deg=275.0, expected_pa_unc_deg=0.5,
        label="anti-solar/anti-motion")

    assert measurement["tail"]["physical_direction_consistent"] is True
    assert measurement["tail"]["expected_pa_separation_deg"] == 0.0
    assert measurement["tail"]["motion_artifact_likely"] is False


def test_group_frames_uses_time_gap_and_primary_ids_are_explicit():
    wcs = cometary.canonical_wcs(10.0, 20.0, 31, 0.25)
    frames = []
    for index, offset_min in enumerate((0, 4, 8, 130, 134)):
        frames.append(cometary.FrameData(
            position=cometary.Position(
                "red_image.%04d.new.fits" % index,
                2460000.5 + offset_min / 1440.0, 10.0, 20.0),
            path=Path("frame_%d.fits" % index), header=fits.Header(),
            data=np.zeros((31, 31), dtype=np.float32), wcs=wcs,
            pixel_scale_arcsec=0.25, exptime=30, catalog=[]))
    blocks = cometary.group_frames(frames, gap_minutes=30)
    assert [len(block) for block in blocks] == [3, 2]


def test_exposure_weighted_midpoint_and_folder_date_warning():
    wcs = cometary.canonical_wcs(10.0, 20.0, 31, 0.25)
    records = []
    for index, (offset_s, exposure_s) in enumerate(((0.0, 100.0),
                                                    (120.0, 300.0))):
        frame = cometary.FrameData(
            position=cometary.Position(
                "frame.%04d.fits" % index,
                2460000.5 + offset_s / 86400.0, 10.0, 20.0),
            path=Path("frame.%04d.fits" % index), header=fits.Header(),
            data=np.zeros((31, 31), dtype=np.float32), wcs=wcs,
            pixel_scale_arcsec=0.25, exptime=exposure_s, catalog=[])
        records.append({"frame": frame})
    midpoint = Time(cometary.weighted_midpoint_iso(records), scale="utc")
    expected = Time(2460000.5 + 90.0 / 86400.0, format="jd", scale="utc")

    assert abs((midpoint - expected).sec) < 1e-4
    warning = cometary.folder_date_warning(
        "/data/20260717UT_APO_McLeod/PP", "2026-07-16")
    assert "20260717UT" in warning
    assert "2026-07-16 UTC" in warning
    assert cometary.folder_date_warning(
        "/data/20260716UT_APO_McLeod/PP", "2026-07-16") is None
