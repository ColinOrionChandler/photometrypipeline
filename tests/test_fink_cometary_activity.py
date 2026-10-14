import math

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


import pptool_fink_cometary_activity as fink_activity


def gaussian_psf(size=41, sigma=2.0):
    yy, xx = np.indices((size, size), dtype=float)
    center = (size - 1) / 2.0
    image = np.exp(-0.5 * (
        ((xx - center) / sigma) ** 2 +
        ((yy - center) / sigma) ** 2))
    return image / image.sum()


def test_catalog_activity_metrics_measure_extension():
    alert = {
        "r:psfFlux": 4000.0,
        "r:psfFluxErr": 200.0,
        "r:apFlux": 7000.0,
        "r:apFluxErr": 500.0,
        "r:scienceFlux": 4100.0,
        "r:scienceFluxErr": 190.0,
        "r:templateFlux": 20.0,
        "r:templateFluxErr": 80.0,
        "r:ixx": 0.40,
        "r:iyy": 0.32,
        "r:ixxPSF": 0.12,
        "r:iyyPSF": 0.10,
        "r:extendedness": 0.99,
        "r:psfChi2": 1100.0,
        "r:psfNdata": 1000,
        "r:reliability": 0.999,
    }
    result = fink_activity.catalog_activity_metrics(alert)

    assert result["aperture_minus_psf_njy"] == 3000.0
    assert result["aperture_excess_fraction"] == 3.0 / 7.0
    assert result["aperture_minus_psf_snr_independent_error_assumption"] > 5
    assert result["moment_width_ratio"] == math.sqrt(0.72 / 0.22)
    assert result["quadrature_intrinsic_fwhm_arcsec"] > 0
    assert result["quality_clean"] is True


def test_variance_weighted_fit_leaves_broad_outer_excess():
    rng = np.random.default_rng(17)
    shape = (30, 30)
    center_x, center_y = 14.3, 14.7
    psf = gaussian_psf(sigma=1.9)
    psf_model = fink_activity.psf_on_image(
        psf, shape, center_x, center_y, dx=0.25, dy=-0.15)
    broad = gaussian_psf(sigma=5.2)
    broad_model = fink_activity.psf_on_image(
        broad, shape, center_x, center_y, dx=0.25, dy=-0.15)
    variance = np.full(shape, 12.0 ** 2)
    data = (5000.0 * psf_model + 2600.0 * broad_model + 7.0 +
            rng.normal(0.0, 12.0, shape))
    cutout = {
        "data": data,
        "variance": variance,
        "psf": psf,
        "pixel_scale_arcsec": 0.2,
    }

    fit = fink_activity.fit_psf_model(
        cutout, center_x, center_y, initial_flux=5000.0)
    radial = fink_activity.radial_residual_measurement(cutout, fit)

    assert fit["success"] is True
    assert abs(fit["centroid_dx_px"] - 0.25) < 0.35
    assert abs(fit["centroid_dy_px"] + 0.15) < 0.35
    assert 4500 < fit["amplitude_njy"] < 7000
    assert radial["outer_residual_flux_njy"] > 0
    assert radial["formal_outer_residual_snr"] > 3


def test_residual_asymmetry_detects_one_sided_excess():
    shape = (31, 31)
    center = 15.0
    yy, xx = np.indices(shape, dtype=float)
    variance = np.full(shape, 4.0)
    symmetric = 10.0 * np.exp(-0.5 * (
        np.hypot(xx - center, yy - center) / 4.0) ** 2)
    one_sided = symmetric + 20.0 * np.exp(-0.5 * (
        ((xx - center - 5.0) / 1.5) ** 2 +
        ((yy - center) / 1.5) ** 2))
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [center + 1.0, center + 1.0]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = [-0.2 / 3600.0, 0.2 / 3600.0]
    cutout = {"variance": variance, "pixel_scale_arcsec": 0.2,
              "wcs": wcs}
    base_fit = {"centroid_x": center, "centroid_y": center}
    symmetric_result = fink_activity.residual_asymmetry_measurement(
        cutout, {**base_fit, "residual": symmetric}, 0.6, 2.4)
    asymmetric_result = fink_activity.residual_asymmetry_measurement(
        cutout, {**base_fit, "residual": one_sided}, 0.6, 2.4)

    assert symmetric_result["formal_m1_snr"] < 1e-6
    assert asymmetric_result["formal_m1_snr"] > 5
    assert asymmetric_result["signed_m1_fraction"] > 0.1
    assert 260 < asymmetric_result["position_angle_deg"] < 280


def test_smoothed_significance_propagates_constant_variance():
    signal = np.zeros((21, 21), dtype=float)
    variance = np.full(signal.shape, 4.0)
    signal[10, 10] = 20.0
    result = fink_activity.smoothed_significance(signal, variance, sigma_px=1.0)

    assert np.isfinite(result[10, 10])
    assert result[10, 10] > 1.0
    assert np.nanmax(result) == result[10, 10]


def test_compass_vectors_follow_celestial_wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [15.0, 15.0]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = [-0.2 / 3600.0, 0.2 / 3600.0]

    vectors = fink_activity.compass_vectors(wcs, 14.0, 14.0)

    assert abs(np.hypot(*vectors["N"]) - 3.5) < 1e-8
    assert abs(np.hypot(*vectors["E"]) - 3.5) < 1e-8
    assert vectors["N"][1] > 0
    assert abs(vectors["N"][0]) < 1e-3
    assert vectors["E"][0] < 0
    assert abs(vectors["E"][1]) < 1e-3


def test_centered_display_limits_hold_fixed_angular_field():
    for shape, center in [((30, 30), (14.263, 14.711)),
                          ((32, 32), (15.403, 15.983))]:
        x_limits, y_limits = fink_activity.centered_display_limits(
            center[0], center[1], 0.2)
        assert abs((x_limits[1] - x_limits[0]) * 0.2 - 6.0) < 1e-10
        assert abs((y_limits[1] - y_limits[0]) * 0.2 - 6.0) < 1e-10
        assert abs(np.mean(x_limits) - center[0]) < 1e-10
        assert abs(np.mean(y_limits) - center[1]) < 1e-10


def test_load_cutout_requires_and_uses_variance_plane(tmp_path):
    size = 30
    data = np.ones((size, size), dtype=np.float32)
    variance = np.full((size, size), 9.0, dtype=np.float32)
    psf = gaussian_psf().astype(np.float64)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [15.0, 15.0]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = [-0.2 / 3600.0, 0.2 / 3600.0]
    header = wcs.to_header()
    del header["CTYPE1"]
    del header["CTYPE2"]
    del header["CUNIT1"]
    del header["CUNIT2"]
    header["BUNIT"] = "nJy"
    path = tmp_path / "difference.fits"
    fits.HDUList([
        fits.PrimaryHDU(data=data, header=header),
        fits.ImageHDU(data=variance,
                      header=fits.Header({"UTYPE": "VarianceUncertainty"})),
        fits.ImageHDU(data=psf, name="PSFIMAGE"),
    ]).writeto(path)

    loaded = fink_activity.load_cutout(path)

    assert loaded["data"].shape == (size, size)
    assert loaded["variance"][0, 0] == 9.0
    assert abs(loaded["psf"].sum() - 1.0) < 1e-10
    assert abs(loaded["pixel_scale_arcsec"] - 0.2) < 1e-6
    assert loaded["wcs_labels_repaired"] is True


def test_local_motion_estimate_converts_pair_to_exposure_trail():
    alert = {
        "r:midpointMjdTai": 61000.0,
        "r:ra": 100.0,
        "r:dec": 20.0,
    }
    queries = [{
        "mjd_tai": 61000.0 + 1.0 / 24.0,
        "ra_deg": 100.01,
        "dec_deg": 20.0,
        "query_id": "paired",
    }]
    result = fink_activity.local_motion_estimate(
        alert, queries, exposure_s=30.0)

    expected_speed = 0.01 * math.cos(math.radians(20.0)) * 3600.0
    assert abs(result["speed_arcsec_hour"] - expected_speed) < 1e-8
    assert abs(result["predicted_exposure_trail_arcsec"] -
               expected_speed / 120.0) < 1e-8
