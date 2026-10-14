
import numpy as np


import pptool_cometary_student_stack as student


def test_sector_profile_and_mask_reject_compact_background_star():
    scale = 0.25
    shape = (201, 201)
    east, north, radius, _ = student.radial_geometry(shape, scale)
    clean = np.exp(-0.5 * (radius / 2.0) ** 2)
    star = 15.0 * np.exp(-0.5 * (
        ((east + 9.5) / 0.8) ** 2 + (north / 0.8) ** 2))
    contaminated = clean + star
    mask = student.circular_mask(shape, scale, -9.5, 0.0, 3.0)

    clean_profile = student.sector_radial_profile(
        clean, np.zeros(shape, dtype=bool), scale)["profile"]
    robust_profile = student.sector_radial_profile(
        contaminated, mask, scale)["profile"]
    _, unmasked = student.mean_radial_profile(
        contaminated, np.zeros(shape, dtype=bool), scale)
    annulus = (np.arange(len(clean_profile)) * 0.5 + 0.25 >= 8.5) & \
        (np.arange(len(clean_profile)) * 0.5 + 0.25 <= 10.5)

    assert np.nanmax(np.abs(robust_profile[annulus] -
                            clean_profile[annulus])) < 1e-3
    assert np.nanmax(unmasked[annulus]) > 0.1


def test_compact_contaminant_location_is_recovered():
    scale = 0.25
    shape = (201, 201)
    east, north, _, _ = student.radial_geometry(shape, scale)
    image = np.random.default_rng(3).normal(0.0, 0.01, shape)
    image += 3.0 * np.exp(-0.5 * (
        ((east + 9.5) / 0.7) ** 2 + ((north - 0.5) / 0.7) ** 2))

    result = student.find_compact_contaminant(
        image, scale, sky_sigma=0.01, minimum_sigma=5.0)

    assert abs(result["east_arcsec"] + 9.5) <= scale
    assert abs(result["north_arcsec"] - 0.5) <= scale
    assert result["peak_significance"] > 5.0


def test_first_crossover_interpolates_after_significant_run():
    radii = np.asarray([2.25, 2.75, 3.25, 3.75, 4.25, 4.75])
    excess = np.asarray([0.8, 0.6, 0.4, 0.2, -0.2, -0.5])
    edges = np.arange(2.0, 5.5, 0.5)

    crossover = student.first_profile_crossover(
        radii, excess, run=(0, 3), edges=edges)

    assert crossover == 4.0
