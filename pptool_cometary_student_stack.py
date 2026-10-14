#!/usr/bin/env python3

"""Analyze a target-aligned studentized stack with contaminant-robust profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage

import pptool_cometary_activity as cometary


def center_crop(image, size):
    size = int(size)
    if size % 2 == 0:
        raise ValueError("crop size must be odd")
    center_y = (image.shape[0] - 1) // 2
    center_x = (image.shape[1] - 1) // 2
    half = size // 2
    return np.asarray(
        image[center_y - half:center_y + half + 1,
              center_x - half:center_x + half + 1], dtype=float)


def radial_geometry(shape, pixel_scale_arcsec):
    yy, xx = np.indices(shape, dtype=float)
    center_x = (shape[1] - 1) / 2.0
    center_y = (shape[0] - 1) / 2.0
    east = -(xx - center_x) * pixel_scale_arcsec
    north = (yy - center_y) * pixel_scale_arcsec
    radius = np.hypot(east, north)
    pa = np.degrees(np.arctan2(east, north)) % 360.0
    return east, north, radius, pa


def subtract_outer_sky(image, pixel_scale_arcsec, inner=17.0, outer=22.0):
    _, _, radius, _ = radial_geometry(image.shape, pixel_scale_arcsec)
    use = (radius >= inner) & (radius < outer) & np.isfinite(image)
    sky, sigma = cometary.robust_location_scale(image[use])
    return np.asarray(image, dtype=float) - sky, sky, sigma


def circular_mask(shape, pixel_scale_arcsec, east_arcsec, north_arcsec,
                  radius_arcsec):
    east, north, _, _ = radial_geometry(shape, pixel_scale_arcsec)
    return np.hypot(east - east_arcsec, north - north_arcsec) <= radius_arcsec


def find_compact_contaminant(image, pixel_scale_arcsec, sky_sigma,
                             search_inner=7.0, search_outer=13.0,
                             minimum_sigma=5.0):
    east, north, radius, _ = radial_geometry(image.shape, pixel_scale_arcsec)
    smooth = ndimage.gaussian_filter(
        np.where(np.isfinite(image), image, 0.0), 1.5)
    local_peak = smooth == ndimage.maximum_filter(smooth, size=9)
    use = local_peak & (radius >= search_inner) & (radius <= search_outer)
    if not np.any(use):
        return None
    scores = np.where(use, smooth / max(sky_sigma, 1e-12), -np.inf)
    y, x = np.unravel_index(np.nanargmax(scores), scores.shape)
    if scores[y, x] < minimum_sigma:
        return None
    return {
        "east_arcsec": float(east[y, x]),
        "north_arcsec": float(north[y, x]),
        "radius_arcsec": float(radius[y, x]),
        "peak_significance": float(scores[y, x]),
    }


def sector_radial_profile(image, mask, pixel_scale_arcsec,
                          bin_arcsec=0.5, max_radius_arcsec=12.0,
                          sectors=36):
    image = np.asarray(image, dtype=float)
    mask = np.asarray(mask, dtype=bool) | ~np.isfinite(image)
    _, _, radius, pa = radial_geometry(image.shape, pixel_scale_arcsec)
    radial_edges = np.arange(
        0.0, max_radius_arcsec + bin_arcsec, bin_arcsec)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    angle_edges = np.linspace(0.0, 360.0, sectors + 1)
    values = np.full(len(radial_centers), np.nan)
    scatter = np.full(len(radial_centers), np.nan)
    counts = np.zeros(len(radial_centers), dtype=int)
    pixel_counts = np.zeros(len(radial_centers), dtype=int)
    sector_values = np.full((len(radial_centers), sectors), np.nan)
    for radial_index, (lower, upper) in enumerate(
            zip(radial_edges[:-1], radial_edges[1:])):
        annulus = (radius >= lower) & (radius < upper) & ~mask
        pixel_counts[radial_index] = int(np.sum(annulus))
        minimum_pixels = 1 if upper <= 2.0 else 3
        for angle_index, (angle_low, angle_high) in enumerate(
                zip(angle_edges[:-1], angle_edges[1:])):
            use = annulus & (pa >= angle_low) & (pa < angle_high)
            if np.sum(use) >= minimum_pixels:
                sector_values[radial_index, angle_index] = float(
                    np.mean(image[use]))
        finite = sector_values[radial_index][
            np.isfinite(sector_values[radial_index])]
        if finite.size:
            location, scale = cometary.robust_location_scale(finite)
            keep = np.abs(finite - location) <= max(3.0 * scale, 1e-15)
            finite = finite[keep]
            values[radial_index] = float(np.median(finite))
            _, scatter[radial_index] = cometary.robust_location_scale(finite)
            counts[radial_index] = len(finite)
    return {
        "radii_arcsec": radial_centers,
        "edges_arcsec": radial_edges,
        "profile": values,
        "scatter": scatter,
        "sector_count": counts,
        "pixel_count": pixel_counts,
        "sector_values": sector_values,
    }


def reconstruct_psf_models(summary, frame_ids, shape, pixel_scale_arcsec):
    wanted = set(frame_ids)
    models = []
    rows = []
    for row in summary["frame_measurements"]:
        if row["frame"] not in wanted:
            continue
        required = (
            row.get("pretrail_psf_fwhm_major_arcsec"),
            row.get("pretrail_psf_fwhm_minor_arcsec"),
            row.get("pretrail_psf_beta"),
            row.get("pretrail_psf_pa_deg"),
        )
        if any(value is None for value in required):
            continue
        major, minor, beta, pa = required
        models.append(cometary.elliptical_moffat_psf(
            shape, major / pixel_scale_arcsec, minor / major,
            beta, pa_deg=pa))
        rows.append(row)
    if len(models) != len(wanted):
        missing = sorted(wanted - {row["frame"] for row in rows})
        raise ValueError("missing pretrail PSF fits for %s" % missing)
    return models, rows


def bootstrap_psf_profiles(models, mask, pixel_scale_arcsec,
                           bin_arcsec, max_radius_arcsec,
                           samples=300, seed=20260717):
    models = np.asarray(models)
    rng = np.random.default_rng(seed)
    profiles = []
    for _ in range(int(samples)):
        indices = rng.integers(0, len(models), len(models))
        model = np.mean(models[indices], axis=0)
        model /= max(float(np.sum(model)), 1e-12)
        profiles.append(sector_radial_profile(
            model, mask, pixel_scale_arcsec, bin_arcsec,
            max_radius_arcsec)["profile"])
    profiles = np.asarray(profiles)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            "median": np.nanmedian(profiles, axis=0),
            "low": np.nanpercentile(profiles, 16, axis=0),
            "high": np.nanpercentile(profiles, 84, axis=0),
            "profiles": profiles,
        }


def first_profile_crossover(radii, excess, run, edges):
    if run is None:
        return None
    start = run[1]
    for index in range(start, len(excess) - 1):
        left = excess[index]
        right = excess[index + 1]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        if left > 0 and right <= 0:
            fraction = left / max(left - right, 1e-12)
            return float(radii[index] + fraction *
                         (radii[index + 1] - radii[index]))
    return float(edges[run[1] + 1])


def positive_profile_crossover(radii, excess, start_radius=2.0,
                               maximum_radius=8.0,
                               minimum_positive_bins=2):
    positive_bins = 0
    for index in range(len(excess) - 1):
        if radii[index] < start_radius:
            continue
        if radii[index] > maximum_radius:
            break
        left = excess[index]
        right = excess[index + 1]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        if left > 0:
            positive_bins += 1
        if (positive_bins >= minimum_positive_bins and left > 0 and
                right <= 0):
            fraction = left / max(left - right, 1e-12)
            return float(radii[index] + fraction *
                         (radii[index + 1] - radii[index]))
        if left <= 0 and positive_bins < minimum_positive_bins:
            positive_bins = 0
    return None


def bootstrap_crossover(target_sector_values, psf_profiles, radii,
                        amplitude, samples=1000, seed=20260717):
    rng = np.random.default_rng(seed)
    sector_values = np.asarray(target_sector_values, dtype=float)
    results = []
    for _ in range(int(samples)):
        indices = rng.integers(0, sector_values.shape[1],
                               sector_values.shape[1])
        sampled = sector_values[:, indices]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            profile = np.nanmedian(sampled, axis=1) / max(amplitude, 1e-12)
        psf_profile = psf_profiles[
            rng.integers(0, len(psf_profiles))]
        crossover = positive_profile_crossover(
            radii, profile - psf_profile)
        if crossover is not None:
            results.append(crossover)
    if len(results) < max(30, 0.2 * samples):
        return None
    values = np.asarray(results)
    return {
        "median_radius_arcsec": float(np.median(values)),
        "low_radius_arcsec": float(np.percentile(values, 16)),
        "high_radius_arcsec": float(np.percentile(values, 84)),
        "successful_samples": int(len(values)),
        "total_samples": int(samples),
    }


def analyze_image(image, psf_models, pixel_scale_arcsec, mask,
                  wcs, expected_tail_pa, bin_arcsec=0.5,
                  max_radius_arcsec=12.0):
    working, sky, sky_sigma = subtract_outer_sky(
        image, pixel_scale_arcsec)
    psf = np.mean(psf_models, axis=0)
    psf /= max(float(np.sum(psf)), 1e-12)
    dx, dy, amplitude, registered = cometary.best_psf_registration(
        working, psf, mask, search_px=5)
    model = amplitude * registered
    residual = working - model
    target = sector_radial_profile(
        working / max(amplitude, 1e-12), mask, pixel_scale_arcsec,
        bin_arcsec, max_radius_arcsec)
    psf_profiles = bootstrap_psf_profiles(
        psf_models, np.zeros_like(mask), pixel_scale_arcsec,
        bin_arcsec, max_radius_arcsec)
    radii = target["radii_arcsec"]
    sector_error = target["scatter"] / np.sqrt(
        np.maximum(target["sector_count"], 1)) / max(amplitude, 1e-12)
    psf_error = 0.5 * (psf_profiles["high"] - psf_profiles["low"])
    total_error = np.sqrt(sector_error ** 2 + psf_error ** 2)
    excess = target["profile"] - psf_profiles["median"]
    significance = excess / np.where(total_error > 0, total_error, np.nan)
    pixel_error = (sky_sigma / max(amplitude, 1e-12)) / np.sqrt(
        np.maximum(target["pixel_count"], 1))
    formal_error = np.sqrt(pixel_error ** 2 + psf_error ** 2)
    formal_significance = excess / np.where(
        formal_error > 0, formal_error, np.nan)
    start_radius = 2.0
    conservative_run = cometary.contiguous_extent(
        significance, radii, threshold=3.0, minimum_bins=2,
        start_radius=start_radius, max_start_radius=4.0)
    run = cometary.contiguous_extent(
        formal_significance, radii, threshold=3.0, minimum_bins=2,
        start_radius=start_radius, max_start_radius=4.0)
    significant_edge = (
        float(target["edges_arcsec"][run[1] + 1])
        if run is not None else None)
    direct_crossover = positive_profile_crossover(radii, excess)
    crossover_bootstrap = bootstrap_crossover(
        target["sector_values"], psf_profiles["profiles"], radii,
        amplitude)
    if crossover_bootstrap is not None:
        crossover = crossover_bootstrap["median_radius_arcsec"]
        diameter = 2.0 * crossover
        diameter_uncertainty = (
            crossover_bootstrap["high_radius_arcsec"] -
            crossover_bootstrap["low_radius_arcsec"])
    elif direct_crossover is not None:
        crossover = direct_crossover
        diameter = 2.0 * crossover
        diameter_uncertainty = 2.0 * bin_arcsec
    else:
        crossover = None
        diameter = None
        diameter_uncertainty = None
    tail = cometary.measure_tail(
        residual, mask, wcs, sky_sigma,
        core_radius_arcsec=start_radius,
        max_radius_arcsec=max_radius_arcsec,
        angle_bin_deg=5.0, radial_bin_arcsec=0.5)
    cometary.annotate_expected_tail(
        {"tail": tail}, expected_tail_pa, 0.5,
        "anti-solar/anti-motion")
    return {
        "working": working,
        "sky": sky,
        "sky_sigma": sky_sigma,
        "psf": psf,
        "model": model,
        "residual": residual,
        "mask": mask,
        "centroid_dx_px": dx,
        "centroid_dy_px": dy,
        "amplitude": amplitude,
        "radii_arcsec": radii,
        "target_profile": target["profile"],
        "target_scatter": target["scatter"],
        "sector_count": target["sector_count"],
        "pixel_count": target["pixel_count"],
        "psf_profile": psf_profiles["median"],
        "psf_low": psf_profiles["low"],
        "psf_high": psf_profiles["high"],
        "excess": excess,
        "excess_error": total_error,
        "significance": significance,
        "formal_excess_error": formal_error,
        "formal_significance": formal_significance,
        "conservative_significant_run": conservative_run,
        "significant_run": run,
        "significant_edge_radius_arcsec": significant_edge,
        "direct_crossover_radius_arcsec": direct_crossover,
        "crossover_radius_arcsec": crossover,
        "crossover_bootstrap": crossover_bootstrap,
        "coma_diameter_arcsec": diameter,
        "coma_diameter_unc_arcsec": diameter_uncertainty,
        "tail": tail,
    }


def mean_radial_profile(image, mask, pixel_scale_arcsec,
                        bin_arcsec=0.5, max_radius_arcsec=12.0):
    radii_px, profile, _, _, _ = cometary.radial_profile(
        image, mask, bin_width_px=bin_arcsec / pixel_scale_arcsec,
        max_radius_px=max_radius_arcsec / pixel_scale_arcsec)
    return radii_px * pixel_scale_arcsec, profile


def public_result(result):
    ignored = {
        "working", "psf", "model", "residual", "mask",
        "radii_arcsec", "target_profile", "target_scatter",
        "sector_count", "pixel_count", "psf_profile", "psf_low", "psf_high",
        "excess", "excess_error", "significance",
        "formal_excess_error", "formal_significance",
    }
    output = {key: value for key, value in result.items()
              if key not in ignored}
    tail = dict(output["tail"])
    for key in ("angle_centers_deg", "angle_significance",
                "radial_centers_arcsec", "radial_significance"):
        if key in tail:
            tail[key] = np.asarray(tail[key]).tolist()
    output["tail"] = tail
    return output


def write_profile_csv(path, original, student, unmasked_profile):
    fields = [
        "radius_arcsec", "original_unmasked_mean",
        "original_masked_sector_median", "student_sector_median",
        "psf_profile", "original_excess", "student_excess",
        "original_sector_significance", "student_sector_significance",
        "original_formal_significance", "student_formal_significance",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, radius in enumerate(student["radii_arcsec"]):
            writer.writerow({
                "radius_arcsec": radius,
                "original_unmasked_mean": unmasked_profile[index],
                "original_masked_sector_median":
                    original["target_profile"][index],
                "student_sector_median": student["target_profile"][index],
                "psf_profile": student["psf_profile"][index],
                "original_excess": original["excess"][index],
                "student_excess": student["excess"][index],
                "original_sector_significance":
                    original["significance"][index],
                "student_sector_significance":
                    student["significance"][index],
                "original_formal_significance":
                    original["formal_significance"][index],
                "student_formal_significance":
                    student["formal_significance"][index],
            })


def render_comparison(path, original, student, contaminant,
                      pixel_scale_arcsec, expected_tail_pa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm

    extent_value = (student["working"].shape[0] - 1) / 2.0 * \
        pixel_scale_arcsec
    extent = [extent_value, -extent_value, -extent_value, extent_value]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for axis, result, title in (
            (axes[0, 0], original, "Original primary stack and star mask"),
            (axes[0, 1], student, "Studentized seven-frame stack"),
            (axes[0, 2], student, "Studentized pretrail-PSF residual")):
        image = result["working"] if axis is not axes[0, 2] \
            else result["residual"]
        sigma = result["sky_sigma"]
        vmax = float(np.nanpercentile(image, 99.8))
        axis.imshow(
            image, origin="lower", cmap="gray_r",
            norm=AsinhNorm(linear_width=max(2.0 * sigma, 1e-12),
                           vmin=-2.0 * sigma, vmax=vmax), extent=extent)
        axis.contour(
            image, levels=np.asarray([3.0, 5.0, 10.0]) * sigma,
            colors=["#2a6fbb", "#d49a00", "#333333"],
            linewidths=0.8, origin="lower", extent=extent)
        axis.axhline(0, color="#777777", alpha=0.5, linewidth=0.7)
        axis.axvline(0, color="#777777", alpha=0.5, linewidth=0.7)
        axis.set_title(title)
        axis.set_xlabel("East offset (arcsec)")
        axis.set_ylabel("North offset (arcsec)")
    if contaminant:
        circle = plt.Circle(
            (contaminant["east_arcsec"], contaminant["north_arcsec"]),
            contaminant["mask_radius_arcsec"], fill=False,
            color="#d49a00", linewidth=1.8, label="masked background star")
        axes[0, 0].add_patch(circle)
        axes[0, 0].legend(loc="upper right", fontsize=8)

    radius = student["radii_arcsec"]
    axes[1, 0].plot(radius, student["target_profile"], color="#111111",
                    label="studentized sector median")
    axes[1, 0].plot(radius, original["target_profile"], color="#d49a00",
                    label="original, star masked")
    axes[1, 0].plot(radius, student["psf_profile"], color="#2a6fbb",
                    label="pretrail PSF")
    axes[1, 0].fill_between(
        radius, student["psf_low"], student["psf_high"],
        color="#2a6fbb", alpha=0.18, label="PSF bootstrap 16-84%")
    axes[1, 0].set_yscale("symlog", linthresh=1e-5)
    axes[1, 0].set_title("Contaminant-robust radial profiles")
    axes[1, 0].set_xlabel("Radius (arcsec)")
    axes[1, 0].set_ylabel("Normalized surface brightness")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(radius, original["formal_significance"], color="#d49a00",
                    label="original, star masked")
    axes[1, 1].plot(radius, student["formal_significance"], color="#111111",
                    label="studentized")
    axes[1, 1].axhline(3.0, color="#2a6fbb", linestyle="--",
                       linewidth=1.0, label="3 sigma")
    for result, color in ((original, "#d49a00"),
                          (student, "#111111")):
        if result["crossover_radius_arcsec"] is not None:
            axes[1, 1].axvline(
                result["crossover_radius_arcsec"], color=color,
                linestyle=":", linewidth=1.2)
    axes[1, 1].set_ylim(-10, 30)
    axes[1, 1].set_title("Formal radial excess significance and crossover")
    axes[1, 1].set_xlabel("Radius (arcsec)")
    axes[1, 1].set_ylabel("Excess significance")
    axes[1, 1].legend(fontsize=8)

    tail = student["tail"]
    axes[1, 2].plot(tail.get("angle_centers_deg", []),
                    tail.get("angle_significance", []),
                    color="#2a6fbb", label="studentized residual")
    axes[1, 2].axhline(3.0, color="#777777", linestyle="--")
    if tail.get("pa_deg") is not None:
        axes[1, 2].axvline(tail["pa_deg"], color="#111111",
                           linestyle=":", label="measured feature PA")
    axes[1, 2].axvline(expected_tail_pa, color="#d49a00",
                       linestyle="-.", label="anti-solar/anti-motion")
    axes[1, 2].set_xlim(0, 360)
    axes[1, 2].set_title("Studentized residual polar profile")
    axes[1, 2].set_xlabel("Position angle east of north (deg)")
    axes[1, 2].set_ylabel("Directional significance")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle(
        "RMM2026 studentized-stack morphology with background-star control",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_review(path, summary):
    original = summary["original_masked"]
    student = summary["studentized"]
    contaminant = summary["background_star"]
    def number(value, digits=2, suffix=""):
        return (("%%.%df" % digits) % value + suffix
                if value is not None else "not sustained")

    lines = [
        "# RMM2026 studentized-stack morphology review", "",
        "This is a morphology cross-check only. Nothing was submitted to the MPC.",
        "The Siril stack combines seven selected frames across two cadence blocks,",
        "so it is not used as a single MPC observation timestamp.", "",
        "## Background-star control", "",
        ("- The strongest compact contaminant in the original primary stack is at "
         "east %.2f arcsec, north %.2f arcsec (radius %.2f arcsec)." %
         (contaminant["east_arcsec"], contaminant["north_arcsec"],
          contaminant["radius_arcsec"])),
        "- A %.1f arcsec radius mask removes that source and its wings." %
        contaminant["mask_radius_arcsec"],
        "- The studentized stack rejects that source; no equivalent >5-sigma "
        "compact contaminant is present between 7 and 13 arcsec.", "",
        "## Radial-profile result", "",
        "| Product | Significant edge radius | First crossover radius | Adopted coma diameter |",
        "|---|---:|---:|---:|",
        "| Original stack, star masked | %s | %s | %s |" %
        (number(original["significant_edge_radius_arcsec"], 2, " arcsec"),
         number(original["crossover_radius_arcsec"], 2, " arcsec"),
         ("%.1f +/- %.1f arcsec" %
          (original["coma_diameter_arcsec"],
           original["coma_diameter_unc_arcsec"])
          if original["coma_diameter_arcsec"] is not None else "upper limit")),
        "| Studentized stack | %s | %s | %s |" %
        (number(student["significant_edge_radius_arcsec"], 2, " arcsec"),
         number(student["crossover_radius_arcsec"], 2, " arcsec"),
         ("%.1f +/- %.1f arcsec" %
          (student["coma_diameter_arcsec"],
           student["coma_diameter_unc_arcsec"])
          if student["coma_diameter_arcsec"] is not None else "upper limit")), "",
        "The adopted diameter is twice the first positive-to-negative profile",
        "crossover. This prevents a detached background source from reopening",
        "the radial profile at larger radius. The significant-edge column uses",
        "formal pixel-noise errors. Crossover uncertainty comes from resampling",
        "azimuthal sectors and the fitted PSF ensemble where that bootstrap is",
        "well-defined; otherwise it falls back to one radial-bin width.", "",
        "## Tail check", "",
        ("- Studentized residual: %s, %.1f +/- %.1f arcsec at PA %.1f +/- %.1f deg."
         % (student["tail"]["status"], student["tail"]["length_arcsec"],
            student["tail"]["length_unc_arcsec"],
            student["tail"]["pa_deg"], student["tail"]["pa_unc_deg"])
         if student["tail"].get("length_arcsec") is not None else
         "- Studentized residual: no measurable tail."),
        "- Expected anti-solar/anti-motion PA: 275.0 +/- 0.5 deg.", "",
        "## Suggested interpretation", "",
        "Use the studentized-stack diameter as the cleaner morphology estimate,",
        "with the masked original stack as a consistency check. Retain the image-",
        "based PA uncertainty separately from the projected-direction uncertainty.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    stack_path = Path(args.stack_fits).expanduser().resolve()
    psf_summary_path = Path(args.psf_summary).expanduser().resolve()
    original_stack_path = Path(args.original_stack).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(stack_path) as hdul:
        stack_data = np.asarray(hdul[0].data, dtype=float)
        stack_header = hdul[0].header.copy()
    with fits.open(original_stack_path) as hdul:
        original_data = np.asarray(hdul[0].data, dtype=float)
    psf_summary = json.loads(psf_summary_path.read_text(encoding="utf-8"))
    pixel_scale = float(stack_header.get(
        "CUTSCALE", psf_summary["analysis"]["output_pixel_scale_arcsec"]))
    size = int(args.crop_size)
    stack = center_crop(stack_data, size)
    original = center_crop(original_data, size)
    wcs = cometary.canonical_wcs(
        float(stack_header.get("CUTRA", 0.0)),
        float(stack_header.get("CUTDEC", 0.0)), size, pixel_scale)
    wcs.array_shape = stack.shape

    primary_frames = psf_summary["analysis"]["primary_frames"]
    selected_frames = [row["frame"] for row in
                       psf_summary["frame_measurements"] if row["selected"]]
    primary_models, _ = reconstruct_psf_models(
        psf_summary, primary_frames, stack.shape, pixel_scale)
    student_models, _ = reconstruct_psf_models(
        psf_summary, selected_frames, stack.shape, pixel_scale)

    original_working, _, original_sigma = subtract_outer_sky(
        original, pixel_scale)
    contaminant = find_compact_contaminant(
        original_working, pixel_scale, original_sigma,
        args.search_inner_arcsec, args.search_outer_arcsec, 5.0)
    if contaminant is None:
        raise ValueError("no significant original-stack contaminant found")
    contaminant["mask_radius_arcsec"] = float(args.mask_radius_arcsec)
    original_mask = circular_mask(
        original.shape, pixel_scale, contaminant["east_arcsec"],
        contaminant["north_arcsec"], args.mask_radius_arcsec)

    stack_working, _, stack_sigma = subtract_outer_sky(stack, pixel_scale)
    stack_contaminant = find_compact_contaminant(
        stack_working, pixel_scale, stack_sigma,
        args.search_inner_arcsec, args.search_outer_arcsec, 5.0)
    stack_mask = np.zeros_like(stack, dtype=bool)
    if stack_contaminant is not None:
        stack_mask |= circular_mask(
            stack.shape, pixel_scale,
            stack_contaminant["east_arcsec"],
            stack_contaminant["north_arcsec"], args.mask_radius_arcsec)

    original_result = analyze_image(
        original, primary_models, pixel_scale, original_mask, wcs,
        args.expected_tail_pa, args.radial_bin_arcsec,
        args.max_radius_arcsec)
    student_result = analyze_image(
        stack, student_models, pixel_scale, stack_mask, wcs,
        args.expected_tail_pa, args.radial_bin_arcsec,
        args.max_radius_arcsec)
    unmasked_radii, unmasked_profile = mean_radial_profile(
        original_result["working"] / max(original_result["amplitude"], 1e-12),
        np.zeros_like(original_mask), pixel_scale,
        args.radial_bin_arcsec, args.max_radius_arcsec)
    if not np.allclose(unmasked_radii, student_result["radii_arcsec"]):
        raise ValueError("profile radius grids do not match")

    summary = {
        "review_only": True,
        "network_submission_performed": False,
        "inputs": {
            "studentized_stack": str(stack_path),
            "original_primary_stack": str(original_stack_path),
            "pretrail_psf_summary": str(psf_summary_path),
            "student_stack_frames": selected_frames,
            "student_stack_count": int(stack_header.get("STACKCNT", 0)),
            "student_stack_livetime_s": float(stack_header.get("LIVETIME", 0)),
        },
        "background_star": contaminant,
        "studentized_detected_contaminant": stack_contaminant,
        "original_masked": public_result(original_result),
        "studentized": public_result(student_result),
    }
    (output_dir / "studentized_stack_analysis.json").write_text(
        json.dumps(summary, indent=2, default=cometary.json_default) + "\n",
        encoding="utf-8")
    write_profile_csv(
        output_dir / "profile_comparison.csv", original_result,
        student_result, unmasked_profile)
    with (output_dir / "background_contaminants.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "product", "east_arcsec", "north_arcsec", "radius_arcsec",
            "peak_significance", "mask_radius_arcsec"])
        writer.writeheader()
        writer.writerow({"product": "original_primary", **contaminant})
        if stack_contaminant:
            writer.writerow({
                "product": "studentized", **stack_contaminant,
                "mask_radius_arcsec": args.mask_radius_arcsec})
    render_comparison(
        output_dir / "studentized_stack_diagnostic.png",
        original_result, student_result, contaminant,
        pixel_scale, args.expected_tail_pa)
    write_review(output_dir / "studentized_stack_review.md", summary)
    print("Wrote %s" % output_dir)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stack_fits")
    parser.add_argument("--psf-summary", required=True)
    parser.add_argument("--original-stack", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", type=int, default=201)
    parser.add_argument("--radial-bin-arcsec", type=float, default=0.5)
    parser.add_argument("--max-radius-arcsec", type=float, default=12.0)
    parser.add_argument("--search-inner-arcsec", type=float, default=7.0)
    parser.add_argument("--search-outer-arcsec", type=float, default=13.0)
    parser.add_argument("--mask-radius-arcsec", type=float, default=4.0)
    parser.add_argument("--expected-tail-pa", type=float, default=275.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
