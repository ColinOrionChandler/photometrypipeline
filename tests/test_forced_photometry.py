import csv
import os
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["PHOTPIPEDIR"] = str(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pptool_forced_photometry as forced


def _wcs_header():
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 266.25
    header["CRVAL2"] = 4.75
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CDELT1"] = -1.0 / 3600.0
    header["CDELT2"] = 1.0 / 3600.0
    header["EXPTIME"] = 10.0
    header["FILTER"] = "I"
    header["MAGZP"] = 24.0
    header["MAGZPSIG"] = 0.2
    return header


def _world(header, x, y):
    wcs = WCS(header)
    ra, dec = wcs.all_pix2world([[x, y]], 0)[0]
    return float(ra), float(dec)


def _write_photometry_row(path, token, header, x=25, y=25):
    ra, dec = _world(header, x, y)
    path.write_text(
        "# header\n"
        " %s 2452435.0115075 99.0000 99.0000 %.8f %+.8f "
        "0.00 0.00 0.00 0.00 10.00 24.0000 0.2000 "
        "99.0000 99.0000 manual_zp I 0 CFHTCFH12K APER 1.00 "
        "99.0000 0.2000 99.0000 0.0100 0.0100 0.0141 "
        "0.0200 0.0200 0.0283 0.0224 0.0224 0.0316\n"
        "# footer\n" % (token, ra, dec))


def _write_reference_catalog(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "mag"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_forced_photometry_recovers_injected_fixed_position_source(tmp_path):
    header = _wcs_header()
    image = np.full((60, 60), 100.0, dtype=np.float32)
    image[25, 25] += 400.0
    for x, y in [(10, 10), (45, 12), (12, 45), (44, 44)]:
        image[y, x] += 1000.0
    fits.PrimaryHDU(data=image, header=header).writeto(tmp_path / "image.fits")
    photometry_path = tmp_path / "photometry_2025_MH348.dat"
    _write_photometry_row(photometry_path, "image.ldac", header)
    catalog_path = tmp_path / "refs.csv"
    _write_reference_catalog(catalog_path, [
        {"x": 10, "y": 10, "mag": 20.0},
        {"x": 45, "y": 12, "mag": 20.0},
        {"x": 12, "y": 45, "mag": 20.0},
        {"x": 44, "y": 44, "mag": 20.0},
    ])

    result = forced.build_recovery(
        photometry_path, "2025 MH348", output_dir=tmp_path / "recovery",
        aperture_radii=[1.1], reference_catalog=catalog_path,
        sky_inner_radius=3, sky_outer_radius=6, snr_threshold=5)
    row = result["results"][0]

    assert result["n_accepted_photometry"] == 1
    assert row["accepted_photometry"] is True
    assert row["calibration"]["catalog"]["usable"] is True
    assert np.isclose(row["calibration"]["zeropoint"], 25.0, atol=0.03)
    assert np.isclose(row["mag"], 20.99, atol=0.08)
    assert Path(row["diagnostic_png"]).exists()
    assert Path(result["csv_path"]).exists()
    assert Path(result["report_path"]).exists()


def test_forced_photometry_non_detection_returns_limit_not_magnitude(tmp_path):
    header = _wcs_header()
    image = np.full((40, 40), 100.0, dtype=np.float32)
    for x, y in [(8, 8), (30, 8), (8, 30)]:
        image[y, x] += 1000.0
    fits.PrimaryHDU(data=image, header=header).writeto(tmp_path / "image.fits")
    photometry_path = tmp_path / "photometry_2025_MH348.dat"
    _write_photometry_row(photometry_path, "image.ldac", header, x=20, y=20)
    catalog_path = tmp_path / "refs.csv"
    _write_reference_catalog(catalog_path, [
        {"x": 8, "y": 8, "mag": 20.0},
        {"x": 30, "y": 8, "mag": 20.0},
        {"x": 8, "y": 30, "mag": 20.0},
    ])

    result = forced.build_recovery(
        photometry_path, "2025 MH348", output_dir=tmp_path / "recovery",
        aperture_radii=[1.1], reference_catalog=catalog_path,
        sky_inner_radius=3, sky_outer_radius=6, snr_threshold=5)
    row = result["results"][0]

    assert result["n_accepted_photometry"] == 0
    assert row["mag"] is None
    assert row["limiting_mag_3sigma"] is not None


def test_catalog_zeropoint_rejects_outlier_reference_star(tmp_path):
    header = _wcs_header()
    image = np.full((50, 50), 100.0, dtype=np.float32)
    for x, y in [(8, 8), (40, 8), (8, 40), (40, 40)]:
        image[y, x] += 1000.0
    fits.PrimaryHDU(data=image, header=header).writeto(tmp_path / "image.fits")
    catalog_rows = [
        {"x": "8", "y": "8", "mag": "20.0"},
        {"x": "40", "y": "8", "mag": "20.0"},
        {"x": "8", "y": "40", "mag": "20.0"},
        {"x": "40", "y": "40", "mag": "16.0"},
    ]

    result = forced.calibrate_zeropoint_from_catalog(
        tmp_path / "image.fits", "I", 1.1, catalog_rows,
        sky_inner_radius=3, sky_outer_radius=6, min_stars=3)

    assert result["usable"] is True
    assert result["n_rejected"] == 1
    assert np.isclose(result["zeropoint"], 25.0, atol=0.03)
