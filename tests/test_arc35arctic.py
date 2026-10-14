import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS


import pptool_arc35arctic as arc35


def test_png_to_fits_pixel_flips_y_and_scales():
    row = {"x": 10, "y": 20, "width": 101, "height": 101}
    x, y = arc35.png_to_fits_pixel(row, (201, 201))
    assert x == 20
    assert y == 160


def test_resolve_clicked_fits_checks_parent(tmp_path):
    click_dir = tmp_path / "click"
    click_dir.mkdir()
    csv_path = click_dir / "click_points.csv"
    csv_path.write_text("image,x,y,width,height,status\n")
    expected = tmp_path / "stack_1stHalf.fits"
    expected.write_bytes(b"fits")
    assert arc35.resolve_clicked_fits(csv_path, "stack_1stHalf.png") == expected


def simple_header(date_obs, exptime=100.0, crval=(10.0, 20.0)):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [40.0, 40.0]
    wcs.wcs.cdelt = [-0.23 / 3600.0, 0.23 / 3600.0]
    wcs.wcs.crval = list(crval)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["DATE-OBS"] = date_obs
    header["EXPTIME"] = exptime
    header["INSTRUME"] = "arctic"
    header["OBJNAME"] = "2025 MK387"
    header["FILTER"] = "CU VR"
    return header


def write_frame(path, date_obs, exptime=100.0, source=None, amplitude=100.0,
                noise=2.0, seed=1):
    rng = np.random.default_rng(seed)
    data = rng.normal(1000.0, noise, (80, 80))
    if source is not None:
        yy, xx = np.indices(data.shape)
        sigma = 4.0 / 2.354820045
        data += amplitude * np.exp(-0.5 * (
            (xx - source[0]) ** 2 + (yy - source[1]) ** 2) / sigma ** 2)
    fits.PrimaryHDU(data=data.astype(np.float32),
                    header=simple_header(date_obs, exptime)).writeto(path)
    return path


def test_infils_discovery_sorting_and_pairing(tmp_path):
    late = write_frame(tmp_path / "late.fits", "2026-07-16T05:00:00")
    early = write_frame(tmp_path / "early.fits", "2026-07-16T03:00:00")
    stack = tmp_path / "stack.fits"
    header = simple_header("2026-07-16T03:00:00")
    header["INFIL001"] = late.name
    header["INFIL002"] = early.name
    fits.PrimaryHDU(data=np.zeros((80, 80)), header=header).writeto(stack)
    seeds = [{"stack_fits": str(stack)}, {"stack_fits": str(stack)}]
    paths = arc35.discover_originals(seeds)
    assert paths == [early, late]
    assert arc35.group_adjacent(paths, 2) == [[early, late]]
    assert seeds[0]["midpoint_jd"] == seeds[1]["midpoint_jd"]


def test_stamp_pair_propagates_timing_metadata_and_provenance(tmp_path):
    first = write_frame(tmp_path / "first.fits", "2026-07-16T03:00:00",
                        exptime=100)
    second = write_frame(tmp_path / "second.fits", "2026-07-16T03:10:00",
                         exptime=300)
    pair = tmp_path / "pair.fits"
    fits.PrimaryHDU(data=np.zeros((80, 80)),
                    header=simple_header("2026-07-16T00:00:00")).writeto(pair)
    record = arc35.stamp_pair(pair, [first, second], "2025 MK387")
    header = fits.getheader(pair)
    expected = np.average([
        arc35.midpoint_jd(fits.getheader(first)),
        arc35.midpoint_jd(fits.getheader(second)),
    ], weights=[100, 300])
    assert np.isclose(header["MIDTIMJD"], expected)
    assert header["EXPTIME"] == 400
    assert header["TEL_KEYW"] == "ARC35ARCTIC"
    assert header["OBJNAME"] == "2025 MK387"
    assert header["INFIL001"] == "first.fits"
    assert header["INFIL002"] == "second.fits"
    assert record["exptime"] == 400


def synthetic_verification_case(tmp_path, positions, amplitude=100.0,
                                blank_constituent=False):
    records = []
    seeds = []
    base = Time("2026-07-16T03:00:00", format="isot", scale="utc").jd
    for index, (x, y) in enumerate(positions):
        date = Time(base + index / 24.0, format="jd").isot
        sources = []
        for constituent in range(2):
            source = tmp_path / ("source_%d_%d.fits" % (index, constituent))
            use_source = None if (blank_constituent and constituent == 1) else (x, y)
            write_frame(source, date, source=use_source, amplitude=amplitude,
                        seed=100 + index * 2 + constituent)
            sources.append(source)
        pair = tmp_path / ("pair_%d.fits" % index)
        write_frame(pair, date, exptime=200, source=(x, y), amplitude=amplitude,
                    noise=np.sqrt(2.0), seed=200 + index)
        records.append({"index": index + 1, "path": str(pair),
                        "png": str(pair.with_suffix(".png")),
                        "sources": [str(path) for path in sources],
                        "midpoint_jd": base + index / 24.0,
                        "midpoint_isot": date, "exptime": 200.0})
        world = WCS(fits.getheader(pair)).all_pix2world([[x, y]], 0)[0]
        seeds.append({"midpoint_jd": base + index / 24.0,
                      "ra_deg": float(world[0]), "dec_deg": float(world[1])})
    return records, [seeds[0], seeds[-1]]


def test_moving_psf_track_passes_verification(tmp_path):
    records, seeds = synthetic_verification_case(
        tmp_path, [(30, 40), (34, 40), (38, 40)])
    result = arc35.verify_candidate(records, seeds)
    assert result["verification_status"] == "verified"
    assert all(result["checks"].values())
    assert result["track"]["speed_arcsec_per_hour"] > 0.5


def test_stationary_source_fails_motion_gate(tmp_path):
    records, seeds = synthetic_verification_case(
        tmp_path, [(34, 40), (34, 40), (34, 40)])
    result = arc35.verify_candidate(records, seeds)
    assert result["verification_status"] == "not_verified"
    assert result["checks"]["nonstationary_3sigma"] is False


def test_missing_constituent_source_fails_cosmic_ray_gate(tmp_path):
    records, seeds = synthetic_verification_case(
        tmp_path, [(30, 40), (34, 40), (38, 40)], blank_constituent=True)
    result = arc35.verify_candidate(records, seeds)
    assert result["verification_status"] == "not_verified"
    assert result["checks"]["constituent_evidence"] is False


def test_marginal_track_fails_signal_gate(tmp_path):
    records, seeds = synthetic_verification_case(
        tmp_path, [(30, 40), (34, 40), (38, 40)], amplitude=1.0)
    result = arc35.verify_candidate(records, seeds)
    assert result["verification_status"] == "not_verified"
    assert not (result["checks"]["joint_snr"] and
                result["checks"]["two_pair_detections"])


def test_ades_input_masks_marginal_photometry_but_keeps_astrometry(tmp_path):
    path = tmp_path / "photometry_2025_MK387.dat"
    path.write_text(
        "# header\n"
        " good 2461237.1 23.1 0.20 10.0 20.0 0 0 0 0 300 25 0.03 "
        "0 0 PANSTARRS r 0 ARC35ARCTIC APER 1.0\n"
        " weak 2461237.2 23.4 0.50 10.1 20.1 0 0 0 0 300 25 0.03 "
        "0 0 PANSTARRS r 0 ARC35ARCTIC APER 1.0\n")
    output = arc35.write_ades_photometry_input(path, snr_min=3.0)
    rows = [line.split() for line in output.read_text().splitlines()
            if line and not line.startswith("#")]
    assert rows[0][2:4] == ["23.1", "0.20"]
    assert rows[1][2:4] == ["99.0000", "99.0000"]
    assert rows[1][4:6] == ["10.1", "20.1"]


def workflow_args(tmp_path):
    click = tmp_path / "click_points.csv"
    click.write_text("image,x,y,width,height,status\n")
    return argparse.Namespace(
        click_points_csv=str(click), target="2025 MK387", pair_size=2,
        stacker=str(tmp_path / "stacker.py"), output_dir=str(tmp_path / "out"),
        search_radius=4.0, fwhm=4.0, joint_snr_min=5.0,
        pair_snr_min=3.0, skip_remote=True)


def patch_workflow_dependencies(monkeypatch, tmp_path, status):
    source = tmp_path / "source.fits"
    source.write_bytes(b"source")
    verification = {"verification_status": status, "checks": {},
                    "measurements": []}
    monkeypatch.setattr(arc35, "read_click_seeds", lambda path: [{"seed": 1}])
    monkeypatch.setattr(arc35, "discover_originals", lambda seeds: [source])
    monkeypatch.setattr(arc35, "group_adjacent", lambda paths, size: [[source]])
    monkeypatch.setattr(
        arc35, "write_checksums",
        lambda root, csv, seeds, originals: (tmp_path / "checksums.json",
                                              {str(source): arc35.sha256(source)}))
    monkeypatch.setattr(arc35, "run_stacker", lambda *args: ["stacker"])
    monkeypatch.setattr(arc35, "collect_pair_records", lambda *args: [])
    monkeypatch.setattr(arc35, "verify_candidate", lambda *args, **kwargs: verification)
    monkeypatch.setattr(arc35, "write_verification",
                        lambda root, result: (tmp_path / "verify.json",
                                              tmp_path / "verify.csv"))
    monkeypatch.setattr(arc35, "render_verification",
                        lambda root, result: {"annotated_cutouts": [],
                                              "blink": None})
    return verification


def test_not_verified_manifest_skips_pp(monkeypatch, tmp_path):
    patch_workflow_dependencies(monkeypatch, tmp_path, "not_verified")
    monkeypatch.setattr(
        arc35, "run_pp_products",
        lambda *args: (_ for _ in ()).throw(AssertionError("PP must not run")))
    result = arc35.run_workflow(workflow_args(tmp_path))
    assert result["workflow_status"] == "not_verified"
    assert result["pp"] is None
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["verification"]["verification_status"] == "not_verified"
    assert manifest["originals_unchanged"] is True


def test_verified_manifest_runs_review_only_pp(monkeypatch, tmp_path):
    verification = patch_workflow_dependencies(monkeypatch, tmp_path, "verified")
    called = []
    monkeypatch.setattr(
        arc35, "run_pp_products",
        lambda root, target, value: called.append(value) or
        {"ades_review": "review.json", "submit_script": None})
    result = arc35.run_workflow(workflow_args(tmp_path))
    assert called == [verification]
    assert result["workflow_status"] == "complete"
    assert result["pp"]["submit_script"] is None
