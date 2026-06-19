import json
from io import BytesIO
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

import pptool_reference_cutouts as ref


def _wcs_header(ra=301.0, dec=-15.0, n=40, pixscale=0.262):
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = n
    h["NAXIS2"] = n
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRVAL1"] = ra
    h["CRVAL2"] = dec
    h["CRPIX1"] = (n + 1) / 2.0
    h["CRPIX2"] = (n + 1) / 2.0
    h["CD1_1"] = -pixscale / 3600.0
    h["CD1_2"] = 0.0
    h["CD2_1"] = 0.0
    h["CD2_2"] = pixscale / 3600.0
    return h


def _fits_bytes(value=5.0, shape=(20, 20), header=None):
    buf = BytesIO()
    fits.PrimaryHDU(data=np.full(shape, value, dtype=np.float32),
                    header=header).writeto(buf)
    return buf.getvalue()


def test_url_builder_includes_bands_only_for_fits():
    jpg = ref.legacy_cutout_url(301.0, -15.0, "jpg", size_arcsec=126.0,
                                pixscale=0.262)
    fts = ref.legacy_cutout_url(301.0, -15.0, "fits", size_arcsec=126.0,
                                pixscale=0.262)
    assert "cutout.jpg?" in jpg and "bands=" not in jpg
    assert "cutout.fits?" in fts and "bands=grz" in fts
    assert "size=481" in jpg  # 126 / 0.262 rounded


def test_builds_reference_cutouts_from_positions_csv(tmp_path):
    positions = tmp_path / "report.csv"
    positions.write_text(
        "ra_deg,dec_deg,julian_date\n"
        "301.378238,-15.812082,2458637.75635\n"
        "301.282936,-15.839301,2458642.87600\n",
        encoding="utf-8")

    calls = []

    def fake_fetch(url, timeout=60):
        calls.append(url)
        return _fits_bytes() if "cutout.fits" in url else b"\xff\xd8jpgdata"

    result = ref.build_reference_cutouts(
        tmp_path, "2025 NN80", positions=positions,
        position_source="sbsar_centroid", fetcher=fake_fetch)

    out = tmp_path / "reference_cutouts_sbsar_centroid"
    assert Path_exists(out)
    assert result["output_dir"] == str(out)
    assert result["n_cutouts"] == 2
    assert result["n_no_coverage"] == 0          # mock FITS has nonzero data
    assert len(list(out.glob("*_ref.jpg"))) == 2
    assert len(list(out.glob("*_ref.fits"))) == 2
    rows = [json.loads(line)
            for line in (out / "reference_cutouts.jsonl").read_text().splitlines()]
    assert rows[0]["layer"] == "ls-dr10"
    assert rows[0]["has_coverage"] is True
    assert rows[0]["position_source"] == "sbsar_centroid"


def test_flags_missing_coverage(tmp_path):
    positions = tmp_path / "report.csv"
    positions.write_text("ra_deg,dec_deg\n10.0,80.0\n", encoding="utf-8")

    def fake_fetch(url, timeout=60):
        return _fits_bytes(value=0.0) if "cutout.fits" in url else b"jpg"

    result = ref.build_reference_cutouts(tmp_path, "2025 NN80",
                                         positions=positions, fetcher=fake_fetch)
    assert result["n_no_coverage"] == 1
    assert any("no Legacy Surveys coverage" in w for w in result["warnings"])


def test_builds_side_by_side_and_blink_gif(tmp_path):
    # source DECam-like frame with a bright point source at center
    src = np.full((40, 40), 100.0, dtype=np.float32)
    src[20, 20] = 5000.0
    src_path = tmp_path / "chip.fits"
    fits.PrimaryHDU(data=src, header=_wcs_header()).writeto(src_path)

    positions = tmp_path / "report.csv"
    positions.write_text(
        "ra_deg,dec_deg,fits_file\n301.0,-15.0,%s\n" % src_path,
        encoding="utf-8")

    # reference cutout: same WCS, flat (no source)
    ref_bytes = _fits_bytes(value=10.0, shape=(40, 40), header=_wcs_header())

    def fake_fetch(url, timeout=60):
        return ref_bytes if "cutout.fits" in url else b"jpgdata"

    result = ref.build_reference_cutouts(
        tmp_path, "2025 NN80", positions=positions, make_comparisons=True,
        blink_interval_ms=500, fetcher=fake_fetch)

    out = Path(result["output_dir"])
    assert len(list(out.glob("*_compare.png"))) == 1
    gifs = list(out.glob("*_blink.gif"))
    assert len(gifs) == 1
    with Image.open(gifs[0]) as gif:
        assert gif.n_frames == 2
        assert gif.info.get("duration") == 500
    row = json.loads((out / "reference_cutouts.jsonl").read_text().splitlines()[0])
    assert row["output_compare_png"] and row["output_blink_gif"]


def test_ps1_survey_uses_independent_backend_and_named_dir(tmp_path):
    positions = tmp_path / "report.csv"
    positions.write_text("ra_deg,dec_deg\n301.378,-15.812\n", encoding="utf-8")

    table = (
        "projcell subcell ra dec filter mjd type filename shortname\n"
        "1234 056 301.378 -15.812 g 55000 stack /rings/g.stk.fits g\n"
        "1234 056 301.378 -15.812 i 55000 stack /rings/i.stk.fits i\n"
        "1234 056 301.378 -15.812 z 55000 stack /rings/z.stk.fits z\n"
    )

    seen = []

    def fake_fetch(url, timeout=60):
        seen.append(url)
        if "ps1filenames" in url:
            return table.encode()
        if "format=fits" in url:
            return _fits_bytes(value=7.0)
        return b"jpgdata"

    result = ref.build_reference_cutouts(
        tmp_path, "2025 NN80", positions=positions, survey="ps1",
        position_source="sbsar_centroid", fetcher=fake_fetch)

    # PS1 gets its own survey-named subdir, distinct from the Legacy default
    assert Path(result["output_dir"]) == tmp_path / "reference_cutouts_ps1_sbsar_centroid"
    assert result["survey"] == "ps1"
    assert result["n_cutouts"] == 1
    assert any("ps1images.stsci.edu" in u for u in seen)   # independent service
    rows = [json.loads(line) for line in
            (Path(result["output_dir"]) / "reference_cutouts.jsonl")
            .read_text().splitlines()]
    assert rows[0]["survey"] == "ps1"
    assert "fitscut.cgi" in rows[0]["fits_url"] and "red=" in rows[0]["fits_url"]
    assert len(list(Path(result["output_dir"]).glob("*_ref.fits"))) == 1


def test_ps1_no_coverage_flagged(tmp_path):
    positions = tmp_path / "report.csv"
    positions.write_text("ra_deg,dec_deg\n10.0,80.0\n", encoding="utf-8")

    def fake_fetch(url, timeout=60):
        if "ps1filenames" in url:
            return b"projcell subcell ra dec filter mjd type filename shortname\n"
        return b"jpgdata"

    result = ref.build_reference_cutouts(tmp_path, "2025 NN80",
                                         positions=positions, survey="ps1",
                                         fetcher=fake_fetch)
    assert any("no coverage" in w for w in result["warnings"])


def Path_exists(p):
    return Path(p).exists()
