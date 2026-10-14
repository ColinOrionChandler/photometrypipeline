"""End-to-end pipeline run on the bundled VATT4K example frames.

Needs Source Extractor and SCAMP on PATH (or next to the Python executable)
and network access to the online catalogs, so it only runs with
``python -m pytest -m integration``.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "example_data" / "vatt4k"
REFERENCE = Path(__file__).resolve().parent / "data" / "vatt4k_reference.json"

pytestmark = [pytest.mark.integration, pytest.mark.network]


def read_photometry(path):
    """Return the accepted rows of a pp_distill photometry_*.dat file."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        rows.append(
            {
                "filename": fields[0],
                "julian_date": float(fields[1]),
                "mag": float(fields[2]),
                "sig": float(fields[3]),
                "ra_deg": float(fields[4]),
                "dec_deg": float(fields[5]),
            }
        )
    return rows


@pytest.fixture(scope="module")
def vatt4k_run(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("vatt4k")
    frames = sorted(EXAMPLE_DIR.glob("*.fits"))
    assert frames, "example_data/vatt4k is missing"
    for frame in frames:
        # PP edits FITS headers in place; never run on the tracked copies.
        shutil.copy(frame, workdir)
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "pp_run.py"), *[f.name for f in frames]],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return workdir


def test_vatt4k_pipeline_produces_target_photometry(vatt4k_run):
    outputs = sorted(vatt4k_run.glob("photometry_*.dat"))
    assert [p.name for p in outputs] == ["photometry_3552.dat"]
    rows = read_photometry(outputs[0])
    assert len(rows) == 2
    for row in rows:
        assert np.isfinite([row["mag"], row["sig"], row["ra_deg"], row["dec_deg"]]).all()
        assert 0 < row["sig"] < 0.5
    assert (vatt4k_run / "diagnostics.html").exists()


def test_vatt4k_pipeline_matches_reference(vatt4k_run):
    if not REFERENCE.exists():
        pytest.skip("reference photometry not captured yet (tests/data/vatt4k_reference.json)")
    reference = json.loads(REFERENCE.read_text())
    rows = read_photometry(vatt4k_run / "photometry_3552.dat")
    assert [r["filename"] for r in rows] == [r["filename"] for r in reference["rows"]]
    for row, ref in zip(rows, reference["rows"], strict=True):
        assert row["julian_date"] == pytest.approx(ref["julian_date"], abs=1e-6)
        assert row["mag"] == pytest.approx(ref["mag"], abs=reference["mag_tolerance"])
        dra = (row["ra_deg"] - ref["ra_deg"]) * np.cos(np.radians(ref["dec_deg"])) * 3600
        ddec = (row["dec_deg"] - ref["dec_deg"]) * 3600
        assert np.hypot(dra, ddec) < reference["position_tolerance_arcsec"]
