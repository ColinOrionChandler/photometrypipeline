"""Shared test setup for the legacy top-level pipeline modules.

The pipeline modules still read their configuration from ``PHOTPIPEDIR`` at
import time, so it is set here, before pytest imports any test module.
"""

import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Always point at this checkout so a PHOTPIPEDIR from the user's shell cannot
# leak another copy of the configuration into the tests.
os.environ["PHOTPIPEDIR"] = str(REPO_ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")
# pptool_sdss falls back to the macOS-only /private/tmp without TMPDIR.
os.environ.setdefault("TMPDIR", tempfile.gettempdir())
