# NTT/EFOSC2 Temporary-Object Workflow

Photometry Pipeline supports EFOSC2 on the ESO 3.58-m NTT through the
`NTTEFOSC2` telescope key and the reusable `pptool_efosc2.py` driver.

The telescope configuration uses MPC station `809`, the native unbinned scale
of `0.1204` arcsec/pixel, ESO detector-binning keywords, `EFOSC`/`EFOSC2`
instrument detection, and the `r#784` to `r` filter translation. Astrometry is
registered against Gaia and photometric calibration tries Pan-STARRS first.

## Inputs

The workflow expects:

- four EFOSC2 science FITS files and their same-stem PNG previews;
- a `click_points.csv` table with `image,x,y,width,height,status` columns;
- an `initial_positions.txt` file containing seven timestamp, RA, and Dec rows.

Rows marked `clicked` are interpreted as zero-based PNG coordinates. EFOSC2
previews are vertically flipped relative to FITS, so the driver applies
`fits_y = height - 1 - png_y`. It searches within eight pixels of each click
and computes a positive-flux centroid in a three-pixel aperture.

The first three initial-position rows form the Rubin/X05 baseline. The final
four initial rows are retained only for coordinate comparison; the augmented
orbit fit uses the three baseline observations plus the four final PP
observations from station 809.

## Running

Run from the repository root in the Python environment that contains PP,
Source Extractor, SCAMP, and SWarp:

```bash
python pptool_efosc2.py /path/to/EFOSC2_PP/click_points.csv \
  --initial-positions /path/to/initial_positions.txt \
  --pp-dir /path/to/EFOSC2_PP \
  --target McLeod
```

`--find-orb` can select a different Project Pluto `fo` executable. The default
is `/Users/colinchandler/GitHub/find_orb/fo`.

## Processing and Preservation

Before processing, the driver creates `originals_backup.tar` containing the
input FITS, PNG, CSV, and initial-position files. It records SHA-256 checksums
before and after the run and fails if an original changes.

Only the primary science array is copied into the PP working files under
`r/`; additional unlabeled image arrays are ignored. Working headers record
the original filename, normalized telescope/filter/object metadata, exposure
midpoint, click pixels, centroid pixels, and target-coordinate provenance.

PP runs in two stages:

1. Prepare and register all four frames against Gaia.
2. Convert the stored centroid pixels through the refined WCS, write the target
   positions file, and run photometry, calibration, and distillation while
   preserving that WCS.

The workflow requires all four Gaia registrations and four active PP target
rows. Its JSON manifest records per-frame WCS shifts, centroid diagnostics,
source-match residuals, zeropoints, calibrated magnitudes, checksum results,
and acceptance flags.

## Orbit Products

Find_Orb runs in short `/private/tmp` workspaces and copies its complete output
back to:

```text
orbit_solver_runs/<target>/baseline/
orbit_solver_runs/<target>/augmented/
```

Each case retains its ADES PSV input, stdout, stderr, configuration, JSON and
text elements, residuals, and a parsed summary. The parent directory contains
JSON and Markdown comparisons. `preliminary_vs_pp_positions.csv` compares the
four excluded initial rows with the final PP positions using timestamp matches
and tangent-plane offsets.

## Review-Only ADES

Temporary objects use `trkSub` instead of `provID`. The EFOSC2 workflow creates
four station-809 rows in `mpc_<target>_ADES.psv` and XML, using observer
`UNKNOWN`, a blank `prog`, and calibrated `r` photometry when available.

The underlying submission helper also exposes this mode directly:

```bash
python pptool_mpcsubmission.py /path/to/photometry_McLeod.dat \
  --target McLeod \
  --trk-sub McLeod \
  --review-only \
  --observatory-code 809 \
  --observer UNKNOWN
```

Review-only mode skips automatic program-code lookup, does not require a
packed provisional designation, does not create 80-column output for a
temporary tracklet, and never creates a live MPC submission script. The
summary and validation JSON explicitly mark the artifacts as not
submission-ready.

## Main Artifacts

- `originals_backup.tar`
- `initial_positions_baseline.txt`
- `r/positions_<target>_r.dat`
- `photometry_<target>.dat`
- `preliminary_vs_pp_positions.csv`
- `orbit_solver_runs/<target>/`
- `mpc_<target>_ADES.psv` and `mpc_<target>_ADES.xml`
- `mpc_<target>_ADES_review.json`
- `efosc2_workflow_manifest.json`
