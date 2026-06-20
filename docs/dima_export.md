# Dima Lightcurve Export

`pptool_dima_export.py` converts PP combined photometry CSV files into the
Dima lightcurve CSV shape:

```text
designation,epoch,band,mag,rmsmag,flag,topodist,heliodist,phase_angle
```

By default, the exporter also fetches public MPC ADES photometry for the same
target and enriches all PP and MPC rows with JPL Horizons geometry. PP rows use
flag `1`; MPC rows use flag `0`.

## Basic Usage

Run from the repository root:

```bash
python pptool_dima_export.py \
  --target "1998 QJ1" \
  --input /path/to/photometry_1998_QJ1_combined.csv \
  --output /path/to/dima_format_obs.csv
```

The output directory receives:

- `dima_format_obs.csv`
- `mpc_<target>_observations.csv`
- `mpc_<target>_observations_raw.json`

Use `--no-include-mpc` to export only PP rows. Use `--strict-horizons` when a
missing or partial Horizons response should fail the run instead of leaving
geometry blank with a warning.

## Site Codes

PP rows use `observatory_code` or `site_code` from the input CSV when present.
If the CSV contains a single instrument, the exporter tries the local PP
telescope configuration and then built-in fallbacks. `WHTPFIP` falls back to
MPC observatory code `950`.

MPC rows use their ADES `stn` value. `--mpc-site-code-fallback` can supply a
site code for MPC rows that do not include one.

## Validation

The focused test suite is:

```bash
pytest tests/test_dima_export.py
```

One integration-style test uses the real 1998 QJ1 PP combined CSV when it is
available locally, and otherwise skips that case.
