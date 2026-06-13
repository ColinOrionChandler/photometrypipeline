# CFHT CFH12K And PP Cutouts Workflow

This repository has first-class support for CFHT/CFH12K CADC products through
`pptool_cfht_cfh12k.py` and `CFHTCFH12K` in `setup/telescopes.py`.

## CFH12K Processing

Run the helper from the repository root:

```bash
python pptool_cfht_cfh12k.py /path/to/CFHT_CFH12K --target "2025 MH348"
```

The helper reads target rows from `*_locations.csv`, extracts the full detector
chip needed for each target observation, writes PP-ready FITS files into
`PP/<date>/<filter>/`, runs PP per date/filter group, combines the per-filter
photometry into root-level target photometry files, and generates MPC/ADES
sidecars plus Find_Orb-style input files.

CFH12K chip selection uses `ChipSliceNum` as the one-based FITS HDU index. When
cutouts are present, their chip labels are used as a shortcut and validation
source; otherwise, the helper checks image-HDU WCS coverage against the target
RA/Dec. Full chip extraction is performed with CFITSIO `imcopy` because this
dataset has compressed multi-extension files that Astropy can read headers from
but may fail to decompress fully.

## Astrometry-Only Rows

Rows with valid PP measured RA/Dec remain active astrometric observations even
when calibrated photometry is unusable. In PP photometry files, values such as
`mag=99`, `mag_sig=99`, or any magnitude field `>=90` mean "no usable
photometry" rather than "bad astrometry".

`pptool_mpcsubmission.py` keeps these rows when the source FITS resolves and
the measured RA/Dec are finite:

- ADES output leaves `photCat`, `mag`, `rmsMag`, and `band` blank.
- MPC 80-column output leaves the magnitude and band columns blank.
- The summary file reports total observations and astrometry-only observations.

Rows that PP already commented out remain excluded. Rows with missing/nonfinite
source RA/Dec, unresolved source FITS, or otherwise unusable astrometry are
skipped or rejected by the relevant output builder.

## PP Cutouts

`pptool_pp_cutouts.py` builds the standard PP cutout artifact set:

```bash
python pptool_pp_cutouts.py /path/to/PP --target "2025 MH348"
```

By default it writes to `/path/to/PP/PP_cutouts` and uses a `126 arcsec` cutout
size. It reads active PP photometry rows, resolves the source FITS through the
same filename logic used by MPC/ADES generation, and centers each cutout on the
PP measured `source_ra/source_dec` columns rather than the predicted ephemeris
position.

For each usable row it writes:

- one FITS cutout with WCS and provenance headers,
- one zscale PNG preview,
- one JSONL metadata row in `PP_cutouts/cutouts.jsonl`.

The JSONL metadata records source photometry and FITS paths, JD, filter,
SExtractor flag, residuals, measured RA/Dec, output paths, overlap/inside
status, and whether the row is `astrometry_only`.

The CFH12K, CFHT MegaCam, Pan-STARRS1, SDSS, WHT Prime, Spacewatch, and
Catalina/Lemmon helpers call the shared cutout generator after PP photometry is
available and record counts, paths, and warnings in their workflow manifests.

## Validation Checklist

For the 2025 MH348 CFH12K run, the expected successful artifact set is:

- `PP/2002-06-09/I` with 5 processed I-band exposures,
- `PP/2002-06-09/Z` with 5 processed Z-band exposures,
- root `photometry_2025_MH348.dat` and
  `photometry_2025_MH348_all_filters.dat`,
- `mpc_2025_MH348_ADES.psv`, `mpc_2025_MH348_80col.txt`, and
  `mpc_2025_MH348_summary.txt`,
- `find_orb_runs/2025_MH348/mpc_2025_MH348_cfh12k_only.obs80`,
- `PP_cutouts` containing 10 FITS cutouts, 10 PNG previews, and 10 JSONL rows.

If a local Find_Orb executable is unavailable, the workflow leaves prepared
input files and records the missing executable as a manifest warning.
