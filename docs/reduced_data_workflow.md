# Reduced-Directory Workflow

`pptool_reduced_data.py` turns a flat directory of reduced, single-image FITS
files into PP datasets grouped by `OBJNAME` and normalized photometric band.
It copies source files because PP updates working FITS headers in place.

```bash
python pptool_reduced_data.py /path/to/reduced/data \
  --output-dir /path/to/PP \
  --mask-quad lower-left
```

The output directory must be new or empty. Object-directory names replace
whitespace with underscores and slashes with plus signs. If an object has one
normalized band, its FITS files are placed directly in the object directory;
objects with multiple bands receive one subdirectory per band.

The workflow preflights every FITS file before writing output. It rejects
unknown instruments or filters, mixed dimensions or binning, name collisions,
and insufficient free space. TAI `DATE-OBS` values are converted to UTC in the
working copy while `DATEORIG` and `TSYSORIG` retain the source values. Raw
filter names are preserved in `ORIGFILT` before `FILTER` is normalized.

## Masks

`--mask-quad` creates a binary Source Extractor `MAP_WEIGHT` image with zeroes
in the selected quadrant and ones elsewhere. `--mask-file` accepts an existing
finite, nonnegative map with the same dimensions as the science data. Both
forms are run-scoped: they do not modify the instrument's default mask.

## Processing and provenance

PP runs independently for every object/band group. A failed group does not
prevent later groups from running, but the overall command exits nonzero if
any group lacks active target photometry. Each group retains `LOG`,
`pp_run_stdout.txt`, and `pp_run_stderr.txt`.

`reduced_data_workflow_manifest.json` records input and output paths, source
checksums before and after processing, header normalization, grouping, mask
geometry, commands, catalog mask validation, and per-group status.

Use `--prepare-only` to build and validate the tree without launching PP.
If a batch is interrupted or some groups fail, `--resume` reloads the existing
manifest and reruns only incomplete groups with the same staged files and mask.
Prior attempt metadata and logs remain available under attempt-numbered names.

## Compact cutout results

To publish the compact observing-night result layout after PP finishes, run:

```bash
python pptool_reduced_results.py PP/reduced_data_workflow_manifest.json \
  --results-dir results
```

This reuses PP's cached JPL Horizons response at each UTC exposure midpoint
(APO site code 705), failing closed rather than making a new network request,
and writes a 126-arcsec FITS cutout, a same-stem PNG made by
`coc_tools.fits.fits2png`, and a transparent `coc_tools` arrow PDF beneath
`results/<object>/ARCTIC/cutouts`. By default, detector pixels are preserved:
the crop is made in the native pixel grid and oriented approximately north-up
and east-left using only 90-degree rotations and/or flips. Band names are
included in filenames so multi-band products remain distinct. Detailed
provenance is retained beside the PP workflow manifest in
`reduced_results_manifest.json`, outside the compact results tree.
Use `--resume` to retain validated FITS/PNG/PDF sets and retry only incomplete
products after a transient Horizons or rendering failure. It also upgrades a
prior PNG/PDF-only results run by regenerating each product with its FITS
cutout.

Arbitrary-angle north-up/east-left reprojection is opt-in because it performs
bilinear resampling and can blur the data:

```bash
python pptool_reduced_results.py PP/reduced_data_workflow_manifest.json \
  --results-dir results --orientation-mode reproject
```
