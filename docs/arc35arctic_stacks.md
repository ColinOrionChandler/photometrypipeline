# ARC 3.5-m/ARCTIC clicked pair stacks

`pptool_arc35arctic.py` turns approximate clicks on QA half-stacks into a
three-epoch pair-stack verification and, only for a verified moving source,
runs PP and writes review-only ADES products.

The click table must contain the usual
`image,x,y,width,height,status` columns. FITS counterparts may be beside the
CSV or one directory above it. The clicked FITS headers must contain
`INFILnnn` cards naming the original reduced exposures. The originals are
sorted by exposure midpoint and divided into adjacent groups; the default
group size is two.

```bash
python pptool_arc35arctic.py /path/to/stacking/click/click_points.csv \
  --target "2025 MK387" \
  --pair-size 2 \
  --stacker /path/to/coc_tools/fits/stack_by_wcs.py
```

The default output directory is `stacking/PP/<target>/`. It contains the true
pair stacks, original-file checksums, JPL Horizons/SBDB snapshots, annotated
candidate cutouts, a blink GIF, verification CSV/JSON, and a workflow
manifest. The source FITS, PNG, and click files are never modified.

Verification uses a PSF-weighted local search propagated from the clicked sky
positions. A source must have joint S/N of at least 5, positive flux in every
pair, at least two pair detections at S/N 3, a linear track tighter than one
stellar FWHM, a 3-sigma preference over a stationary source, and positive
evidence in both constituent exposures. A failed gate writes all QA products,
skips PP and ADES, and exits with status 2.

For a passing candidate, the pair working copies are Gaia-registered under
`ARC35ARCTIC`, `CU VR` is treated as approximate `r`, and PP uses a positions
sidecar derived from the refined WCS. ADES output uses MPC station 705,
observer `UNKNOWN` when the headers have no observer, and review-only mode.
No live submission script is created or executed.

