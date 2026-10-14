# MPC cometary-activity measurements

`pptool_cometary_activity.py` builds a review package for the Minor Planet
Center cometary-activity form. It measures radial excess relative to a stellar
PSF, searches PSF-subtracted residuals for directional tail structure, and
writes a JSON object with the form's field names. It never sends a request to
the MPC.

The stellar comparison is motion-aware and has two tracking modes:

- `empirical-motion` is the original/default mode for sidereally tracked
  images. It convolves the empirical stellar PSF with the target's predicted
  motion.
- `pretrail-target-tracked` is for exposures tracked on the moving target.
  It fits an elliptical Moffat seeing PSF convolved with the known
  non-sidereal field-star trail. The recovered pretrail PSF—not the already
  trailed stellar image—is compared with the target, avoiding a second,
  artificial motion convolution. Bootstrap fits to the comparison stars
  provide the PSF-profile envelope.

Within each cadence block, the tool also builds a leave-one-out stationary-sky
template. Predicted target locations are masked from each input to prevent
self-subtraction. Frames whose template fails coverage/noise QA use the direct
image and are marked accordingly.

## RMM2026 example

Run from the photometrypipeline checkout with the COC Python environment:

```bash
/Users/colinchandler/opt/anaconda3/envs/COC/bin/python \
  pptool_cometary_activity.py \
  "/path/to/20260717UT_APO_McLeod/PP" \
  --target McLeod \
  --designation RMM2026 \
  --positions "/path/to/PP/object_centered_cutouts/cutout_positions.csv" \
  --ephemeris "/path/to/PP/positions_McLeod_augmented.dat" \
  --selection "/path/to/PP/apo_observation_selection.csv" \
  --observer "C. O. Chandler" \
  --email coc123@uw.edu \
  --observatory-code 705 \
  --aperture-m 3.5 \
  --primary-frames 0146,0147,0148,0149,0150,0151 \
  --secondary-frame 0172 \
  --psf-mode pretrail-target-tracked \
  --expected-tail-pa 275 \
  --expected-tail-pa-unc 0.5 \
  --expected-tail-label "anti-solar/anti-motion"
```

The default output directory is `cometary_activity_<designation>/` below the
PP root. Target-tracked pretrail runs automatically use
`cometary_activity_<designation>_pretrail_psf/`, so they do not replace a
previous empirical-motion package. Each package contains:

- `frame_measurements.csv`: per-frame PSF, subtraction, blend, coma, and tail
  measurements.
- `cometary_activity_summary.json`: rich provenance, QA, measurements, and
  artifact paths.
- `mpc_cometary_payload.json`: flat form-compatible values for manual review.
- `mpc_cometary_review.md`: a readable evidence table and warnings.
- `primary_stack.fits` and `primary_stack.png`: the target-aligned primary
  stack.
- `diagnostics/`: six-panel plots for every frame and the primary stack.
- `psf_fits/`: target-tracked mode diagnostics showing the trailed empirical
  stellar PSF, fitted trailed model, recovered pretrail PSF, and residuals.

## Interpretation

Coma diameter is not FWHM. It is twice the outer radius of a contiguous radial
excess detected at at least 3 sigma in two neighboring radial bins after the
tracking-appropriate target PSF is removed. The uncertainty reflects the comparison
star profile envelope and radial binning. When no excess is resolved, the
summary records an upper limit and leaves the form's numeric coma diameter
blank.

Tail PA is measured east of celestial north using each stamp's WCS. The tail
length is the outermost supported radial bin in the strongest directional
residual. A low-significance but measurable feature is retained as `marginal`
in the evidence package, while invalid subtraction/profile results remain
blank rather than being guessed. Broad-tail lower and upper PA bounds are
written into the two MPC PA fields. An optional expected physical PA is
reported as independent context; its uncertainty is not substituted for the
image-based PA uncertainty.

The prepared payload is review-gated. Inspect the primary-stack diagnostic,
the individual-frame support, blend flags, motion alignment, and any timestamp
warnings before copying values into the MPC form.

## Studentized-stack cross-check

`pptool_cometary_student_stack.py` compares a separately prepared
studentized stack with the original primary stack without replacing either
analysis package. It identifies and masks compact contaminants in the original
stack, reconstructs the appropriate pretrail PSF ensemble from the activity
summary, and measures the first positive-to-negative radial-profile crossover.

```bash
/Users/colinchandler/opt/anaconda3/envs/COC/bin/python \
  pptool_cometary_student_stack.py \
  "/path/to/McLeod-LSST_stacked_student.fit" \
  --psf-summary "/path/to/cometary_activity_RMM2026_pretrail_psf/cometary_activity_summary.json" \
  --original-stack "/path/to/cometary_activity_RMM2026_pretrail_psf/primary_stack.fits" \
  --output-dir "/path/to/cometary_activity_RMM2026_studentized_stack"
```

This product is a morphology cross-check rather than an MPC observation if the
input stack spans more than one cadence block. Its review sheet reports that
limitation explicitly and no network submission code is present.

## Fink alert-cutout analysis

`pptool_fink_cometary_activity.py` is the small-stamp counterpart for Fink
alert bundles. It does not require comparison stars: each difference/science
FITS cutout supplies a normalized `PSFIMAGE` and an `UNCERT` image labeled
`VarianceUncertainty`. The tool fits that PSF with a free subpixel offset and
local background plane, measures variance-weighted radial residuals, and
cross-checks the result against the alert's aperture/PSF fluxes, second
moments, extendedness, template flux, and quality flags.

```bash
/Users/colinchandler/opt/anaconda3/envs/COC/bin/python \
  pptool_fink_cometary_activity.py \
  "/path/to/alerts/fink" \
  --output-dir "/path/to/fink_activity_results"
```

When a `queries.csv` file is present two levels above the alert root, it is
used for a same-night motion-smear check. The default 30-second exposure can be
changed with `--exposure-s`. Outputs include per-alert catalog metrics,
per-cutout PSF-fit measurements, a time-series figure, per-alert diagnostics,
a JSON provenance summary, and a technical Markdown report.

The diagnostic plots overlay Gaussian-smoothed, variance-propagated
significance contours on both the target and the PSF residual. Solid cyan
contours mark positive +3, +5, +10, and +20 sigma structure; dashed magenta
contours mark the corresponding negative residuals when present. A gold arrow
shows the signed first angular harmonic (`m=1`) of the outer residual, and the
adjacent panel reports its formal significance and sky PA east of north. This
is an asymmetry diagnostic, not a validated tail PA. White north and east
compass arrows on the target and residual panels are computed from each
cutout's celestial WCS rather than assuming an image-axis orientation.
Native alert dimensions are allowed to vary. The analysis always uses the full
native arrays, while diagnostic image panels use a fixed 6-arcsec field
centered on the alert WCS position. Larger cutouts are cropped for display
only; smaller or incomplete cutouts show uncovered border rather than being
rescaled to fill the panel.

These small-stamp results are activity proxies, not MPC-ready coma diameters.
Their formal pixel significances do not include PSF-model uncertainty or all
resampling covariance, and no observing-geometry correction is applied.
