
# photometrypipeline

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ColinOrionChandler/photometrypipeline/smoke-test.yml)](https://github.com/ColinOrionChandler/photometrypipeline/actions/workflows/smoke-test.yml)

The Photometry Pipeline (PP) is a Python package for automated photometric
analysis of imaging data from small to medium-sized observatories. It uses
[Source Extractor](https://www.astromatic.net/software/sextractor) and
[SCAMP](https://www.astromatic.net/software/scamp) to register and
photometrically calibrate images based on catalogs that are available online;
photometry is measured using Source Extractor aperture photometry. PP has been
designed for asteroid observations, but can be used with any kind of imaging
data.

This repository is Colin Orion Chandler's actively developed fork of
[mommermi/photometrypipeline](https://github.com/mommermi/photometrypipeline),
adding instrument workflows (`pptool_*`), MPC ADES submission tooling, and
cutout/cometary-activity products. The upstream documentation remains at
[photometrypipeline.readthedocs.io](https://photometrypipeline.readthedocs.io/);
see its [list of supported instruments and catalogs](https://photometrypipeline.readthedocs.io/en/latest/supported.html).

## Installation

PP needs Python 3.11 or newer and these external programs on your `PATH`:

* [Source Extractor](https://www.astromatic.net/software/sextractor) (`sex`, `sextractor` or `source-extractor`)
* [SCAMP](https://www.astromatic.net/software/scamp)
* [SWarp](https://www.astromatic.net/software/swarp) (only for `pp_combine`)

The simplest way to get all of them is conda-forge. From a clone of this
repository:

```
conda env create -f environment.yml
conda activate photometrypipeline
pip install -e '.[dev]'
```

Until the package layout migration is finished, PP still locates its
configuration through the `PHOTPIPEDIR` environment variable, which must point
at the repository root (see `doc/install.rst`).

## Development

Install the pre-commit hooks once per clone:

```
pre-commit install
```

and run the test suite with `python -m pytest`. Tests that need Source
Extractor, SCAMP and network catalog access are marked `integration` and are
skipped by default; run them with `python -m pytest -m integration`. See
`DEVELOPMENT_GUIDE.md` and the LINCC Frameworks
[template documentation](https://lincc-ppt.readthedocs.io/en/latest/) for the
conventions this repository follows.

## Acknowledgments

If you are using PP for your research, please acknowledge PP by citing

* Mommert 2017, PHOTOMETRYPIPELINE: An Automated Pipeline for Calibrated
  Photometry, [Astronomy & Computing](http://www.sciencedirect.com/science/article/pii/S2213133716300816), 18, 47.

PP was supported by NASA grants NNX15AE90G and NNX14AN82G and was developed in
the framework of the Mission Accessible Near-Earth Objects Survey
([MANOS](http://manosobs.wordpress.com/)).

## License

The Photometry Pipeline is distributed under the GNU General Public License,
version 3 or (at your option) any later version; see `LICENSE`.

Copyright (C) 2016-2022 Michael Mommert; fork modifications copyright (C)
2017-2026 Colin Orion Chandler.
