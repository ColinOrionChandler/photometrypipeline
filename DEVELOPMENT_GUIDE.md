# photometrypipeline Development Guide

This document provides a guide for contributing to the development of this project, and can include information such as design principles, core abstractions, API conventions, development workflow, and best-practices.

## Design Goals and North Stars
**CRITICAL: Always keep these design principles in mind when making changes to the project.**

**Correctness is the Top Priority.** The code should include thorough tests. Any approximations should be clearly documented and communicated to the user.

**Code should be modular and use a few consistent APIs.**

**Make Easy Things Easy, Hard Things Possible.** Common use cases should require minimal configuration or set up, but users should also be able to perform complex work.

**Code should follow a standard format** with automatic formatting supported by `ruff`.

**Code should be documented.** Docstrings should provide sufficient API information, such as parameter explanations, for users to understand the code.



## Common Commands

```bash
# All tests
python -m pytest

# Parallel tests
python -m pytest -n auto

# Lint and format (let the linter fix style — do not hand-tune)
ruff check src/ tests/
ruff format src/ tests/
# Pre-commit (runs ruff lint/format, workflow/pyproject schema checks, pytest, etc.)
pre-commit run --all-files

# Build docs
sphinx-build -M html ./docs ./_readthedocs
```

## Repository Structure

```
src/photometrypipeline/             Main package
tests/photometrypipeline/           Test suite
docs/                             Sphinx documentation sources
```


## Testing Conventions

- **File naming:** `tests/photometrypipeline/<SUBDIR>/test_<name>.py`, mirroring the `src/photometrypipeline/` layout.
- **Fixtures:** defined in `tests/conftest.py`. Use existing fixtures, do not duplicate test data.
- Default test run: `python -m pytest`


## Project Specifics

**The repository is mid-migration to the template's `src/` layout.** Until that
finishes, the pipeline still lives in flat top-level modules: `_pp_conf.py`,
`toolbox.py`, `catalog.py`, `diagnostics.py`, the pipeline stages `pp_*.py`
and the fork workflow tools `pptool_*.py`. Telescope definitions and the
Source Extractor/SCAMP/SWarp configuration files live in `setup/`. pytest puts
the repository root on `sys.path` (`pythonpath = ["."]` in `pyproject.toml`),
and `tests/conftest.py` points `PHOTPIPEDIR` at this checkout.

**External programs.** The pipeline calls Source Extractor, SCAMP and SWarp.
`environment.yml` installs them from conda-forge. Unit tests must not need
them, so fake `subprocess` calls and `pp_extract._sextractor_cmd` with
`monkeypatch`. Tests that run the real binaries or query online catalogs are
marked `integration` (and `network`), are skipped by default, and run with
`python -m pytest -m integration`.

**Data files.** Never commit pipeline products (`PP/`, `LOG`, `*.ldac`,
diagnostics) or new large FITS files; the pre-commit hook rejects files over
500 kB. Test inputs should be synthesized in `tmp_path`.

**Lint baseline.** `pyproject.toml` carries a temporary per-file ruff baseline
for the legacy modules plus a `format.exclude` list. Entries may only be
removed, never added: new and touched code must pass the full rule set.

**Branches.** `main` is the mainline and is protected by the
`no-commit-to-branch` pre-commit hook; work on a feature branch and open a pull
request against `ColinOrionChandler/photometrypipeline` (not upstream).
