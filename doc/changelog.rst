Changelog
=========

Major changes to the pipeline since 2016-10-01 (see `Mommert 2017`_) are
documented here.

* 2026-06-25: ``pptool_mpcsubmission.py`` now writes an executable
  ``submit_mpc_<target>_ADES.sh`` sidecar next to each generated ADES XML
  file. The script targets the live MPC XML endpoint, prints the resolved
  ACK/AC2/object-type/program-code fields and curl command, requires typing
  ``submit`` unless ``--no-interaction`` is passed, and fails generation if
  the site code cannot be mapped to the MPC XML base-62 ``prog`` value.

* 2026-06-15: ``pptool_mpcsubmission.py`` now snapshots successful
  MPFit/BandK2000 ``*.obs80`` inputs next to generated ADES and
  80-column submission files, with a JSON manifest. New solver-specific
  runs should live under ``orbit_solver_runs/<target>``; older
  ``find_orb_runs/<target>`` layouts are still scanned for compatibility.

* 2026-06-03: added WHT Prime Focus Imaging Platform documentation for
  CASU reduced products under ``WHTPFIP``. The workflow stages
  target-containing frames into ``PP/<date>/<filter>``, stamps WHT/PFIP
  headers, writes per-group target position sidecars, and can preserve
  externally solved WCS with ``--keep-wcs``/``-keep_wcs``. WHT/PFIP now
  translates ``B``, ``V``, ``R``, and ``I`` filter keywords directly while
  retaining the legacy ``unknown`` to ``R`` fallback.

* 2026-06-01: added SDSS 2.5-m imaging support under ``SDSS`` and the
  ``pptool_sdss.py`` workflow wrapper. The helper stages confirmed
  target-containing full frames from CADC cutout names into
  ``PP/<DATE-OBS>/<filter>``, stamps SDSS provenance headers, keeps the
  archive WCS, and uses ``SDSS-R9`` for photometric calibration. MPC
  submission output now reports SDSS observations as
  ``TEL 2.5-m SDSS telescope + CCD``; the initial 2003 WW218 run
  produced SDSS-only ADES/80-column outputs plus current-MPC+SDSS
  Find_Orb comparison artifacts.

* 2026-05-31: added Spacewatch 0.9-m/PDS support under
  ``SPACEWATCH09`` and the ``pptool_spacewatch.py`` workflow wrapper.
  The helper stages full-frame archive FITS products into
  ``PP/<date>/<band>``, writes PP-ready working files and
  CATCH-position sidecars, keeps the archive WCS, maps
  ``Schott OG-515`` to an r-like PP band, and leaves Spacewatch header
  ``MAGZP`` values as provenance while PP derives zeropoints from
  catalog stars.

* 2026-05-26: added Catalina Lemmon 60-inch support under
  ``CATALINALEM60`` and the ``pptool_catalina_lemmon60.py`` workflow
  wrapper, including safe raw-file backup, compressed extension-FITS
  flattening, ``-keep_wcs`` processing, header zeropoints from
  ``MAGZP``/``PHOTIRMS``, and group-position fallback support. Added
  per-frame header zeropoints to ``pp_calibrate`` and ``pp_run`` via
  ``-magzp_keyword``, ``-magzp_sig_keyword``, and ``-magzp_sig``.
  Updated ``pptool_mpcsubmission.py`` defaults for Catalina Lemmon
  submission emails, measurers, ACK text, AC2 contacts, and fallback
  astrometric uncertainty handling.
  Added ``pptool_submit_ades_to_mpc.py`` for guarded ADES XML endpoint
  submissions with base-62 program-code lookup, fixed AC2 contacts, and
  the ``Small Body Search and Rescue`` ACK text.

* 2026-05-29: updated ``pptool_mpcsubmission.py`` FITS-row matching so
  MPC export works when PP truncates long ``catalog_token`` image names
  in ``photometry_<target>.dat`` rows (common for OMEGACam-style
  filenames). Added regression coverage in
  ``tests/test_mpcsubmission.py`` for truncated-token prefix matching.

* 2026-05-23: imported local telescope/configuration updates, including
  DECam support; ``pp_run`` now passes through ``-telescope``,
  ``-nodeblending``, ``-variable_stars``, ``-positions``,
  ``-fixedtargets``, and ``-offset``; LDAC source coordinates are
  recomputed from the FITS WCS when available, which is important for TPV
  headers used with ``-keep_wcs``; extraction also writes ``*.ldac.csv``
  inspection files; distilled photometry tables include magnitude and
  position uncertainty components and quadrature totals; added
  ``pptool_mpcsubmission.py`` to produce paired ADES PSV and MPC1992
  80-column submission files from distilled PP photometry, and updated
  DECam's MPC observatory code to ``W84``

* 2018-12-02: major overhaul of diagnostic output

* 2018-11-23: implementation of ``pp_setup.py``, which will eventually
  replace ``_pp_conf.py``; photometric catalog data used in the photometric
  calibration can now by output as ascii table and/or into final .db file;
  ``pillow`` has been replaced by ``skimage``; ``pandas`` is now a required
  Python module

* 2017-11-24: implementation of `-solar` option in ``pp_run`` and
  ``pp_calibrate`` to obtain photometric calibration only from stars
  with Sun-like colors

* 2017-10-20: implementation of ``pp_stackedphotometry``, providing
  automated image stacking and subsequent photometry

* 2017-10-04: implementation of ``pp_combine``, which enables
  automated image combination

* 2017-08-29: implementation of Gaia/TGAS as an alternative
  astrometric catalog for shallow widefield observations

* 2017-06-01: extraction of serendipitously observed targets
  (asteroids and variable stars) implemented in ``pp_distill``

* 2017-03-19: if no filtername is provided (``None``) or the
  ``-instrumental`` option of ``pp_calibrate`` is used, ``pp_run``
  will complete all pipeline tasks using these instrumental magnitudes

* 2017-03-16: implementation of `Pan-STARRS DR1`_ for the photometric
  calibration; currently, PP uses a home-built access of MAST at
  STScI, which is limited to catalog queries with a maximum cone
  radius of 0.5 deg; please note that this kind of query is rather
  slow compared to Vizier queries of SDSS or APASS

* 2017-02-24: ``pillow`` is now a required python module; the pipeline
  now supports default distortion parameters for wide-field cameras

* 2017-02-04: catalog.data is now an astropy.table, catalog downloads
  using astroquery.vizier (no effect to the user), pp now supports
  Gaia DR1 (CMC, USNOB1, and PPMX have been removed)


  
.. _Mommert 2017: http://adsabs.harvard.edu/abs/2017A%26C....18...47M
.. _Pan-STARRS DR1: http://panstarrs.stsci.edu/
