WHT Prime CASU Workflow
=======================

The ``WHTPFIP`` telescope configuration supports WHT Prime Focus Imaging
Platform CASU products from Roque de los Muchachos (MPC code ``950``).
The workflow helper ``pptool_wht_prime.py`` stages target-containing
CASU reduced frames into a PP-style tree, normalizes the FITS headers,
writes one ``positions_<target>.dat`` file per date/filter group, and
can run PP on each group.

Data Layout
-----------

Keep large WHT processing trees outside the repository, for example in a
Dropbox object directory. The helper expects a CASU WHT Prime directory
containing ``cutouts.jsonl`` and a ``reduced/`` directory with
``*_reduced.fits`` files. The output tree is organized as::

  PP/<UTC date>/<filter>/

For each staged science frame, the helper stamps:

* ``PPINSTRU`` and ``TEL_KEYW`` as ``WHTPFIP``
* normalized ``TELESCOP`` and ``INSTRUME`` values
* ``OBJECT`` as the requested target name
* ISO ``DATE-OBS`` exposure start and ``MIDTIMJD`` exposure midpoint
* normalized ``FILTER`` values
* ``JPLRA`` and ``JPLDEC`` from the cutout metadata, when available

Filters
-------

WHT Prime ``FILTER`` values ``B``, ``V``, ``R``, and ``I`` are passed
through to PP. Historical or missing ``unknown`` filter values are treated
as ``R`` so older reduced products remain compatible.

Running With Preserved WCS
--------------------------

If the input frames already have good WCS, or if they were solved outside
PP with a local astrometry.net installation, rerun PP with preserved WCS
instead of asking PP/SCAMP to register the frames again. The helper exposes
this as ``--keep-wcs``; the lower-level PP option is ``-keep_wcs``.

A typical helper run is::

  python pptool_wht_prime.py /path/to/CASU/WHT/Prime \
      --target "1998 QJ1" \
      --output-root /path/to/dropbox/object/PP \
      --keep-wcs

For a manual or externally solved tree, run PP from each date/filter
directory with the matching positions file, for example::

  python /path/to/photometrypipeline/pp_run.py \
      -target "1998 QJ1" \
      -telescope WHTPFIP \
      -positions positions_1998_QJ1.dat \
      -keep_wcs \
      *_reduced.fits

This preserves the solved WCS cards and still lets PP extract sources,
derive photometric zeropoints, distill target photometry, and write the
diagnostic HTML for each group.

Submission Products
-------------------

After PP writes ``photometry_<target>.dat`` files, use
``pptool_mpcsubmission.py`` on the WHT PP output root with observatory
code ``950``. It writes paired ADES PSV and MPC1992 80-column sidecars
plus a summary file. If the object already has MPC observations, compare
the generated WHT source-frame start times against the existing MPC rows
and submit only genuinely new measurements.

For orbit-impact checks, keep the generated Find_Orb inputs and logs in a
side directory under the same Dropbox processing tree, not in the Git
checkout.
