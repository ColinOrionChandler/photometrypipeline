import os
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PHOTPIPEDIR", str(REPO_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _pp_conf
from catalog import catalog
import pp_calibrate
import pptool_catalina_lemmon60 as cl60


def test_catalina_lemmon60_telescope_config_maps_clear_to_v():
    obsparam = _pp_conf.telescope_parameters["CATALINALEM60"]

    assert obsparam["observatory_code"] == "G96"
    assert obsparam["filter_translations"]["NONE"] == "V"
    assert obsparam["filter_translations"]["clear"] == "V"
    assert (_pp_conf.instrument_identifiers["SN 110-106/165685-06"] ==
            "CATALINALEM60")


def test_header_zeropoints_are_applied_per_frame(tmp_path):
    fits_a = tmp_path / "a.fits"
    fits_b = tmp_path / "b.fits"
    fits.PrimaryHDU(header=fits.Header({"MAGZP": 28.5, "PHOTIRMS": 0.02})
                    ).writeto(fits_a)
    fits.PrimaryHDU(header=fits.Header({"MAGZP": 29.0, "PHOTIRMS": 0.03})
                    ).writeto(fits_b)

    zeropoints = pp_calibrate.manual_zeropoints_from_headers(
        [str(fits_a), str(fits_b)], magzp_keyword="MAGZP",
        magzp_sig_keyword="PHOTIRMS")

    cat_a = catalog(pp_calibrate.ldac_filename_for_fits(str(fits_a)))
    cat_a.add_fields(["MAG_APER", "MAGERR_APER"], [[1.0], [0.1]])
    cat_b = catalog(pp_calibrate.ldac_filename_for_fits(str(fits_b)))
    cat_b.add_fields(["MAG_APER", "MAGERR_APER"], [[1.0], [0.1]])

    pp_calibrate.apply_manual_zeropoints([cat_a, cat_b], "V", zeropoints)

    assert np.isclose(cat_a["Vmag"][0], 29.5)
    assert np.isclose(cat_b["Vmag"][0], 30.0)
    assert np.isclose(cat_a["e_Vmag"][0], np.hypot(0.1, 0.02))
    assert np.isclose(cat_b["e_Vmag"][0], np.hypot(0.1, 0.03))


def test_astrometric_sigmas_use_matched_sources_before_ref_join(monkeypatch,
                                                                tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pp_calibrate.conf, "save_caldata", False)

    ref_cat = catalog("ref")
    ref_cat.add_fields(
        ["ra_deg", "dec_deg", "Rmag", "e_Rmag", "ident"],
        [
            [359.99990, 0.00010, 0.00020, 359.99970, 0.00030, 359.99991],
            [0.00000, 0.00008, -0.00011, 0.00022, -0.00028, 0.00001],
            [18.0, 18.5, 19.0, 17.5, 19.5, 18.05],
            [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            [1, 2, 3, 4, 5, 6],
        ],
    )
    inst_cat = catalog("synthetic.ldac")
    inst_cat.add_fields(
        ["ra_deg", "dec_deg", "MAG_APER", "MAGERR_APER"],
        [
            [359.99991, 0.00011, 0.00019, 359.99969, 0.00031],
            [0.00001, 0.00009, -0.00010, 0.00021, -0.00027],
            [-10.0, -9.5, -9.0, -10.5, -8.5],
            [0.01, 0.01, 0.01, 0.01, 0.01],
        ],
    )

    pp_calibrate.derive_zeropoints(ref_cat, [inst_cat], "R", 3)

    assert np.nanmax(inst_cat["ASTR_SIG_POS"]) < 1.0
    assert np.nanmax(inst_cat["ASTR_SIG_RA"]) < 1.0


def test_convert_compressed_extension_to_primary_fits(tmp_path):
    raw_path = tmp_path / "raw_image.fz"
    working_path = tmp_path / "cl60_0001.fits"
    header = fits.Header()
    header["TELESCOP"] = "60-INCH TELESCOPE"
    header["INSTRUME"] = "SN 110-106/165685-06"
    header["FILTER"] = "NONE"
    header["MAGZP"] = 28.509
    header["PHOTIRMS"] = 0.019
    hdulist = fits.HDUList([
        fits.PrimaryHDU(),
        fits.CompImageHDU(data=np.arange(4).reshape(2, 2),
                          header=header),
    ])
    hdulist.writeto(raw_path)

    record = cl60.convert_to_pp_fits(raw_path, working_path, 1)

    with fits.open(working_path) as hdulist:
        assert len(hdulist) == 1
        assert hdulist[0].data.shape == (2, 2)
        assert hdulist[0].header["PPINSTRU"] == "CATALINALEM60"
        assert hdulist[0].header["ORIGFILE"] == raw_path.name
        assert hdulist[0].header["ORIGEXT"] == 1
        assert hdulist[0].header["MAGZP"] == 28.509
        assert hdulist[0].header["PHOTIRMS"] == 0.019

    assert record["working_name"] == "cl60_0001.fits"
    assert record["image_index"] == 0
