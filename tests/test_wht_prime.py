import json
import os
from pathlib import Path
import sys

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["PHOTPIPEDIR"] = str(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _pp_conf
import pptool_wht_prime as wht


def write_wht_reduced(path):
    header = fits.Header()
    header["TELESCOP"] = "WHT"
    header["INSTRUME"] = "PRIME IMAGING"
    header["OBJECT"] = "1998QJ1 R"
    header["DATE-OBS"] = "15/10/98"
    header["MJD-OBS"] = 51101.847708333335
    header["RA"] = "20:54:13.31"
    header["DEC"] = "-00:55:17.0"
    header["EXPTIME"] = 60.0
    header["AIRMASS"] = 1.15
    header["CCDXBIN"] = 1
    header["CCDYBIN"] = 1
    fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.float32),
                    header=header).writeto(path)


def write_cutouts_jsonl(path):
    rows = [
        {
            "kind": "source",
            "inside": True,
            "input_path": "reduced/source_reduced_wcs.fits",
        },
        {
            "kind": "target",
            "inside": True,
            "input_path": "reduced/wht19981015_00268029_reduced_wcs.fits",
            "output_path": "cutouts/target_cutout.fits",
            "source_filename": "wht19981015_00268029.fits.fz",
            "obs_time": "1998-10-15T20:20:42.000",
            "mjd": 51101.847708333335,
            "exposure": 60.0,
            "filter_name": "unknown",
            "ra_deg": 313.551038636,
            "dec_deg": -0.92545099,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_wht_telescope_config_is_registered():
    obsparam = _pp_conf.telescope_parameters["WHTPFIP"]

    assert obsparam["observatory_code"] == "950"
    assert obsparam["secpix"] == (0.421, 0.421)
    assert obsparam["date_keyword"] == "DATE-OBS"
    assert obsparam["filter_translations"]["B"] == "B"
    assert obsparam["filter_translations"]["V"] == "V"
    assert obsparam["filter_translations"]["R"] == "R"
    assert obsparam["filter_translations"]["I"] == "I"
    assert _pp_conf.instrument_identifiers["PRIME IMAGING"] == "WHTPFIP"


def test_prepare_pp_tree_copies_and_stamps_wht_headers(tmp_path):
    reduced_dir = tmp_path / "reduced"
    reduced_dir.mkdir()
    write_wht_reduced(reduced_dir / "wht19981015_00268029_reduced.fits")
    write_cutouts_jsonl(tmp_path / "cutouts.jsonl")

    manifest = wht.prepare_pp_tree(tmp_path, tmp_path / "PP", "1998 QJ1",
                                   overwrite=True)

    record = manifest["records"][0]
    working = Path(record["working"])
    assert working == (tmp_path / "PP" / "1998-10-15" / "R" /
                       "wht19981015_00268029_reduced.fits")
    assert record["filter"] == "R"
    expected_start = Time(51101.847708333335, format="mjd", scale="utc")
    expected_mid = expected_start + 30.0 * u.second
    assert np.isclose(record["midtimjd"], expected_mid.jd)

    with fits.open(working) as hdulist:
        header = hdulist[0].header
        assert header["PPINSTRU"] == "WHTPFIP"
        assert header["TEL_KEYW"] == "WHTPFIP"
        assert header["TELESCOP"] == "4.2-m William Herschel Telescope"
        assert header["INSTRUME"] == "WHT Prime Focus Imaging Platform"
        assert header["OBJECT"] == "1998 QJ1"
        assert header["DATE-OBS"].startswith("1998-10-15T20:20:42")
        assert np.isclose(header["MIDTIMJD"], expected_mid.jd)
        assert header["EQUINOX"] == 2000.0
        assert header["FILTER"] == "R"
        assert header["JPLRA"] == 313.551038636
        assert header["JPLDEC"] == -0.92545099


def test_write_positions_file_uses_jpl_position_and_midtime(tmp_path):
    records = [{
        "working_name": "wht19981015_00268029_reduced.fits",
        "midtimjd": 2451102.34805589,
        "target_position": {
            "ra_deg": 313.551038636,
            "dec_deg": -0.92545099,
        },
    }]

    path = wht.write_positions_file(tmp_path, records, "1998 QJ1")

    assert path.name == "positions_1998_QJ1.dat"
    assert path.read_text().strip() == (
        "wht19981015_00268029_reduced.fits 313.5510386360 "
        "-0.9254509900 2451102.34805589 1998_QJ1")


def test_apply_scamp_head_preserves_numeric_wcs_cards(tmp_path):
    fits_path = tmp_path / "wht19981015_00268029_reduced.fits"
    write_wht_reduced(fits_path)
    with fits.open(fits_path, mode="update") as hdulist:
        hdulist[0].header["CRVAL1"] = ("313.555", "fake string WCS")
        hdulist[0].header["A_ORDER"] = 2

    head = fits.Header()
    head["EQUINOX"] = 2000.0
    head["RADESYS"] = "ICRS"
    head["CTYPE1"] = "RA---TAN"
    head["CTYPE2"] = "DEC--TAN"
    head["CUNIT1"] = "deg"
    head["CUNIT2"] = "deg"
    head["CRVAL1"] = 313.5557863567
    head["CRVAL2"] = -0.9179208556
    head["CRPIX1"] = 512.0
    head["CRPIX2"] = 512.0
    head["CD1_1"] = 0.0001169444444444
    head["CD1_2"] = 0.0
    head["CD2_1"] = 0.0
    head["CD2_2"] = 0.0001169444444444
    head["PV1_0"] = 0.0
    head_path = tmp_path / "wht19981015_00268029_reduced.head"
    head_path.write_text(head.tostring(sep="\n", endcard=True,
                                       padding=False) + "\n")

    wht.apply_scamp_head_file(fits_path, head_path)

    header = fits.getheader(fits_path)
    assert isinstance(header["CRVAL1"], float)
    assert header["CRVAL1"] == 313.5557863567
    assert header["RADESYS"] == "ICRS"
    assert header["RADECSYS"] == "ICRS"
    assert header["PV1_0"] == 0.0
    assert "A_ORDER" not in header
    assert header["REGCAT"] == "GAIA"


def test_remake_cutouts_from_manifest_preserves_existing_name(tmp_path):
    source_dir = tmp_path / "PP_local_astrometry" / "1998-10-15" / "R"
    source_dir.mkdir(parents=True)
    source = source_dir / "wht19981015_00268029_reduced.fits"

    header = fits.Header()
    header["DATE-OBS"] = "1998-10-15T20:20:42.000"
    header["EXPTIME"] = 60.0
    header["FILTER"] = "R"
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 21.0
    header["CRPIX2"] = 21.0
    header["CRVAL1"] = 313.55111
    header["CRVAL2"] = -0.92540
    header["CD1_1"] = -1.0 / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0 / 3600.0
    fits.PrimaryHDU(data=np.arange(41 * 41, dtype=np.float32).reshape(41, 41),
                    header=header).writeto(source)

    cutouts_dir = tmp_path / "cutouts"
    cutouts_dir.mkdir()
    (tmp_path / "cutouts.jsonl").write_text(json.dumps({
        "kind": "target",
        "source_filename": "wht19981015_00268029.fits.fz",
        "output_path": "cutouts/existing_name.fits",
    }) + "\n")

    manifest = {
        "target": "1998 QJ1",
        "records": [{
            "date_obs": "1998-10-15T20:20:42.000",
            "eph": {
                "ra_deg": 313.55111,
                "dec_deg": -0.92540,
                "v_mag": 20.4,
            },
            "exptime": 60.0,
            "filter": "R",
            "midtime_jd": 2451102.3480555555,
            "working": str(source),
            "working_dir": str(source_dir),
            "working_name": source.name,
        }],
    }
    manifest_path = tmp_path / "wht_local_astrometry_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    result = wht.remake_cutouts_from_manifest(
        tmp_path, manifest_path, size_arcsec=10.0)

    output_fits = cutouts_dir / "existing_name.fits"
    output_png = output_fits.with_suffix(".png")
    assert result["n_records"] == 1
    assert result["n_inside"] == 1
    assert output_fits.exists()
    assert output_png.exists()

    with fits.open(output_fits) as hdulist:
        header = hdulist[0].header
        assert hdulist[0].data.shape == (11, 11)
        assert header["CUTSRC"] == source.name
        assert header["CUTIN"]
        assert np.isclose(header["CUTRA"], 313.55111)

    rows = [
        json.loads(line) for line in
        (tmp_path / "cutouts.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["output_path"] == "cutouts/existing_name.fits"
    assert rows[0]["filter_name"] == "R"
