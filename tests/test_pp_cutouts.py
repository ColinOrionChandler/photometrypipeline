import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from pptool_pp_cutouts import (
    build_pp_cutouts,
    default_output_dir,
    load_positions,
)


def write_wcs_fits(path, ra_deg=144.17425351, dec_deg=13.91718331):
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = 41
    header["NAXIS2"] = 41
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = ra_deg
    header["CRVAL2"] = dec_deg
    header["CRPIX1"] = 21.0
    header["CRPIX2"] = 21.0
    header["CD1_1"] = -1.0 / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0 / 3600.0
    header["TELESCOP"] = "Synthetic"
    header["INSTRUME"] = "SyntheticCam"
    data = np.arange(41 * 41, dtype=np.float32).reshape(41, 41)
    fits.PrimaryHDU(data=data, header=header).writeto(path)


def pp_row(token, jd, mag, mag_sig, ra="144.17425351",
           dec="+13.91718331"):
    return (
        " {token} {jd:.7f} {mag} {mag_sig} {ra} {dec} -0.25 2.36 "
        "0.00 0.00 47.00 29.4913 0.0342 -8.1628 0.0808 "
        "SDSS-R9 r 0 SyntheticCam APER 0.87 0.0808 0.0342 "
        "0.0877 0.0121 0.0121 0.0171 0.0638 0.0816 0.1036 "
        "0.0650 0.0825 0.1050\n"
    ).format(token=token, jd=jd, mag=mag, mag_sig=mag_sig,
             ra=ra, dec=dec)


def test_builds_pp_cutouts_from_source_ra_dec_and_preserves_astrometry_only(
        tmp_path):
    write_wcs_fits(tmp_path / "src1.fits")
    write_wcs_fits(tmp_path / "src2.fits")
    photometry = tmp_path / "photometry_2025_MH348.dat"
    photometry.write_text(
        "# header\n" +
        pp_row("src1", 2457398.7899512, "21.3285", "0.0877") +
        pp_row("src2", 2457398.7909512, "99.0000", "99.0000",
               ra="144.17415351") +
        pp_row("missing", 2457398.7919512, "21.0000", "0.0500",
               ra="nan") +
        "# footer\n")

    result = build_pp_cutouts(tmp_path, "2025 MH348", size_arcsec=10.0)

    assert result["n_rows"] == 3
    assert result["n_cutouts"] == 2
    assert result["n_fits"] == 2
    assert result["n_png"] == 2
    assert result["n_astrometry_only"] == 1
    assert result["warnings"]

    output_dir = tmp_path / "PP_cutouts"
    assert len(list(output_dir.glob("*.fits"))) == 2
    assert len(list(output_dir.glob("*.png"))) == 2
    jsonl_path = output_dir / "cutouts.jsonl"
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["ra_deg"] == 144.17425351
    assert rows[0]["astrometry_only"] is False
    assert rows[0]["mag"] == 21.3285
    assert rows[1]["ra_deg"] == 144.17415351
    assert rows[1]["astrometry_only"] is True
    assert rows[1]["mag"] is None
    assert rows[1]["mag_sig"] is None

    with fits.open(rows[0]["output_fits"]) as hdulist:
        assert hdulist[0].data.shape == (11, 11)
        assert hdulist[0].header["CUTRA"] == rows[0]["ra_deg"]
        assert hdulist[0].header["CUTDEC"] == rows[0]["dec_deg"]
        assert hdulist[0].header["ASTONLY"] is False


def test_discovers_multiple_per_directory_photometry_files(tmp_path):
    first = tmp_path / "night1"
    second = tmp_path / "night2"
    first.mkdir()
    second.mkdir()
    write_wcs_fits(first / "src1.fits")
    write_wcs_fits(second / "src2.fits")
    (first / "photometry_2025_MH348.dat").write_text(
        "# header\n" +
        pp_row("src1", 2457398.7899512, "21.3285", "0.0877"))
    (second / "photometry_2025_MH348.dat").write_text(
        "# header\n" +
        pp_row("src2", 2457398.7909512, "22.1000", "0.1000"))

    result = build_pp_cutouts(tmp_path, "2025 MH348", size_arcsec=10.0)

    assert result["n_rows"] == 2
    assert result["n_cutouts"] == 2
    assert len(result["photometry_files"]) == 2


def test_position_source_selects_distinct_directory(tmp_path):
    write_wcs_fits(tmp_path / "src1.fits")
    (tmp_path / "photometry_2025_MH348.dat").write_text(
        "# header\n" + pp_row("src1", 2457398.7899512, "21.3285", "0.0877"))

    result = build_pp_cutouts(tmp_path, "2025 MH348", size_arcsec=10.0,
                              position_source="sbsar click")

    expected = tmp_path / "PP_cutouts_sbsar_click"
    assert default_output_dir(tmp_path, "sbsar click") == expected
    assert Path(result["output_dir"]) == expected
    assert result["position_source"] == "sbsar click"
    assert not (tmp_path / "PP_cutouts").exists()
    assert len(list(expected.glob("*_cutout.fits"))) == 1

    rows = [json.loads(line)
            for line in (expected / "cutouts.jsonl").read_text().splitlines()]
    assert rows[0]["position_source"] == "sbsar click"
    with fits.open(rows[0]["output_fits"]) as hdulist:
        assert hdulist[0].header["CUTPSRC"] == "sbsar click"


def test_positions_csv_centers_on_sbsar_centroid(tmp_path):
    write_wcs_fits(tmp_path / "src1.fits")
    click_ra, click_dec = 144.17500000, 13.91600000
    centroid_ra, centroid_dec = 144.17425351, 13.91718331
    positions = tmp_path / "click_observations.csv"
    positions.write_text(
        "image,fits_file,click_ra_deg,click_dec_deg,"
        "centroid_ra_deg,centroid_dec_deg,click_mjd_utc\n"
        "src1.png,{fits},{cra},{cdec},{tra},{tdec},57398.2899512\n".format(
            fits=tmp_path / "src1.fits", cra=click_ra, cdec=click_dec,
            tra=centroid_ra, tdec=centroid_dec),
        encoding="utf-8")

    # auto prefers the refined centroid over the raw click
    specs, ra_col = load_positions(positions)
    assert ra_col == "centroid_ra_deg"
    assert specs[0]["ra_deg"] == centroid_ra
    assert specs[0]["julian_date"] == 57398.2899512 + 2400000.5

    result = build_pp_cutouts(tmp_path, "2025 MH348", size_arcsec=10.0,
                              positions=positions,
                              position_source="sbsar_centroid")

    assert result["n_cutouts"] == 1
    assert result["positions_file"] == str(positions.resolve())
    rows = [json.loads(line) for line in
            (Path(result["output_dir"]) / "cutouts.jsonl").read_text()
            .splitlines()]
    assert rows[0]["ra_deg"] == centroid_ra
    assert rows[0]["astrometry_only"] is True
    with fits.open(rows[0]["output_fits"]) as hdulist:
        assert hdulist[0].header["CUTRA"] == centroid_ra

    # explicit click column overrides the auto centroid preference
    click_specs, click_col = load_positions(positions,
                                            position_column="click")
    assert click_col == "click_ra_deg"
    assert click_specs[0]["ra_deg"] == click_ra
