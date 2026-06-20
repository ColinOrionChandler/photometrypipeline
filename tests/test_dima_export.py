import csv
import os
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["PHOTPIPEDIR"] = str(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pptool_dima_export as dima


class FakeTable:
    colnames = ["delta", "r", "alpha"]

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, column):
        return [row[column] for row in self.rows]


def test_pp_only_conversion_preserves_dima_header_and_flag(tmp_path):
    combined = tmp_path / "photometry_1998_QJ1_combined.csv"
    combined.write_text(
        "catalog_token,julian_date,mag,mag_sig,band,instrument\n"
        "wht1.ldac,2451102.3480556,19.8456,0.0343,R,WHTPFIP\n",
        encoding="utf-8",
    )

    rows = dima.pp_rows_to_dima(
        combined,
        target="1998 QJ1",
        pp_flag="1",
    )
    output = tmp_path / "dima.csv"
    dima.write_dima_csv(output, rows)

    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)

    assert reader.fieldnames == dima.DIMA_HEADERS
    assert written == [{
        "designation": "1998_QJ1",
        "epoch": "2451102.3480556",
        "band": "R",
        "mag": "19.8456",
        "rmsmag": "0.0343",
        "flag": "1",
        "topodist": "",
        "heliodist": "",
        "phase_angle": "",
    }]
    assert rows[0].site_code == "950"


def test_mpc_parsing_filters_photometry_and_saves_observations(tmp_path):
    payload = [{
        "ADES_DF": [
            {
                "obstime": "1998-10-15T20:21:12.384Z",
                "stn": "950",
                "mag": "19.90",
                "rmsmag": "0.10",
                "band": "R",
                "provid": "1998 QJ1",
                "obsid": "obs1",
            },
            {
                "obstime": "1998-10-15T20:22:00.000Z",
                "stn": "950",
                "mag": "",
                "band": "R",
                "obsid": "no-mag",
            },
            {
                "obstime": "1998-10-15T20:23:00.000Z",
                "stn": "950",
                "mag": "20.0",
                "band": None,
                "obsid": "no-band",
            },
        ],
    }]

    observations = dima.extract_mpc_ades_observations(payload)
    rows, warnings = dima.mpc_observations_to_dima_rows(
        observations,
        target="1998 QJ1",
        mpc_flag="0",
    )
    csv_path, json_path = dima.save_mpc_observations(
        tmp_path,
        "1998 QJ1",
        payload,
        observations,
    )

    assert warnings == []
    assert len(rows) == 1
    assert rows[0].designation == "1998_QJ1"
    assert rows[0].epoch_text == "2451102.34806"
    assert rows[0].mag == "19.90"
    assert rows[0].rmsmag == "0.10"
    assert rows[0].flag == "0"
    assert rows[0].site_code == "950"
    assert csv_path.exists()
    assert json_path.exists()
    assert len(list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))) == 3


def test_horizons_enrichment_aligns_by_input_order():
    rows = [
        dima.DimaRow(
            designation="1998_QJ1",
            epoch=2.0,
            epoch_text="2",
            band="R",
            mag="20.0",
            rmsmag="0.1",
            flag="1",
            site_code="950",
            source="pp",
            source_id="a",
        ),
        dima.DimaRow(
            designation="1998_QJ1",
            epoch=1.0,
            epoch_text="1",
            band="R",
            mag="20.1",
            rmsmag="",
            flag="0",
            site_code="950",
            source="mpc",
            source_id="b",
        ),
    ]
    calls = []

    def fake_query(target, site_code, epochs, id_type):
        calls.append((target, site_code, epochs, id_type))
        return FakeTable([
            {"delta": 1.1, "r": 2.1, "alpha": 3.1},
            {"delta": 1.2, "r": 2.2, "alpha": 3.2},
        ])

    warnings = dima.enrich_rows_with_horizons(
        rows,
        target="1998 QJ1",
        id_type="smallbody",
        query_func=fake_query,
    )

    assert warnings == []
    assert calls == [("1998 QJ1", "950", [1.0, 2.0], "smallbody")]
    assert rows[0].topodist == "1.2"
    assert rows[0].heliodist == "2.2"
    assert rows[0].phase_angle == "3.2"
    assert rows[1].topodist == "1.1"
    assert rows[1].heliodist == "2.1"
    assert rows[1].phase_angle == "3.1"


def test_export_with_mocked_mpc_and_horizons_uses_current_pp_shape(tmp_path, monkeypatch):
    combined = Path(
        "/Users/colinchandler/Colinchandler Dropbox/Colin Chandler/School/"
        "Research/Objects/Lost Small Bodies/objects/Centaurs/1998_QJ1/CASU/"
        "WHT/Prime/PP/photometry_1998_QJ1_combined.csv"
    )
    if not combined.exists():
        pytest.skip("real PP combined CSV is not available")

    raw_payload = {"mock": True}
    observations = [{
        "obstime": "1998-10-15T20:21:12.384Z",
        "stn": "950",
        "mag": "19.90",
        "band": "R",
        "obsid": "mpc1",
    }]
    monkeypatch.setattr(
        dima,
        "fetch_mpc_observations",
        lambda target: (raw_payload, observations),
    )
    monkeypatch.setattr(
        dima,
        "query_horizons_ephemerides",
        lambda target, site_code, epochs, id_type: FakeTable(
            {"delta": 1.0 + index, "r": 2.0 + index, "alpha": 3.0 + index}
            for index, _epoch in enumerate(epochs)
        ),
    )

    result = dima.export_dima(
        input_path=combined,
        output_path=tmp_path / "dima_format_obs.csv",
        target="1998 QJ1",
        include_mpc=True,
        pp_flag="1",
        mpc_flag="0",
        pp_site_code=None,
        mpc_site_code_fallback=None,
        horizons_id_type="smallbody",
        strict_horizons=True,
        horizons_batch_size=50,
    )

    output_rows = list(csv.DictReader(result.output_path.open(newline="", encoding="utf-8")))
    assert result.pp_rows == 5
    assert result.mpc_rows == 1
    assert result.total_rows == 6
    assert len(output_rows) == 6
    assert output_rows[0]["designation"] == "1998_QJ1"
