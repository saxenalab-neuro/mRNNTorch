"""JSON configuration validation and network construction tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mrnntorch.mrnn.leaky_mrnn import mRNN


FIXTURES = Path(__file__).parent / "fixtures"


def test_valid_config_builds_network_and_applies_defaults():
    model = mRNN(config=str(FIXTURES / "config_valid.json"), device="cpu")

    assert list(model.region_dict) == ["r1", "r2"]
    assert list(model.inp_dict) == ["i1"]
    assert model.total_num_units == 5
    assert model.total_num_inputs == 4
    assert model.get_region_indices("r2") == (2, 5)
    assert model.region_dict["r1"].sign == "pos"
    assert model.region_dict["r2"].sign == "neg"
    assert model.region_dict["r1"]["r2"].weight_mask.shape == (3, 2)
    assert model.inp_dict["i1"]["r1"].weight_mask.shape == (2, 4)
    assert model.W_rec.shape == (5, 5)
    assert model.W_inp.shape == (5, 4)


def test_invalid_config_reports_all_field_errors():
    with pytest.raises(ValidationError) as exc_info:
        mRNN(config=str(FIXTURES / "config_invalid.json"), device="cpu")

    errors = {error["loc"]: error["type"] for error in exc_info.value.errors()}
    assert errors[("recurrent_regions", 0, "num_units")] == "int_type"
    assert errors[("recurrent_connections", 0, "sparsity")] == "less_than_equal"


@pytest.mark.parametrize(
    ("config", "location", "error_type"),
    [
        ({"recurrent_regions": [{"num_units": 2}]}, ("recurrent_regions", 0, "name"), "missing"),
        ({"input_regions": [{"name": "i", "num_units": 0}]}, ("input_regions", 0, "num_units"), "greater_than"),
        ({"input_connections": [{"src_region": "i", "dst_region": "r", "sparsity": -0.1}]}, ("input_connections", 0, "sparsity"), "greater_than_equal"),
        ({"recurrent_regons": []}, ("recurrent_regons",), "extra_forbidden"),
        ({"input_regions": {}}, ("input_regions",), "list_type"),
    ],
)
def test_invalid_json_values_report_location(tmp_path, config, location, error_type):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))

    with pytest.raises(ValidationError) as exc_info:
        mRNN(config=str(path), device="cpu")

    assert any(
        error["loc"] == location and error["type"] == error_type
        for error in exc_info.value.errors()
    )


def test_empty_config_allows_manual_network_construction(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("{}")

    model = mRNN(config=str(path), device="cpu")
    assert model.region_dict == {}
    assert model.inp_dict == {}
    model.add_recurrent_region(name="manual", num_units=2)
    assert model.get_region_size("manual") == 2
