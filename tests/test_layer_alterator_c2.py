import json

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from src.layer_alterator.simulation_runner import run_layer_alterator
from src.layer_alterator.validator import validate_c2_inputs
from src.agent.transition_planner import calculate_transition_plan
from src.layer_alterator_agent.rules_generator import generate_partial_percentage_rules
from src.layer_alterator_agent.reference_loader import (
    FRACTION_COLUMNS,
    PREDICTOR_COLUMNS,
)


def _write_raster(path, data, nodata=-999.0):
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32632",
        "transform": from_origin(0, 2, 1, 1),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.asarray(data, dtype="float32"), 1)


def _c2_fixture(tmp_path):
    ucp = tmp_path / "ucp"
    fractions = tmp_path / "fractions"
    output = tmp_path / "output"
    ucp.mkdir()
    fractions.mkdir()

    first = {name: 0.0 for name in PREDICTOR_COLUMNS}
    second = {name: 0.0 for name in PREDICTOR_COLUMNS}
    unchanged = {name: 0.0 for name in PREDICTOR_COLUMNS}
    first.update({"TCH": 50.0, "F_G": 100.0})
    second.update({"TCH": -50.0, "F_TV": 100.0})
    # Verify compatibility with the typo in the professor's vector_pct example.
    first["SFV"] = first.pop("SVF")
    second["SFV"] = second.pop("SVF")
    unchanged["SFV"] = unchanged.pop("SVF")
    frame = gpd.GeoDataFrame(
        [
            {"fid": 1, **first, "geometry": box(0, 0, 1, 2)},
            {"fid": 2, **second, "geometry": box(1, 0, 2, 2)},
            {"fid": 3, **unchanged, "geometry": box(2, 0, 3, 2)},
        ],
        crs="EPSG:32632",
    )
    vector = tmp_path / "vector_pct.geojson"
    frame.to_file(vector, driver="GeoJSON")

    base_ucp = np.array([[0.4, 0.4, 0.0], [0.4, -999.0, 0.0]], dtype="float32")
    for name in [item for item in PREDICTOR_COLUMNS if item not in FRACTION_COLUMNS]:
        _write_raster(ucp / f"{name}.tif", base_ucp)

    fraction_values = {
        "F_AC": 0.1,
        "F_S": 0.1,
        "F_M": 0.1,
        "F_BS": 0.1,
        "F_G": 0.2,
        "F_TV": 0.3,
        "F_W": 0.1,
    }
    for name, value in fraction_values.items():
        _write_raster(fractions / f"{name}.tif", np.full((2, 3), value))

    rules = tmp_path / "operation_rules_C2.json"
    rules.write_text(
        json.dumps({f"{name}.tif": "pct" for name in PREDICTOR_COLUMNS}),
        encoding="utf-8",
    )
    return vector, rules, ucp, fractions, output


def test_c2_applies_each_polygon_percentage_and_preserves_outside_and_nodata(tmp_path):
    vector, rules, ucp, fractions, output = _c2_fixture(tmp_path)
    result = run_layer_alterator(vector, rules, ucp, fractions, output)

    assert result.success
    assert len(result.raster_outputs) == 12
    with rasterio.open(result.raster_outputs["TCH"]) as src:
        data = src.read(1)
        assert src.nodata == -999.0
    assert np.allclose(data[:, 0], [0.6, 0.6])
    assert np.isclose(data[0, 1], 0.2)
    assert data[1, 1] == -999.0
    # Polygon 3 is explicitly unchanged. Its true zeros must not be raised to
    # the notebook's 0.01 calculation baseline.
    assert np.allclose(data[:, 2], [0.0, 0.0])


def test_c2_fraction_outputs_sum_to_one_inside_and_leave_outside_unchanged(tmp_path):
    vector, rules, ucp, fractions, output = _c2_fixture(tmp_path)
    result = run_layer_alterator(vector, rules, ucp, fractions, output)
    arrays = {}
    for name in FRACTION_COLUMNS:
        with rasterio.open(result.raster_outputs[name]) as src:
            arrays[name] = src.read(1)
    stack = np.stack([arrays[name] for name in FRACTION_COLUMNS])
    assert np.allclose(stack.sum(axis=0), 1.0)
    assert arrays["F_G"][0, 0] > 0.2
    assert arrays["F_TV"][0, 1] > 0.3
    for name, original in {
        "F_AC": 0.1, "F_S": 0.1, "F_M": 0.1, "F_BS": 0.1,
        "F_G": 0.2, "F_TV": 0.3, "F_W": 0.1,
    }.items():
        assert np.isclose(arrays[name][0, 2], original)


def test_c2_rejects_missing_or_non_pct_rules(tmp_path):
    vector, rules, ucp, fractions, _ = _c2_fixture(tmp_path)
    frame = gpd.read_file(vector)
    rule_data = json.loads(rules.read_text(encoding="utf-8"))
    rule_data["TCH.tif"] = "none"
    try:
        validate_c2_inputs(frame, rule_data, ucp, fractions)
    except ValueError as exc:
        assert "every predictor rule" in str(exc)
    else:
        raise AssertionError("C2 accepted a non-pct rule")


def test_transition_percentages_come_from_current_mean_target_and_intensity(tmp_path):
    vector, _, ucp, fractions, _ = _c2_fixture(tmp_path)
    target = {name: 0.4 for name in PREDICTOR_COLUMNS}
    target.update({
        "F_AC": 0.1, "F_S": 0.1, "F_M": 0.1, "F_BS": 0.1,
        "F_G": 0.1, "F_TV": 0.4, "F_W": 0.1,
    })
    target["TCH"] = 0.6
    plan = calculate_transition_plan(
        vector, "fid", 1, target, 50, ucp, fractions
    )
    # Current TCH=0.4, half-way target=0.5, therefore +25%.
    assert np.isclose(plan["percentages"]["TCH"], 25.0)
    assert plan["method"].startswith("current polygon raster means")
    assert not plan["blockers"]


def test_transition_inverse_accounts_for_notebook_zero_ucp_baseline(tmp_path):
    vector, _, ucp, fractions, _ = _c2_fixture(tmp_path)
    target = {name: 0.1 for name in PREDICTOR_COLUMNS}
    target.update({
        "F_AC": 0.1, "F_S": 0.1, "F_M": 0.1, "F_BS": 0.1,
        "F_G": 0.2, "F_TV": 0.3, "F_W": 0.1,
    })
    target["TCH"] = 0.2
    plan = calculate_transition_plan(
        vector, "fid", 3, target, 50, ucp, fractions, active_predictors=["TCH"]
    )
    # Polygon 3 contains true zero TCH pixels. The notebook first raises them
    # to 0.01; +900% then produces the requested halfway target of 0.1.
    assert np.isclose(plan["percentages"]["TCH"], 900.0)
    assert np.isclose(plan["estimated_output_means"]["TCH"], 0.1)


def test_full_c3_uses_pct_and_none_and_preserves_none_ucp(tmp_path):
    vector, _, ucp, fractions, output = _c2_fixture(tmp_path)
    frame = gpd.read_file(vector)
    frame["F_G"] = 100.0
    frame["IMD"] = 50.0
    frame.to_file(vector, driver="GeoJSON")
    rules = generate_partial_percentage_rules(
        str(tmp_path / "operation_rules_C3.json"), ["F_G", "IMD"]
    )
    result = run_layer_alterator(vector, rules, ucp, fractions, output)
    assert len(result.raster_outputs) == 12
    with rasterio.open(result.raster_outputs["TCH"]) as source:
        tch = source.read(1)
    assert np.isclose(tch[0, 0], 0.4)  # NONE UCP is copied exactly.
    with rasterio.open(result.raster_outputs["IMD"]) as source:
        imd = source.read(1)
    assert np.isclose(imd[0, 0], 0.6)
    fraction_stack = []
    for name in FRACTION_COLUMNS:
        with rasterio.open(result.raster_outputs[name]) as source:
            fraction_stack.append(source.read(1))
    assert np.allclose(np.stack(fraction_stack).sum(axis=0), 1.0)
