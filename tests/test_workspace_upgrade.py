import json
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from src.layer_alterator.router import classify_rules
from src.layer_alterator.simulation_runner import run_layer_alterator
from src.agent.execution_manager import validate_raster_outputs
from src.web.workspace_helpers import (
    build_qgis_qml, generation_checklist, mark_unchanged, polygon_rows,
    workflow_progress,
)


def test_polygon_workspace_rows_and_preflight():
    frame = gpd.GeoDataFrame([
        {"fid": 7, "transf_type": "parking", "geometry": box(0, 0, 1, 1)},
        {"fid": 8, "transf_type": "dump", "geometry": box(1, 0, 2, 1)},
    ], crs="EPSG:4326")
    rows = polygon_rows(frame, "fid", {7: "Dense trees"}, {8})
    assert [row["Status"] for row in rows] == ["confirmed", "unchanged"]
    checks, ready = generation_checklist(True, "cooling", [7, 8], {7: "Dense trees"}, {8})
    assert ready and all(checks.values())
    unchanged = []
    mark_unchanged(unchanged, 8)
    mark_unchanged(unchanged, 8)
    assert unchanged == [8]


def test_workflow_progress_does_not_count_empty_preallocated_decisions():
    flags, progress = workflow_progress(
        True, False, [7, 8, 9], {7: "", 8: "", 9: ""}, [], False
    )
    assert flags == [True, False, False, False]
    assert progress == 0.25
    _, progress = workflow_progress(
        True, True, [7, 8, 9],
        {7: "Dense trees", 8: "Water"}, [9], False,
    )
    assert progress == 0.75
    flags, progress = workflow_progress(
        False, True, [7], {7: "Dense trees"}, [], True
    )
    assert flags == [False, False, False, False]
    assert progress == 0.0


def test_qgis_style_and_structured_percentage_rules():
    qml = build_qgis_qml()
    assert "target_lcz_type" in qml and "Dense trees" in qml
    rules = {
        "TCH.tif": {"action": "pct", "percentage": 20},
        "IMD.tif": "none",
    }
    assert classify_rules(rules).code == "C3"


def test_c3_runner_changes_pct_copies_none_and_normalizes_fractions(tmp_path):
    ucp, fractions, output = tmp_path / "u", tmp_path / "f", tmp_path / "o"
    ucp.mkdir(); fractions.mkdir()
    profile = {
        "driver": "GTiff", "height": 1, "width": 1, "count": 1,
        "dtype": "float32", "crs": "EPSG:4326",
        "transform": from_origin(0, 1, 1, 1),
    }
    for name, value in {"F_G.tif": 0.5, "F_TV.tif": 0.5}.items():
        with rasterio.open(fractions / name, "w", **profile) as dst:
            dst.write(np.array([[value]], dtype="float32"), 1)
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({
        "F_G.tif": {"action": "pct", "percentage": 100},
        "F_TV.tif": "none",
    }))
    result = run_layer_alterator("", rules_path, ucp, fractions, output)
    with rasterio.open(result.raster_outputs["F_G"]) as src:
        grass = src.read(1)[0, 0]
    with rasterio.open(result.raster_outputs["F_TV"]) as src:
        trees = src.read(1)[0, 0]
    assert np.allclose(grass + trees, 1.0)
    assert grass > trees


def test_different_output_grids_are_warning_not_execution_failure(tmp_path):
    profile = {
        "driver": "GTiff", "height": 1, "width": 1, "count": 1,
        "dtype": "float32", "crs": "EPSG:4326",
        "transform": from_origin(0, 1, 1, 1),
    }
    first = tmp_path / "a.tif"
    second = tmp_path / "b.tif"
    with rasterio.open(first, "w", **profile) as dst:
        dst.write(np.ones((1, 1), dtype="float32"), 1)
    shifted = dict(profile)
    shifted["transform"] = from_origin(1, 1, 1, 1)
    with rasterio.open(second, "w", **shifted) as dst:
        dst.write(np.ones((1, 1), dtype="float32"), 1)
    report = validate_raster_outputs({"A": str(first), "B": str(second)})
    assert report.valid
    assert any("different raster grid" in warning for warning in report.warnings)
