import json
import numpy as np
import geopandas as gpd
from shapely.geometry import box

from src.agent.audit_response import AuditableAnswer
from src.agent.llm_requirements import parse_extraction
from src.layer_alterator.percentage import normalize_fraction_stack
from src.layer_alterator.router import classify_rules, require_executable
from src.layer_alterator_agent.output_schema import migrate_legacy_columns


def test_llm_json_is_strict_and_auditable():
    parsed = parse_extraction('{"primary_function":"cooling","confidence":0.8}')
    assert parsed["primary_function"] == "cooling"
    assert parse_extraction('{"made_up":"value","confidence":1}') is None
    answer = AuditableAnswer("Scattered trees", reference_facts=["LCZ 102: P"])
    assert answer.to_dict()["reference_facts"] == ["LCZ 102: P"]


def test_schema_preserves_original_transformation_type():
    frame = gpd.GeoDataFrame(
        [{"transf_type": "open parking", "matched_urban_type": "Dense trees",
          "user_description": "wooded park", "geometry": box(0, 0, 1, 1)}],
        crs="EPSG:4326",
    )
    migrated = migrate_legacy_columns(frame)
    assert migrated.loc[0, "original_transf_type"] == "open parking"
    assert migrated.loc[0, "target_lcz_type"] == "Dense trees"
    assert migrated.loc[0, "user_request"] == "wooded park"


def test_notebook_rule_cases_c0_to_c5():
    assert classify_rules({"A": "none"}).code == "C0"
    assert classify_rules({"A": "mask", "B": "mask"}).code == "C1"
    assert classify_rules({"A": "pct"}).code == "C2"
    assert classify_rules({"A": "pct", "B": "none"}).code == "C3"
    assert classify_rules({"A": "mask", "B": "none"}).code == "C4"
    assert classify_rules({"A": "mask", "B": "pct"}).code == "C5"
    try:
        require_executable({"A": "mask", "B": "pct"})
        assert False
    except ValueError as error:
        assert "C5" in str(error)


def test_fraction_stack_is_jointly_normalized():
    result = normalize_fraction_stack([
        np.array([[0.4]], dtype="float32"),
        np.array([[0.4]], dtype="float32"),
        np.array([[0.4]], dtype="float32"),
    ])
    assert np.allclose(result[:, 0, 0].sum(), 1.0)
