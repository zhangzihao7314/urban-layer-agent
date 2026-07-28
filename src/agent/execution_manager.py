"""Execution plans and post-generation validation for Agent actions."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List
import json

import geopandas as gpd
import rasterio

from src.layer_alterator_agent.reference_loader import (
    FRACTION_COLUMNS,
    PREDICTOR_COLUMNS,
    get_predictor_values,
)


@dataclass
class ExecutionStep:
    name: str
    status: str = "pending"
    detail: str = ""


@dataclass
class ValidationReport:
    valid: bool
    checks: Dict[str, bool]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def build_execution_plan() -> List[ExecutionStep]:
    return [
        ExecutionStep("validate_polygon_ids"),
        ExecutionStep("retrieve_predictor_values"),
        ExecutionStep("write_geojson_attributes"),
        ExecutionStep("generate_rules"),
        ExecutionStep("generate_config"),
        ExecutionStep("validate_outputs"),
    ]


def validate_generated_inputs(
    source_vector: str,
    updated_vector: str,
    rules_path: str,
    config_path: str,
    reference_df,
    id_column: str,
    modified_decisions: Dict[int, str],
) -> ValidationReport:
    checks, errors, warnings = {}, [], []
    try:
        source = gpd.read_file(source_vector)
        updated = gpd.read_file(updated_vector)
        checks["feature_count_preserved"] = len(source) == len(updated)
        checks["geometry_count_preserved"] = len(source.geometry) == len(updated.geometry)
        checks["polygon_ids_preserved"] = set(source[id_column]) == set(updated[id_column])
        checks["all_predictors_present"] = all(
            name in updated.columns for name in PREDICTOR_COLUMNS
        )
        if checks["all_predictors_present"]:
            checks["all_predictors_complete"] = not updated[PREDICTOR_COLUMNS].isna().any().any()
        for polygon_id, urban_type in modified_decisions.items():
            row = updated[updated[id_column] == polygon_id]
            expected = get_predictor_values(reference_df, urban_type)
            if row.empty:
                errors.append(f"Modified polygon ID missing from output: {polygon_id}")
                continue
            checks[f"predictors_match_{polygon_id}"] = all(
                abs(float(row.iloc[0][name]) - expected[name]) < 1e-6
                for name in PREDICTOR_COLUMNS
            )
            fraction_sum = sum(float(row.iloc[0][name]) for name in FRACTION_COLUMNS)
            checks[f"fraction_sum_{polygon_id}"] = abs(fraction_sum - 1.0) <= 0.01
        rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        checks["rules_are_json_object"] = isinstance(rules, dict)
        checks["config_paths_exist"] = all(
            Path(config[key]).exists() for key in ["vector_mask", "rules"] if key in config
        )
    except Exception as exc:
        errors.append(str(exc))
    failed = [name for name, ok in checks.items() if not ok]
    errors.extend(f"Failed validation check: {name}" for name in failed)
    return ValidationReport(not errors, checks, errors, warnings)


def validate_raster_outputs(paths: Dict[str, str]) -> ValidationReport:
    checks, errors, warnings = {}, [], []
    metadata_signature = None
    reference_name = None
    for name, path in paths.items():
        try:
            with rasterio.open(path) as source:
                data = source.read(1, masked=True)
                checks[f"{name}_readable"] = True
                checks[f"{name}_has_crs"] = source.crs is not None
                checks[f"{name}_nonempty"] = data.size > 0
                signature = (source.crs, source.width, source.height, source.transform)
                if metadata_signature is None:
                    metadata_signature = signature
                    reference_name = name
                elif signature != metadata_signature:
                    # C1 can safely write each source raster on its own grid.
                    # Cross-layer alignment matters only for downstream models
                    # that stack all predictors, so report it as compatibility
                    # guidance rather than claiming Layer Alterator failed.
                    warnings.append(
                        f"{name} uses a different raster grid from {reference_name}. "
                        "The output is valid for individual GIS inspection, but "
                        "align all layers before a downstream stacked simulation."
                    )
                if data.count() == 0:
                    warnings.append(f"{name} contains only NoData values.")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    failed = [name for name, ok in checks.items() if not ok]
    errors.extend(f"Failed validation check: {name}" for name in failed)
    return ValidationReport(not errors, checks, errors, warnings)


def recommend_replanning(validation: ValidationReport) -> List[str]:
    if validation.valid:
        return ["Outputs passed structural validation; evaluate goal performance with the downstream LST model."]
    recommendations = []
    if any("predictors_match" in error for error in validation.errors):
        recommendations.append("Regenerate predictor attributes from the reference CSV.")
    if any("fraction_sum" in error for error in validation.errors):
        recommendations.append("Recheck fraction fields before rerunning Layer Alterator.")
    if any("aligned" in error or "crs" in error.lower() for error in validation.errors):
        recommendations.append("Align raster CRS, resolution, transform, and extent.")
    return recommendations or ["Inspect the validation errors and revise the plan before rerunning."]
