"""Generate a C2 percentage vector from confirmed conversational decisions."""

from pathlib import Path
import json

import geopandas as gpd

from src.layer_alterator_agent.reference_loader import PREDICTOR_COLUMNS
from src.layer_alterator_agent.rules_generator import (
    generate_partial_percentage_rules,
    generate_percentage_rules,
)


def write_c2_percentage_vector(
    source_vector: str,
    id_column: str,
    confirmed_percentages: dict,
    output_path: str,
) -> str:
    vector = gpd.read_file(source_vector)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    for predictor in PREDICTOR_COLUMNS:
        vector[predictor] = 0.0
    for polygon_id, percentages in confirmed_percentages.items():
        mask = vector[id_column].astype(str) == str(polygon_id)
        if not mask.any():
            raise ValueError(f"Unknown polygon ID while writing C2 vector: {polygon_id}")
        for predictor in PREDICTOR_COLUMNS:
            vector.loc[mask, predictor] = float(percentages.get(predictor, 0.0))
    vector.to_file(output, driver="GeoJSON")
    return str(output)


def generate_c2_inputs(source_vector, id_column, confirmed_percentages, output_dir):
    """Write the conversational C2 vector, all-PCT rules, and compact config."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    vector_path = write_c2_percentage_vector(
        source_vector,
        id_column,
        confirmed_percentages,
        str(folder / "c2_percentage_vector.geojson"),
    )
    rules_path = generate_percentage_rules(str(folder / "operation_rules_C2.json"))
    config_path = folder / "simulation_config_C2.json"
    config_path.write_text(
        json.dumps(
            {"workflow": "C2", "vector_mask": vector_path, "rules": rules_path},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    vector = gpd.read_file(vector_path)
    return {
        "workflow": "C2",
        "updated_vector_path": vector_path,
        "rules_path": rules_path,
        "config_path": str(config_path),
        "attribute_table": vector.drop(columns="geometry").to_dict(orient="records"),
    }


def generate_c3_inputs(source_vector, id_column, confirmed_percentages, pct_predictors, output_dir):
    """Write conversational C3 inputs with a global PCT/NONE rule selection."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    vector_path = write_c2_percentage_vector(
        source_vector, id_column, confirmed_percentages,
        str(folder / "c3_percentage_vector.geojson"),
    )
    rules_path = generate_partial_percentage_rules(
        str(folder / "operation_rules_C3.json"), pct_predictors
    )
    config_path = folder / "simulation_config_C3.json"
    config_path.write_text(
        json.dumps({"workflow": "C3", "vector_mask": vector_path, "rules": rules_path}, indent=2) + "\n",
        encoding="utf-8",
    )
    vector = gpd.read_file(vector_path)
    return {
        "workflow": "C3", "updated_vector_path": vector_path,
        "rules_path": rules_path, "config_path": str(config_path),
        "attribute_table": vector.drop(columns="geometry").to_dict(orient="records"),
    }
