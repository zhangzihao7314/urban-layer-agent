import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import geopandas as gpd
import rasterio

from src.layer_alterator.masking import apply_masking
from src.layer_alterator.percentage import (
    apply_polygon_percentage_fractions,
    apply_polygon_percentage_ucp,
)
from src.layer_alterator.validator import validate_c1_inputs, validate_c2_inputs, validate_c3_inputs
from src.layer_alterator_agent.reference_loader import FRACTION_COLUMNS


@dataclass
class LayerAlteratorResult:
    output_dir: str
    raster_outputs: Dict[str, str]
    success: bool = True


def run_c1_layer_alterator(
    vector_mask_path: str,
    rules_path: str,
    ucp_folder: str,
    fractions_folder: str,
    output_folder: str,
) -> LayerAlteratorResult:
    # Use Rasterio's matching PROJ database instead of an unrelated system
    # PostGIS/QGIS installation that may be present in the environment.
    os.environ.pop("PROJ_LIB", None)
    os.environ["PROJ_DATA"] = str(Path(rasterio.__file__).resolve().parent / "proj_data")

    gdf = gpd.read_file(vector_mask_path)
    with open(rules_path, encoding="utf-8") as file:
        rules = json.load(file)

    layer_paths = validate_c1_inputs(gdf, rules, ucp_folder, fractions_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for attribute_name, raster_path in layer_paths.items():
        output_path = output_dir / f"{attribute_name}_mask.tif"
        outputs[attribute_name] = apply_masking(
            gdf,
            str(raster_path),
            attribute_name,
            str(output_path),
        )
    return LayerAlteratorResult(str(output_dir), outputs)


def run_c2_layer_alterator(
    vector_mask_path: str,
    rules_path: str,
    ucp_folder: str,
    fractions_folder: str,
    output_folder: str,
) -> LayerAlteratorResult:
    """Run the professor notebook's all-PCT, per-polygon C2 workflow."""
    os.environ.pop("PROJ_LIB", None)
    os.environ["PROJ_DATA"] = str(Path(rasterio.__file__).resolve().parent / "proj_data")
    gdf = gpd.read_file(vector_mask_path)
    with open(rules_path, encoding="utf-8") as file:
        rules = json.load(file)
    layer_paths, percentage_columns = validate_c2_inputs(
        gdf, rules, ucp_folder, fractions_folder
    )
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = apply_polygon_percentage_fractions(
        gdf,
        {name: layer_paths[name] for name in FRACTION_COLUMNS},
        percentage_columns,
        str(output_dir),
    )
    for name, raster_path in layer_paths.items():
        if name in FRACTION_COLUMNS:
            continue
        output_path = output_dir / f"{name}_pct.tif"
        outputs[name] = apply_polygon_percentage_ucp(
            gdf,
            str(raster_path),
            percentage_columns[name],
            str(output_path),
        )
    return LayerAlteratorResult(str(output_dir), outputs)


def run_c3_layer_alterator(vector_mask_path, rules_path, ucp_folder, fractions_folder, output_folder):
    """Run the professor notebook's mixed PCT/NONE C3 workflow."""
    os.environ.pop("PROJ_LIB", None)
    os.environ["PROJ_DATA"] = str(Path(rasterio.__file__).resolve().parent / "proj_data")
    gdf = gpd.read_file(vector_mask_path)
    with open(rules_path, encoding="utf-8") as file:
        rules = json.load(file)
    layer_paths, columns, actions = validate_c3_inputs(
        gdf, rules, ucp_folder, fractions_folder
    )
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    active_fractions = [name for name in FRACTION_COLUMNS if actions[name] == "pct"]
    outputs = apply_polygon_percentage_fractions(
        gdf,
        {name: layer_paths[name] for name in FRACTION_COLUMNS},
        columns,
        str(output_dir),
        active_predictors=active_fractions,
    )
    for name, raster_path in layer_paths.items():
        if name in FRACTION_COLUMNS:
            continue
        output_path = output_dir / f"{name}_pct.tif"
        if actions[name] == "pct":
            outputs[name] = apply_polygon_percentage_ucp(
                gdf, str(raster_path), columns[name], str(output_path)
            )
        else:
            with rasterio.open(raster_path) as source:
                data = source.read(1)
                profile = source.profile.copy()
            with rasterio.open(output_path, "w", **profile) as target:
                target.write(data, 1)
            outputs[name] = str(output_path)
    return LayerAlteratorResult(str(output_dir), outputs)
