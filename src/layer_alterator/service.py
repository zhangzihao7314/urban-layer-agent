import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import geopandas as gpd
import rasterio

from src.layer_alterator.masking import apply_masking
from src.layer_alterator.validator import validate_c1_inputs


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
