from pathlib import Path
from typing import Dict

import geopandas as gpd
import numpy as np
import rasterio

from src.layer_alterator_agent.reference_loader import FRACTION_COLUMNS, PREDICTOR_COLUMNS


def normalize_rules(rules: Dict[str, str]) -> Dict[str, str]:
    return {key if key.endswith(".tif") else f"{key}.tif": value.lower() for key, value in rules.items()}


def validate_c1_inputs(
    gdf: gpd.GeoDataFrame,
    rules: Dict[str, str],
    ucp_folder: str,
    fractions_folder: str,
) -> Dict[str, Path]:
    if gdf.empty:
        raise ValueError("The vector mask contains no polygons.")
    if gdf.crs is None:
        raise ValueError("The vector mask has no CRS.")

    normalized = normalize_rules(rules)
    if not normalized or set(normalized.values()) != {"mask"}:
        raise ValueError("C1 requires every rule to be 'mask'.")

    layer_paths = {}
    for predictor in PREDICTOR_COLUMNS:
        filename = f"{predictor}.tif"
        if normalized.get(filename) != "mask":
            raise ValueError(f"Missing C1 mask rule for {filename}.")
        if predictor not in gdf.columns:
            raise ValueError(f"Vector attribute is missing: {predictor}")
        folder = fractions_folder if predictor.startswith("F_") else ucp_folder
        path = Path(folder) / filename
        if not path.exists():
            raise FileNotFoundError(f"Raster layer not found: {path}")
        with rasterio.open(path) as src:
            if src.crs != gdf.crs:
                raise ValueError(f"CRS mismatch for {filename}: raster={src.crs}, vector={gdf.crs}")
        layer_paths[predictor] = path

    for index, row in gdf.iterrows():
        values = {name: float(row[name]) for name in PREDICTOR_COLUMNS}
        outside = {name: value for name, value in values.items() if not 0.0 <= value <= 1.0}
        if outside:
            raise ValueError(f"Feature {index} has values outside [0,1]: {outside}")
        if values["IMD"] < values["BSF"]:
            raise ValueError(f"Feature {index} violates IMD >= BSF.")
        fraction_sum = sum(values[name] for name in FRACTION_COLUMNS)
        if not np.isclose(fraction_sum, 1.0, atol=1e-6):
            raise ValueError(f"Feature {index} fraction sum is {fraction_sum}, expected 1.0.")

    return layer_paths

