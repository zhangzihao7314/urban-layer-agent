from pathlib import Path
from typing import Dict

import geopandas as gpd
import numpy as np
import rasterio

from src.layer_alterator_agent.reference_loader import FRACTION_COLUMNS, PREDICTOR_COLUMNS


def normalize_rules(rules: Dict[str, str]) -> Dict[str, str]:
    return {
        key if key.endswith(".tif") else f"{key}.tif": (
            str(value.get("action", "none")).lower()
            if isinstance(value, dict)
            else str(value).lower()
        )
        for key, value in rules.items()
    }


def _resolve_percentage_column(gdf: gpd.GeoDataFrame, predictor: str) -> str:
    """Return the vector attribute used by the professor's C2 workflow."""
    if predictor in gdf.columns:
        return predictor
    # The professor-provided vector_pct example contains SFV while the raster
    # and rule names use SVF. Supporting the alias keeps the example usable
    # without silently treating the percentage as zero.
    if predictor == "SVF" and "SFV" in gdf.columns:
        return "SFV"
    raise ValueError(f"Vector percentage attribute is missing: {predictor}")


def validate_c2_inputs(
    gdf: gpd.GeoDataFrame,
    rules: Dict[str, str],
    ucp_folder: str,
    fractions_folder: str,
) -> tuple[Dict[str, Path], Dict[str, str]]:
    """Validate the all-PCT, per-polygon C2 inputs from layers_simulator.ipynb."""
    if gdf.empty:
        raise ValueError("The percentage vector contains no polygons.")
    if gdf.crs is None:
        raise ValueError("The percentage vector has no CRS.")

    normalized = normalize_rules(rules)
    if set(normalized) != {f"{name}.tif" for name in PREDICTOR_COLUMNS}:
        missing = sorted(
            {f"{name}.tif" for name in PREDICTOR_COLUMNS} - set(normalized)
        )
        extra = sorted(set(normalized) - {f"{name}.tif" for name in PREDICTOR_COLUMNS})
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unexpected={extra}")
        raise ValueError("C2 requires exactly the 12 predictor rules (" + ", ".join(details) + ").")
    if set(normalized.values()) != {"pct"}:
        raise ValueError("C2 requires every predictor rule to be 'pct'.")

    columns = {
        predictor: _resolve_percentage_column(gdf, predictor)
        for predictor in PREDICTOR_COLUMNS
    }
    for index, row in gdf.iterrows():
        for predictor, column in columns.items():
            value = row[column]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                raise ValueError(
                    f"Feature {index} percentage is missing for {predictor}."
                )
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Feature {index} percentage for {predictor} is not numeric: {value!r}"
                ) from exc
            if not np.isfinite(numeric):
                raise ValueError(
                    f"Feature {index} percentage for {predictor} is not finite."
                )

    layer_paths: Dict[str, Path] = {}
    fraction_grid = None
    for predictor in PREDICTOR_COLUMNS:
        filename = f"{predictor}.tif"
        folder = fractions_folder if predictor in FRACTION_COLUMNS else ucp_folder
        path = Path(folder) / filename
        if not path.exists():
            raise FileNotFoundError(f"Raster layer not found: {path}")
        with rasterio.open(path) as src:
            if src.crs != gdf.crs:
                raise ValueError(
                    f"CRS mismatch for {filename}: raster={src.crs}, vector={gdf.crs}"
                )
            if predictor in FRACTION_COLUMNS:
                grid = (src.crs, src.width, src.height, src.transform)
                if fraction_grid is None:
                    fraction_grid = grid
                elif grid != fraction_grid:
                    raise ValueError(
                        "C2 fraction rasters must share CRS, dimensions, and transform."
                    )
        layer_paths[predictor] = path
    return layer_paths, columns


def validate_c3_inputs(gdf, rules, ucp_folder, fractions_folder):
    """Validate a complete 12-layer mix of PCT and NONE rules."""
    normalized = normalize_rules(rules)
    expected = {f"{name}.tif" for name in PREDICTOR_COLUMNS}
    if set(normalized) != expected:
        raise ValueError("C3 requires rules for exactly all 12 predictors.")
    values = set(normalized.values())
    if values != {"pct", "none"}:
        raise ValueError("C3 requires a mix of 'pct' and 'none' rules.")
    # Validate raster grids and numeric percentage attributes with an all-PCT
    # structural copy; NONE attributes are written as zero by our generator.
    all_pct = {key: "pct" for key in normalized}
    layer_paths, columns = validate_c2_inputs(
        gdf, all_pct, ucp_folder, fractions_folder
    )
    actions = {key.removesuffix(".tif"): value for key, value in normalized.items()}
    return layer_paths, columns, actions


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
