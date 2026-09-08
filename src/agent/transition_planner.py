"""Ground C2/C3 percentages in current rasters and LCZ reference targets."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

from src.layer_alterator_agent.reference_loader import FRACTION_COLUMNS, PREDICTOR_COLUMNS


def _raster_path(name, ucp_folder, fractions_folder):
    return Path(fractions_folder if name in FRACTION_COLUMNS else ucp_folder) / f"{name}.tif"


def polygon_raster_means(vector_path, id_column, polygon_id, ucp_folder, fractions_folder):
    vector = gpd.read_file(vector_path)
    selected = vector[vector[id_column].astype(str) == str(polygon_id)]
    if selected.empty:
        raise ValueError(f"Unknown Polygon ID: {polygon_id}")
    means = {}
    fraction_samples = {}
    ucp_samples = {}
    for name in PREDICTOR_COLUMNS:
        path = _raster_path(name, ucp_folder, fractions_folder)
        if not path.exists():
            raise FileNotFoundError(f"Raster layer not found: {path}")
        with rasterio.open(path) as source:
            geometry = selected.to_crs(source.crs).geometry
            inside = geometry_mask(
                geometry, out_shape=source.shape, transform=source.transform, invert=True
            )
            data = source.read(1).astype("float64")
            valid = inside & np.isfinite(data)
            if source.nodata is not None:
                valid &= data != source.nodata
            values = data[valid]
            if not values.size:
                raise ValueError(f"Polygon {polygon_id} has no valid cells in {name}.")
            means[name] = float(values.mean())
            if name in FRACTION_COLUMNS:
                fraction_samples[name] = values
            else:
                ucp_samples[name] = values
    return means, fraction_samples, ucp_samples


def _fraction_factors(samples, desired, active):
    names = list(FRACTION_COLUMNS)
    size = min(len(samples[name]) for name in names)
    if size > 200_000:
        indices = np.linspace(0, size - 1, 200_000, dtype=int)
    else:
        indices = np.arange(size)
    stack = np.stack([samples[name][:size][indices] for name in names])
    active = set(active)
    factors = np.ones(len(names), dtype="float64")
    for i, name in enumerate(names):
        current = stack[i].mean()
        if name in active and current > 1e-9:
            factors[i] = max(desired[name] / current, 1e-6)
    # Iterative proportional fitting accounts for Layer Alterator's per-pixel
    # normalization while keeping NONE factors fixed at one for C3.
    for _ in range(40):
        transformed = stack * factors[:, None]
        sums = transformed.sum(axis=0)
        normalized = transformed / np.where(sums == 0, 1.0, sums)
        actual = normalized.mean(axis=1)
        for i, name in enumerate(names):
            if name in active and actual[i] > 1e-9:
                factors[i] *= max(desired[name] / actual[i], 1e-6)
        if active == set(names):
            # All-fraction factors have one redundant common scale.
            factors /= np.exp(np.mean(np.log(np.maximum(factors, 1e-12))))
    transformed = stack * factors[:, None]
    normalized = transformed / np.where(transformed.sum(axis=0) == 0, 1.0, transformed.sum(axis=0))
    achieved = {name: float(normalized[i].mean()) for i, name in enumerate(names)}
    return {name: float((factors[i] - 1.0) * 100.0) for i, name in enumerate(names)}, achieved


def calculate_transition_plan(
    vector_path,
    id_column,
    polygon_id,
    target_values,
    intensity,
    ucp_folder,
    fractions_folder,
    active_predictors=None,
):
    """Calculate auditable C2/C3 percentages without LLM-generated numbers."""
    current, fraction_samples, ucp_samples = polygon_raster_means(
        vector_path, id_column, polygon_id, ucp_folder, fractions_folder
    )
    alpha = float(intensity) / 100.0
    active = set(PREDICTOR_COLUMNS if active_predictors is None else active_predictors)
    desired = {
        name: current[name] + alpha * (float(target_values[name]) - current[name])
        if name in active else current[name]
        for name in PREDICTOR_COLUMNS
    }
    percentages = {name: 0.0 for name in PREDICTOR_COLUMNS}
    achieved = dict(desired)
    blockers = []
    warnings = []
    for name in [item for item in PREDICTOR_COLUMNS if item not in FRACTION_COLUMNS]:
        if name not in active:
            continue
        # The professor notebook replaces every valid zero UCP pixel with 0.01
        # before multiplying. Base the inverse calculation on that effective
        # pixel distribution, not only on the original polygon mean.
        samples = np.asarray(ucp_samples[name], dtype="float64")
        raised = np.where(samples == 0.0, 0.01, samples)
        baseline = float(raised.mean())
        if baseline <= 1e-12:
            blockers.append(f"{name} has no usable UCP baseline for a percentage transition.")
            continue
        factor = max(float(desired[name]) / baseline, 0.0)
        percentages[name] = (factor - 1.0) * 100.0
        achieved[name] = float(np.clip(raised * factor, 0.0, 1.0).mean())
        zero_count = int(np.count_nonzero(samples == 0.0))
        if zero_count:
            warnings.append(
                f"{name} contains {zero_count} zero pixel(s); the inverse calculation "
                "includes the professor notebook's 0.01 UCP baseline."
            )
    fraction_pct, fraction_achieved = _fraction_factors(
        fraction_samples, desired, active & set(FRACTION_COLUMNS)
    )
    for name in FRACTION_COLUMNS:
        if name in active:
            if current[name] <= 1e-9 and desired[name] > 1e-9:
                blockers.append(
                    f"{name} is zero in the current polygon; multiplicative C2/C3 cannot introduce it."
                )
            percentages[name] = fraction_pct[name]
        achieved[name] = fraction_achieved[name]
    extreme = [name for name, value in percentages.items() if abs(value) > 500]
    if extreme:
        warnings.append(
            "Extreme relative changes exceed 500% for: " + ", ".join(extreme)
            + ". Consider C1 replacement or a lower intensity."
        )
    return {
        "polygon_id": polygon_id,
        "intensity": float(intensity),
        "current_values": current,
        "target_values": {name: float(target_values[name]) for name in PREDICTOR_COLUMNS},
        "intermediate_targets": desired,
        "percentages": percentages,
        "estimated_output_means": achieved,
        "active_predictors": sorted(active),
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "method": "current polygon raster means + professor LCZ reference + confirmed transition intensity",
    }
