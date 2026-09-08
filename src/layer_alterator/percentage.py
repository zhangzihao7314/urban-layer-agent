"""C2/C3 percentage operations ported from the professor's simulator notebook."""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize

from src.layer_alterator_agent.reference_loader import FRACTION_COLUMNS


def apply_percentage(raster_path: str, percentage: float, output_path: str) -> str:
    """Multiply valid pixels by ``1 + percentage/100`` and preserve NoData."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype("float32")
        meta = src.meta.copy()
        nodata = src.nodata
        valid = np.ones(data.shape, dtype=bool) if nodata is None else data != nodata
        data[valid] = np.clip(data[valid] * (1.0 + float(percentage) / 100.0), 0.0, 1.0)
        with rasterio.open(output, "w", **meta) as dst:
            dst.write(data, 1)
    return str(output)


def apply_polygon_percentage_ucp(
    gdf,
    raster_path: str,
    percentage_column: str,
    output_path: str,
    exceed_handling: str = "clip",
    zero_handling: str = "raise",
    zero_value: float = 0.01,
) -> str:
    """Apply the notebook C2 percentage rule inside each polygon only."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata
        nodata_mask = np.zeros(src.shape, dtype=bool)
        if nodata is not None:
            nodata_mask = data == nodata
        modified_mask = np.zeros(src.shape, dtype=bool)

        for _, row in gdf.iterrows():
            percentage = float(row[percentage_column])
            # A zero attribute explicitly means this polygon is unchanged.
            # Do not apply the notebook's 0 -> 0.01 UCP baseline in that case.
            if percentage == 0.0:
                continue
            polygon_mask = rasterize(
                [(row.geometry, 1)],
                out_shape=src.shape,
                transform=src.transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
            update_mask = polygon_mask & ~nodata_mask
            modified_mask |= update_mask
            if zero_handling == "raise":
                data[update_mask & (data == 0.0)] = float(zero_value)
            elif zero_handling != "preserve":
                raise ValueError("zero_handling must be 'raise' or 'preserve'.")
            factor = 1.0 + percentage / 100.0
            data[update_mask] *= factor

        valid = modified_mask & ~nodata_mask
        if exceed_handling == "clip":
            data[valid] = np.clip(data[valid], 0.0, 1.0)
        elif exceed_handling == "normalize":
            values = data[valid]
            if values.size and values.max() != values.min():
                data[valid] = (values - values.min()) / (values.max() - values.min())
            elif values.size:
                data[valid] = 0.0
        elif exceed_handling != "ignore":
            raise ValueError("exceed_handling must be 'clip', 'normalize', or 'ignore'.")
        if nodata is not None:
            data[nodata_mask] = nodata
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(data.astype(profile["dtype"]), 1)
    return str(output)


def apply_polygon_percentage_fractions(
    gdf,
    raster_paths,
    percentage_columns,
    output_folder: str,
    active_predictors=None,
) -> dict:
    """Apply C2 percentages and normalize seven fractions per polygon pixel."""
    names = list(FRACTION_COLUMNS)
    arrays = []
    profiles = []
    nodata_masks = []
    for name in names:
        with rasterio.open(raster_paths[name]) as src:
            array = src.read(1).astype("float32")
            arrays.append(array)
            profiles.append(src.profile.copy())
            nodata_masks.append(
                np.zeros(src.shape, dtype=bool)
                if src.nodata is None
                else array == src.nodata
            )
            shape, transform = src.shape, src.transform

    stack = np.stack(arrays)
    joint_valid = ~np.any(np.stack(nodata_masks), axis=0)
    active = set(names if active_predictors is None else active_predictors)
    for _, row in gdf.iterrows():
        # Rows written for non-intervention polygons contain 0% in every
        # active fraction field. Copy them exactly; normalizing them would
        # otherwise alter their original values even though they are marked
        # unchanged.
        if not any(
            float(row[percentage_columns[name]]) != 0.0
            for name in names
            if name in active
        ):
            continue
        polygon_mask = rasterize(
            [(row.geometry, 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        pixels = polygon_mask & joint_valid
        if not np.any(pixels):
            continue
        values = stack[:, pixels]
        factors = np.array(
            [
                1.0 + float(row[percentage_columns[name]]) / 100.0
                if name in active else 1.0
                for name in names
            ],
            dtype="float32",
        )[:, None]
        values *= factors
        sums = values.sum(axis=0)
        nonzero = sums != 0.0
        values[:, nonzero] /= sums[nonzero]
        values[:, ~nonzero] = 0.0
        stack[:, pixels] = values

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for index, name in enumerate(names):
        profile = profiles[index]
        nodata = profile.get("nodata")
        if nodata is not None:
            stack[index][nodata_masks[index]] = nodata
        output = output_dir / f"{name}_pct.tif"
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(stack[index].astype(profile["dtype"]), 1)
        outputs[name] = str(output)
    return outputs


def normalize_fraction_stack(arrays, nodata=None):
    """Normalize jointly so valid fraction pixels sum to one."""
    stack = np.stack(arrays).astype("float32")
    valid = np.ones(stack.shape[1:], dtype=bool)
    if nodata is not None:
        valid = ~np.any(stack == nodata, axis=0)
    sums = stack.sum(axis=0)
    adjustable = valid & (sums > 0)
    stack[:, adjustable] /= sums[adjustable]
    return stack


def normalize_fraction_rasters(raster_paths):
    """Normalize a set of output fraction rasters in place."""
    if not raster_paths:
        return
    arrays, profiles, nodata = [], [], None
    for path in raster_paths:
        with rasterio.open(path) as src:
            arrays.append(src.read(1))
            profiles.append(src.profile)
            nodata = src.nodata if nodata is None else nodata
    normalized = normalize_fraction_stack(arrays, nodata)
    for path, array, profile in zip(raster_paths, normalized, profiles):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array.astype(profile["dtype"]), 1)
