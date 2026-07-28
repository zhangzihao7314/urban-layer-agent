"""C2/C3 percentage operations ported from the professor's simulator notebook."""

from pathlib import Path
import numpy as np
import rasterio


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
