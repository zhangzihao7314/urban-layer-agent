from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


def apply_masking(
    gdf: gpd.GeoDataFrame,
    raster_path: str,
    attribute_name: str,
    output_path: str,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        metadata = src.meta.copy()
        nodata = src.nodata
        nodata_mask = data == nodata if nodata is not None else np.zeros(src.shape, dtype=bool)

        for _, row in gdf.iterrows():
            polygon_mask = rasterize(
                [(row.geometry, 1)],
                out_shape=src.shape,
                transform=src.transform,
                fill=0,
                dtype=np.uint8,
            ).astype(bool)
            update_mask = polygon_mask & ~nodata_mask
            data = np.where(update_mask, float(row[attribute_name]), data)

        with rasterio.open(output_path, "w", **metadata) as dst:
            dst.write(data, 1)

    return str(output_path)

