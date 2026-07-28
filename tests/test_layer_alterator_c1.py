import json

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from src.layer_alterator.service import run_c1_layer_alterator
from src.layer_alterator_agent.reference_loader import PREDICTOR_COLUMNS


def test_c1_end_to_end_writes_all_rasters(tmp_path):
    ucp = tmp_path / "ucp"
    fractions = tmp_path / "fractions"
    outputs = tmp_path / "outputs"
    ucp.mkdir()
    fractions.mkdir()

    values = {
        "TCH": 0.2, "IMD": 0.4, "BH": 0.1, "BSF": 0.2, "SVF": 0.8,
        "F_AC": 0.2, "F_S": 0.1, "F_M": 0.1, "F_BS": 0.1,
        "F_G": 0.2, "F_TV": 0.2, "F_W": 0.1,
    }
    frame = gpd.GeoDataFrame(
        [{"fid": 7, **values, "geometry": box(0, 0, 2, 2)}],
        crs="EPSG:32632",
    )
    vector = tmp_path / "mask.geojson"
    frame.to_file(vector, driver="GeoJSON")

    metadata = {
        "driver": "GTiff", "height": 2, "width": 2, "count": 1,
        "dtype": "float32", "crs": "EPSG:32632",
        "transform": from_origin(0, 2, 1, 1),
    }
    for name in PREDICTOR_COLUMNS:
        folder = fractions if name.startswith("F_") else ucp
        with rasterio.open(folder / f"{name}.tif", "w", **metadata) as dst:
            dst.write(np.zeros((2, 2), dtype=np.float32), 1)

    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({f"{name}.tif": "mask" for name in PREDICTOR_COLUMNS}))
    result = run_c1_layer_alterator(vector, rules, ucp, fractions, outputs)

    assert result.success
    assert len(result.raster_outputs) == 12
    with rasterio.open(result.raster_outputs["TCH"]) as src:
        assert np.allclose(src.read(1), 0.2)

