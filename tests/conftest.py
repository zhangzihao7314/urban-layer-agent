import os
import sys
from pathlib import Path

import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent a system PostGIS installation from overriding this Python
# environment's PROJ database during raster tests.
os.environ.pop("PROJ_LIB", None)
os.environ["PROJ_DATA"] = str(Path(rasterio.__file__).resolve().parent / "proj_data")
