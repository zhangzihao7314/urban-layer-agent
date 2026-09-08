"""Validate the essential invariants of a completed C2 Layer Alterator run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layer_alterator_agent.reference_loader import (
    FRACTION_COLUMNS,
    PREDICTOR_COLUMNS,
)


def _input_path(name: str, ucp_folder: Path, fractions_folder: Path) -> Path:
    folder = fractions_folder if name in FRACTION_COLUMNS else ucp_folder
    return folder / f"{name}.tif"


def validate_outputs(
    vector_path: str,
    ucp_folder: str,
    fractions_folder: str,
    output_folder: str,
    tolerance: float = 1e-5,
) -> dict:
    """Return a compact, JSON-serializable C2 validation report."""
    vector = gpd.read_file(vector_path)
    ucp_dir = Path(ucp_folder)
    fraction_dir = Path(fractions_folder)
    output_dir = Path(output_folder)

    missing = [
        name
        for name in PREDICTOR_COLUMNS
        if not (output_dir / f"{name}_pct.tif").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing C2 outputs: {', '.join(missing)}")

    outside_max_change = {}
    fraction_polygon_mask = None
    common_valid = None
    output_fractions = []

    for name in PREDICTOR_COLUMNS:
        source_path = _input_path(name, ucp_dir, fraction_dir)
        output_path = output_dir / f"{name}_pct.tif"
        with rasterio.open(source_path) as source, rasterio.open(output_path) as result:
            if (
                source.crs != result.crs
                or source.shape != result.shape
                or source.transform != result.transform
            ):
                raise ValueError(f"Grid metadata changed for {name}.")
            if vector.crs != source.crs:
                raise ValueError(f"Vector/raster CRS mismatch for {name}.")

            original = source.read(1).astype("float64")
            changed = result.read(1).astype("float64")
            polygon_mask = geometry_mask(
                vector.geometry,
                out_shape=source.shape,
                transform=source.transform,
                invert=True,
            )

            valid = np.ones(source.shape, dtype=bool)
            if source.nodata is not None:
                valid &= original != source.nodata
            if result.nodata is not None:
                valid &= changed != result.nodata
            outside = valid & ~polygon_mask
            outside_max_change[name] = float(
                np.max(np.abs(changed[outside] - original[outside]))
                if np.any(outside)
                else 0.0
            )

            if name in FRACTION_COLUMNS:
                output_fractions.append(changed)
                common_valid = valid if common_valid is None else common_valid & valid
                fraction_polygon_mask = polygon_mask

    fraction_stack = np.stack(output_fractions)
    fraction_pixels = fraction_polygon_mask & common_valid
    fraction_sums = fraction_stack[:, fraction_pixels].sum(axis=0)
    max_fraction_sum_error = float(
        np.max(np.abs(fraction_sums - 1.0)) if fraction_sums.size else 0.0
    )
    max_outside_change = max(outside_max_change.values(), default=0.0)

    return {
        "passed": bool(
            not missing
            and max_outside_change <= tolerance
            and max_fraction_sum_error <= tolerance
        ),
        "output_count": len(PREDICTOR_COLUMNS) - len(missing),
        "polygon_count": int(len(vector)),
        "fraction_polygon_pixel_count": int(np.count_nonzero(fraction_polygon_mask)),
        "max_outside_polygon_absolute_change": max_outside_change,
        "max_fraction_sum_error_inside_polygons": max_fraction_sum_error,
        "tolerance": tolerance,
        "outside_change_by_predictor": outside_max_change,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vector")
    parser.add_argument("ucp_folder")
    parser.add_argument("fractions_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--report", help="Optional path for the JSON report")
    args = parser.parse_args()

    report = validate_outputs(
        args.vector,
        args.ucp_folder,
        args.fractions_folder,
        args.output_folder,
        args.tolerance,
    )
    rendered = json.dumps(report, indent=2)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
