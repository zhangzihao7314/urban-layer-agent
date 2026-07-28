from pathlib import Path
import geopandas as gpd

from src.layer_alterator_agent.attribute_generator import generate_polygon_attributes
from src.layer_alterator_agent.output_schema import preserve_source_semantics


ID_COLUMN_CANDIDATES = ["polygon_id", "fid", "id", "name"]


def resolve_id_column(gdf: gpd.GeoDataFrame, id_column: str = None) -> str:
    """Select and validate a stable polygon identifier column."""
    if id_column:
        if id_column not in gdf.columns:
            raise ValueError(f"ID column not found: {id_column}")
        selected = id_column
    else:
        columns_by_lower = {str(col).lower(): col for col in gdf.columns}
        selected = next(
            (columns_by_lower[name] for name in ID_COLUMN_CANDIDATES if name in columns_by_lower),
            None,
        )

        if selected is None:
            selected = "polygon_id"
            gdf[selected] = range(1, len(gdf) + 1)

    if gdf[selected].isna().any():
        raise ValueError(f"ID column '{selected}' contains empty values.")
    if not gdf[selected].is_unique:
        raise ValueError(f"ID column '{selected}' contains duplicate values.")

    return selected


def load_vector(vector_path: str) -> gpd.GeoDataFrame:
    """
    Load vector polygon file.
    Supported by geopandas: GeoJSON, GPKG, SHP, etc.
    """
    return gpd.read_file(vector_path)


def get_polygon_ids(gdf: gpd.GeoDataFrame, id_column: str = None) -> list:
    """
    Get polygon IDs from a validated source identifier column.
    """
    id_column = resolve_id_column(gdf, id_column)
    return gdf[id_column].tolist()


def write_attributes_to_vector(
    vector_path: str,
    reference_df,
    polygon_descriptions: dict,
    output_path: str,
    id_column: str = None,
) -> str:
    """
    Write generated Layer Alterator attributes back to vector.

    polygon_descriptions example:
    {
        1: "park area",
        2: "parking lot"
    }
    """
    gdf = preserve_source_semantics(load_vector(vector_path))

    id_column = resolve_id_column(gdf, id_column)

    for idx, row in gdf.iterrows():
        polygon_id = row[id_column]

        # Polygons explicitly marked "unchanged" are intentionally omitted
        # from polygon_descriptions and keep their source attributes.
        if polygon_id not in polygon_descriptions:
            continue

        attributes = generate_polygon_attributes(
            reference_df=reference_df,
            polygon_id=polygon_id,
            user_description=polygon_descriptions[polygon_id],
        )

        for key, value in attributes.items():
            gdf.loc[idx, key] = value

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".geojson":
        gdf.to_file(output_path, driver="GeoJSON")
    elif output_path.suffix.lower() == ".gpkg":
        gdf.to_file(output_path, driver="GPKG")
    else:
        gdf.to_file(output_path)

    return str(output_path)
