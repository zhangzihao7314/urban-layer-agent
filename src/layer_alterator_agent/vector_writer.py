from pathlib import Path
import geopandas as gpd

from src.layer_alterator_agent.attribute_generator import generate_polygon_attributes


def load_vector(vector_path: str) -> gpd.GeoDataFrame:
    """
    Load vector polygon file.
    Supported by geopandas: GeoJSON, GPKG, SHP, etc.
    """
    return gpd.read_file(vector_path)


def get_polygon_ids(gdf: gpd.GeoDataFrame, id_column: str = "polygon_id") -> list:
    """
    Get polygon IDs from vector.
    If polygon_id does not exist, create IDs from row index.
    """
    if id_column not in gdf.columns:
        gdf[id_column] = range(1, len(gdf) + 1)

    return gdf[id_column].tolist()


def write_attributes_to_vector(
    vector_path: str,
    reference_df,
    polygon_descriptions: dict,
    output_path: str,
    id_column: str = "polygon_id",
) -> str:
    """
    Write generated Layer Alterator attributes back to vector.

    polygon_descriptions example:
    {
        1: "park area",
        2: "parking lot"
    }
    """
    gdf = load_vector(vector_path)

    if id_column not in gdf.columns:
        gdf[id_column] = range(1, len(gdf) + 1)

    for idx, row in gdf.iterrows():
        polygon_id = row[id_column]

        if polygon_id not in polygon_descriptions:
            raise ValueError(
                f"Missing user description for polygon_id={polygon_id}"
            )

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