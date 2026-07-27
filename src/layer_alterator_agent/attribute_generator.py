from src.layer_alterator_agent.matcher import match_urban_type

from src.layer_alterator_agent.reference_loader import (
    get_predictor_values,
    validate_predictor_values,
)


def generate_polygon_attributes(
    reference_df,
    polygon_id,
    user_description,
):
    """
    Generate Layer Alterator attributes for one polygon.

    Workflow:
    user description
        -> matched urban type
        -> predictor values
        -> validated attributes
    """

    matched_urban_type = match_urban_type(user_description)

    predictor_values = get_predictor_values(
        reference_df,
        matched_urban_type,
    )

    validate_predictor_values(predictor_values)

    result = {
        "polygon_id": polygon_id,
        "user_description": user_description,
        "matched_urban_type": matched_urban_type,
    }

    result.update(predictor_values)

    return result