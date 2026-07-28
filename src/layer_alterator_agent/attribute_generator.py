from src.layer_alterator_agent.matcher import match_urban_type

from src.layer_alterator_agent.reference_loader import (
    get_predictor_values,
    validate_predictor_values,
)


def generate_polygon_attributes(
    reference_df,
    polygon_id,
    user_description,
    planning_typology=None,
    confidence=None,
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
        "user_request": user_description,
        "user_description": user_description,
        "target_planning_typology": planning_typology,
        "target_lcz_type": matched_urban_type,
        "matched_urban_type": matched_urban_type,
        "decision_confidence": confidence,
    }

    matched_row = reference_df[
        reference_df["Class Name"].astype(str).str.lower() == matched_urban_type.lower()
    ].iloc[0]
    if "LCZ" in reference_df.columns:
        result["target_lcz_code"] = matched_row["LCZ"]

    result.update(predictor_values)

    return result
