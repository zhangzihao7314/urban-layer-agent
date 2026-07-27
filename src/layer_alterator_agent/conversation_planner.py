def recommend_urban_types(user_goal: str):
    text = user_goal.lower()

    if any(k in text for k in ["heat", "hot", "cool", "temperature", "热岛", "降温", "更凉快"]):
        return {
            "goal": "Urban heat mitigation",
            "recommended_types": ["Dense trees", "Low Plants", "Open low-rise"],
            "explanation": (
                "To reduce urban heat, the system recommends increasing vegetation "
                "and reducing impervious or dense built-up surfaces."
            )
        }

    if any(k in text for k in ["green", "vegetation", "tree", "park", "绿化", "树", "公园"]):
        return {
            "goal": "Urban greening",
            "recommended_types": ["Dense trees", "Low Plants", "Park area"],
            "explanation": (
                "The user intention is related to urban greening, so vegetation-based "
                "urban types are recommended."
            )
        }

    if any(k in text for k in ["parking", "car", "road", "停车", "道路"]):
        return {
            "goal": "Transport or paved surface",
            "recommended_types": ["Parking lot", "Open low-rise"],
            "explanation": (
                "The user intention suggests a paved or transport-related transformation."
            )
        }

    return {
        "goal": "Unclear",
        "recommended_types": [],
        "explanation": (
            "The system could not confidently identify the transformation goal. "
            "Please describe whether you want greening, cooling, densification, water, or parking."
        )
    }