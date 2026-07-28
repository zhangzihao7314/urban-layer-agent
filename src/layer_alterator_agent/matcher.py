from difflib import get_close_matches


SUPPORTED_URBAN_TYPES = [
    "Compact midrise",
    "Compact low-rise",
    "Open midrise",
    "Open low-rise",
    "Large low-rise",
    "Dense trees",
    "Scattered trees",
    "Low Plants",
    "Bare rock or paved",
    "Bare Soil or sand",
    "Water",
]


URBAN_TYPE_ALIASES = {
    "compact midrise": "Compact midrise",
    "compact middle rise": "Compact midrise",

    "compact low-rise": "Compact low-rise",
    "compact low rise": "Compact low-rise",

    "open midrise": "Open midrise",
    "open middle rise": "Open midrise",

    "open low-rise": "Open low-rise",
    "open low rise": "Open low-rise",

    "large low-rise": "Large low-rise",
    "large low rise": "Large low-rise",
    "industrial": "Large low-rise",
    "commercial": "Large low-rise",
    "warehouse": "Large low-rise",

    "dense trees": "Dense trees",
    "forest": "Dense trees",
    "many trees": "Dense trees",
    "tree area": "Dense trees",
    # Generic park language is resolved from the professor's correspondence
    # table because several LCZ types can legitimately represent a park.

    "scattered trees": "Scattered trees",
    "sparse trees": "Scattered trees",

    "low plants": "Low Plants",
    "grass": "Low Plants",
    "grassland": "Low Plants",
    "lawn": "Low Plants",

    "bare rock or paved": "Bare rock or paved",
    "paved": "Bare rock or paved",
    "paved area": "Bare rock or paved",
    "parking": "Bare rock or paved",
    "parking lot": "Bare rock or paved",
    "road": "Bare rock or paved",

    "bare soil": "Bare Soil or sand",
    "sand": "Bare Soil or sand",
    "soil": "Bare Soil or sand",

    "water": "Water",
    "pond": "Water",
    "lake": "Water",
    "river": "Water",
}


def match_urban_type(user_text: str) -> str:
    """
    Match user natural language input to one supported LCZ urban type.
    Longer aliases are checked first to avoid:
    'parking lot' being matched as 'park'.
    """
    text = user_text.lower().strip()

    if not text:
        raise ValueError("Empty urban type description.")

    sorted_aliases = sorted(
        URBAN_TYPE_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for alias, urban_type in sorted_aliases:
        if alias in text:
            return urban_type

    close_matches = get_close_matches(
        text,
        [item.lower() for item in SUPPORTED_URBAN_TYPES],
        n=1,
        cutoff=0.65,
    )

    if close_matches:
        matched_lower = close_matches[0]
        for urban_type in SUPPORTED_URBAN_TYPES:
            if urban_type.lower() == matched_lower:
                return urban_type

    raise ValueError(
        f"Cannot match '{user_text}' to supported urban typologies. "
        f"Supported types: {SUPPORTED_URBAN_TYPES}"
    )
