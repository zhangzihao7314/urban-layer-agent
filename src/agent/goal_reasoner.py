"""Data-backed goal interpretation and LCZ suitability scoring."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.agent.knowledge_base import PROJECT_ROOT
from src.layer_alterator_agent.reference_loader import add_missing_water_fraction


DEFAULT_PREDICTOR_TABLE = PROJECT_ROOT / "data/reference/ref_predictor_values_mean.csv"

GOAL_PROFILES = {
    "Urban heat increase": {
        "keywords": [
            "increase heat", "increase the city heat", "make the city hotter",
            "raise temperature", "higher temperature",
        ],
        "weights": {"F_TV": -0.38, "F_G": -0.22, "F_W": -0.18, "IMD": 0.22},
    },
    "Urban heat mitigation": {
        "keywords": ["heat", "hot", "cool", "temperature", "heat island"],
        "weights": {"F_TV": 0.38, "F_G": 0.22, "F_W": 0.18, "IMD": -0.22},
    },
    "Urban greening": {
        "keywords": ["green", "vegetation", "tree", "park", "biodiversity"],
        "weights": {"F_TV": 0.55, "F_G": 0.35, "IMD": -0.10},
    },
    "Blue infrastructure": {
        "keywords": ["water", "blue", "flood", "drainage", "pond"],
        "weights": {"F_W": 0.70, "F_G": 0.15, "F_TV": 0.15},
    },
    "Open morphology": {
        "keywords": ["open", "ventilation", "less dense", "air flow"],
        "weights": {"SVF": 0.35, "IMD": -0.30, "BSF": -0.20, "F_G": 0.15},
    },
}


@dataclass(frozen=True)
class GoalRecommendation:
    goal: str
    ranked_types: List[str]
    scores: Dict[str, float]
    explanation: str


def infer_goal(text: str) -> str:
    normalized = text.lower()
    heat_terms = ("heat", "temperature", "hot", "warm", "cool")
    increase_terms = ("increase", "raise", "higher", "hotter", "more heat")
    decrease_terms = ("decrease", "reduce", "lower", "cool", "mitigate", "less heat")
    if any(term in normalized for term in heat_terms):
        if any(
            phrase in normalized
            for phrase in (
                "do not increase", "don't increase", "avoid increasing",
                "prevent heat increase", "avoid higher temperature",
            )
        ):
            return "Urban heat mitigation"
        if any(
            phrase in normalized
            for phrase in (
                "do not decrease", "don't decrease", "avoid cooling",
                "prevent cooling",
            )
        ):
            return "Urban heat increase"
        if any(term in normalized for term in increase_terms):
            return "Urban heat increase"
        if any(term in normalized for term in decrease_terms):
            return "Urban heat mitigation"
    matches = [
        (sum(keyword in normalized for keyword in profile["keywords"]), name)
        for name, profile in GOAL_PROFILES.items()
    ]
    score, name = max(matches)
    return name if score else "Unclear"


def recommend_for_goal(
    text: str,
    predictor_table: str | Path = DEFAULT_PREDICTOR_TABLE,
    limit: int = 5,
) -> GoalRecommendation:
    goal = infer_goal(text)
    if goal == "Unclear":
        return GoalRecommendation(goal, [], {}, "No supported goal profile was detected.")
    frame = add_missing_water_fraction(pd.read_csv(predictor_table))
    weights = GOAL_PROFILES[goal]["weights"]
    scores = {}
    for _, row in frame.iterrows():
        scores[str(row["Class Name"])] = round(sum(float(row[key]) * weight for key, weight in weights.items()), 4)
    ranked = sorted(scores, key=scores.get, reverse=True)[:limit]
    drivers = ", ".join(f"{key} ({weight:+.2f})" for key, weight in weights.items())
    return GoalRecommendation(
        goal,
        ranked,
        scores,
        "Professor-provided predictor values were ranked with system-defined "
        f"heuristic weights: {drivers}.",
    )
