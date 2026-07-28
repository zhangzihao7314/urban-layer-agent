"""Compatibility wrapper for data-driven goal recommendations."""

from src.agent.goal_reasoner import recommend_for_goal


def recommend_urban_types(user_goal: str):
    recommendation = recommend_for_goal(user_goal)
    return {
        "goal": recommendation.goal,
        "recommended_types": recommendation.ranked_types,
        "scores": recommendation.scores,
        "explanation": recommendation.explanation,
    }
