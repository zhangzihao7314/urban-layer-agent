"""Explicit tools available to the Urban Layer Agent."""

from dataclasses import asdict
from pathlib import Path
from typing import Dict

import geopandas as gpd

from src.agent.context_parser import parse_planning_context
from src.agent.goal_reasoner import recommend_for_goal
from src.agent.knowledge_base import TypologyKnowledgeBase, get_knowledge_base
from src.agent.scenario_planner import compare_scenarios
from src.layer_alterator_agent.reference_loader import get_predictor_values, load_reference_table


class UrbanLayerTools:
    def __init__(self, typology_table=None, predictor_table=None):
        self.kb = get_knowledge_base(str(typology_table)) if typology_table else get_knowledge_base()
        self.predictor_table = predictor_table
        self._predictors = load_reference_table(str(predictor_table)) if predictor_table else None

    def inspect_uploaded_vector(self, path: str, id_column: str | None = None) -> dict:
        frame = gpd.read_file(path)
        chosen = id_column or ("fid" if "fid" in frame.columns else frame.columns[0])
        return {"path": path, "id_column": chosen, "polygon_ids": frame[chosen].astype(int).tolist(),
                "feature_count": len(frame), "crs": str(frame.crs)}

    def search_planning_typology(self, text: str) -> dict:
        context = parse_planning_context(text, self.kb)
        candidates = self.kb.candidates_for(context.planning_typology) if context.planning_typology else []
        return {"context": asdict(context), "candidates": [asdict(item) for item in candidates]}

    def rank_lcz_candidates(self, text: str) -> list[dict]:
        result = self.search_planning_typology(text)
        preferences = result["context"]["preferences"]
        boosts = {"trees": {"Dense trees": 2.0, "Scattered trees": 0.75},
                  "grass": {"Low Plants": 1.5}, "water": {"Water": 2.0},
                  "open": {"Open midrise": 1.0, "Open low-rise": 1.0},
                  "paved": {"Bare rock or paved": 1.5}}
        for candidate in result["candidates"]:
            candidate["base_score"] = candidate["score"]
            candidate["preference_boost"] = sum(boosts.get(p, {}).get(candidate["urban_type"], 0) for p in preferences)
            candidate["score"] += candidate["preference_boost"]
        return sorted(result["candidates"], key=lambda item: (-item["score"], item["lcz_code"]))

    def get_predictor_values(self, urban_type: str) -> dict:
        if self._predictors is None:
            raise ValueError("A predictor reference table is required for this tool.")
        return get_predictor_values(self._predictors, urban_type)

    def recommend_goal_types(self, goal: str) -> dict:
        recommendation = recommend_for_goal(goal, self.predictor_table) if self.predictor_table else recommend_for_goal(goal)
        return asdict(recommendation)

    def compare_scenarios(self, goal: str, scenarios: Dict[str, Dict[int, str]]) -> list[dict]:
        recommendation = recommend_for_goal(goal, self.predictor_table) if self.predictor_table else recommend_for_goal(goal)
        return [asdict(item) for item in compare_scenarios(scenarios, recommendation)]
