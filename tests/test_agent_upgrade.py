from pathlib import Path

import pytest

from src.agent.context_parser import parse_planning_context
from src.agent.goal_reasoner import recommend_for_goal
from src.agent.knowledge_base import TypologyKnowledgeBase
from src.agent.scenario_planner import compare_scenarios
from src.agent.tools import UrbanLayerTools
from src.agent.orchestrator import UrbanLayerAgent
from src.agent.typology_resolver import propose_typology
from src.agent.schemas import AgentState
from src.agent.state_machine import set_proposal
from src.agent.state_machine import confirm_multi_plan, undo_last_decision
from src.agent.flexible_dialogue import (
    detect_conflicts,
    is_natural_confirmation,
    parse_flexible_message,
    resolve_natural_selection,
)
from src.agent.conversation_controller import ConversationController, explain_grounded_choice
from src.agent.proposal_engine import explain_scenario_comparison
from src.agent.intent_router import route_intent
from src.agent.schemas import Intent


ROOT = Path(__file__).resolve().parents[1]
TYPOLOGY = ROOT / "data/reference/LCZ_urban_types_match.xlsx"
PREDICTORS = ROOT / "data/reference/ref_predictor_values_mean.csv"


def test_revise_command_accepts_punctuation_and_cancel_wording():
    assert route_intent("Revise.").intent == Intent.REVISE
    assert route_intent("cancel please").intent == Intent.REVISE


def test_informational_question_is_not_a_goal_or_polygon_command():
    assert route_intent("What is Water?").intent == Intent.CHAT
    assert route_intent("How to increase the city heat").intent == Intent.CHAT


def test_professor_typology_workbook_is_loaded():
    kb = TypologyKnowledgeBase(TYPOLOGY)
    assert kb.aggregated_classes["Urban Green System"] == [101, 102, 104]
    assert len(kb.urban_types) == 11
    assert "Urban Park" in kb.planning_typologies


def test_urban_park_uses_primary_secondary_correspondence():
    candidates = TypologyKnowledgeBase(TYPOLOGY).candidates_for("Urban Park")
    marks = {item.urban_type: item.correspondence for item in candidates}
    assert marks["Scattered trees"] == "P"
    assert marks["Low Plants"] == "P"
    assert marks["Dense trees"] == "S"


def test_context_parser_separates_park_and_tree_preference():
    context = parse_planning_context("Polygon 7 should become a forest park")
    assert context.polygon_id == 7
    assert context.planning_typology == "Urban Forest"
    assert "trees" in context.preferences


def test_agent_tool_boosts_tree_candidate_for_wooded_park():
    tools = UrbanLayerTools(TYPOLOGY, PREDICTORS)
    ranked = tools.rank_lcz_candidates("Polygon 7 should become a wooded urban park")
    assert ranked[0]["urban_type"] in {"Dense trees", "Scattered trees"}
    assert ranked[0]["preference_boost"] > 0


def test_generic_park_requires_table_backed_clarification():
    proposal = propose_typology(7, "create a park")
    assert proposal.needs_clarification
    assert proposal.planning_typology == "Urban Park"
    assert proposal.candidate_evidence
    assert {"Scattered trees", "Low Plants"}.issubset(set(proposal.candidate_types))


def test_goal_reasoning_is_data_backed():
    result = recommend_for_goal("reduce urban heat by increasing vegetation", PREDICTORS)
    assert result.goal == "Urban heat mitigation"
    assert result.ranked_types[0] in {"Dense trees", "Water", "Scattered trees", "Low Plants"}
    assert result.scores["Dense trees"] > result.scores["Compact midrise"]


def test_heat_goal_respects_increase_and_decrease_direction():
    cooling = recommend_for_goal("I want to decrease the city heat", PREDICTORS)
    heating = recommend_for_goal("How to increase the city heat", PREDICTORS)
    assert cooling.goal == "Urban heat mitigation"
    assert heating.goal == "Urban heat increase"
    assert cooling.scores["Dense trees"] > cooling.scores["Bare rock or paved"]
    assert heating.scores["Dense trees"] < heating.scores["Bare rock or paved"]
    assert recommend_for_goal("Do not increase city heat", PREDICTORS).goal == "Urban heat mitigation"
    assert recommend_for_goal("Avoid cooling the city", PREDICTORS).goal == "Urban heat increase"


def test_heat_goal_supports_chinese_direction():
    cooling = recommend_for_goal("我希望降低城市热并保留休闲空间", PREDICTORS)
    heating = recommend_for_goal("让这个区域升温", PREDICTORS)
    assert cooling.goal == "Urban heat mitigation"
    assert heating.goal == "Urban heat increase"


def test_scenario_comparison_adds_feasibility_warnings():
    recommendation = recommend_for_goal("reduce urban heat", PREDICTORS)
    results = compare_scenarios({
        "all trees": {7: "Dense trees", 8: "Dense trees"},
        "blue green": {7: "Dense trees", 8: "Water"},
    }, recommendation)
    by_name = {item.name: item for item in results}
    assert any("spatial diversity" in warning for warning in by_name["all trees"].warnings)
    assert any("feasibility" in warning for warning in by_name["blue green"].warnings)


def test_unknown_planning_typology_is_rejected():
    kb = TypologyKnowledgeBase(TYPOLOGY)
    with pytest.raises(ValueError, match="Unknown planning typology"):
        kb.candidates_for("Airport on Mars")


def test_agent_selects_and_records_domain_tools():
    agent = UrbanLayerAgent(UrbanLayerTools(TYPOLOGY, PREDICTORS))
    run = agent.run("Polygon 7 should become a grass park")
    assert [call["tool"] for call in run.tool_calls] == [
        "search_planning_typology", "rank_lcz_candidates"
    ]
    assert run.result["ranked_candidates"][0]["urban_type"] == "Low Plants"
    assert run.needs_user_confirmation


def test_new_polygon_command_replaces_pending_clarification():
    state = AgentState()
    state.polygon_ids = [7]
    grass = propose_typology(7, "a grass park")
    set_proposal(state, grass)
    assert state.pending_proposal.candidate_types[0] == "Low Plants"

    wooded = propose_typology(7, "a wooded park")
    set_proposal(state, wooded)
    assert state.pending_proposal.candidate_types[0] == "Dense trees"


def test_multi_polygon_message_is_parsed():
    parsed = parse_flexible_message(
        "Make Polygon 7 a grass park, Polygon 8 a wooded park, and leave Polygon 9 unchanged."
    )
    assert [(item.polygon_id, item.action) for item in parsed.instructions] == [
        (7, "change"), (8, "change"), (9, "keep")
    ]
    assert parsed.instructions[1].description == "a wooded park"


def test_controller_builds_complete_multi_plan():
    state = AgentState(polygon_ids=[7, 8, 9])
    result = ConversationController().interpret(
        "Make Polygon 7 a grass park, Polygon 8 a wooded park, and leave Polygon 9 unchanged.",
        state,
    )
    assert result.kind == "multi_plan"
    assert result.proposals[7].recommended_type == "Low Plants"
    assert result.proposals[8].recommended_type == "Dense trees"
    assert result.keep_unchanged == [9]


def test_same_choice_reference_uses_previous_decision():
    state = AgentState(polygon_ids=[7, 8], confirmed_decisions={7: "Low Plants"})
    parsed = parse_flexible_message("Apply the same choice to Polygon 8")
    assert parsed.instructions[0].action == "copy_previous"


def test_natural_selection_and_confirmation():
    candidates = ["Low Plants", "Dense trees", "Water"]
    assert resolve_natural_selection("The second option is fine", candidates) == "Dense trees"
    assert is_natural_confirmation("Looks good")


def test_constraints_produce_grounded_warning():
    parsed = parse_flexible_message("Keep Polygon 7 accessible for pedestrians")
    warnings = detect_conflicts(parsed.instructions, parsed.constraints)
    assert any("cannot be verified" in item for item in warnings)


def test_explanation_separates_fact_reasoning_and_limitation():
    proposal = propose_typology(7, "a grass park")
    explanation = explain_grounded_choice(proposal)
    assert "Reference-table facts" in explanation
    assert "Agent reasoning" in explanation
    assert "Limitation" in explanation


def test_multi_confirmation_and_undo():
    state = AgentState(polygon_ids=[7, 8])
    state.pending_multi_plan = {
        7: propose_typology(7, "grass park"),
        8: propose_typology(8, "wooded park"),
    }
    confirm_multi_plan(state)
    assert state.confirmed_decisions == {7: "Low Plants", 8: "Dense trees"}
    undo_last_decision(state)
    assert state.confirmed_decisions == {}


def test_undo_restores_unchanged_state_together_with_decisions():
    state = AgentState(polygon_ids=[7, 8, 9], confirmed_decisions={7: "Scattered trees"})
    state.pending_multi_plan = {8: propose_typology(8, "grass park")}
    state.pending_unchanged_polygon_ids = [9]
    confirm_multi_plan(state)
    assert state.unchanged_polygon_ids == [9]
    undo_last_decision(state)
    assert state.confirmed_decisions == {7: "Scattered trees"}
    assert state.unchanged_polygon_ids == []


def test_exclusion_constraint_removes_water_candidate():
    proposal = propose_typology(7, "a park", ["exclude:Water"])
    assert "Water" not in proposal.candidate_types
