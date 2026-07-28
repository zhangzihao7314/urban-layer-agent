import pandas as pd
import pytest

from src.agent.intent_router import route_intent
from src.agent.schemas import AgentStage, AgentState, Intent
from src.agent.state_machine import confirm_proposal, register_goal, register_vector, set_proposal
from src.agent.typology_resolver import propose_typology
from src.layer_alterator_agent.matcher import match_urban_type
from src.layer_alterator_agent.reference_loader import get_predictor_values, validate_predictor_values
from src.layer_alterator_agent.vector_writer import get_polygon_ids, resolve_id_column


def test_matcher_accepts_supported_descriptions():
    assert match_urban_type("create a pond") == "Water"
    assert match_urban_type("parking lot") == "Bare rock or paved"


def test_matcher_rejects_unknown_description():
    with pytest.raises(ValueError):
        match_urban_type("flying airport on mars")


def test_source_fid_is_preserved():
    frame = pd.DataFrame({"fid": [7, 8, 9]})
    assert resolve_id_column(frame) == "fid"
    assert get_polygon_ids(frame, "fid") == [7, 8, 9]


def test_duplicate_ids_are_rejected():
    frame = pd.DataFrame({"fid": [7, 7, 9]})
    with pytest.raises(ValueError, match="duplicate"):
        resolve_id_column(frame)


def test_intent_router():
    assert route_intent("Generate").intent == Intent.GENERATE
    decision = route_intent("Polygon 7 should become Water")
    assert decision.intent == Intent.SET_POLYGON
    assert decision.polygon_id == 7
    assert decision.target_description == "Water"


def test_ambiguous_park_requires_clarification():
    proposal = propose_typology(7, "create a park")
    assert proposal.needs_clarification
    assert proposal.recommended_type is None
    assert "Dense trees" in proposal.candidate_types


def test_state_machine_requires_confirmation():
    state = register_vector(AgentState(), "test.geojson", "fid", [7])
    register_goal(state, "reduce urban heat")
    proposal = propose_typology(7, "Dense trees")
    proposal.predictor_values = {"TCH": 0.5}
    set_proposal(state, proposal)
    assert state.stage == AgentStage.WAITING_FOR_CONFIRMATION
    confirm_proposal(state)
    assert state.confirmed_decisions == {7: "Dense trees"}
    assert state.stage == AgentStage.READY_TO_GENERATE


def test_reference_values_satisfy_c1(reference_table_path):
    frame = pd.read_csv(reference_table_path)
    frame["F_W"] = 1.0 - frame[["F_AC", "F_S", "F_M", "F_BS", "F_G", "F_TV"]].sum(axis=1)
    for urban_type in frame["Class Name"]:
        values = get_predictor_values(frame, urban_type)
        validate_predictor_values(values)


@pytest.fixture
def reference_table_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "data/reference/ref_predictor_values_mean.csv"

