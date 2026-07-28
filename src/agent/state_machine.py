from src.agent.schemas import AgentStage, AgentState, TypologyProposal


def register_vector(state: AgentState, path: str, id_column: str, polygon_ids: list[int]) -> AgentState:
    state.vector_path = path
    state.id_column = id_column
    state.polygon_ids = polygon_ids
    state.confirmed_decisions.clear()
    state.unchanged_polygon_ids.clear()
    state.decision_snapshots.clear()
    state.pending_multi_plan.clear()
    state.pending_unchanged_polygon_ids.clear()
    state.active_requirement = None
    state.complete_proposals.clear()
    state.pending_proposal = None
    state.stage = AgentStage.WAITING_FOR_GOAL
    return state


def register_goal(state: AgentState, goal: str) -> AgentState:
    if not state.vector_path:
        raise ValueError("Upload a vector before setting the goal.")
    state.overall_goal = goal.strip()
    state.conversation_history.append({"role": "user", "content": goal.strip()})
    state.stage = AgentStage.WAITING_FOR_POLYGON
    return state


def set_proposal(state: AgentState, proposal: TypologyProposal) -> AgentState:
    if proposal.polygon_id not in state.polygon_ids:
        raise ValueError(f"Unknown polygon ID: {proposal.polygon_id}")
    state.pending_proposal = proposal
    if proposal.polygon_id in state.unchanged_polygon_ids:
        state.unchanged_polygon_ids.remove(proposal.polygon_id)
    state.tool_history.append({"tool": "propose_polygon_change", "polygon_id": proposal.polygon_id,
                               "candidate_types": proposal.candidate_types})
    state.stage = (
        AgentStage.WAITING_FOR_CLARIFICATION
        if proposal.needs_clarification
        else AgentStage.WAITING_FOR_CONFIRMATION
    )
    return state


def clarify_proposal(state: AgentState, selected_type: str, predictor_values: dict) -> AgentState:
    if not state.pending_proposal:
        raise ValueError("There is no proposal to clarify.")
    if selected_type not in state.pending_proposal.candidate_types:
        raise ValueError(f"Unsupported clarification choice: {selected_type}")
    state.pending_proposal.recommended_type = selected_type
    state.pending_proposal.predictor_values = predictor_values
    state.pending_proposal.needs_clarification = False
    state.stage = AgentStage.WAITING_FOR_CONFIRMATION
    return state


def confirm_proposal(state: AgentState) -> AgentState:
    proposal = state.pending_proposal
    if not proposal or not proposal.recommended_type:
        raise ValueError("There is no complete proposal to confirm.")
    state.decision_snapshots.append({
        "decisions": dict(state.confirmed_decisions),
        "unchanged": list(state.unchanged_polygon_ids),
    })
    state.confirmed_decisions[proposal.polygon_id] = proposal.recommended_type
    state.decision_records.append({
        "polygon_id": proposal.polygon_id,
        "original_request": proposal.original_description,
        "planning_typology": proposal.planning_typology,
        "candidate_evidence": proposal.candidate_evidence,
        "final_lcz_type": proposal.recommended_type,
        "predictor_values": proposal.predictor_values,
        "confirmed_by_user": True,
    })
    state.pending_proposal = None
    state.stage = (
        AgentStage.READY_TO_GENERATE
        if set(state.confirmed_decisions) | set(state.unchanged_polygon_ids) == set(state.polygon_ids)
        else AgentStage.WAITING_FOR_POLYGON
    )
    return state


def revise_proposal(state: AgentState) -> AgentState:
    state.pending_proposal = None
    state.stage = AgentStage.WAITING_FOR_POLYGON
    return state


def record_tool_call(state: AgentState, tool: str, arguments: dict, result_summary: str) -> AgentState:
    state.tool_history.append({"tool": tool, "arguments": arguments, "result": result_summary})
    return state


def add_constraint(state: AgentState, constraint: str) -> AgentState:
    if constraint not in state.constraints:
        state.constraints.append(constraint)
    return state


def confirm_multi_plan(state: AgentState) -> AgentState:
    if not state.pending_multi_plan:
        raise ValueError("There is no multi-polygon plan to confirm.")
    state.decision_snapshots.append({
        "decisions": dict(state.confirmed_decisions),
        "unchanged": list(state.unchanged_polygon_ids),
    })
    for polygon_id, proposal in state.pending_multi_plan.items():
        if proposal.recommended_type:
            state.confirmed_decisions[polygon_id] = proposal.recommended_type
            if polygon_id in state.unchanged_polygon_ids:
                state.unchanged_polygon_ids.remove(polygon_id)
    for polygon_id in state.pending_unchanged_polygon_ids:
        state.confirmed_decisions.pop(polygon_id, None)
        if polygon_id not in state.unchanged_polygon_ids:
            state.unchanged_polygon_ids.append(polygon_id)
    state.pending_multi_plan = {}
    state.pending_unchanged_polygon_ids = []
    state.stage = (
        AgentStage.READY_TO_GENERATE
        if set(state.confirmed_decisions) | set(state.unchanged_polygon_ids) == set(state.polygon_ids)
        else AgentStage.WAITING_FOR_POLYGON
    )
    return state


def undo_last_decision(state: AgentState) -> AgentState:
    if not state.decision_snapshots:
        raise ValueError("There is no confirmed decision to undo.")
    snapshot = state.decision_snapshots.pop()
    state.confirmed_decisions = dict(snapshot["decisions"])
    state.unchanged_polygon_ids = list(snapshot["unchanged"])
    state.pending_proposal = None
    state.pending_multi_plan = {}
    state.pending_unchanged_polygon_ids = []
    state.stage = (
        AgentStage.READY_TO_GENERATE
        if state.polygon_ids and set(state.confirmed_decisions) | set(state.unchanged_polygon_ids) == set(state.polygon_ids)
        else AgentStage.WAITING_FOR_POLYGON
    )
    return state
