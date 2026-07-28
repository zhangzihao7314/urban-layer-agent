from src.agent.schemas import AgentStage, AgentState, TypologyProposal


def register_vector(state: AgentState, path: str, id_column: str, polygon_ids: list[int]) -> AgentState:
    state.vector_path = path
    state.id_column = id_column
    state.polygon_ids = polygon_ids
    state.confirmed_decisions.clear()
    state.pending_proposal = None
    state.stage = AgentStage.WAITING_FOR_GOAL
    return state


def register_goal(state: AgentState, goal: str) -> AgentState:
    if not state.vector_path:
        raise ValueError("Upload a vector before setting the goal.")
    state.overall_goal = goal.strip()
    state.stage = AgentStage.WAITING_FOR_POLYGON
    return state


def set_proposal(state: AgentState, proposal: TypologyProposal) -> AgentState:
    if proposal.polygon_id not in state.polygon_ids:
        raise ValueError(f"Unknown polygon ID: {proposal.polygon_id}")
    state.pending_proposal = proposal
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
    state.confirmed_decisions[proposal.polygon_id] = proposal.recommended_type
    state.pending_proposal = None
    state.stage = (
        AgentStage.READY_TO_GENERATE
        if set(state.confirmed_decisions) == set(state.polygon_ids)
        else AgentStage.WAITING_FOR_POLYGON
    )
    return state


def revise_proposal(state: AgentState) -> AgentState:
    state.pending_proposal = None
    state.stage = AgentStage.WAITING_FOR_POLYGON
    return state

