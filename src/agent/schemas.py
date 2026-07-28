from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Intent(str, Enum):
    CHAT = "chat"
    SET_GOAL = "set_goal"
    APPLY_ALL = "apply_all"
    SET_POLYGON = "set_polygon"
    ASK_EXPLANATION = "ask_explanation"
    GENERATE = "generate"
    CONFIRM = "confirm"
    REVISE = "revise"
    UPLOAD_HELP = "upload_help"


class AgentStage(str, Enum):
    WAITING_FOR_VECTOR = "waiting_for_vector"
    WAITING_FOR_GOAL = "waiting_for_goal"
    WAITING_FOR_POLYGON = "waiting_for_polygon"
    WAITING_FOR_CLARIFICATION = "waiting_for_clarification"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    READY_TO_GENERATE = "ready_to_generate"
    INPUTS_GENERATED = "inputs_generated"
    RUNNING_LAYER_ALTERATOR = "running_layer_alterator"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class IntentDecision:
    intent: Intent
    polygon_id: Optional[int] = None
    target_description: Optional[str] = None
    confidence: float = 0.0


@dataclass
class TypologyProposal:
    polygon_id: int
    original_description: str
    candidate_types: List[str]
    recommended_type: Optional[str]
    confidence: float
    needs_clarification: bool
    clarification_question: Optional[str] = None
    predictor_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentState:
    stage: AgentStage = AgentStage.WAITING_FOR_VECTOR
    vector_path: Optional[str] = None
    id_column: Optional[str] = None
    polygon_ids: List[int] = field(default_factory=list)
    overall_goal: str = ""
    confirmed_decisions: Dict[int, str] = field(default_factory=dict)
    pending_proposal: Optional[TypologyProposal] = None
    outputs: Dict[str, str] = field(default_factory=dict)
    last_error: Optional[str] = None

