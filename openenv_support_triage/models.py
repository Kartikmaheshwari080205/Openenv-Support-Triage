from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionType(str, Enum):
    CLASSIFY = "classify"
    ASSIGN = "assign"
    DRAFT_REPLY = "draft_reply"
    CLOSE = "close"
    NOOP = "noop"


class TicketObservation(BaseModel):
    ticket_id: str
    customer: str
    subject: str
    body: str
    priority: int = Field(ge=1, le=5)
    sla_hours: int = Field(gt=0)
    predicted_category: Optional[str] = None
    assigned_team: Optional[str] = None
    reply_draft: Optional[str] = None
    closed: bool = False


class Observation(BaseModel):
    task_id: str
    task_name: str
    difficulty: Difficulty
    step_count: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    tickets: List[TicketObservation]
    recent_actions: List[str]
    guidance: str


class Action(BaseModel):
    action_type: ActionType
    ticket_id: Optional[str] = None
    value: Optional[str] = None
    message: Optional[str] = None


class Reward(BaseModel):
    total: float = Field(ge=-1.0, le=1.0)
    components: Dict[str, float]
    rationale: str


class TicketState(BaseModel):
    ticket_id: str
    customer: str
    subject: str
    body: str
    priority: int = Field(ge=1, le=5)
    sla_hours: int = Field(gt=0)
    true_category: str
    true_team: str
    required_keywords: List[str]
    forbidden_keywords: List[str]
    predicted_category: Optional[str] = None
    assigned_team: Optional[str] = None
    reply_draft: Optional[str] = None
    closed: bool = False


class EnvironmentState(BaseModel):
    task_id: str
    task_name: str
    difficulty: Difficulty
    step_count: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    done: bool = False
    tickets: Dict[str, TicketState]
    recent_actions: List[str]
    cumulative_reward: float = 0.0


class StepInfo(BaseModel):
    valid_action: bool
    grader_score: Optional[float] = None
    message: str
    reward_breakdown: Dict[str, float]
    action_trace: Dict[str, Any]


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: StepInfo


class TaskSummary(BaseModel):
    id: str
    task_id: str
    name: str
    difficulty: Difficulty
    description: str
    objective: str
    grader: Dict[str, str] = {
        "id": "support-ticket-grader",
        "endpoint": "/grader",
        "type": "deterministic",
    }


class TasksResponse(BaseModel):
    tasks: List[TaskSummary]
    action_schema: Dict[str, Any]


class BaselineTaskResult(BaseModel):
    task_id: str
    score: float
    steps: int
    done: bool


class BaselineResult(BaseModel):
    model: str
    seed_note: str
    average_score: float
    task_results: List[BaselineTaskResult]
