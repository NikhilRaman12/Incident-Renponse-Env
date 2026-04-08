from pydantic import BaseModel
from typing import List, Dict, Optional
from enum import Enum


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class SeverityLevel(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class AlertCategory(str, Enum):
    infrastructure = "infrastructure"
    application = "application"
    database = "database"
    network = "network"
    security = "security"


class ActionType(str, Enum):
    CLASSIFY_ALERT = "classify_alert"
    ANALYZE_LOGS = "analyze_logs"
    IDENTIFY_ROOT_CAUSE = "identify_root_cause"
    GENERATE_RUNBOOK = "generate_runbook"
    ESCALATE = "escalate"
    RESOLVE = "resolve"


class TaskType(str, Enum):
    alert_classification = "alert_classification"
    root_cause_analysis = "root_cause_analysis"
    runbook_generation = "runbook_generation"


# ─────────────────────────────────────────────
# Core Models
# ─────────────────────────────────────────────

class MetricSnapshot(BaseModel):
    name: str
    value: float
    timestamp: Optional[str] = None


class Alert(BaseModel):
    alert_id: str
    title: str
    description: str
    service: str
    environment: str
    timestamp: str
    metrics: Dict[str, float]
    logs: List[str]
    metric_details: Optional[List[MetricSnapshot]] = []
    raw_log_count: Optional[int] = 0
    firing_duration_seconds: Optional[int] = 0
    previous_incidents: Optional[List[str]] = []

class ServiceDependency(BaseModel):
    service: str
    depends_on: Optional[List[str]] = []
    status: Optional[str] = "unknown"

    # Allow extra fields like latency, error_rate
    class Config:
        extra = "allow"

class IncidentContext(BaseModel):
    service_graph: List[ServiceDependency]
    recent_deployments: List[Dict]
    runbook_hints: List[str]
    similar_past_incidents: List[str]
    on_call_teams: Dict[str, List[str]]


class Observation(BaseModel):
    task_id: str
    task_type: TaskType
    step: int
    max_steps: int
    alert: Alert
    context: IncidentContext
    available_actions: List[str]
    progress_hint: str
    episode_history: List[Dict]


# ─────────────────────────────────────────────
# Action + Runbook
# ─────────────────────────────────────────────

class RunbookSection(BaseModel):
    diagnosis_steps: List[str]
    remediation_steps: List[str]
    rollback_plan: List[str]
    escalation_criteria: List[str]
    prevention_measures: List[str]
    commands: List[str]


class Action(BaseModel):
    action_type: ActionType

    severity: Optional[SeverityLevel] = None
    category: Optional[AlertCategory] = None
    team: Optional[str] = None

    root_cause_component: Optional[str] = None
    root_cause_type: Optional[str] = None
    evidence: Optional[List[str]] = None
    impact: Optional[str] = None
    affected_services: Optional[List[str]] = None

    runbook: Optional[RunbookSection] = None

    notes: Optional[str] = None
    confidence: Optional[float] = None


# ─────────────────────────────────────────────
# Reward Models
# ─────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    severity_accuracy: float = 0.0
    category_accuracy: float = 0.0
    team_accuracy: float = 0.0

    rca_component: float = 0.0
    rca_type: float = 0.0
    evidence_quality: float = 0.0
    impact_accuracy: float = 0.0

    runbook_diagnosis: float = 0.0
    runbook_remediation: float = 0.0
    runbook_rollback: float = 0.0
    runbook_escalation: float = 0.0
    runbook_prevention: float = 0.0

    completeness_bonus: float = 0.0
    penalty_misclass: float = 0.0
    penalty_incomplete: float = 0.0

    @property
    def total(self) -> float:
        score = (
            self.severity_accuracy +
            self.category_accuracy +
            self.team_accuracy +
            self.rca_component +
            self.rca_type +
            self.evidence_quality +
            self.impact_accuracy +
            self.runbook_diagnosis +
            self.runbook_remediation +
            self.runbook_rollback +
            self.runbook_escalation +
            self.runbook_prevention +
            self.completeness_bonus
        )

        score -= (self.penalty_misclass + self.penalty_incomplete)

        return max(0.0, min(1.0, score))


class Reward(BaseModel):
    score: float
    breakdown: RewardBreakdown
    feedback: str
    partial_progress: bool = False
    task_complete: bool = False


# ─────────────────────────────────────────────
# Episode State
# ─────────────────────────────────────────────

class EpisodeState(BaseModel):
    episode_id: str
    task_type: TaskType
    scenario_id: str
    step: int
    max_steps: int
    cumulative_reward: float
    done: bool
    started_at: float
    actions_taken: List[Dict] = []
    rewards_history: List[float] = []
    ground_truth: Optional[Dict] = None