"""
IncidentResponseEnv — Task Graders
Deterministic, multi-dimensional scoring (0.0–1.0)
"""

from __future__ import annotations

import re
from typing import List, Optional

from environment.data_generator import ScenarioGroundTruth
from environment.models import (
    Action, Reward, RewardBreakdown,
    RunbookSection, SeverityLevel,
)

# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

_SEVERITY_ORDER = {
    SeverityLevel.P1: 0,
    SeverityLevel.P2: 1,
    SeverityLevel.P3: 2,
    SeverityLevel.P4: 3,
}

def _severity_distance(a, b):
    return abs(_SEVERITY_ORDER[a] - _SEVERITY_ORDER[b])


def _keyword_overlap(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    text = text.lower()
    return sum(1 for k in keywords if k.lower() in text) / len(keywords)


def _list_keyword_overlap(items: List[str], keywords: List[str]) -> float:
    if not items:
        return 0.0
    return _keyword_overlap(" ".join(items), keywords)


def _step_quality(steps: List[str], min_steps=2):
    if not steps:
        return 0.0

    valid = [s for s in steps if len(s.strip()) > 15]
    quantity = min(1.0, len(valid) / min_steps)

    specific = sum(
        1 for s in valid if any(k in s.lower() for k in
            ["kubectl","grep","curl","restart","rollback","verify","check"]
        )
    )
    specificity = specific / len(valid) if valid else 0.0

    return 0.5 * quantity + 0.5 * specificity


def _service_overlap(pred, actual):
    if not pred or not actual:
        return 0.0
    p, a = set(map(str.lower, pred)), set(map(str.lower, actual))
    return len(p & a) / len(p | a)


def _fuzzy(pred, actual):
    if not pred or not actual:
        return 0.0
    if pred.lower() == actual.lower():
        return 1.0
    if actual.lower() in pred.lower():
        return 0.8
    return 0.3


# ─────────────────────────────────────────────
# Task 1 — Alert Classification
# ─────────────────────────────────────────────

class AlertClassificationGrader:

    def grade(self, action: Action, gt: ScenarioGroundTruth) -> Reward:
        bd = RewardBreakdown()
        fb = []

        # Severity
        if action.severity:
            if action.severity == gt.severity:
                bd.severity_accuracy = 0.40
                fb.append("Severity correct")
            else:
                dist = _severity_distance(action.severity, gt.severity)
                bd.severity_accuracy = 0.20 if dist == 1 else 0.0

                if gt.severity == SeverityLevel.P1 and action.severity in [SeverityLevel.P3, SeverityLevel.P4]:
                    bd.penalty_misclass = 0.20
        else:
            fb.append("Missing severity")

        # Category
        if action.category == gt.category:
            bd.category_accuracy = 0.30
        else:
            fb.append("Category mismatch")

        # Team
        if action.team:
            bd.team_accuracy = 0.20 * _fuzzy(action.team, gt.team)

        # Completeness
        if action.severity and action.category and action.team:
            bd.completeness_bonus = 0.10

        score = bd.total

        return Reward(
            score=round(score, 4),
            breakdown=bd,
            feedback=" | ".join(fb),
            partial_progress=0.0 < score < 0.9,
            task_complete=score >= 0.85,
        )


# ─────────────────────────────────────────────
# Task 2 — Root Cause Analysis
# ─────────────────────────────────────────────

class RootCauseAnalysisGrader:

    def grade(self, action: Action, gt: ScenarioGroundTruth) -> Reward:
        bd = RewardBreakdown()
        fb = []

        # Component
        comp_score = _fuzzy(action.root_cause_component, gt.root_cause_component)
        bd.rca_component = comp_score * 0.25

        # Type
        type_score = _fuzzy(action.root_cause_type, gt.root_cause_type)
        bd.rca_type = type_score * 0.25

        # Evidence
        ev_score = _list_keyword_overlap(action.evidence or [], gt.evidence_keywords)
        bd.evidence_quality = ev_score * 0.25

        # Impact
        impact_score = _keyword_overlap(action.impact or "", gt.impact.split())
        bd.impact_accuracy = impact_score * 0.15

        # Services
        svc_score = _service_overlap(action.affected_services or [], gt.affected_services)
        bd.evidence_quality += svc_score * 0.10

        score = bd.total

        return Reward(
            score=round(score, 4),
            breakdown=bd,
            feedback="RCA evaluated",
            partial_progress=0.0 < score < 0.9,
            task_complete=score >= 0.75,
        )


# ─────────────────────────────────────────────
# Task 3 — Runbook Generation
# ─────────────────────────────────────────────

class RunbookGenerationGrader:

    def grade(self, action: Action, gt: ScenarioGroundTruth) -> Reward:
        bd = RewardBreakdown()

        if not action.runbook:
            return Reward(score=0.0, breakdown=bd, feedback="No runbook")

        rb: RunbookSection = action.runbook

        bd.runbook_diagnosis = _step_quality(rb.diagnosis_steps) * 0.20
        bd.runbook_remediation = _step_quality(rb.remediation_steps) * 0.25
        bd.runbook_rollback = _step_quality(rb.rollback_plan) * 0.15
        bd.runbook_escalation = _step_quality(rb.escalation_criteria) * 0.15
        bd.runbook_prevention = _step_quality(rb.prevention_measures) * 0.15

        if rb.commands:
            bd.completeness_bonus = min(1.0, len(rb.commands) / 3) * 0.10
        else:
            bd.penalty_incomplete = 0.05

        score = bd.total

        return Reward(
            score=round(score, 4),
            breakdown=bd,
            feedback="Runbook evaluated",
            partial_progress=0.0 < score < 0.85,
            task_complete=score >= 0.70,
        )


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

GRADERS = {
    "alert_classification": AlertClassificationGrader(),
    "root_cause_analysis": RootCauseAnalysisGrader(),
    "runbook_generation": RunbookGenerationGrader(),
}


def grade(task_type: str, action: Action, ground_truth: ScenarioGroundTruth) -> Reward:
    return GRADERS[task_type].grade(action, ground_truth)