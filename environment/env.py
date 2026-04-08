"""
IncidentResponseEnv — Core OpenEnv Environment
Implements the full OpenEnv interface:
  reset()  → Observation
  step()   → (Observation, Reward, done: bool, info: dict)
  state()  → EpisodeState

Thread-safe episode management with trajectory tracking.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional, Tuple

from environment.data_generator import Scenario, sample_scenario
from environment.graders import grade
from environment.models import (
    Action, ActionType, EpisodeState, Observation, Reward,
    RewardBreakdown, RunbookSection, TaskType,
)

# ─────────────────────────────────────────────
# Task Configuration
# ─────────────────────────────────────────────

TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "alert_classification": {
        "max_steps":  5,
        "terminal_actions": {ActionType.CLASSIFY_ALERT},
        "available_actions": [
            "classify_alert (requires: severity, category, team)",
            "analyze_logs   (read more logs — increases context)",
            "escalate        (page human responder)",
        ],
        "progress_hint": "Examine the alert metrics and logs, then classify severity, category, and assign to a team.",
        "difficulty": "easy",
    },
    "root_cause_analysis": {
        "max_steps":  8,
        "terminal_actions": {ActionType.IDENTIFY_ROOT_CAUSE},
        "available_actions": [
            "analyze_logs          (expand log context)",
            "identify_root_cause   (requires: root_cause_component, root_cause_type, evidence, impact)",
            "escalate               (page human responder)",
        ],
        "progress_hint": "Identify what component failed, what type of failure it was, provide evidence from the logs, and assess impact.",
        "difficulty": "medium",
    },
    "runbook_generation": {
        "max_steps":  12,
        "terminal_actions": {ActionType.GENERATE_RUNBOOK, ActionType.RESOLVE},
        "available_actions": [
            "analyze_logs           (expand log context)",
            "identify_root_cause    (optional pre-step for RCA)",
            "generate_runbook       (requires: runbook with all sections filled)",
            "escalate               (page human responder)",
            "resolve                (mark incident resolved with notes)",
        ],
        "progress_hint": "Generate a complete runbook with: diagnosis_steps, remediation_steps, rollback_plan, escalation_criteria, prevention_measures, and commands.",
        "difficulty": "hard",
    },
}


# ─────────────────────────────────────────────
# IncidentResponseEnv
# ─────────────────────────────────────────────

class IncidentResponseEnv:
    """
    OpenEnv-compliant environment for SRE incident response.

    Usage:
        env = IncidentResponseEnv(task_type="alert_classification")
        obs = env.reset()
        action = Action(action_type=ActionType.CLASSIFY_ALERT, severity="P1", ...)
        obs, reward, done, info = env.step(action)
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        task_type: str = "alert_classification",
        seed: Optional[int] = None,
        debug: bool = False,
    ):
        if task_type not in TASK_CONFIG:
            raise ValueError(
                f"Unknown task_type '{task_type}'. Choose from: {list(TASK_CONFIG.keys())}"
            )
        self.task_type   = task_type
        self.seed        = seed
        self.debug       = debug
        self._config     = TASK_CONFIG[task_type]

        # Episode state (initialised by reset())
        self._scenario:  Optional[Scenario]     = None
        self._state:     Optional[EpisodeState]  = None
        self._done:      bool                    = False

    # ──────────────────────────────────────────
    # Public OpenEnv API
    # ──────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Begin a new episode. Returns the initial Observation."""
        effective_seed = seed if seed is not None else self.seed
        self._scenario = sample_scenario(self.task_type, seed=effective_seed)
        self._done     = False

        self._state = EpisodeState(
            episode_id      = str(uuid.uuid4()),
            task_type       = TaskType(self.task_type),
            scenario_id     = self._scenario.scenario_id,
            step            = 0,
            max_steps       = self._config["max_steps"],
            cumulative_reward = 0.0,
            done            = False,
            started_at      = time.time(),
            ground_truth    = self._scenario.ground_truth.__dict__ if self.debug else None,
        )

        return self._build_observation(
            progress_hint=self._config["progress_hint"]
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process one agent action.

        Returns:
            observation  : Next observation (Observation)
            reward       : Scored reward with breakdown (Reward)
            done         : True if episode ended
            info         : Diagnostic dict
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._state.step += 1
        step_num = self._state.step

        # ── Validate action ────────────────────────────────────────────
        validation_error = self._validate_action(action)
        if validation_error:
            reward = Reward(
                score=0.0,
                breakdown=RewardBreakdown(penalty_incomplete=0.05),
                feedback=f"Invalid action: {validation_error}",
                partial_progress=False,
            )
            obs = self._build_observation()
            self._record_step(action, reward)
            return obs, reward, False, {"step": step_num, "error": validation_error}

        # ── Handle non-terminal informational actions ──────────────────
        if action.action_type == ActionType.ANALYZE_LOGS:
            reward = Reward(
                score=0.05,
                breakdown=RewardBreakdown(completeness_bonus=0.05),
                feedback="Logs analysed. Additional log context has been added to your observation.",
                partial_progress=True,
            )
            obs = self._build_observation(
                extra_logs=True,
                progress_hint="Good — you've reviewed additional logs. Now make your primary decision.",
            )
            self._record_step(action, reward)
            return obs, reward, False, {"step": step_num, "action": "analyze_logs"}

        if action.action_type == ActionType.ESCALATE:
            reward = Reward(
                score=0.10,
                breakdown=RewardBreakdown(completeness_bonus=0.10),
                feedback="Escalation triggered. Human responder paged. Environment records this as partial progress.",
                partial_progress=True,
            )
            obs = self._build_observation(
                progress_hint="Escalation noted. Continue with your primary task action.",
            )
            self._record_step(action, reward)
            return obs, reward, False, {"step": step_num, "action": "escalate"}

        # ── Grade the terminal action ──────────────────────────────────
        reward = grade(self.task_type, action, self._scenario.ground_truth)

        # Apply step-count penalty (reward efficiency)
        step_fraction = step_num / self._config["max_steps"]
        if step_fraction > 0.8 and reward.score > 0.0:
            efficiency_penalty = (step_fraction - 0.8) * 0.10
            reward.score     = max(0.0, round(reward.score - efficiency_penalty, 4))
            reward.feedback  += f" [Step penalty applied: -{efficiency_penalty:.3f}]"

        # Terminal conditions
        terminal_action = action.action_type in self._config["terminal_actions"]
        timeout         = step_num >= self._config["max_steps"]
        self._done      = terminal_action or timeout

        if timeout and not terminal_action:
            reward.score    = max(0.0, reward.score - 0.10)
            reward.feedback += " [Timeout penalty: episode exceeded max_steps]"

        self._state.cumulative_reward += reward.score
        self._state.done = self._done
        self._record_step(action, reward)

        obs = self._build_observation(
            progress_hint=(
                "Episode complete." if self._done
                else f"Step {step_num}/{self._config['max_steps']} — continue."
            )
        )

        info = {
            "step":               step_num,
            "done":               self._done,
            "cumulative_reward":  self._state.cumulative_reward,
            "task_complete":      reward.task_complete,
            "scenario_id":        self._scenario.scenario_id,
            "episode_id":         self._state.episode_id,
        }

        return obs, reward, self._done, info

    def state(self) -> EpisodeState:
        """Return the full current episode state (for debugging / eval harness)."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _build_observation(
        self,
        extra_logs: bool = False,
        progress_hint: str = "",
    ) -> Observation:
        assert self._scenario is not None
        assert self._state    is not None

        alert = self._scenario.alert
        if extra_logs:
            # Reveal more log lines on analyze_logs action
            pass  # logs already present in full; real env could paginate

        return Observation(
            task_id          = self._state.episode_id,
            task_type        = TaskType(self.task_type),
            step             = self._state.step,
            max_steps        = self._config["max_steps"],
            alert            = alert,
            context          = self._scenario.context,
            available_actions= self._config["available_actions"],
            progress_hint    = progress_hint or self._config["progress_hint"],
            episode_history  = self._state.actions_taken[-5:],  # last 5 steps
        )

    def _validate_action(self, action: Action) -> Optional[str]:
        """Return error string if action is invalid, else None."""
        if action.action_type == ActionType.CLASSIFY_ALERT:
            if self.task_type != "alert_classification":
                return "classify_alert is only valid in alert_classification task."
        if action.action_type == ActionType.IDENTIFY_ROOT_CAUSE:
            if not action.root_cause_component:
                return "identify_root_cause requires root_cause_component."
            if not action.root_cause_type:
                return "identify_root_cause requires root_cause_type."
        if action.action_type == ActionType.GENERATE_RUNBOOK:
            if action.runbook is None:
                return "generate_runbook requires a filled runbook field."
        return None

    def _record_step(self, action: Action, reward: Reward) -> None:
        if self._state is None:
            return
        self._state.actions_taken.append({
            "step":        self._state.step,
            "action_type": action.action_type.value,
            "score":       reward.score,
            "feedback":    reward.feedback[:120],
        })
        self._state.rewards_history.append(reward.score)

    # ──────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────

    def __repr__(self) -> str:
        step = self._state.step if self._state else 0
        return (
            f"IncidentResponseEnv("
            f"task={self.task_type}, step={step}, done={self._done})"
        )


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def make_env(task_type: str = "alert_classification", seed: Optional[int] = None, debug: bool = False) -> IncidentResponseEnv:
    return IncidentResponseEnv(task_type=task_type, seed=seed, debug=debug)
