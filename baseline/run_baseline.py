"""
IncidentResponseEnv — Baseline Inference Script
Runs an LLM agent (gpt-4o-mini) against all 3 tasks and prints
reproducible baseline scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline/run_baseline.py

    # Custom model / seed:
    python baseline/run_baseline.py --model gpt-4o --seed 42 --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from environment.env import make_env
from environment.models import Action, ActionType, AlertCategory, RunbookSection, SeverityLevel

# ─────────────────────────────────────────────
# LLM Agent
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Site Reliability Engineer (SRE) performing incident response.
You will receive production alert data and must respond with a precise JSON action.

RULES:
1. Respond ONLY with valid JSON — no markdown, no explanation outside JSON.
2. Always include "action_type" at the top level.
3. For alert_classification: include severity (P1/P2/P3/P4), category, team, notes.
4. For root_cause_analysis: include root_cause_component, root_cause_type, evidence (list), impact, affected_services (list).
5. For runbook_generation: include a full "runbook" object with:
   diagnosis_steps, remediation_steps, rollback_plan, escalation_criteria, prevention_measures, commands
   (each is a list of strings).
6. Be specific and technical. Use real CLI commands. Name exact services.
"""

TASK_INSTRUCTIONS = {
    "alert_classification": (
        "TASK: Classify this alert.\n"
        "Return JSON with: action_type='classify_alert', severity (P1/P2/P3/P4), "
        "category (infrastructure/application/database/network/security), team, notes."
    ),
    "root_cause_analysis": (
        "TASK: Identify the root cause.\n"
        "Return JSON with: action_type='identify_root_cause', root_cause_component, "
        "root_cause_type, evidence (list of 3-5 log/metric observations), impact, "
        "affected_services (list), blast_radius, notes."
    ),
    "runbook_generation": (
        "TASK: Generate a complete incident runbook.\n"
        "Return JSON with: action_type='generate_runbook', runbook: {"
        "  diagnosis_steps: [...], remediation_steps: [...], rollback_plan: [...], "
        "  escalation_criteria: [...], prevention_measures: [...], commands: [...], "
        "  expected_resolution_time: '...'}"
    ),
}


class LLMAgent:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set.\n"
                "Set it with: export OPENAI_API_KEY=sk-..."
            )
        self.client      = OpenAI(api_key=api_key)
        self.model       = model
        self.temperature = temperature

    def act(self, task_type: str, observation_text: str) -> Tuple[Action, str]:
        """Call LLM and parse response into an Action."""
        task_instruction = TASK_INSTRUCTIONS[task_type]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{task_instruction}\n\n{observation_text}",
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = messages,
                temperature = self.temperature,
                max_tokens  = 2048,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

        raw_text = response.choices[0].message.content.strip()

        # Parse JSON → Action
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw_text[:200]}") from e

        action = _parse_action(data, task_type)
        return action, raw_text


def _parse_action(data: Dict[str, Any], task_type: str) -> Action:
    """Safely parse LLM JSON output into a typed Action."""
    action_type_str = data.get("action_type", "")

    # Map task type → expected terminal action
    default_action = {
        "alert_classification": "classify_alert",
        "root_cause_analysis":  "identify_root_cause",
        "runbook_generation":   "generate_runbook",
    }.get(task_type, action_type_str)

    try:
        action_type = ActionType(action_type_str or default_action)
    except ValueError:
        action_type = ActionType(default_action)

    # Severity
    severity = None
    if sev_raw := data.get("severity"):
        try:
            severity = SeverityLevel(str(sev_raw).upper())
        except ValueError:
            pass

    # Category
    category = None
    if cat_raw := data.get("category"):
        try:
            category = AlertCategory(str(cat_raw).lower())
        except ValueError:
            pass

    # Runbook
    runbook = None
    if rb_data := data.get("runbook"):
        if isinstance(rb_data, dict):
            runbook = RunbookSection(
                diagnosis_steps     = _to_list(rb_data.get("diagnosis_steps",     [])),
                remediation_steps   = _to_list(rb_data.get("remediation_steps",   [])),
                rollback_plan       = _to_list(rb_data.get("rollback_plan",       [])),
                escalation_criteria = _to_list(rb_data.get("escalation_criteria", [])),
                prevention_measures = _to_list(rb_data.get("prevention_measures", [])),
                commands            = _to_list(rb_data.get("commands",            [])),
                expected_resolution_time = rb_data.get("expected_resolution_time"),
            )

    return Action(
        action_type           = action_type,
        severity              = severity,
        category              = category,
        team                  = data.get("team"),
        root_cause_component  = data.get("root_cause_component"),
        root_cause_type       = data.get("root_cause_type"),
        evidence              = _to_list(data.get("evidence",          [])) or None,
        impact                = data.get("impact"),
        affected_services     = _to_list(data.get("affected_services", [])) or None,
        blast_radius          = data.get("blast_radius"),
        runbook               = runbook,
        notes                 = data.get("notes"),
    )


def _to_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x) for x in v if x]
    if isinstance(v, str) and v:
        return [v]
    return []


# ─────────────────────────────────────────────
# Evaluation Harness
# ─────────────────────────────────────────────

TASKS = ["alert_classification", "root_cause_analysis", "runbook_generation"]
TASK_DIFFICULTIES = {"alert_classification": "Easy", "root_cause_analysis": "Medium", "runbook_generation": "Hard"}


def run_task(
    agent: LLMAgent,
    task_type: str,
    seed: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    env = make_env(task_type=task_type, seed=seed)
    obs = env.reset(seed=seed)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Task : {task_type} [{TASK_DIFFICULTIES[task_type]}]")
        print(f"  Seed : {seed}")
        print(f"  Scenario: {env._scenario.scenario_id}")
        print(f"{'─'*60}")

    total_reward  = 0.0
    steps_taken   = 0
    episode_done  = False
    step_log: List[Dict] = []

    while not episode_done:
        prompt = obs.to_agent_prompt()
        steps_taken += 1

        try:
            action, raw_json = agent.act(task_type, prompt)
        except Exception as e:
            print(f"  [step {steps_taken}] Agent error: {e}")
            break

        obs, reward, episode_done, info = env.step(action)
        total_reward += reward.score

        step_log.append({
            "step":        steps_taken,
            "action_type": action.action_type.value,
            "score":       reward.score,
            "feedback":    reward.feedback[:200],
        })

        if verbose:
            print(f"  Step {steps_taken}: action={action.action_type.value}  score={reward.score:.4f}")
            print(f"    Feedback: {reward.feedback[:150]}")
            if reward.breakdown:
                bd = reward.breakdown
                nonzero = {k: round(v, 3) for k, v in bd.model_dump().items() if v != 0.0}
                if nonzero:
                    print(f"    Breakdown: {nonzero}")

        # Rate limit courtesy
        time.sleep(0.3)

    final_score = round(min(1.0, total_reward), 4)
    if verbose:
        print(f"\n  ✓ Final score: {final_score}")

    return {
        "task_type":    task_type,
        "difficulty":   TASK_DIFFICULTIES[task_type],
        "seed":         seed,
        "final_score":  final_score,
        "steps_taken":  steps_taken,
        "step_log":     step_log,
    }


def run_full_eval(
    model:   str = "gpt-4o-mini",
    seed:    int = 42,
    runs:    int = 1,
    verbose: bool = True,
    output:  Optional[str] = None,
) -> Dict[str, Any]:
    agent  = LLMAgent(model=model)
    results: Dict[str, List[float]] = {t: [] for t in TASKS}
    run_details: List[Dict] = []

    print(f"\n{'═'*60}")
    print(f"  IncidentResponseEnv — Baseline Evaluation")
    print(f"  Model : {model}")
    print(f"  Seed  : {seed}  |  Runs : {runs}")
    print(f"{'═'*60}")

    for run_idx in range(runs):
        run_seed = seed + run_idx
        for task_type in TASKS:
            result = run_task(agent, task_type, seed=run_seed, verbose=verbose)
            results[task_type].append(result["final_score"])
            run_details.append({**result, "run": run_idx + 1})

    # Aggregate
    summary = {}
    overall_scores = []
    for task_type in TASKS:
        scores = results[task_type]
        avg    = round(sum(scores) / len(scores), 4)
        summary[task_type] = {"avg": avg, "scores": scores, "difficulty": TASK_DIFFICULTIES[task_type]}
        overall_scores.extend(scores)

    overall_avg = round(sum(overall_scores) / len(overall_scores), 4)
    summary["overall"] = {"avg": overall_avg}

    print(f"\n{'═'*60}")
    print("  RESULTS SUMMARY")
    print(f"{'═'*60}")
    for task_type, s in summary.items():
        if task_type == "overall":
            continue
        bar_len = int(s["avg"] * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {s['difficulty']:7s} {task_type:<25s} {bar}  {s['avg']:.4f}")
    print(f"{'─'*60}")
    print(f"  {'Overall':>34s}                         {overall_avg:.4f}")
    print(f"{'═'*60}\n")

    full_results = {
        "model":       model,
        "seed":        seed,
        "runs":        runs,
        "summary":     summary,
        "run_details": run_details,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(full_results, f, indent=2)
        print(f"Results saved to: {output}")

    return full_results


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IncidentResponseEnv Baseline Evaluation")
    parser.add_argument("--model",   default="gpt-4o-mini",  help="OpenAI model name")
    parser.add_argument("--seed",    type=int, default=42,    help="Random seed for reproducibility")
    parser.add_argument("--runs",    type=int, default=1,     help="Number of evaluation runs per task")
    parser.add_argument("--quiet",   action="store_true",     help="Suppress per-step output")
    parser.add_argument("--output",  default="baseline/results/baseline_scores.json",
                        help="Path to save JSON results")
    args = parser.parse_args()

    run_full_eval(
        model   = args.model,
        seed    = args.seed,
        runs    = args.runs,
        verbose = not args.quiet,
        output  = args.output,
    )
