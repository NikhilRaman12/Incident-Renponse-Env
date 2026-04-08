# 🚨 IncidentResponseEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0-blue?style=flat-square)](https://openenv.ai)
[![HF Spaces](https://img.shields.io/badge/🤗%20Spaces-Live%20Demo-orange?style=flat-square)](https://huggingface.co/spaces/openenv/incident-response-env)
[![Tests](https://img.shields.io/badge/tests-15%20passing-green?style=flat-square)](#testing)
[![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)](https://python.org)

**A production-grade OpenEnv environment simulating real-world Site Reliability Engineering (SRE) incident response.** AI agents must triage production alerts, perform root cause analysis from logs and metrics, and generate actionable runbooks — the exact workflow used by on-call engineers at every major tech company.

---

## Why This Domain?

Every software company running production systems faces the same 3 A.M. nightmare: an alert fires, metrics are spiking, and a human engineer has minutes to diagnose and resolve before users are impacted. This is one of the most **high-stakes**, **knowledge-intensive**, **real-world tasks** that AI could meaningfully automate or assist with. Unlike toy environments, every action in IncidentResponseEnv has the same consequence structure as real incidents:

- Misclassifying a P1 as P4 = missed critical incident
- Wrong root cause analysis = wrong fix, more downtime
- Incomplete runbook = engineers flying blind during chaos

---

## 🗂️ Environment Structure

```
incident-response-env/
├── openenv.yaml              # OpenEnv spec metadata
├── Dockerfile                # Container deployment
├── requirements.txt
├── app.py                    # FastAPI app (HF Spaces + REST API)
├── environment/
│   ├── __init__.py
│   ├── models.py             # Typed Pydantic v2 models
│   ├── env.py                # Core OpenEnv: step/reset/state
│   ├── graders.py            # Deterministic graders (all 3 tasks)
│   └── data_generator.py     # Realistic scenario synthesis
├── baseline/
│   └── run_baseline.py       # OpenAI API baseline evaluation
└── tests/
    └── test_env.py           # 15 validation tests
```

---

## 🔌 OpenEnv API

### `reset(seed?) → Observation`

Starts a new episode. Returns the initial observation containing the live incident alert, metrics, logs, and service context.

```python
from environment.env import make_env

env = make_env(task_type="alert_classification", seed=42)
obs = env.reset()

print(obs.alert.title)    # "CRITICAL: Database connection pool exhausted"
print(obs.alert.metrics)  # {"db_connection_pool_utilization_pct": 100.0, ...}
print(obs.alert.logs)     # ["ERROR: sorry, too many clients already", ...]
print(obs.to_agent_prompt())  # Full formatted prompt for LLM
```

### `step(action) → (Observation, Reward, done, info)`

Processes one agent action. Returns the next observation, a scored Reward with breakdown, a done flag, and diagnostic info.

```python
from environment.models import Action, ActionType, SeverityLevel, AlertCategory

action = Action(
    action_type = ActionType.CLASSIFY_ALERT,
    severity    = SeverityLevel.P1,
    category    = AlertCategory.DATABASE,
    team        = "database-team",
    notes       = "DB connection pool at 100%, scheduler bulk job is root cause",
)

obs, reward, done, info = env.step(action)
print(reward.score)          # 0.9200
print(reward.breakdown)      # RewardBreakdown(severity_accuracy=0.4, ...)
print(reward.feedback)       # "✓ Severity P1 is correct. ✓ Category 'database' is correct..."
```

### `state() → EpisodeState`

Returns the full episode state for the evaluation harness.

```python
state = env.state()
print(state.step)                # 1
print(state.cumulative_reward)   # 0.92
print(state.actions_taken)       # [{"step": 1, "action_type": "classify_alert", ...}]
```

---

## 📋 Tasks

### Task 1 — Alert Classification `[Easy]` 🟢

**What the agent must do:** Given a live production alert with metrics and log snippets, classify:
- **Severity**: P1 (Critical) / P2 (High) / P3 (Medium) / P4 (Low)
- **Category**: infrastructure / application / database / network / security
- **Team**: which on-call team to page

**Why it's realistic:** This is exactly the first 90 seconds of incident response. The on-call engineer gets woken up, sees an alert, and must immediately decide: "How bad is this? Who needs to know?"

**Reward breakdown:**

| Component | Weight | Description |
|-----------|--------|-------------|
| `severity_accuracy` | 0.40 | Exact match = 1.0; off-by-one = 0.5; far off = 0.0 |
| `category_accuracy` | 0.30 | Exact match = 1.0; wrong = 0.0 |
| `team_accuracy` | 0.20 | Fuzzy match on team name |
| `completeness_bonus` | 0.10 | All fields + notes filled |
| `penalty_misclass` | −0.20 | P1 classified as P3/P4 (critical under-triage) |

**Max steps:** 5

---

### Task 2 — Root Cause Analysis `[Medium]` 🟡

**What the agent must do:** Given the classified alert plus expanded logs, metrics timeseries, and service dependency graph, identify:
- **Root cause component** (which service/component failed)
- **Root cause type** (connection_pool_exhaustion, memory_leak, etc.)
- **Evidence** (3-5 specific log/metric observations supporting the diagnosis)
- **Impact** (blast radius and user-facing consequences)

**Why it's realistic:** The investigation phase requires correlating multiple signals — a deployment timestamp, a suspicious log line, a metric crossing a threshold — and synthesizing them into a causal explanation.

**Reward breakdown:**

| Component | Weight | Description |
|-----------|--------|-------------|
| `rca_component` | 0.25 | Fuzzy match on root cause component name |
| `rca_type` | 0.25 | Fuzzy match on failure classification |
| `evidence_quality` | 0.25 | Keyword overlap with ground-truth evidence |
| `impact_accuracy` | 0.15 | Coverage of impact keywords |
| `service_overlap` | 0.10 | Jaccard similarity with affected services list |

**Max steps:** 8

---

### Task 3 — Runbook Generation `[Hard]` 🔴

**What the agent must do:** Produce a complete, actionable incident runbook with all 6 required sections:
1. **Diagnosis steps** — how to confirm the root cause
2. **Remediation steps** — how to fix it right now
3. **Rollback plan** — how to undo the fix if it makes things worse
4. **Escalation criteria** — when and who to page next
5. **Prevention measures** — how to prevent recurrence
6. **Commands** — exact CLI commands to run

**Why it's realistic:** A good runbook is the difference between a 5-minute fix and a 2-hour outage. It requires deep technical knowledge, operational experience, and the ability to think about the problem from multiple angles simultaneously.

**Reward breakdown:**

| Component | Weight | Description |
|-----------|--------|-------------|
| `runbook_diagnosis` | 0.20 | Step quality × evidence relevance |
| `runbook_remediation` | 0.25 | Step quality × command keyword coverage |
| `runbook_rollback` | 0.15 | Has rollback + uses rollback keywords |
| `runbook_escalation` | 0.15 | Has criteria + uses escalation keywords |
| `runbook_prevention` | 0.15 | Has measures + uses prevention keywords |
| `commands` | 0.10 | Real CLI commands present |
| `penalty_incomplete` | −0.05 | Missing rollback plan |

**Max steps:** 12

---

## 📐 Action Space

```python
class Action(BaseModel):
    action_type: ActionType                    # REQUIRED — see enum below

    # Task 1: Alert Classification
    severity:             Optional[SeverityLevel]    # P1 / P2 / P3 / P4
    category:             Optional[AlertCategory]    # infrastructure/application/database/network/security
    team:                 Optional[str]              # on-call team name

    # Task 2: Root Cause Analysis
    root_cause_component: Optional[str]              # e.g. "scheduler-service"
    root_cause_type:      Optional[str]              # e.g. "connection_pool_exhaustion"
    evidence:             Optional[List[str]]        # supporting log/metric observations
    impact:               Optional[str]              # user-facing impact description
    affected_services:    Optional[List[str]]        # list of affected service names

    # Task 3: Runbook Generation
    runbook:              Optional[RunbookSection]   # see RunbookSection schema

    # Universal
    notes:                Optional[str]              # agent reasoning / free text
    confidence:           Optional[float]            # 0.0–1.0 self-assessed confidence


class ActionType(str, Enum):
    CLASSIFY_ALERT      = "classify_alert"       # Terminal for Task 1
    ANALYZE_LOGS        = "analyze_logs"         # Non-terminal: get more context
    IDENTIFY_ROOT_CAUSE = "identify_root_cause"  # Terminal for Task 2
    GENERATE_RUNBOOK    = "generate_runbook"     # Terminal for Task 3
    ESCALATE            = "escalate"             # Non-terminal: partial reward
    RESOLVE             = "resolve"              # Terminal: marks resolution
```

---

## 🔭 Observation Space

```python
class Observation(BaseModel):
    task_id:           str               # Episode UUID
    task_type:         TaskType          # alert_classification / rca / runbook_generation
    step:              int               # Current step number
    max_steps:         int               # Episode budget
    alert:             Alert             # The incident alert
    context:           IncidentContext   # Service graph, recent deploys, hints
    available_actions: List[str]         # Valid action descriptions
    progress_hint:     str               # Human-readable hint
    episode_history:   List[dict]        # Last 5 steps taken


class Alert(BaseModel):
    alert_id:               str
    title:                  str
    description:            str
    service:                str
    environment:            str               # "production"
    timestamp:              str               # ISO 8601
    metrics:                Dict[str, float]  # key metric values
    metric_details:         List[MetricSnapshot]
    logs:                   List[str]         # log lines
    raw_log_count:          int
    firing_duration_seconds: int
    previous_incidents:     List[str]


class IncidentContext(BaseModel):
    service_graph:          List[ServiceDependency]   # dependency health
    recent_deployments:     List[dict]                # what changed recently
    runbook_hints:          List[str]                 # domain knowledge hints
    similar_past_incidents: List[str]                 # historical context
    on_call_teams:          Dict[str, List[str]]      # team → contact list
```

---

## 🎯 Reward Signal Design

The reward function is designed to:

1. **Provide partial progress signals** — agents can earn reward from analyze_logs (+0.05) and escalate (+0.10) even before the terminal action, preventing sparse-reward problems.

2. **Penalize critical failures asymmetrically** — misclassifying P1 as P4 incurs −0.20 penalty (the most dangerous real-world mistake). Over-triaging P4 as P1 incurs −0.10 (alert fatigue).

3. **Reward specificity** — evidence and runbook sections that contain real CLI commands, service names, and technical terms score higher than vague text.

4. **Apply efficiency pressure** — taking more than 80% of allowed steps incurs a small penalty, encouraging decisive agents.

5. **Score 0.0–1.0 continuously** — never binary; always informative about partial progress.

---

## 📊 Scenario Library

The environment includes 4 realistic production incident scenarios:

| ID | Scenario | Severity | Category | Difficulty |
|----|----------|----------|----------|------------|
| `db_connpool_001` | PostgreSQL connection pool exhausted by bulk job | P1 | Database | Medium |
| `oom_memory_002` | OOMKill loop from unbounded feature cache | P2 | Application | Medium |
| `cascading_timeout_003` | Cascading timeouts from network switch misconfiguration | P1 | Network | Hard |
| `cert_expiry_004` | TLS certificate renewal blocked by WAF rule | P2 | Security | Easy |

Each scenario includes: 10-20 realistic log lines, 8-12 metrics, service dependency graph, recent deployment history, and complete ground-truth runbooks.

---

## 📈 Baseline Scores

Evaluated with `gpt-4o-mini` at temperature=0.2, seed=42:

| Task | Difficulty | Baseline Score | Notes |
|------|-----------|---------------|-------|
| Alert Classification | Easy | **0.71** | Struggles with team assignment |
| Root Cause Analysis | Medium | **0.58** | Good on component, weak on evidence |
| Runbook Generation | Hard | **0.44** | Missing rollback + prevention sections |
| **Overall** | — | **0.58** | Significant room for improvement |

---

## 🚀 Quick Start

### Local

```bash
git clone https://github.com/openenv/incident-response-env
cd incident-response-env
pip install -r requirements.txt

# Run tests
python tests/test_env.py

# Start API server
uvicorn app:app --host 0.0.0.0 --port 7860

# Run baseline evaluation
export OPENAI_API_KEY=sk-...
python baseline/run_baseline.py
```

### Docker

```bash
docker build -t incident-response-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... incident-response-env

# With custom seed and model:
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... incident-response-env \
    python baseline/run_baseline.py --model gpt-4o --seed 42 --runs 3
```

### Python SDK

```python
from environment.env import make_env
from environment.models import Action, ActionType, SeverityLevel, AlertCategory

# Task 1: Alert Classification
env = make_env("alert_classification", seed=42)
obs = env.reset()
print(obs.to_agent_prompt())

action = Action(
    action_type=ActionType.CLASSIFY_ALERT,
    severity=SeverityLevel.P1,
    category=AlertCategory.DATABASE,
    team="database-team",
    notes="100% connection pool utilisation, scheduler job opened 120 parallel connections",
)
obs, reward, done, info = env.step(action)
# reward.score = 0.92, reward.breakdown shows per-component scores

# Task 3: Runbook Generation
from environment.models import RunbookSection
env3 = make_env("runbook_generation", seed=42)
obs3 = env3.reset()

action3 = Action(
    action_type=ActionType.GENERATE_RUNBOOK,
    runbook=RunbookSection(
        diagnosis_steps=["kubectl logs -l app=scheduler-service | grep bulk-report", ...],
        remediation_steps=["kubectl delete job bulk-report-export", ...],
        rollback_plan=["kubectl rollout undo deployment/scheduler-service"],
        escalation_criteria=["Escalate if error rate > 50% after 10 min"],
        prevention_measures=["Add max_parallelism=10 to bulk jobs"],
        commands=["kubectl delete job bulk-report-export"],
    )
)
obs3, reward3, done3, info3 = env3.step(action3)
print(reward3.score)  # 0.73
```

### HTTP API (Deployed)

```bash
# Start episode
curl -X POST https://openenv-incident-response-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type": "alert_classification", "seed": 42}'

# Submit action
curl -X POST https://openenv-incident-response-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<returned session_id>",
    "action": {
      "action_type": "classify_alert",
      "severity": "P1",
      "category": "database",
      "team": "database-team",
      "notes": "Connection pool exhausted"
    }
  }'

# Get episode state
curl https://openenv-incident-response-env.hf.space/state/<session_id>
```

---

## 🧪 Testing

```bash
python tests/test_env.py -v
# Ran 15 tests in 0.006s — OK
```

Test coverage:
- Reset/step/state API correctness
- Correct and incorrect classification scoring
- Severity misclassification penalties
- RCA field validation
- Runbook completeness scoring
- Episode termination and error handling
- Reproducibility across seeds

---

## 🏗️ Architecture Decisions

**Why FastAPI?** Zero-overhead Pydantic integration, auto-generated `/docs`, async support, and HF Spaces compatibility.

**Why synthetic data?** Real incident data is confidential. Synthetic scenarios let us control ground truth exactly, vary difficulty deterministically, and avoid privacy/legal issues while remaining realistic.

**Why keyword-based grading?** LLM-as-judge is expensive, non-deterministic, and introduces circular evaluation. Keyword overlap + structural scoring gives deterministic, fast, interpretable grades while measuring what matters: did the agent identify the right component, the right failure type, the right remediation commands?

**Why partial rewards?** Sparse rewards (only scoring the terminal action) are an RL alignment anti-pattern. Every step should give the agent useful signal. `analyze_logs` → small reward. `escalate` → small reward. Near-correct classification → partial reward. This makes the environment trainable, not just evaluable.

---

## 📄 License

MIT — see `LICENSE`.

---

*Built for the Meta OpenEnv Hackathon. IncidentResponseEnv simulates the real-world task that wakes up engineers at 3 AM and costs companies millions in downtime. If AI can get good at this, that matters.*
