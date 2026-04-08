from __future__ import annotations

import os
import uuid
import time
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from environment.env import make_env, IncidentResponseEnv, TASK_CONFIG
from environment.models import Action

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

APP_VERSION = "1.1.0"
SESSION_TTL = int(os.getenv("SESSION_TTL", 1800))  # 30 min
DEBUG_MODE  = os.getenv("DEBUG", "false").lower() == "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("incident-env")

# ─────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────

app = FastAPI(
    title="IncidentResponseEnv",
    description="OpenEnv — SRE Incident Response Simulation",
    version=APP_VERSION,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# session_id → {env, last_access}
_sessions: Dict[str, Dict[str, Any]] = {}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def cleanup_sessions():
    """Remove expired sessions"""
    now = time.time()
    expired = [
        sid for sid, data in _sessions.items()
        if now - data["last_access"] > SESSION_TTL
    ]
    for sid in expired:
        _sessions.pop(sid, None)
    if expired:
        logger.info(f"Cleaned {len(expired)} expired sessions")

def get_env(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found. Call /reset.")
    session["last_access"] = time.time()
    return session["env"]

# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_type: str = "alert_classification"
    seed: Optional[int] = 42
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    session_id: str
    action: Action

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "active_sessions": len(_sessions),
        "tasks": list(TASK_CONFIG.keys())
    }

@app.get("/metrics")
def metrics():
    return {
        "total_sessions": len(_sessions),
        "ttl_seconds": SESSION_TTL
    }

@app.post("/reset")
def reset(req: ResetRequest):
    cleanup_sessions()

    if req.task_type not in TASK_CONFIG:
        raise HTTPException(
            400,
            f"Invalid task_type '{req.task_type}'. Available: {list(TASK_CONFIG.keys())}"
        )

    session_id = req.session_id or str(uuid.uuid4())

    try:
        env = make_env(task_type=req.task_type, seed=req.seed, debug=DEBUG_MODE)
        obs = env.reset(seed=req.seed)

        _sessions[session_id] = {
            "env": env,
            "last_access": time.time()
        }

        return {
            "session_id": session_id,
            "observation": obs.model_dump()
        }

    except Exception as e:
        logger.exception("Reset failed")
        raise HTTPException(500, str(e))


@app.post("/step")
def step(req: StepRequest):
    env = get_env(req.session_id)

    try:
        obs, reward, done, info = env.step(req.action)

        if done:
            _sessions.pop(req.session_id, None)

        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }

    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Step failed")
        raise HTTPException(500, str(e))


@app.get("/state/{session_id}")
def state(session_id: str):
    env = get_env(session_id)

    try:
        return env.state().model_dump()
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.get("/tasks")
def tasks():
    return TASK_CONFIG


# ─────────────────────────────────────────────
# Demo UI (unchanged)
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def ui():
    return """YOUR EXISTING HTML (UNCHANGED)"""


# ─────────────────────────────────────────────
# Dev entry
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=True
    )
