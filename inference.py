import os
import requests
import json
from dotenv import load_dotenv   # <-- add this

# Load variables from .env file
load_dotenv()

# Environment variables with safe defaults
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "default-model")
HF_TOKEN = os.getenv("HF_TOKEN", "")

TASKS = [
    "alert_classification",
    "root_cause_analysis",
    "runbook_generation"
]


def run_task(task):
    print("[START]")
    print(json.dumps({"task": task}))

    if not API_BASE_URL or API_BASE_URL.strip() == "":
        raise ValueError("API_BASE_URL is not set. Please configure it in your environment or code.")

    # RESET
    res = requests.post(
        f"{API_BASE_URL}/reset",
        json={"task_type": task, "seed": 42}
    )
    data = res.json()
    session_id = data.get("session_id")

    print("[STEP]")
    print(json.dumps({"event": "reset", "task": task}))

    done = False
    step_count = 0

    while not done and step_count < 5:
        action = {
            "action_type": "analyze_logs"
        }

        res = requests.post(
            f"{API_BASE_URL}/step",
            json={
                "session_id": session_id,
                "action": action
            }
        )

        result = res.json()
        done = result.get("done", False)

        print("[STEP]")
        print(json.dumps({
            "step": step_count,
            "reward": result.get("reward", {}),
            "done": done
        }))

        step_count += 1

    print("[END]")
    print(json.dumps({"task": task, "completed": True}))


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
