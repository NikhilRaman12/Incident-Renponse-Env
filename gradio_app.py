import gradio as gr
import requests
import json

API_URL = "http://127.0.0.1:7860"  # FastAPI backend
session_id = None

def start_episode(task_type):
    global session_id
    response = requests.post(
        f"{API_URL}/reset",
        json={"task_type": task_type, "seed": 42}
    )
    data = response.json()
    session_id = data["session_id"]
    obs = data["observation"]
    return (
        session_id,
        json.dumps(obs, indent=2)
    )

def take_action(action_json):
    global session_id
    try:
        action = json.loads(action_json)
    except:
        return "Invalid JSON", "", ""
    response = requests.post(
        f"{API_URL}/step",
        json={
            "session_id": session_id,
            "action": action
        }
    )
    data = response.json()
    obs = data.get("observation", {})
    reward = data.get("reward", {})
    done = data.get("done", False)
    return (
        json.dumps(obs, indent=2),
        json.dumps(reward, indent=2),
        str(done)
    )

with gr.Blocks() as demo:
    gr.Markdown("# Incident Response Env (OpenEnv UI)")

    with gr.Row():
        task = gr.Dropdown(
            ["alert_classification", "root_cause_analysis", "runbook_generation"],
            label="Select Task"
        )
        start_btn = gr.Button("Start Episode")

    session_box = gr.Textbox(label="Session ID")
    observation = gr.Textbox(label="Observation", lines=20)

    start_btn.click(
        start_episode,
        inputs=[task],
        outputs=[session_box, observation]
    )

    gr.Markdown("## Action Input (JSON)")

    action_input = gr.Textbox(
        label="Action JSON",
        lines=10,
        value=json.dumps({
            "action_type": "classify_alert",
            "severity": "P1",
            "category": "database",
            "team": "database-team",
            "notes": "Connection pool exhausted"
        }, indent=2)
    )

    step_btn = gr.Button("Take Step")
    next_obs = gr.Textbox(label="Next Observation", lines=20)
    reward_box = gr.Textbox(label="Reward", lines=10)
    done_box = gr.Textbox(label="Done")

    step_btn.click(
        take_action,
        inputs=[action_input],
        outputs=[next_obs, reward_box, done_box]
    )

# Let Gradio auto-select a free port
demo.launch(share=True)
