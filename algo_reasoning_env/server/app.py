"""
FastAPI application for the Algo Reasoning Environment.

This module creates an HTTP server that exposes the AlgoReasoningEnvironment
over HTTP endpoints with session-based state management.

Endpoints:
    - GET /: Interactive landing page explaining the environment
    - POST /reset: Reset the environment and get a new problem
    - POST /step: Evaluate an agent's submission
    - GET /state: Get current environment state
    - GET /health: Health check
    - POST /evaluate: Combined reset+step in a single request

Usage:
    # Development:
    uvicorn algo_reasoning_env.server.app:app --reload --host 0.0.0.0 --port 7860

    # Production (single worker for session state):
    uvicorn algo_reasoning_env.server.app:app --host 0.0.0.0 --port 7860 --workers 1
"""

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from algo_reasoning_env import (
    AlgoReasoningAction,
    AlgoReasoningObservation,
)
from .session_store import create_session, get_session, delete_session, session_count


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    """Request body for /reset."""

    seed: Optional[int] = Field(default=None, description="Random seed (unused)")
    episode_id: Optional[str] = Field(default=None, max_length=255)


class StepRequestBody(BaseModel):
    """Action payload nested inside /step request."""

    solution_code: str = Field(..., description="Rust impl Solution block")
    reasoning_steps: str = Field(
        ..., description="Step-by-step reasoning (step-1, step-2, etc.)"
    )
    time_complexity: str = Field(
        ..., description="Time complexity, e.g. O(n) or O(n^2)"
    )


class StepRequest(BaseModel):
    """Request body for /step."""

    session_id: str = Field(..., description="Returned by /reset")
    action: StepRequestBody = Field(..., description="Agent submission")


class EvaluateRequest(BaseModel):
    """Request body for /evaluate (stateless combined reset+step)."""

    solution_code: str = Field(..., description="Rust impl Solution block")
    reasoning_steps: str = Field(
        ..., description="Step-by-step reasoning (step-1, step-2, etc.)"
    )
    time_complexity: str = Field(
        ..., description="Time complexity, e.g. O(n) or O(n^2)"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs_to_dict(obs: AlgoReasoningObservation) -> Dict[str, Any]:
    """Serialize an observation to a plain dict for JSON response."""
    d: Dict[str, Any] = {
        "problem_id": obs.problem_id,
        "task_id": obs.task_id,
        "difficulty": obs.difficulty,
        "problem_description": obs.problem_description,
        "starter_code": obs.starter_code,
        "expected_complexity": obs.expected_complexity,
        "ground_truth_explanation": obs.ground_truth_explanation,
        "tags": obs.tags,
        "test_harness": obs.test_harness,
        "done": obs.done,
        "reward": obs.reward,
    }
    if obs.reasoning_score is not None:
        d["reasoning_score"] = obs.reasoning_score
    if obs.complexity_score is not None:
        d["complexity_score"] = obs.complexity_score
    if obs.correctness_reward is not None:
        d["correctness_reward"] = obs.correctness_reward
    if obs.evaluation is not None:
        ev = obs.evaluation
        d["evaluation"] = {
            "reasoning_score": ev.reasoning_score,
            "complexity_score": ev.complexity_score,
            "correctness_reward": ev.correctness_reward,
            "predicted_complexity": ev.predicted_complexity,
            "compilation_error": ev.compilation_error,
            "test_output": ev.test_output,
        }
    return d


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Algo Reasoning Environment",
    version="1.0.0",
    description=(
        "OpenEnv-compatible environment for evaluating AI agents on "
        "Rust code correctness, reasoning quality, and time complexity."
    ),
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> HTMLResponse:
    """Landing page explaining how the environment works."""
    html = _LANDING_HTML
    return HTMLResponse(content=html)


@app.get("/health", tags=["Health"])
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "active_sessions": session_count(),
    }


@app.post("/reset", tags=["Environment Control"])
async def reset(request: ResetRequest = None) -> Dict[str, Any]:
    """
    Reset the environment and return the first problem observation.

    Creates a new session. The caller must store the returned ``session_id``
    and pass it to ``/step``.
    """
    data_dir = os.getenv("DATA_DIR", "/data")
    api_key = os.getenv("LIGHTNING_API_KEY")

    try:
        session_id, env = create_session(data_dir=data_dir, api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {e}")

    try:
        obs = env.reset()
    except Exception as e:
        delete_session(session_id)
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")

    return {
        "session_id": session_id,
        "observation": _obs_to_dict(obs),
        "reward": None,
        "done": False,
    }


@app.post("/step", tags=["Environment Control"])
async def step(request: StepRequest) -> Dict[str, Any]:
    """
    Evaluate an agent's submission.

    Requires the ``session_id`` returned by ``/reset``.
    The action is nested under the ``action`` key.
    """
    try:
        env = get_session(request.session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Build the typed action
    action = AlgoReasoningAction(
        solution_code=request.action.solution_code,
        reasoning_steps=request.action.reasoning_steps,
        time_complexity=request.action.time_complexity,
    )

    try:
        obs = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")

    response: Dict[str, Any] = {
        "observation": _obs_to_dict(obs),
        "reward": obs.reward,
        "done": obs.done,
    }

    # Clean up session after episode is done
    if obs.done:
        delete_session(request.session_id)

    return response


@app.get("/state", tags=["Environment Control"])
async def state() -> Dict[str, Any]:
    """Return environment metadata and current state."""
    return {
        "active_sessions": session_count(),
        "num_problems": None,  # Unknown without an active session
    }


@app.post("/evaluate", tags=["Environment Control"])
async def evaluate(request: EvaluateRequest) -> Dict[str, Any]:
    """
    Stateless combined reset+step in a single request.

    Creates a temporary session, loads the next problem, evaluates the
    submission, and returns the result. No ``session_id`` is needed.
    """
    data_dir = os.getenv("DATA_DIR", "/data")
    api_key = os.getenv("LIGHTNING_API_KEY")

    try:
        session_id, env = create_session(data_dir=data_dir, api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {e}")

    try:
        env.reset()

        action = AlgoReasoningAction(
            solution_code=request.solution_code,
            reasoning_steps=request.reasoning_steps,
            time_complexity=request.time_complexity,
        )

        obs = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
    finally:
        delete_session(session_id)

    return {
        "observation": _obs_to_dict(obs),
        "reward": obs.reward,
        "done": obs.done,
    }


# ---------------------------------------------------------------------------
# Landing page HTML
# ---------------------------------------------------------------------------

_LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>algo_reasoning_env — Rust LeetCode Evaluator</title>
<style>
  :root {
    --bg-primary: #ffffff;
    --bg-secondary: #f7f8fc;
    --border: #e2e4ef;
    --text-primary: #1a1a2e;
    --text-secondary: #5c5c7a;
    --accent: #c45c26;
    --pill-t-bg: #e8f5ee;
    --pill-t-color: #0a5c36;
    --pill-t-border: #0a5c36;
    --pill-p-bg: #f0edfe;
    --pill-p-color: #3c2d91;
    --pill-p-border: #534ab7;
    --pill-a-bg: #fef3ed;
    --pill-a-color: #633806;
    --pill-a-border: #c45c26;
    --pill-r-bg: #fce8e8;
    --pill-r-color: #791f1f;
    --pill-r-border: #a32d2d;
    --pill-g-bg: #e8f5ee;
    --pill-g-color: #0a5c36;
    --pill-g-border: #0a5c36;
    --card-border: #e2e4ef;
    --radius: 10px;
    --mono-font: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg-secondary);
    color: var(--text-primary);
    line-height: 1.6;
    padding: 2rem 1rem;
  }
  .container { max-width: 860px; margin: 0 auto; }
  header {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    border-left: 4px solid var(--accent);
  }
  h1 { font-size: 1.4rem; font-weight: 600; color: var(--text-primary); }
  h1 span { color: var(--accent); }
  header p { font-size: 0.9rem; color: var(--text-secondary); margin-top: 0.3rem; }
  .controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
  }
  #dots { display: flex; gap: 6px; }
  .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--border); cursor: pointer;
    transition: background 0.2s;
  }
  .dot.active { background: var(--accent); }
  .controls-right { display: flex; gap: 8px; }
  button {
    padding: 0 14px; height: 32px; font-size: 0.85rem;
    border-radius: 6px; border: 1px solid var(--border);
    background: var(--bg-primary); color: var(--text-primary);
    cursor: pointer;
  }
  button:disabled { opacity: 0.4; cursor: default; }
  button:hover:not(:disabled) { background: var(--bg-secondary); }
  #step-label { font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.5rem; }
  .sp { display: none; }
  .sp.on { display: block; }
  .step-title { font-size: 1rem; font-weight: 500; margin-bottom: 0.5rem; }
  .note { font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 1rem; }
  .note strong { font-weight: 500; color: var(--text-primary); }
  .code {
    font-family: var(--mono-font); font-size: 0.8rem;
    background: var(--bg-primary); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 0.75rem 1rem;
    line-height: 1.6; white-space: pre; overflow-x: auto;
    margin-bottom: 1rem;
  }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 1rem; }
  .card {
    background: var(--bg-primary); border: 1px solid var(--card-border);
    border-radius: var(--radius); padding: 0.75rem 1rem;
    flex: 1; min-width: 140px;
  }
  .card-label { font-size: 0.7rem; color: var(--text-secondary); margin-bottom: 2px; }
  .card-val { font-size: 0.85rem; font-weight: 500; }
  .pill {
    display: inline-block; border-radius: 12px; font-size: 0.7rem;
    font-weight: 500; padding: 1px 8px;
  }
  .pill-t { background: var(--pill-t-bg); color: var(--pill-t-color); border: 0.5px solid var(--pill-t-border); }
  .pill-p { background: var(--pill-p-bg); color: var(--pill-p-color); border: 0.5px solid var(--pill-p-border); }
  .pill-a { background: var(--pill-a-bg); color: var(--pill-a-color); border: 0.5px solid var(--pill-a-border); }
  .pill-r { background: var(--pill-r-bg); color: var(--pill-r-color); border: 0.5px solid var(--pill-r-border); }
  .pill-g { background: var(--pill-g-bg); color: var(--pill-g-color); border: 0.5px solid var(--pill-g-border); }
  .footer { text-align: center; font-size: 0.75rem; color: var(--text-secondary); margin-top: 1.5rem; }
</style>
</head>
<body>
<div class="container">

<header>
  <h1><span>algo_reasoning_env</span> — Rust LeetCode Evaluator</h1>
  <p>OpenEnv environment for evaluating AI agents on Rust code correctness, reasoning, and time complexity</p>
</header>

<div class="controls">
  <div id="dots"></div>
  <div class="controls-right">
    <button id="prev-btn" onclick="go(-1)">&#8592; prev</button>
    <button id="next-btn" onclick="go(1)">next &#8594;</button>
  </div>
</div>
<div id="step-label">step 1 of 6</div>

<!-- STEP 1: reset() -->
<div class="sp on" id="s0">
  <div class="step-title">reset() — episode starts</div>
  <div class="note">Server loads a Rust LeetCode problem from the dataset. The observation contains the problem description, starter code template, test harness, and expected time complexity. <strong>No complexity pattern name is ever included in the observation.</strong></div>
  <div class="code">POST /reset
{}

Response:
{
  "session_id": "abc-123-...",
  "observation": {
    "task_id": "two-sum",
    "difficulty": "Easy",
    "problem_description": "Given an array of integers nums ...",
    "starter_code": "impl Solution { pub fn two_sum(...)",
    "test_harness": "fn test_two_sum() { ... }",
    "expected_complexity": "O(n)",
    "ground_truth_explanation": "Use a hash map for constant-time lookup...",
    "tags": ["Array", "Hash Table"],
    "done": false,
    "reward": null
  },
  "reward": null,
  "done": false
}</div>
  <div class="note">The agent sees only the problem description and starter code — it must infer the correct algorithm and complexity from scratch. Store the <strong>session_id</strong> for the next step.</div>
</div>

<!-- STEP 2: agent forms action -->
<div class="sp" id="s1">
  <div class="step-title">agent produces action</div>
  <div class="note">The agent (LLM) receives the observation and must return an <span class="pill pill-a">action</span> with three fields — Rust solution code, step-by-step reasoning, and time complexity label.</div>
  <div class="code">POST /step
{
  "session_id": "abc-123-...",
  "action": {
    "solution_code": "impl Solution { pub fn two_sum(...) }",
    "reasoning_steps": "step-1: Create a HashMap. step-2: Iterate and check complement.",
    "time_complexity": "O(n)"
  }
}</div>
  <div class="note">The <strong>reasoning_steps field</strong> is evaluated by an LLM judge for clarity and logical coherence. The <strong>time_complexity field</strong> is matched against ground truth Big-O notation.</div>
</div>

<!-- STEP 3: sandbox execution -->
<div class="sp" id="s2">
  <div class="step-title">step() — Rust compilation &amp; test execution</div>
  <div class="note">The server receives the action. Rust code is compiled with <strong>rustc</strong> inside a restricted subprocess — 30s timeout for both compilation and tests. No network access, no external dependencies.</div>
  <div class="code">result = sandbox.run(
  code = action.solution_code,
  timeout_sec = 30,
)
# Returns: { compilation_ok, test_output, compilation_error }</div>
  <div class="row">
    <div class="card"><div class="card-label">on timeout</div><div class="card-val">reward = 0.0, done = True</div></div>
    <div class="card"><div class="card-label">on compile error</div><div class="card-val">reward = 0.0, error returned</div></div>
    <div class="card"><div class="card-label">on success</div><div class="card-val">outputs passed to grader</div></div>
  </div>
</div>

<!-- STEP 4: grading -->
<div class="sp" id="s3">
  <div class="step-title">grader — computing reward</div>
  <div class="note">Three independent scores combined into one reward signal. Each component gives <strong>partial credit</strong> — wrong complexity still earns output score if the code passes tests.</div>
  <div class="code">reward = 0.50 * correctness_reward
          + 0.30 * reasoning_score
          + 0.20 * complexity_score

# correctness (0.5 weight):
#   1.0 = all tests pass
#   0.3 = compiles but tests fail
#   0.0 = compilation error

# reasoning (0.3 weight):
#   LLM judge scores 0.0–1.0 based on explanation clarity

# complexity (0.2 weight):
#   1 = Big-O matches ground truth
#   0 = mismatch</div>
  <div class="row">
    <div class="card" style="border-left:3px solid #0a5c36;border-radius:0 var(--radius) var(--radius) 0;">
      <div class="card-label">correctness <span class="pill pill-t">50%</span></div>
      <div class="card-val">1.0 / 0.3 / 0.0</div>
    </div>
    <div class="card" style="border-left:3px solid #534ab7;border-radius:0 var(--radius) var(--radius) 0;">
      <div class="card-label">reasoning <span class="pill pill-p">30%</span></div>
      <div class="card-val">LLM judge 0.0–1.0</div>
    </div>
    <div class="card" style="border-left:3px solid #c45c26;border-radius:0 var(--radius) var(--radius) 0;">
      <div class="card-label">complexity <span class="pill pill-a">20%</span></div>
      <div class="card-val">1 or 0</div>
    </div>
  </div>
</div>

<!-- STEP 5: observation returned -->
<div class="sp" id="s4">
  <div class="step-title">step() returns observation</div>
  <div class="note">The agent gets back a rich observation showing what went wrong — enough signal to self-correct on the next attempt. <strong>Problems cycle indefinitely</strong> after the dataset is exhausted.</div>
  <div class="code">Response:
{
  "observation": {
    "task_id": "two-sum",
    "problem_description": "...",
    "starter_code": "...",
    "expected_complexity": "O(n)",
    "done": true,
    "reward": 0.83,
    "correctness_reward": 1.0,
    "reasoning_score": 0.65,
    "complexity_score": 1,
    "evaluation": {
      "reasoning_score": 0.65,
      "complexity_score": 1,
      "correctness_reward": 1.0,
      "predicted_complexity": "O(n)",
      "compilation_error": null,
      "test_output": "test result: ok. 5 passed; 0 failed"
    }
  },
  "reward": 0.83,
  "done": true
}</div>
  <div class="note">Session is auto-deleted after <strong>done: true</strong>. Call <strong>/reset</strong> again for the next problem.</div>
</div>

<!-- STEP 6: dataset overview -->
<div class="sp" id="s5">
  <div class="step-title">dataset overview &amp; difficulty tiers</div>
  <div class="note" style="margin-bottom:1rem;">The environment cycles through problems in order. After problem N-1, it wraps back to 0. Weighted scoring rewards <strong>frontier-challenging hard problems</strong> — a task where frontier models score 0.3 is better than one where they score 0.9.</div>
  <div class="row">
    <div class="card" style="border-left:3px solid #0a5c36;border-radius:0 var(--radius) var(--radius) 0;">
      <div class="card-label"><span class="pill pill-g">Easy</span></div>
      <div class="card-val" style="margin-bottom:4px;">Sort / reverse / search</div>
      <div style="font-size:0.8rem;color:var(--text-secondary);">O(n) or O(n log n). Frontier models score ~0.9.</div>
    </div>
    <div class="card" style="border-left:3px solid #c45c26;border-radius:0 var(--radius) var(--radius) 0;">
      <div class="card-label"><span class="pill pill-a">Medium</span></div>
      <div class="card-val" style="margin-bottom:4px;">Two-pointer / sliding window</div>
      <div style="font-size:0.8rem;color:var(--text-secondary);">O(n). Requires real induction.</div>
    </div>
    <div class="card" style="border-left:3px solid #a32d2d;border-radius:0 var(--radius) var(--radius) 0;">
      <div class="card-label"><span class="pill pill-r">Hard</span></div>
      <div class="card-val" style="margin-bottom:4px;">DP / graph algorithms</div>
      <div style="font-size:0.8rem;color:var(--text-secondary);">Frontier models score ~0.3–0.4.</div>
    </div>
  </div>
  <div class="note">All problems come from the <code>complexity_reasoning_data/</code> dataset embedded in the Docker image. Problems cycle forever.</div>
</div>

<div class="footer">algo_reasoning_env &middot; OpenEnv &middot; Rust LeetCode Evaluator</div>
</div>

<script>
var cur = 0, N = 6;
var panels = document.querySelectorAll('.sp');
var dotsEl = document.getElementById('dots');
var slEl = document.getElementById('step-label');
var prevBtn = document.getElementById('prev-btn');
var nextBtn = document.getElementById('next-btn');

for (var i = 0; i < N; i++) {
  var d = document.createElement('span');
  d.className = 'dot' + (i === 0 ? ' active' : '');
  d.setAttribute('data-i', i);
  d.onclick = (function(idx) { return function() { show(idx); }; })(i);
  dotsEl.appendChild(d);
}

function show(i) {
  panels[cur].classList.remove('on');
  dotsEl.children[cur].classList.remove('active');
  cur = (i + N) % N;
  panels[cur].classList.add('on');
  dotsEl.children[cur].classList.add('active');
  slEl.textContent = 'step ' + (cur + 1) + ' of ' + N;
  prevBtn.disabled = cur === 0;
  nextBtn.disabled = cur === N - 1;
}

function go(d) { show(cur + d); }

prevBtn.disabled = true;
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Direct execution entry point
# ---------------------------------------------------------------------------


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution.

    Args:
        host: Host address to bind to
        port: Port number to listen on
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    main(host=args.host, port=args.port)
