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
    problem_id: Optional[int] = Field(
        default=None,
        description="Specific problem ID to load. If omitted, loads next in sequence.",
    )


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

    Pass ``problem_id`` to load a specific problem, or omit it to load
    the next problem in sequence.
    """
    data_dir = os.getenv("DATA_DIR", "/data")
    api_key = os.getenv("LIGHTNING_API_KEY")

    try:
        session_id, env = create_session(data_dir=data_dir, api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {e}")

    try:
        problem_id = request.problem_id if request else None
        obs = env.reset(problem_id=problem_id)
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
<title>Algo Reasoning Env</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0e0e10;
    --bg2: #16161a;
    --bg3: #1e1e24;
    --border: rgba(255,255,255,0.07);
    --border2: rgba(255,255,255,0.12);
    --text: #f0eff4;
    --muted: #8b8a94;
    --dim: #555460;
    --purple-light: #EEEDFE;
    --purple-mid: #7F77DD;
    --purple-dark: #534AB7;
    --teal-light: #E1F5EE;
    --teal-mid: #1D9E75;
    --teal-dark: #0F6E56;
    --coral-light: #FAECE7;
    --coral-mid: #D85A30;
    --amber-mid: #EF9F27;
    --amber-light: #FAEEDA;
    --green-mid: #639922;
    --green-light: #EAF3DE;
    --blue-mid: #378ADD;
    --blue-light: #E6F1FB;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 16px;
    line-height: 1.7;
    min-height: 100vh;
  }

  a { color: var(--purple-mid); text-decoration: none; }
  a:hover { text-decoration: underline; }

  .container { max-width: 900px; margin: 0 auto; padding: 0 2rem; }

  /* NAV */
  nav {
    border-bottom: 1px solid var(--border);
    padding: 1rem 0;
    position: sticky;
    top: 0;
    background: rgba(14,14,16,0.92);
    backdrop-filter: blur(8px);
    z-index: 100;
  }
  nav .inner {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 600;
    font-size: 15px;
    color: var(--text);
  }
  .logo-icon {
    width: 28px; height: 28px;
    background: var(--purple-dark);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
  }
  .nav-links { display: flex; gap: 1.5rem; }
  .nav-links a { font-size: 14px; color: var(--muted); }
  .nav-links a:hover { color: var(--text); text-decoration: none; }

  /* HERO */
  .hero {
    padding: 5rem 0 3rem;
    text-align: center;
  }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(127,119,221,0.12);
    border: 1px solid rgba(127,119,221,0.25);
    color: #AFA9EC;
    font-size: 12px;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 100px;
    margin-bottom: 1.5rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
  }
  .badge-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #7F77DD;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
  h1 {
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 700;
    line-height: 1.15;
    letter-spacing: -0.02em;
    margin-bottom: 1.25rem;
    color: var(--text);
  }
  .hero-accent { color: var(--purple-mid); }
  .hero-sub {
    font-size: 1.1rem;
    color: var(--muted);
    max-width: 580px;
    margin: 0 auto 2.5rem;
    line-height: 1.6;
  }
  .hero-cta {
    display: flex;
    gap: 12px;
    justify-content: center;
    flex-wrap: wrap;
  }
  .btn-primary {
    background: var(--purple-dark);
    color: #fff;
    border: none;
    padding: 10px 24px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    text-decoration: none;
    display: inline-block;
    transition: background 0.15s;
  }
  .btn-primary:hover { background: var(--purple-mid); text-decoration: none; color: #fff; }
  .btn-outline {
    background: transparent;
    color: var(--muted);
    border: 1px solid var(--border2);
    padding: 10px 24px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    text-decoration: none;
    display: inline-block;
    transition: border-color 0.15s, color 0.15s;
  }
  .btn-outline:hover { border-color: var(--muted); color: var(--text); text-decoration: none; }

  /* STAT STRIP */
  .stat-strip {
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    padding: 2rem 0;
    margin: 3rem 0;
  }
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
  }
  .stat-item {
    text-align: center;
    padding: 0 1rem;
    border-right: 1px solid var(--border);
  }
  .stat-item:last-child { border-right: none; }
  .stat-num {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 4px;
  }
  .stat-num .accent { color: var(--purple-mid); }
  .stat-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }

  /* SECTION */
  section { padding: 4rem 0; }
  .section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--purple-mid);
    margin-bottom: 0.75rem;
  }
  h2 {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
    line-height: 1.2;
  }
  .section-desc {
    color: var(--muted);
    font-size: 1rem;
    max-width: 560px;
    margin-bottom: 2.5rem;
    line-height: 1.65;
  }

  /* WHY RUST */
  .why-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  .why-card {
    background: var(--bg2);
    padding: 1.5rem;
  }
  .why-icon {
    font-size: 20px;
    margin-bottom: 0.75rem;
    display: block;
  }
  .why-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.5rem;
  }
  .why-desc {
    font-size: 13px;
    color: var(--muted);
    line-height: 1.55;
  }

  /* PIPELINE */
  .pipeline {
    display: flex;
    flex-direction: column;
    gap: 0;
    position: relative;
  }
  .pipeline::before {
    content: '';
    position: absolute;
    left: 20px;
    top: 24px;
    bottom: 24px;
    width: 1px;
    background: linear-gradient(to bottom, var(--purple-dark), var(--teal-dark));
    opacity: 0.4;
  }
  .pipeline-step {
    display: flex;
    gap: 1.25rem;
    padding: 1.25rem 0;
    align-items: flex-start;
  }
  .step-num {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--bg3);
    border: 1px solid var(--border2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 600;
    color: var(--purple-mid);
    flex-shrink: 0;
    position: relative;
    z-index: 1;
  }
  .step-content { flex: 1; padding-top: 8px; }
  .step-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 4px;
  }
  .step-desc { font-size: 13px; color: var(--muted); line-height: 1.55; }

  /* REWARD */
  .reward-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }
  .reward-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .reward-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }
  .reward-card.correctness::before { background: var(--purple-mid); }
  .reward-card.reasoning::before { background: var(--teal-mid); }
  .reward-card.complexity::before { background: var(--amber-mid); }
  .reward-weight {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-bottom: 4px;
  }
  .correctness .reward-weight { color: var(--purple-mid); }
  .reasoning .reward-weight { color: var(--teal-mid); }
  .complexity .reward-weight { color: var(--amber-mid); }
  .reward-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .reward-desc { font-size: 12px; color: var(--muted); line-height: 1.5; }
  .reward-scale {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    font-size: 11px;
    color: var(--dim);
    font-family: monospace;
  }

  /* DIFFICULTY */
  .diff-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 2rem;
  }
  .diff-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
  }
  .diff-count { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 2px; }
  .diff-card.easy .diff-count { color: var(--teal-mid); }
  .diff-card.medium .diff-count { color: var(--amber-mid); }
  .diff-card.hard .diff-count { color: var(--coral-mid); }
  .diff-label { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
  .diff-card.easy .diff-label { color: var(--teal-mid); }
  .diff-card.medium .diff-label { color: var(--amber-mid); }
  .diff-card.hard .diff-label { color: var(--coral-mid); }
  .diff-share { font-size: 12px; color: var(--dim); }
  .diff-mult {
    margin-top: 0.75rem;
    font-size: 11px;
    font-family: monospace;
    color: var(--dim);
    background: var(--bg3);
    padding: 3px 8px;
    border-radius: 4px;
    display: inline-block;
  }

  /* CODE BLOCK */
  .code-block {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin: 1.5rem 0;
  }
  .code-header {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .code-dot { width: 10px; height: 10px; border-radius: 50%; }
  .code-dot.r { background: #FF5F57; }
  .code-dot.y { background: #FFBD2E; }
  .code-dot.g { background: #28C840; }
  .code-lang { margin-left: auto; font-size: 11px; color: var(--dim); text-transform: uppercase; letter-spacing: 0.05em; }
  pre {
    padding: 1.25rem 1.5rem;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 13px;
    line-height: 1.6;
    overflow-x: auto;
    color: #c9c7d4;
  }
  .kw { color: #AFA9EC; }
  .fn { color: #5DCAA5; }
  .str { color: #FAC775; }
  .cmt { color: var(--dim); }
  .num { color: #F0997B; }
  .typ { color: #85B7EB; }

  /* ACTION / OBS SPACE */
  .space-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  .space-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  .space-header {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .space-header.action { color: var(--purple-mid); }
  .space-header.observe { color: var(--teal-mid); }
  .space-dot { width: 7px; height: 7px; border-radius: 50%; }
  .space-dot.a { background: var(--purple-mid); }
  .space-dot.o { background: var(--teal-mid); }
  .field-list { padding: 0.75rem 1rem; }
  .field-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 5px 0;
    border-bottom: 1px solid var(--border);
  }
  .field-row:last-child { border-bottom: none; }
  .field-name { font-family: monospace; font-size: 12px; color: var(--teal-mid); }
  .field-type { font-size: 11px; color: var(--dim); font-family: monospace; }

  /* ENDPOINTS */
  .endpoint-list { display: flex; flex-direction: column; gap: 8px; }
  .endpoint-row {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.875rem 1rem;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .method {
    font-family: monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    min-width: 44px;
    text-align: center;
  }
  .method.post { background: rgba(127,119,221,0.15); color: var(--purple-mid); }
  .method.get  { background: rgba(29,158,117,0.15); color: var(--teal-mid); }
  .endpoint-path { font-family: monospace; font-size: 13px; color: var(--text); }
  .endpoint-desc { margin-left: auto; font-size: 12px; color: var(--dim); }

  /* TAGS */
  .tag-cloud {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 1.5rem;
  }
  .tag {
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 100px;
    background: var(--bg3);
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .tag span { color: var(--dim); margin-left: 4px; font-size: 11px; }

  /* FOOTER */
  footer {
    border-top: 1px solid var(--border);
    padding: 2rem 0;
    margin-top: 4rem;
  }
  .footer-inner {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 13px;
    color: var(--dim);
  }

  /* DIVIDER */
  .divider { border: none; border-top: 1px solid var(--border); margin: 0; }

  @media (max-width: 640px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
    .stat-item:nth-child(2) { border-right: none; }
    .why-grid { grid-template-columns: 1fr; }
    .reward-grid { grid-template-columns: 1fr; }
    .diff-grid { grid-template-columns: 1fr; }
    .space-grid { grid-template-columns: 1fr; }
    .footer-inner { flex-direction: column; gap: 8px; text-align: center; }
    .nav-links { display: none; }
  }
</style>
</head>
<body>

<nav>
  <div class="inner">
    <div class="nav-logo">
      <div class="logo-icon">&#x1F980;</div>
      Algo Reasoning Env
    </div>
    <div class="nav-links">
      <a href="#why-rust">Why Rust</a>
      <a href="#dataset">Dataset</a>
      <a href="#evaluation">Evaluation</a>
      <a href="#api">API</a>
    </div>
  </div>
</nav>

<!-- HERO -->
<div class="container">
  <section class="hero">
    <div class="badge">
      <span class="badge-dot"></span>
      OpenEnv · RL Environment
    </div>
    <h1>
      Can an AI truly<br>
      <span class="hero-accent">reason about code?</span>
    </h1>
    <p class="hero-sub">
      Most benchmarks ask one question: does the code pass the tests?
      We ask three — correctness, reasoning quality, and complexity understanding —
      in a language where getting things right is genuinely hard: <strong>Rust</strong>.
    </p>
    <div class="hero-cta">
      <a href="#api" class="btn-primary">Try the API</a>
      <a href="#dataset" class="btn-outline">Explore dataset</a>
    </div>
  </section>

  <!-- STAT STRIP -->
  <div class="stat-strip">
    <div class="stat-grid">
      <div class="stat-item">
        <div class="stat-num"><span class="accent">952</span></div>
        <div class="stat-label">Problems</div>
      </div>
      <div class="stat-item">
        <div class="stat-num">3</div>
        <div class="stat-label">Eval dimensions</div>
      </div>
      <div class="stat-item">
        <div class="stat-num">2.6<span style="font-size:1.1rem">k</span></div>
        <div class="stat-label">Test harnesses</div>
      </div>
      <div class="stat-item">
        <div class="stat-num"><span class="accent">0→1</span></div>
        <div class="stat-label">Reward range</div>
      </div>
    </div>
  </div>
</div>

<hr class="divider">

<!-- WHY RUST -->
<div class="container" id="why-rust">
  <section>
    <div class="section-label">Why Rust</div>
    <h2>The compiler doesn't forgive</h2>
    <p class="section-desc">
      Python hides incomplete understanding behind dynamic typing and runtime duck-typing.
      Rust's compiler doesn't. We use that as a hard correctness gate.
    </p>
    <div class="why-grid">
      <div class="why-card">
        <span class="why-icon">&#x26D4;</span>
        <div class="why-title">Compilation is binary</div>
        <div class="why-desc">Code either compiles or it doesn't. No runtime surprises, no partial credit for almost-right syntax. This gives a signal Python's interpreter cannot provide.</div>
      </div>
      <div class="why-card">
        <span class="why-icon">&#x1F9E0;</span>
        <div class="why-title">Types demand precision</div>
        <div class="why-desc">A model must know <code>Vec&lt;i32&gt;</code> vs <code>Vec&lt;Vec&lt;i32&gt;&gt;</code>, and <code>Option&lt;Box&lt;ListNode&gt;&gt;</code> for linked lists. There is no "close enough".</div>
      </div>
      <div class="why-card">
        <span class="why-icon">&#x1F510;</span>
        <div class="why-title">Ownership tests reasoning</div>
        <div class="why-desc">Linked lists and trees require understanding borrowing, ownership transfer, and smart pointers — going beyond algorithm correctness into memory safety.</div>
      </div>
      <div class="why-card">
        <span class="why-icon">&#x26A1;</span>
        <div class="why-title">Deterministic evaluation</div>
        <div class="why-desc"><code>rustc --test</code> compiles to a native binary. No interpreter, no venv, no dependency management. One subprocess, one 30-second timeout, one truth.</div>
      </div>
    </div>
  </section>
</div>

<hr class="divider">

<!-- DATASET -->
<div class="container" id="dataset">
  <section>
    <div class="section-label">Dataset</div>
    <h2>Built from the ground up</h2>
    <p class="section-desc">
      This isn't a wrapper around an existing benchmark.
      952 problems assembled through a five-phase pipeline.
    </p>

    <div class="pipeline">
      <div class="pipeline-step">
        <div class="step-num">1</div>
        <div class="step-content">
          <div class="step-title">Source extraction</div>
          <div class="step-desc">Problem descriptions from LeetCode. Expert explanations and Big-O annotations from the <code>doocs/leetcode</code> repository — detailed algorithmic writeups parsed to capture step-by-step reasoning.</div>
        </div>
      </div>
      <div class="pipeline-step">
        <div class="step-num">2</div>
        <div class="step-content">
          <div class="step-title">Rust starter code</div>
          <div class="step-desc">Function signatures imported from <code>rustgym_eng</code> and <code>doocs/leetcode</code>. Each template provides the exact <code>pub fn</code> signature the model must implement — no guessing required.</div>
        </div>
      </div>
      <div class="pipeline-step">
        <div class="step-num">3</div>
        <div class="step-content">
          <div class="step-title">Test harness conversion</div>
          <div class="step-desc">2,641 Python test cases converted to Rust — type mappings, linked list construction, float tolerance, order-agnostic checking — into the standard <code>#[cfg(test)]</code> format.</div>
        </div>
      </div>
      <div class="pipeline-step">
        <div class="step-num">4</div>
        <div class="step-content">
          <div class="step-title">Solution generation with rollback</div>
          <div class="step-desc">LLM-generated solutions compiled and tested up to 3 times. On failure, compiler errors feed back into the model for self-correction — iterative refinement for higher-quality references.</div>
        </div>
      </div>
      <div class="pipeline-step">
        <div class="step-num">5</div>
        <div class="step-content">
          <div class="step-title">Assembly</div>
          <div class="step-desc">Tag-based boilerplate injection: <code>ListNode</code> prepended for linked-list problems, <code>TreeNode</code> for trees. The model focuses on the algorithm, not data structure definitions.</div>
        </div>
      </div>
    </div>

    <div style="margin-top: 2.5rem;">
      <div class="diff-grid">
        <div class="diff-card easy">
          <div class="diff-count">347</div>
          <div class="diff-label">Easy</div>
          <div class="diff-share">36.5% · Sort / reverse / search</div>
          <div class="diff-mult">×0.3 weight</div>
        </div>
        <div class="diff-card medium">
          <div class="diff-count">481</div>
          <div class="diff-label">Medium</div>
          <div class="diff-share">50.5% · Two-pointer / sliding window</div>
          <div class="diff-mult">×0.5 weight</div>
        </div>
        <div class="diff-card hard">
          <div class="diff-count">124</div>
          <div class="diff-label">Hard</div>
          <div class="diff-share">13.0% · DP / graph algorithms</div>
          <div class="diff-mult">×1.0 weight</div>
        </div>
      </div>

      <div class="tag-cloud">
        <div class="tag">Array <span>1,278</span></div>
        <div class="tag">String <span>510</span></div>
        <div class="tag">Hash Table <span>430</span></div>
        <div class="tag">Dynamic Programming <span>387</span></div>
        <div class="tag">Math <span>355</span></div>
        <div class="tag">Sorting <span>320</span></div>
        <div class="tag">Greedy <span>301</span></div>
        <div class="tag">Binary Search</div>
        <div class="tag">Tree</div>
        <div class="tag">Linked List</div>
        <div class="tag">Graph</div>
        <div class="tag">Backtracking</div>
      </div>
    </div>
  </section>
</div>

<hr class="divider">

<!-- EVALUATION -->
<div class="container" id="evaluation">
  <section>
    <div class="section-label">Evaluation</div>
    <h2>Three dimensions, one score</h2>
    <p class="section-desc">
      Most benchmarks are one-dimensional. We evaluate three independent signals
      because they test genuinely different capabilities.
    </p>

    <div class="reward-grid">
      <div class="reward-card correctness">
        <div class="reward-weight">50%</div>
        <div class="reward-name">Correctness</div>
        <div class="reward-desc">Does the Rust code compile and pass all test cases? The compiler is the judge — no partial credit for almost-right syntax.</div>
        <div class="reward-scale">0.0 compile fail · 0.3 passes compile · 1.0 passes tests</div>
      </div>
      <div class="reward-card reasoning">
        <div class="reward-weight">30%</div>
        <div class="reward-name">Reasoning</div>
        <div class="reward-desc">An LLM judge compares step-by-step reasoning against expert ground truth. Can the model identify the right algorithm and describe the logic flow?</div>
        <div class="reward-scale">0.0 — 1.0 continuous · semantic matching</div>
      </div>
      <div class="reward-card complexity">
        <div class="reward-weight">20%</div>
        <div class="reward-name">Complexity</div>
        <div class="reward-desc">Semantic Big-O matching — O(m×n) equals O(n×m), O(max(m,n)) equals O(m+n). Tests whether the model understands algorithmic efficiency.</div>
        <div class="reward-scale">0 or 1 · binary · semantic normalization</div>
      </div>
    </div>
  </section>
</div>

<hr class="divider">

<!-- API -->
<div class="container" id="api">
  <section>
    <div class="section-label">API</div>
    <h2>Standard OpenEnv interface</h2>
    <p class="section-desc">
      Session-based state management. Each <code>/reset</code> returns a <code>session_id</code>
      that must be passed to <code>/step</code>.
    </p>

    <div class="space-grid" style="margin-bottom: 2rem;">
      <div class="space-card">
        <div class="space-header action">
          <span class="space-dot a"></span>
          Action space
        </div>
        <div class="field-list">
          <div class="field-row">
            <span class="field-name">solution_code</span>
            <span class="field-type">str</span>
          </div>
          <div class="field-row">
            <span class="field-name">reasoning_steps</span>
            <span class="field-type">str</span>
          </div>
          <div class="field-row">
            <span class="field-name">time_complexity</span>
            <span class="field-type">str</span>
          </div>
        </div>
      </div>
      <div class="space-card">
        <div class="space-header observe">
          <span class="space-dot o"></span>
          Observation space
        </div>
        <div class="field-list">
          <div class="field-row">
            <span class="field-name">problem_description</span>
            <span class="field-type">str</span>
          </div>
          <div class="field-row">
            <span class="field-name">starter_code</span>
            <span class="field-type">str</span>
          </div>
          <div class="field-row">
            <span class="field-name">expected_complexity</span>
            <span class="field-type">str</span>
          </div>
          <div class="field-row">
            <span class="field-name">difficulty</span>
            <span class="field-type">Easy | Medium | Hard</span>
          </div>
          <div class="field-row">
            <span class="field-name">tags</span>
            <span class="field-type">list[str]</span>
          </div>
          <div class="field-row">
            <span class="field-name">reward</span>
            <span class="field-type">float 0.0–1.0</span>
          </div>
        </div>
      </div>
    </div>

    <div class="endpoint-list" style="margin-bottom: 2rem;">
      <div class="endpoint-row">
        <span class="method post">POST</span>
        <span class="endpoint-path">/reset</span>
        <span class="endpoint-desc">New episode · returns session_id + observation</span>
      </div>
      <div class="endpoint-row">
        <span class="method post">POST</span>
        <span class="endpoint-path">/step</span>
        <span class="endpoint-desc">Submit solution · requires session_id</span>
      </div>
      <div class="endpoint-row">
        <span class="method post">POST</span>
        <span class="endpoint-path">/evaluate</span>
        <span class="endpoint-desc">Stateless combined reset + step</span>
      </div>
      <div class="endpoint-row">
        <span class="method get">GET</span>
        <span class="endpoint-path">/state</span>
        <span class="endpoint-desc">Active sessions + server state</span>
      </div>
      <div class="endpoint-row">
        <span class="method get">GET</span>
        <span class="endpoint-path">/health</span>
        <span class="endpoint-desc">Health check</span>
      </div>
    </div>

    <div class="code-block">
      <div class="code-header">
        <span class="code-dot r"></span>
        <span class="code-dot y"></span>
        <span class="code-dot g"></span>
        <span class="code-lang">bash</span>
      </div>
      <pre><span class="cmt"># 1. Reset — get a problem</span>
curl -X POST <span class="str">"https://tm23hgf-rust-algo-reasoning.hf.space/reset"</span> \
  -H <span class="str">"Content-Type: application/json"</span> \
  -d <span class="str">'{}'</span>

<span class="cmt"># 2. Step — submit your solution</span>
curl -X POST <span class="str">"https://tm23hgf-rust-algo-reasoning.hf.space/step"</span> \
  -H <span class="str">"Content-Type: application/json"</span> \
  -d <span class="str">'{
    "session_id": "abc-123-...",
    "action": {
      "solution_code": "impl Solution { pub fn two_sum(...) -> Vec&lt;i32&gt; { ... } }",
      "reasoning_steps": "step-1: Use a HashMap for O(n) lookup.",
      "time_complexity": "O(n)"
    }
  }'</span></pre>
    </div>

    <div class="code-block">
      <div class="code-header">
        <span class="code-dot r"></span>
        <span class="code-dot y"></span>
        <span class="code-dot g"></span>
        <span class="code-lang">stdout format</span>
      </div>
      <pre><span class="kw">[START]</span> task=algo_reasoning env=algo_reasoning_env model=gpt-oss-20b
<span class="kw">[STEP]</span>  step=<span class="num">1</span> action=<span class="str">"solution=[len=120] complexity=[O(n)]"</span> reward=<span class="num">0.85</span> done=<span class="fn">true</span> error=<span class="fn">null</span>
<span class="kw">[STEP]</span>  step=<span class="num">2</span> action=<span class="str">"solution=[len=95]  complexity=[O(n²)]"</span> reward=<span class="num">0.30</span> done=<span class="fn">true</span> error=<span class="fn">null</span>
<span class="kw">[END]</span>   success=<span class="fn">true</span> steps=<span class="num">200</span> score=<span class="num">0.45</span> rewards=<span class="num">0.85</span>,<span class="num">0.30</span>,...</pre>
    </div>
  </section>
</div>

<footer>
  <div class="footer-inner">
    <span>Algo Reasoning Env · OpenEnv · 2025</span>
    <span>952 problems · Rust · Easy / Medium / Hard</span>
  </div>
</footer>

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
