"""
FastAPI application for the Algo Reasoning Environment.

This module creates an HTTP server that exposes the AlgoReasoningEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - GET /: Interactive landing page explaining the environment
    - POST /reset: Reset the environment and get a new problem
    - POST /step: Evaluate an agent's submission
    - GET /state: Get current environment state
    - GET /health: Health check

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4
"""

try:
    from openenv.core.env_server.http_server import create_app as create_openenv_app
    from openenv.core.env_server.types import ServerMode
except ImportError as e:
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies with 'pip install openenv>=0.1.0'."
    ) from e

from starlette.responses import HTMLResponse

from algo_reasoning_env import (
    AlgoReasoningAction,
    AlgoReasoningObservation,
    AlgoReasoningEnvironment,
)


def get_landing_html() -> str:
    return """<!DOCTYPE html>
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
    --accent-light: #fef3ed;
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
  <div class="code">observation = {
  "task_id": "two-sum",
  "difficulty": "Easy",
  "problem_description": "Given an array of integers nums and an integer target, ...",
  "starter_code": "impl Solution { pub fn two_sum(...)",
  "test_harness": "fn test_two_sum() { ... }",
  "expected_complexity": "O(n)",
  "ground_truth_explanation": "Use a hash map for constant-time lookup...",
  "tags": ["array", "hash-table"],
  "step": 0,
}</div>
  <div class="note">The agent sees only the problem description and starter code — it must infer the correct algorithm and complexity from scratch.</div>
</div>

<!-- STEP 2: agent forms action -->
<div class="sp" id="s1">
  <div class="step-title">agent produces action</div>
  <div class="note">The agent (LLM) receives the observation and must return a typed <span class="pill pill-a">action</span> with three fields — Rust solution code, step-by-step reasoning, and time complexity label.</div>
  <div class="code">action = AlgoReasoningAction(
  solution_code="""
impl Solution {
    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        for (i, &num) in nums.iter().enumerate() {
            let complement = target - num;
            if let Some(&j) = map.get(&complement) {
                return vec![j as i32, i as i32];
            }
            map.insert(num, i);
        }
        vec![]
    }
}
  """,
  reasoning_steps="step-1: Create a HashMap for O(1) lookup. step-2: Iterate through nums, check if complement exists. step-3: Return indices if found.",
  time_complexity="O(n)",
)</div>
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
# Returns: { compilation_ok: bool, test_output: str, compilation_error: str | None }</div>
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
  <div class="code">StepResult(
  observation = {
    "task_id": "two-sum",
    "problem_description": "...",
    "starter_code": "...",
    "test_harness": "...",
    "expected_complexity": "O(n)",
    "ground_truth_explanation": "...",
    "last_output": [1, 2],          # code produced this output
    "execution_ok": True,
    "step": 1,
  },
  reward = 0.83,
  done   = False,   # continues indefinitely
  info   = {
    "correctness_reward": 1.0,
    "reasoning_score": 0.65,
    "complexity_score": 1,
    "compilation_error": None,
    "test_output": "all tests passed",
  }
)</div>
  <div class="note">The <strong>info dict breakdown</strong> enables RL trainers to shape auxiliary losses per component. The agent sees its own output vs expected, and can refine its complexity claim on the next step.</div>
</div>

<!-- STEP 6: dataset overview -->
<div class="sp" id="s5">
  <div class="step-title">dataset overview &amp; difficulty tiers</div>
  <div class="note" style="margin-bottom:1rem;">The environment cycles through problems in order. After problem N-1, it wraps back to 0. Weighted scoring rewards <strong>frontier-challenging hard problems</strong> — a task where frontier models score 0.3 is better than one where they score 0.9.</div>
  <div class="row">
    <div class="card" style="border-left:3px solid #0a5c36;border-radius:0 var(--radius) var(--radius) 0;">
      <div class="card-label"><span class="pill pill-g">Easy</span></div>
      <div class="card-val" style="margin-bottom:4px;">Sort / reverse / search</div>
      <div style="font-size:0.8rem;color:var(--text-secondary);">3 examples unambiguously reveal the pattern. O(n) or O(n log n). Frontier models score ~0.9.</div>
    </div>
    <div class="card" style="border-left:3px solid #c45c26;border-radius:0 var(--radius) var(--radius) 0;">
      <div class="card-label"><span class="pill pill-a">Medium</span></div>
      <div class="card-val" style="margin-bottom:4px;">Two-pointer / sliding window</div>
      <div style="font-size:0.8rem;color:var(--text-secondary);">Pattern requires seeing the constraint. O(n). Requires real induction.</div>
    </div>
    <div class="card" style="border-left:3px solid #a32d2d;border-radius:0 var(--radius) var(--radius) 0;">
      <div class="card-label"><span class="pill pill-r">Hard</span></div>
      <div class="card-val" style="margin-bottom:4px;">DP / graph algorithms</div>
      <div style="font-size:0.8rem;color:var(--text-secondary);">Examples consistent with multiple algorithm classes. Frontier models score ~0.3–0.4.</div>
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


# Create the FastAPI app
app = create_openenv_app(
    env=AlgoReasoningEnvironment,
    action_cls=AlgoReasoningAction,
    observation_cls=AlgoReasoningObservation,
    env_name="algo_reasoning_env",
    max_concurrent_envs=1,
)


@app.get("/")
async def root():
    return HTMLResponse(get_landing_html())


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
