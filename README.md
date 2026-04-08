---
title: Algo Reasoning Environment
emoji: 🧠
colorFrom: yellow
colorTo: gray
sdk: docker
app_file: app.py
pinned: false
tags:
  - openenv
---

# Algo Reasoning Environment

**Can an AI write correct Rust code, explain its reasoning, AND identify the right time complexity — all at once?**

Most coding benchmarks ask one question: *does the code pass the tests?* We ask three. This environment evaluates AI agents on **code correctness**, **reasoning quality**, and **algorithmic complexity understanding** simultaneously — in a language where getting things right is genuinely hard: Rust.

---

## Why Rust?

LeetCode's native language is Python. So why force models into Rust?

**Compilation is a hard correctness gate.** Rust's compiler doesn't forgive. Code either compiles or it doesn't — no runtime surprises, no duck typing to hide incomplete understanding. This gives us a binary signal that Python's interpreter can't provide: `0.0` for non-compiling code, `0.3` for compiling-but-failing, `1.0` for passing tests.

**The type system demands precision.** A model must understand `Vec<i32>` vs `Vec<Vec<i32>>`, `Option<Box<ListNode>>` for linked lists, `Option<Rc<RefCell<TreeNode>>>` for trees. There's no "close enough" — the type signature is either right or the code doesn't compile.

**Ownership creates novel reasoning challenges.** Linked lists and trees in Rust require understanding borrowing, ownership transfer, and smart pointers. This goes beyond algorithm correctness — it tests whether the model can reason about memory safety.

**Deterministic local evaluation.** `rustc --test` compiles to a binary that runs in a temp directory. No interpreter, no virtual environment, no dependency management. The entire evaluation is a subprocess with a 30-second timeout.

---

## The Dataset — Built from the Ground Up

This isn't a wrapper around an existing benchmark. The entire dataset was constructed from scratch through a multi-phase pipeline:

### Phase 1: Source Extraction
Problem descriptions sourced from LeetCode. Expert explanations and time complexity annotations extracted from the doocs/leetcode repository — detailed algorithmic writeups originally in Chinese and English, parsed to capture step-by-step reasoning and Big-O notation.

### Phase 2: Rust Starter Code
Function signatures imported from `rustgym_eng` and doocs/leetcode's Rust solutions. Each starter code template provides the exact `pub fn` signature the model must implement.

### Phase 3: Test Harness Conversion
2,641 Python test cases converted to Rust using LLMs with detailed conversion rules: type mappings (`list` → `Vec`, `None` → `None`), linked list construction, floating point comparison tolerances, order-agnostic result checking. Transformed from `fn check()` format to Rust's standard `#[cfg(test)] mod tests` framework.

### Phase 4: Solution Generation with Rollback
LLM-generated Rust solutions compiled and tested up to 3 times. On failure, the compiler error or test output is fed back to the model for self-correction — creating higher-quality reference solutions through iterative refinement.

### Phase 5: Assembly
Each problem assembled with tag-based boilerplate injection: `ListNode` struct prepended for linked list problems, `TreeNode` for tree problems. The model doesn't need to define data structures — it focuses on the algorithm.

### Result
**952 usable problems** with descriptions, Rust starter code, test harnesses, expert explanations, and time complexity annotations.

| Difficulty | Count | Share | Typical Challenge |
|-----------|-------|-------|-------------------|
| Easy | 347 | 36.5% | Sort / reverse / search |
| Medium | 481 | 50.5% | Two-pointer / sliding window |
| Hard | 124 | 13.0% | DP / graph algorithms |

**Top tags:** Array (1,278), String (510), Hash Table (430), Dynamic Programming (387), Math (355), Sorting (320), Greedy (301)

---

## Three-Dimensional Evaluation

Most coding benchmarks are one-dimensional: pass/fail. We evaluate three independent dimensions because they test different capabilities:

| Dimension | Weight | Score Range | What It Tests |
|-----------|--------|-------------|---------------|
| **Correctness** | 50% | 0.0 / 0.3 / 1.0 | Does the Rust code compile and pass tests? |
| **Reasoning** | 30% | 0.0 — 1.0 | Can the model explain *why* the algorithm works? |
| **Complexity** | 20% | 0 or 1 | Does the model understand algorithmic efficiency? |

### Why These Weights?

**Correctness at 50%** — the dominant signal. In Rust, producing code that compiles is itself a significant achievement. The model must handle exact type signatures, ownership rules, and borrow checking. The three-tier scoring (0.0 / 0.3 / 1.0) creates a gradient: non-compiling code gets nothing, compiling code gets partial credit, passing code gets full credit.

**Reasoning at 30%** — the understanding signal. An LLM judge compares the model's step-by-step reasoning against expert ground truth explanations. Can it identify the right algorithm? Mention key data structures? Describe the logic flow without misconceptions? Scored 0.0–1.0 on a continuous scale.

**Complexity at 20%** — the efficiency signal. Simpler to achieve (just state "O(n)"), scored binarily (0 or 1). The LLM judge performs semantic matching: `O(m*n)` equals `O(n*m)`, `O(max(m,n))` equals `O(m+n)`. Weighted lower because it's easier, but still essential for evaluating true algorithmic understanding.

### Difficulty Multiplier

Harder problems contribute more to the final score:

| Difficulty | Multiplier |
|-----------|-----------|
| Easy | 0.3x |
| Medium | 0.5x |
| Hard | 1.0x |

A hard problem is worth **3.3x** an easy problem. This incentivizes agents to tackle challenging problems rather than optimizing for easy wins.

---

## Action and Observation Spaces

### Action Space

Agents submit an action with three fields:

| Field | Type | Description |
|-------|------|-------------|
| `solution_code` | `str` | Rust `impl Solution { pub fn ... }` block |
| `reasoning_steps` | `str` | Step-by-step reasoning (step-1, step-2, ...) |
| `time_complexity` | `str` | Time complexity (e.g., "O(n)", "O(n^2) in worst case") |

### Observation Space

After `reset()`, agents receive:

| Field | Type | Description |
|-------|------|-------------|
| `problem_id` | `int` | Problem ID from dataset |
| `task_id` | `str` | Task identifier (e.g., "two-sum") |
| `difficulty` | `str` | "Easy", "Medium", or "Hard" |
| `problem_description` | `str` | Full problem description |
| `starter_code` | `str` | Starter code template with exact function signature |
| `expected_complexity` | `str` | Ground truth time complexity |
| `ground_truth_explanation` | `str` | Expert explanation for reasoning evaluation |
| `tags` | `list[str]` | Problem tags (Array, Hash Table, etc.) |
| `test_harness` | `str` | Rust test cases for validation |

After `step()`, additional fields are populated:

| Field | Type | Description |
|-------|------|-------------|
| `reward` | `float` | Combined reward (0.5 * correctness + 0.3 * reasoning + 0.2 * complexity) |
| `reasoning_score` | `float` | Reasoning quality score (0.0–1.0) |
| `complexity_score` | `int` | Time complexity correctness (0 or 1) |
| `correctness_reward` | `float` | Code correctness: 0.0, 0.3, or 1.0 |

---

## Setup and Usage

### Installation

```bash
git clone https://github.com/your-repo/algo_reasoning_env.git
cd algo_reasoning_env
pip install -e algo_reasoning_env
```

### Running the Server

```bash
export LIGHTNING_API_KEY="your_api_key"
export DATA_DIR="/path/to/data"

uvicorn algo_reasoning_env.server.app:app --host 0.0.0.0 --port 7860
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LIGHTNING_API_KEY` | Yes | API key for LLM judge calls |
| `DATA_DIR` | No | Path to dataset files (default: `/data`) |
| `API_BASE_URL` | No | LLM API base URL (default: `https://lightning.ai/api/v1/`) |
| `MODEL_NAME` | No | LLM model name (default: `lightning-ai/gpt-oss-20b`) |

### Using via HTTP API

The server uses session-based state management. Each `/reset` creates a session and returns a `session_id` that must be passed to `/step`.

```bash
# 1. Reset — get a problem and session_id
curl -X POST "https://your-space.hf.space/reset" \
  -H "Content-Type: application/json" \
  -d '{}'

# Response:
# {"session_id": "abc-123-...", "observation": {"problem_id": 1, "task_id": "two-sum", ...}}

# 2. Step — submit solution with session_id
curl -X POST "https://your-space.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc-123-...",
    "action": {
      "solution_code": "impl Solution { pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> { ... } }",
      "reasoning_steps": "step-1: Create a HashMap. step-2: Iterate and check complement.",
      "time_complexity": "O(n)"
    }
  }'

# Response:
# {"observation": {...}, "reward": 0.85, "done": true}
```

There is also a stateless `/evaluate` endpoint that combines reset+step:

```bash
curl -X POST "https://your-space.hf.space/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "solution_code": "impl Solution { pub fn two_sum(...) -> Vec<i32> { ... } }",
    "reasoning_steps": "step-1: Use hashmap for O(n) lookup.",
    "time_complexity": "O(n)"
  }'
```

### Using with Python (Direct)

```python
from algo_reasoning_env import AlgoReasoningEnvironment, AlgoReasoningAction

env = AlgoReasoningEnvironment(data_dir="/data")
observation = env.reset()
print(f"Problem: {observation.task_id}")
print(f"Description: {observation.problem_description}")

action = AlgoReasoningAction(
    solution_code="impl Solution { pub fn two_sum(...) }",
    reasoning_steps="step-1: Use hashmap...",
    time_complexity="O(n)",
)

result = env.step(action)
print(f"Reward: {result.reward}")
```

---

## Baseline Scores

```bash
export LIGHTNING_API_KEY="your_api_key"
export API_BASE_URL="https://lightning.ai/api/v1/"
export MODEL_NAME="lightning-ai/gpt-oss-20b"

# Run baseline (default: 200 problems)
python inference.py

# Run all 952 problems
python inference.py --num-problems 952

# Quick test
python inference.py --num-problems 10
```

Expected output:
```
[START] task=algo_reasoning env=algo_reasoning_env model=lightning-ai/gpt-oss-20b
[STEP] step=1 action="solution=[len=120] reasoning=[step-1: Create HashMap...] complexity=[O(n)]" reward=+0.85 done=True error=None
[STEP] step=2 action="solution=[len=95] reasoning=[step-1: Use two pointers...] complexity=[O(n)]" reward=+0.30 done=True error=None
...
[END] success=true steps=200 score=0.4500 rewards=[0.85, 0.30, ...]
```

---

## Validation

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://your-space.hf.space
```

The validator runs 3 checks:
1. **HF Space is live** — POST to `/reset`, expect HTTP 200
2. **Docker builds** — `docker build` succeeds within 600s
3. **OpenEnv spec valid** — `openenv validate` passes

---

## Docker Deployment

```bash
docker build -t algo-reasoning-env .

docker run -p 7860:7860 \
  -e LIGHTNING_API_KEY="your_api_key" \
  -v /path/to/data:/data \
  algo-reasoning-env
```

The server runs with `--workers 1` to ensure in-memory session state is shared across HTTP requests.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment, get new problem + `session_id` |
| `/step` | POST | Evaluate agent submission (requires `session_id`) |
| `/evaluate` | POST | Stateless combined reset+step |
| `/state` | GET | Server state (active sessions) |
| `/health` | GET | Health check |

---

## License

BSD-style license (see LICENSE file)
