---
title: Algo Reasoning Environment
emoji: 🧠
colorFrom: yellow
colorTo: gray
sdk: docker
app_file: app.py
pinned: false
---

# Algo Reasoning Environment

An OpenEnv-compatible environment that evaluates AI agents on their ability to solve LeetCode-style problems in Rust, providing accurate reasoning and time complexity analysis.

## Environment Description

The Algo Reasoning Environment presents AI agents with algorithmic problems and evaluates their submissions on three dimensions:

1. **Code Correctness**: Does the Rust code compile and pass test cases?
2. **Reasoning Quality**: Does the agent provide accurate step-by-step reasoning?
3. **Time Complexity**: Does the agent correctly identify the algorithm's time complexity?

### Motivation

This environment was designed to train and evaluate AI agents on:
- Writing correct, efficient Rust code
- Explaining their algorithmic reasoning
- Understanding time and space complexity
- Following problem specifications accurately

## Action and Observation Spaces

### Action Space

Agents submit `AlgoReasoningAction` with:

| Field | Type | Description |
|-------|------|-------------|
| `solution_code` | `str` | The Rust `impl Solution` block |
| `reasoning_steps` | `str` | Step-by-step reasoning (step-1, step-2, etc.) |
| `time_complexity` | `str` | Time complexity (e.g., "O(n)", "O(n^2) in worst case") |

### Observation Space

After `reset()`, agents receive `AlgoReasoningObservation` with:

| Field | Type | Description |
|-------|------|-------------|
| `problem_id` | `int` | Problem ID from dataset |
| `task_id` | `str` | Task identifier (e.g., "two-sum") |
| `difficulty` | `str` | "Easy", "Medium", or "Hard" |
| `problem_description` | `str` | Full problem description |
| `starter_code` | `str` | Starter code template |
| `expected_complexity` | `str` | Ground truth time complexity |
| `ground_truth_explanation` | `str` | Expert explanation for reasoning evaluation |
| `tags` | `list[str]` | Problem tags (for code assembly) |
| `test_harness` | `str` | Test cases for validation |

After `step()`, additional fields are populated:

| Field | Type | Description |
|-------|------|-------------|
| `reward` | `float` | Combined reward (0.5*correctness + 0.3*reasoning + 0.2*complexity) |
| `reasoning_score` | `float` | Reasoning quality score (0.0-1.0) |
| `complexity_score` | `int` | Time complexity correctness (0 or 1) |
| `correctness_reward` | `float` | Code correctness: 0.0, 0.3, or 1.0 |

## Task Descriptions

The environment cycles through problems from the embedded dataset in order (0, 1, 2, ...).

### Example Tasks

**Problem 1: Two Sum (Easy)**
- Find two numbers in an array that add up to a target
- Expected time complexity: O(n)
- Tests multiple input/target combinations

**Problem 2: Add Two Numbers (Medium)**
- Add two linked lists representing reversed numbers
- Expected time complexity: O(max(m, n))

**Problem 3: Longest Substring Without Repeating Characters (Medium)**
- Find the length of the longest substring without repeating characters
- Expected time complexity: O(n)

## Reward Function

The combined reward is computed as:

```
reward = 0.5 * correctness + 0.3 * reasoning + 0.2 * complexity
```

Where:
- **correctness**: 1.0 (compiles + tests pass), 0.3 (compiles + tests fail), 0.0 (doesn't compile)
- **reasoning**: 0.0-1.0 (LLM judge evaluation)
- **complexity**: 0 or 1 (LLM judge semantic matching)

## Setup and Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/algo_reasoning_env.git
cd algo_reasoning_env

# Install dependencies
pip install -e algo_reasoning_env
```

### Running the Server

```bash
# Set required environment variables
export LIGHTNING_API_KEY="your_api_key"
export DATA_DIR="/path/to/data"

# Run the server
uvicorn algo_reasoning_env.server.app:app --host 0.0.0.0 --port 7860
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LIGHTNING_API_KEY` | Yes | API key for LLM judge calls |
| `DATA_DIR` | No | Path to dataset files (default: `/data`) |
| `API_BASE_URL` | No | LLM API base URL (default: `https://lightning.ai/api/v1/`) |
| `MODEL_NAME` | No | LLM model name (default: `lightning-ai/gpt-oss-20b`) |

### Using with OpenEnv Client

```python
from algo_reasoning_env import AlgoReasoningEnvironment, AlgoReasoningAction

# Create environment
env = AlgoReasoningEnvironment(data_dir="/data")

# Get a problem
observation = env.reset()
print(f"Problem: {observation.task_id}")
print(f"Description: {observation.problem_description}")

# Submit a solution
action = AlgoReasoningAction(
    solution_code="impl Solution { pub fn two_sum(...) }",
    reasoning_steps="step-1: Use hashmap...",
    time_complexity="O(n)",
)

# Evaluate
result = env.step(action)
print(f"Reward: {result.reward}")
```

## Docker Deployment

### Build the Docker Image

```bash
# Build from algo_reasoning_env/server/
cd algo_reasoning_env/server
docker build -t algo-reasoning-env .

# Or build from root with Makefile
make docker-build
```

### Run the Container

```bash
docker run -p 7860:7860 \
  -e LIGHTNING_API_KEY="your_api_key" \
  -v /path/to/data:/data \
  algo-reasoning-env
```

## Baseline Scores

To run the baseline inference script:

```bash
# Set environment variables
export LIGHTNING_API_KEY="your_api_key"
export API_BASE_URL="https://lightning.ai/api/v1/"
export MODEL_NAME="lightning-ai/gpt-oss-20b"

# Run baseline
python inference.py
```

Expected output format:
```
[START] task=algo_reasoning env=benchmark model=gpt-oss-20b
[STEP] step=1 action="solution=[...]" reward=0.00 done=false error=None
[STEP] step=2 action="solution=[...]" reward=0.35 done=false error=None
[STEP] step=3 action="solution=[...]" reward=0.85 done=true error=None
[END] success=true steps=3 score=0.85 rewards=[0.0, 0.35, 0.85]
```

## Validation

To validate the submission:

```bash
# Start the server in another terminal
uvicorn algo_reasoning_env.server.app:app --port 7860

# Run validation
chmod +x validate-submission.sh
./validate-submission.sh http://localhost:7860
```

## Dataset Structure

The environment expects the following files in the data directory:

1. `dataset.jsonl` - Problem descriptions with fields:
   - `problem_id`, `task_id`, `difficulty`, `tags`
   - `problem_description`, `time_complexity`, `explanation`

2. `starter_codes.jsonl` - Starter code templates:
   - `problem_id`, `function_name`, `starter_code`

3. `test_harness.jsonl` - Test cases:
   - `problem_id`, `harness`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment and get new problem |
| `/step` | POST | Evaluate agent submission |
| `/state` | GET | Get current environment state |
| `/health` | GET | Health check |

## License

BSD-style license (see LICENSE file)
