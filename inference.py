"""
Baseline inference script for the Algo Reasoning Environment.

Communicates with the environment via HTTP (HF Space endpoints) and
logs scores in the strict [START] / [STEP] / [END] format required
for evaluation.

Usage:
    python inference.py
    python inference.py --output results.jsonl

Required environment variables:
    API_KEY        Injected by evaluator at runtime (via LiteLLM proxy).
    API_BASE_URL   Injected by evaluator at runtime.
    MODEL_NAME     The model identifier (default: Qwen/Qwen2.5-72B-Instruct).
    HF_SPACE_URL   The HF Space URL (defaults to the deployed space).
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_SPACE_URL = os.getenv(
    "HF_SPACE_URL",
    "https://tm23hgf-rust-algo-reasoning.hf.space",
)
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "algo_reasoning_env"
TASKS = ["task_easy", "task_medium", "task_hard"]
SUCCESS_SCORE_THRESHOLD = 0.7
REQUEST_TIMEOUT = 120  # seconds for HTTP calls to HF Space


# ---------------------------------------------------------------------------
# Logging helpers — strict [START] / [STEP] / [END] format
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    action_preview = action[:100] + "..." if len(action) > 100 else action
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f'[STEP] step={step} action="{action_preview}" '
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(
    problem_desc: str,
    starter_code: str,
    expected_complexity: str,
) -> str:
    return f"""You are solving a LeetCode problem in Rust.

Below is the starter code with the exact function signature.

{starter_code}

Problem Description:
{problem_desc}

Expected Time Complexity: {expected_complexity}

Your task:
1. Write the complete Rust implementation
2. Provide step-by-step reasoning
3. State the time complexity

Output format:
```rust
impl Solution {{
    pub fn ... {{
        // implementation
    }}
}}

// reasoning
// step-1: ...
// step-2: ...
// step-3: ...
// step-4: ...
// step-5: ...

// time complexity
// O(...)
```
"""


# ---------------------------------------------------------------------------
# Model interaction
# ---------------------------------------------------------------------------


def get_model_response(client: OpenAI, model: str, prompt: str) -> Optional[str]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            temperature=0.0,
            timeout=120,
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return None


def _extract_impl_block(text: str, start: int) -> str:
    """
    Extract a full ``impl Solution { ... }`` block from *text* starting
    at position *start*.  Handles nested braces by tracking depth.

    Returns the extracted block including the outer ``impl Solution { ... }``,
    or an empty string if the braces never balance.
    """
    # Find the opening { after "impl Solution"
    open_brace = text.find("{", start)
    if open_brace == -1:
        return ""

    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def parse_model_response(response: str) -> Tuple[str, str, str]:
    """
    Parse the model response to extract solution, reasoning, and complexity.

    Returns:
        (solution_code, reasoning_steps, time_complexity)
    """
    solution_code = ""
    reasoning_steps = ""
    time_complexity = ""

    # Try fenced code block first (```rust ... ```)
    code_match = re.search(r"```rust\n(.*?)\n```", response, re.DOTALL)
    if code_match:
        solution_code = code_match.group(1).strip()

    # Fallback: find `impl Solution {` and extract via brace counting
    if not solution_code:
        impl_match = re.search(r"impl\s+Solution\s*\{", response)
        if impl_match:
            solution_code = _extract_impl_block(response, impl_match.start())

    # Extract reasoning
    reasoning_match = re.search(
        r"// reasoning\s*\n(.*?)(?=// time complexity|\Z)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if reasoning_match:
        reasoning_steps = reasoning_match.group(1).strip()

    # Extract time complexity
    complexity_match = re.search(
        r"// time complexity\s*\n?//\s*(O\([^)]+(?:\([^)]*\)[^)]*)*\))",
        response,
        re.IGNORECASE,
    )
    if complexity_match:
        time_complexity = complexity_match.group(1).strip()
    else:
        o_match = re.search(r"(O\([^)]+\))", response)
        if o_match:
            time_complexity = o_match.group(1)

    return solution_code, reasoning_steps, time_complexity


# ---------------------------------------------------------------------------
# HF Space HTTP interaction
# ---------------------------------------------------------------------------


def env_reset(task_name: Optional[str] = None) -> Tuple[str, Dict]:
    """
    Call POST /reset on the HF Space.

    Args:
        task_name: Optional task name ("easy", "medium", "hard").
                   Filters problems by the corresponding difficulty.

    Returns:
        (session_id, observation_dict)
    """
    payload: Dict[str, Any] = {}
    if task_name is not None:
        payload["task_name"] = task_name

    resp = requests.post(
        f"{HF_SPACE_URL}/reset",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["session_id"], data["observation"]


def env_step(session_id: str, action: Dict) -> Dict:
    """
    Call POST /step on the HF Space.

    Returns:
        response dict with observation, reward, done
    """
    resp = requests.post(
        f"{HF_SPACE_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Task evaluation
# ---------------------------------------------------------------------------


def run_task(
    client: OpenAI,
    task_name: str,
    output_path: str,
    task_results: List[Dict[str, Any]],
) -> Tuple[float, List[float], int]:
    """
    Run evaluation for a single task (easy/medium/hard).

    Returns:
        (task_score, task_rewards, steps_taken)
    """
    task_rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        session_id, observation = env_reset(task_name=task_name)
    except Exception as e:
        print(f"[DEBUG] Reset for task {task_name} failed: {e}", flush=True)
        log_step(
            step=1,
            action="RESET_FAILED",
            reward=0.0,
            done=True,
            error=str(e)[:200],
        )
        task_results.append(
            {
                "task": task_name,
                "problem_id": "",
                "task_id": "",
                "difficulty": "",
                "correctness_reward": 0.0,
                "reasoning_score": 0.0,
                "complexity_score": 0,
                "predicted_complexity": "",
                "ground_truth_complexity": "",
                "generated_code": "",
                "reasoning_steps": "",
                "reward": 0.0,
                "error": f"Reset failed: {str(e)[:200]}",
            }
        )
        task_rewards.append(0.0)
        steps_taken = 1
        score = 0.01
        log_end(success=False, steps=steps_taken, score=score, rewards=task_rewards)
        return score, task_rewards, steps_taken

    steps_taken = 1

    try:
        problem_desc = observation.get("problem_description", "")
        starter_code = observation.get("starter_code", "")
        expected_complexity = observation.get("expected_complexity", "")

        # Build prompt and get model response
        prompt = build_prompt(
            problem_desc=problem_desc,
            starter_code=starter_code,
            expected_complexity=expected_complexity,
        )

        model_response = get_model_response(client, MODEL_NAME, prompt)

        if not model_response:
            log_step(
                step=steps_taken,
                action="MODEL_FAILED",
                reward=0.0,
                done=True,
                error="Model request failed",
            )
            task_rewards.append(0.0)
            task_results.append(
                {
                    "task": task_name,
                    "problem_id": observation.get("problem_id", ""),
                    "task_id": observation.get("task_id", ""),
                    "difficulty": observation.get("difficulty", ""),
                    "correctness_reward": 0.0,
                    "reasoning_score": 0.0,
                    "complexity_score": 0,
                    "predicted_complexity": "",
                    "ground_truth_complexity": expected_complexity,
                    "generated_code": "",
                    "reasoning_steps": "",
                    "reward": 0.0,
                    "error": "Model request failed",
                }
            )
            score = 0.01
            log_end(success=False, steps=steps_taken, score=score, rewards=task_rewards)
            return score, task_rewards, steps_taken

        # Parse model response
        solution_code, reasoning_steps, time_complexity = parse_model_response(
            model_response
        )

        if not solution_code:
            log_step(
                step=steps_taken,
                action="PARSE_FAILED",
                reward=0.0,
                done=True,
                error="Could not extract solution code",
            )
            task_rewards.append(0.0)
            task_results.append(
                {
                    "task": task_name,
                    "problem_id": observation.get("problem_id", ""),
                    "task_id": observation.get("task_id", ""),
                    "difficulty": observation.get("difficulty", ""),
                    "correctness_reward": 0.0,
                    "reasoning_score": 0.0,
                    "complexity_score": 0,
                    "predicted_complexity": time_complexity,
                    "ground_truth_complexity": expected_complexity,
                    "generated_code": "",
                    "reasoning_steps": reasoning_steps,
                    "reward": 0.0,
                    "error": "Could not extract solution code",
                }
            )
            score = 0.01
            log_end(success=False, steps=steps_taken, score=score, rewards=task_rewards)
            return score, task_rewards, steps_taken

        # Step — submit solution via HF Space
        action = {
            "solution_code": solution_code,
            "reasoning_steps": reasoning_steps,
            "time_complexity": time_complexity,
        }

        result = env_step(session_id, action)
        reward = result.get("reward", 0.0) or 0.0
        obs = result.get("observation", {})

        task_rewards.append(reward)

        # Collect result
        task_results.append(
            {
                "task": task_name,
                "problem_id": obs.get("problem_id", ""),
                "task_id": obs.get("task_id", ""),
                "difficulty": obs.get("difficulty", ""),
                "correctness_reward": obs.get("correctness_reward", 0.0),
                "reasoning_score": obs.get("reasoning_score", 0.0),
                "complexity_score": obs.get("complexity_score", 0),
                "predicted_complexity": (
                    obs.get("evaluation", {}).get(
                        "predicted_complexity", time_complexity
                    )
                ),
                "ground_truth_complexity": obs.get(
                    "expected_complexity", expected_complexity
                ),
                "generated_code": solution_code,
                "reasoning_steps": reasoning_steps,
                "reward": reward,
                "error": obs.get("evaluation", {}).get("compilation_error"),
            }
        )

        # Log step
        action_str = (
            f"solution=[len={len(solution_code)}] "
            f"reasoning=[{reasoning_steps[:50]}...] "
            f"complexity=[{time_complexity}]"
        )
        log_step(
            step=steps_taken,
            action=action_str,
            reward=reward,
            done=True,
            error=None,
        )

        score = reward
        score = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} failed: {e}", flush=True)
        task_rewards.append(0.0)
        task_results.append(
            {
                "task": task_name,
                "problem_id": "",
                "task_id": "",
                "difficulty": "",
                "correctness_reward": 0.0,
                "reasoning_score": 0.0,
                "complexity_score": 0,
                "predicted_complexity": "",
                "ground_truth_complexity": "",
                "generated_code": "",
                "reasoning_steps": "",
                "reward": 0.0,
                "error": str(e)[:200],
            }
        )
        log_step(
            step=steps_taken,
            action="EXCEPTION",
            reward=0.0,
            done=True,
            error=str(e)[:200],
        )
        score = 0.01
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=task_rewards)
    return score, task_rewards, steps_taken


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_inference(output_path: str = "results.jsonl") -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_results: List[Dict[str, Any]] = []
    all_scores: List[float] = []

    try:
        for task_name in TASKS:
            score, _, _ = run_task(
                client=client,
                task_name=task_name,
                output_path=output_path,
                task_results=all_results,
            )
            all_scores.append(score)
    finally:
        # Save results.jsonl
        if all_results:
            with open(output_path, "w") as f:
                for result in all_results:
                    json.dump(result, f)
                    f.write("\n")
            print(f"\nSaved {len(all_results)} results to {output_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline inference for Algo Reasoning Env"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.jsonl",
        help="Output file for results (default: results.jsonl)",
    )
    args = parser.parse_args()

    run_inference(output_path=args.output)
