"""
Baseline inference script for the Algo Reasoning Environment.

This script runs a baseline agent against all problems in the environment
and logs scores in the required format for evaluation.

Usage:
    python inference.py
    python inference.py --num-problems 10
    python inference.py --api-url https://api.example.com --model my-model

Required environment variables:
    API_BASE_URL: The API endpoint for the LLM
    MODEL_NAME: The model identifier to use
    HF_TOKEN: Your HuggingFace API key (if needed)
    LIGHTNING_API_KEY: API key for Lightning AI
"""

import argparse
import json
import os
import re
from typing import List, Optional

from openai import OpenAI

# Environment configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://lightning.ai/api/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "lightning-ai/gpt-oss-20b")
API_KEY = os.getenv("LIGHTNING_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Benchmark configuration
BENCHMARK = "algo_reasoning_env"
TASK_NAME = "algo_reasoning"
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Logging helpers — strict [START] / [STEP] / [END] format
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    """
    Log the start of a benchmark run.

    Format: [START] task=<task> env=<env> model=<model>
    """
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    """
    Log each step of the benchmark.

    Format: [STEP] step=<n> action="<action>" reward=<reward> done=<done> error=<error>
    """
    action_preview = action[:100] + "..." if len(action) > 100 else action
    error_str = f'error="{error}"' if error else "error=None"
    print(
        f'[STEP] step={step} action="{action_preview}" reward={reward:+.2f} done={done} {error_str}',
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    """
    Log the end of a benchmark run.

    Format: [END] success=<success> steps=<n> score=<score> rewards=<rewards>
    """
    rewards_str = json.dumps([round(r, 4) for r in rewards])
    print(
        f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(
    problem_desc: str,
    starter_code: str,
    expected_complexity: str,
    history: List[str],
) -> str:
    """Build the prompt for the model."""
    history_text = ""
    if history:
        history_text = "\n\nPrevious attempts:\n" + "\n\n".join(history)

    prompt = f"""You are solving a LeetCode problem in Rust.

Below is the starter code with the exact function signature.

{starter_code}

Problem Description:
{problem_desc}

Expected Time Complexity: {expected_complexity}

{history_text}

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
    return prompt


# ---------------------------------------------------------------------------
# Model interaction
# ---------------------------------------------------------------------------


def get_model_response(
    client: OpenAI,
    model: str,
    prompt: str,
) -> Optional[str]:
    """Get response from the model."""
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


def parse_model_response(response: str) -> tuple[str, str, str]:
    """
    Parse the model response to extract solution, reasoning, and complexity.

    Returns:
        Tuple of (solution_code, reasoning_steps, time_complexity)
    """
    solution_code = ""
    reasoning_steps = ""
    time_complexity = ""

    # Extract code block
    code_match = re.search(r"```rust\n(.*?)\n```", response, re.DOTALL)
    if code_match:
        solution_code = code_match.group(1)

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
        # Fallback: find any O(...)
        o_match = re.search(r"(O\([^)]+\))", response)
        if o_match:
            time_complexity = o_match.group(1)

    return solution_code, reasoning_steps, time_complexity


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_inference(num_problems: int = 952) -> None:
    """
    Run baseline inference across multiple problems.

    Args:
        num_problems: Number of problems to evaluate (default: 952)
    """
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    # Import the environment
    try:
        from algo_reasoning_env import AlgoReasoningEnvironment, AlgoReasoningAction
    except ImportError:
        print(
            "[ERROR] algo_reasoning_env not installed. Install with: pip install -e algo_reasoning_env"
        )
        return

    # Initialize environment
    env = AlgoReasoningEnvironment(
        data_dir=os.getenv("DATA_DIR", "/data"),
        api_key=API_KEY,
    )

    all_rewards: List[float] = []
    problems_completed = 0
    problems_succeeded = 0
    total_score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        for problem_idx in range(1, num_problems + 1):
            try:
                # Reset — get next problem
                observation = env.reset()

                problem_desc = observation.problem_description
                starter_code = observation.starter_code
                expected_complexity = observation.expected_complexity

                # Build prompt
                prompt = build_prompt(
                    problem_desc=problem_desc,
                    starter_code=starter_code,
                    expected_complexity=expected_complexity,
                    history=[],
                )

                # Get model response
                model_response = get_model_response(
                    client=client,
                    model=MODEL_NAME,
                    prompt=prompt,
                )

                if not model_response:
                    log_step(
                        step=problem_idx,
                        action="MODEL_FAILED",
                        reward=0.0,
                        done=True,
                        error="Model request failed",
                    )
                    all_rewards.append(0.0)
                    problems_completed += 1
                    continue

                # Parse response
                solution_code, reasoning_steps, time_complexity = parse_model_response(
                    model_response
                )

                if not solution_code:
                    log_step(
                        step=problem_idx,
                        action="PARSE_FAILED",
                        reward=0.0,
                        done=True,
                        error="Could not extract solution code from model response",
                    )
                    all_rewards.append(0.0)
                    problems_completed += 1
                    continue

                # Create action and step
                action = AlgoReasoningAction(
                    solution_code=solution_code,
                    reasoning_steps=reasoning_steps,
                    time_complexity=time_complexity,
                )

                result = env.step(action)
                reward = result.reward or 0.0

                all_rewards.append(reward)
                problems_completed += 1

                if reward >= 0.5:
                    problems_succeeded += 1

                # Log step
                action_str = (
                    f"solution=[len={len(solution_code)}] "
                    f"reasoning=[{reasoning_steps[:50]}...] "
                    f"complexity=[{time_complexity}]"
                )
                log_step(
                    step=problem_idx,
                    action=action_str,
                    reward=reward,
                    done=True,
                    error=None,
                )

            except Exception as e:
                print(f"[DEBUG] Problem {problem_idx} failed: {e}", flush=True)
                all_rewards.append(0.0)
                problems_completed += 1
                log_step(
                    step=problem_idx,
                    action="EXCEPTION",
                    reward=0.0,
                    done=True,
                    error=str(e)[:200],
                )

        # Calculate aggregate score
        if all_rewards:
            total_score = sum(all_rewards) / len(all_rewards)
            total_score = min(max(total_score, 0.0), 1.0)

        success = total_score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(
            success=success,
            steps=problems_completed,
            score=total_score,
            rewards=all_rewards,
        )

        # Summary
        print(f"\n--- Summary ---", flush=True)
        print(f"Problems attempted: {problems_completed}", flush=True)
        print(f"Problems succeeded (reward >= 0.5): {problems_succeeded}", flush=True)
        print(f"Average reward: {total_score:.4f}", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline inference for Algo Reasoning Env"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="API base URL (overrides API_BASE_URL env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides MODEL_NAME env var)",
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=952,
        help="Number of problems to evaluate (default: 952)",
    )
    args = parser.parse_args()

    if args.api_url:
        os.environ["API_BASE_URL"] = args.api_url
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    run_inference(num_problems=args.num_problems)
