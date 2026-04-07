"""
Baseline inference script for the Algo Reasoning Environment.

This script runs a baseline agent against the environment and logs scores
in the required format for evaluation.

Usage:
    python inference.py

Required environment variables:
    API_BASE_URL: The API endpoint for the LLM
    MODEL_NAME: The model identifier to use
    HF_TOKEN: Your HuggingFace API key (if needed)
    LIGHTNING_API_KEY: API key for Lightning AI
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import List, Optional

from openai import OpenAI

# Environment configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://lightning.ai/api/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "lightning-ai/gpt-oss-20b")
API_KEY = os.getenv("LIGHTNING_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Benchmark configuration
IMAGE_NAME = os.getenv("IMAGE_NAME", "algo-reasoning-env:latest")
BENCHMARK = "algo_reasoning_env"
TASK_NAME = "algo_reasoning"
MAX_STEPS = 10
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.7


def log_start(task: str, env: str, model: str) -> None:
    """
    Log the start of a benchmark run.

    Format: [START] task=<task> env=<env> model=<model>
    """
    print(
        f"[START] task={task} env={env} model={model}",
        flush=True,
    )


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
    rewards_str = json.dumps(rewards)
    print(
        f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


async def main() -> None:
    """Run the baseline inference against the environment."""
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    # Import the environment after checking dependencies
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

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        result = env.reset()
        last_observation = result
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            # Get the problem description to send to the model
            problem_desc = last_observation.problem_description
            starter_code = last_observation.starter_code
            expected_complexity = last_observation.expected_complexity

            # Build prompt for the model
            prompt = build_prompt(
                problem_desc=problem_desc,
                starter_code=starter_code,
                expected_complexity=expected_complexity,
                history=history,
            )

            # Get model response
            model_response = await get_model_response(
                client=client,
                model=MODEL_NAME,
                prompt=prompt,
            )

            if not model_response:
                log_step(
                    step=step,
                    action="MODEL_FAILED",
                    reward=0.0,
                    done=True,
                    error="Model request failed",
                )
                break

            # Parse the response
            solution_code, reasoning_steps, time_complexity = parse_model_response(
                model_response
            )

            # Create action
            action = AlgoReasoningAction(
                solution_code=solution_code,
                reasoning_steps=reasoning_steps,
                time_complexity=time_complexity,
            )

            # Execute step
            result = env.step(action)
            obs = result

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_observation = obs
            last_reward = reward

            # Format action for logging
            action_str = f"solution=[len={len(solution_code)}] reasoning=[{reasoning_steps[:50]}...]"

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: reward={reward:+.2f}, "
                f"correctness={result.correctness_reward}, "
                f"reasoning={result.reasoning_score}, "
                f"complexity={result.complexity_score}"
            )

            if done:
                break

        # Calculate final score
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


def build_prompt(
    problem_desc: str,
    starter_code: str,
    expected_complexity: str,
    history: List[str],
) -> str:
    """Build the prompt for the model."""
    history_text = ""
    if history:
        history_text = "\n\n".join(history)

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


async def get_model_response(
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
    import re

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
    args = parser.parse_args()

    if args.api_url:
        os.environ["API_BASE_URL"] = args.api_url
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    asyncio.run(main())
