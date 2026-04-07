"""
Phase 4: Code Generation Pipeline

Generates Rust solution code from starter codes using Lightning AI API (Nemotron model).

Main components:
1. build_generation_prompt() — Create API prompt from problem data
2. generate_one() — Generate code for single problem via API
3. generate_batch() — Process problems in sequence
4. extract_generated_code() — Extract Rust code from raw API response
5. save_generated() — Write generated_solutions.jsonl
6. save_debug() — Write code_gen_debug.jsonl

CLI Usage:
    python pipeline/code_generator.py --num-problems 50

    Or set LIGHTNING_API_KEY env var and run:
    python pipeline/code_generator.py [--num-problems 50]
"""

import json
import os
import re
import argparse
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from openai import OpenAI

from .config import (
    LIGHTNING_BASE_URL,
    LLM_MODEL,
    GENERATION_TIMEOUT,
    get_api_key,
)


def load_starter_codes(starter_codes_path: str) -> Dict[int, Dict]:
    """
    Load starter codes from starter_codes.jsonl.

    Expected format: {"problem_id": 1, "function_name": "two_sum", "starter_code": "..."}

    Args:
        starter_codes_path: Path to starter_codes.jsonl

    Returns:
        Dict mapping problem_id -> {function_name, starter_code}
    """
    starter_map = {}

    if not os.path.isfile(starter_codes_path):
        print(f"Warning: Starter codes file not found: {starter_codes_path}")
        return starter_map

    try:
        with open(starter_codes_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    problem_id = data.get("problem_id")
                    function_name = data.get("function_name")
                    starter_code = data.get("starter_code")
                    if problem_id and function_name and starter_code:
                        starter_map[problem_id] = {
                            "function_name": function_name,
                            "starter_code": starter_code,
                        }
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in starter codes: {e}")
    except IOError as e:
        print(f"Error reading starter codes: {e}")

    return starter_map


def build_generation_prompt(problem: Dict, starter_code_info: Dict) -> str:
    """
    Build a prompt for Lightning AI API to generate Rust solution code.

    Args:
        problem: Problem dict with problem_id, task_id, problem_description
        starter_code_info: Dict with function_name and starter_code

    Returns:
        Formatted prompt for API
    """
    function_name = starter_code_info["function_name"]
    starter_code = starter_code_info["starter_code"]
    problem_description = problem.get("problem_description", "")
    task_id = problem.get("task_id", "unknown")

    prompt = f"""You are solving a LeetCode problem in Rust.
Below is the starter code with the exact function signature.
Fill in the implementation where "// OUR CODE GOES HERE" appears.

═══════════════════════════════════════
STARTER CODE
═══════════════════════════════════════
{starter_code}

═══════════════════════════════════════
PROBLEM DESCRIPTION
═══════════════════════════════════════
{task_id} (LeetCode #{problem.get("problem_id", "?")})

{problem_description}

═══════════════════════════════════════
RULES
═══════════════════════════════════════
1. Output ONLY the complete impl Solution block — no explanation outside the code
2. Keep the exact function signature as given
3. Do NOT add helper functions outside impl Solution
4. Your code must compile with rustc
5. After the impl block, add a reasoning section as comments

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════
Output ONLY a Rust code block — no prose, no explanation outside the code block:

```rust
impl Solution {{
    pub fn {function_name}(...) -> ... {{
        // your implementation
    }}
}}

// reasoning
// step-1: ...
// step-2: ...
// step-3: ...
// step-4: ...
// step-5: ...

// time complexity
// O(...)  [ONLY the Big-O notation, e.g. O(n), O(n^2), O(m*n), O(log n)]
```
"""
    return prompt


def extract_generated_code(raw_output: str) -> str:
    """
    Extract Rust code block from raw API response.

    Tries multiple patterns:
    1. ```rust ... ``` code fence
    2. impl Solution { ... } patterns
    3. Falls back to raw output if neither found

    Args:
        raw_output: Raw response from API

    Returns:
        Extracted and cleaned Rust code
    """
    # Try extracting code fence ```rust ... ```
    pattern = r"```rust\n(.*?)\n```"
    match = re.search(pattern, raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try extracting without language specifier ```...```
    pattern = r"```\n(.*?)\n```"
    match = re.search(pattern, raw_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Check if it looks like Rust impl
        if "impl Solution" in code or "pub fn" in code:
            return code

    # Try extracting impl Solution block
    pattern = r"(impl Solution \{.*?\})"
    match = re.search(pattern, raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: if response contains impl Solution, extract it
    if "impl Solution" in raw_output:
        start = raw_output.find("impl Solution")
        # Find the closing brace of impl block
        brace_count = 0
        end = start
        for i, char in enumerate(raw_output[start:], start):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        if end > start:
            return raw_output[start:end].strip()

    # Last resort: return raw output
    return raw_output.strip()


def generate_one(
    problem: Dict,
    starter_code_info: Dict,
    api_key: Optional[str] = None,
    model: str = LLM_MODEL,
    timeout: int = GENERATION_TIMEOUT,
) -> Dict:
    """
    Generate Rust code for a single problem via Lightning AI API.

    Args:
        problem: Problem dict
        starter_code_info: Dict with function_name and starter_code
        api_key: Lightning AI API key (optional, reads from env if not provided)
        model: Model to use
        timeout: Request timeout in seconds

    Returns:
        Dict with problem_id, generated_code, raw_output, status, error
    """
    prompt = build_generation_prompt(problem, starter_code_info)

    try:
        # Use OpenAI client with Lightning AI base URL
        client = OpenAI(
            base_url=LIGHTNING_BASE_URL,
            api_key=api_key or get_api_key(),
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            timeout=timeout,
        )

        raw = response.choices[0].message.content
        generated_code = extract_generated_code(raw)

        return {
            "problem_id": problem["problem_id"],
            "generated_code": generated_code,
            "raw_output": raw,
            "status": "success",
            "error": None,
        }

    except Exception as e:
        error_str = str(e)
        # Handle specific error types
        if "429" in error_str or "rate limit" in error_str.lower():
            return {
                "problem_id": problem["problem_id"],
                "generated_code": None,
                "raw_output": None,
                "status": "rate_limited",
                "error": "429 Rate Limited",
            }
        elif "timeout" in error_str.lower():
            return {
                "problem_id": problem["problem_id"],
                "generated_code": None,
                "raw_output": None,
                "status": "timeout",
                "error": f"Request timeout ({timeout}s)",
            }
        else:
            return {
                "problem_id": problem["problem_id"],
                "generated_code": None,
                "raw_output": None,
                "status": "error",
                "error": error_str,
            }


def generate_batch(
    problems: List[Dict],
    starter_codes_map: Dict[int, Dict],
    api_key: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate code for problems in sequence.

    Args:
        problems: List of problem dicts
        starter_codes_map: Dict mapping problem_id -> {function_name, starter_code}
        api_key: Lightning AI API key

    Returns:
        Tuple of (successes, failures)
    """
    successes = []
    failures = []

    # Filter problems that have starter codes
    problems_to_generate = [p for p in problems if p["problem_id"] in starter_codes_map]

    print(f"Generating code for {len(problems_to_generate)} problems...")
    print()

    for problem in problems_to_generate:
        problem_id = problem["problem_id"]
        starter_code_info = starter_codes_map[problem_id]

        print(f"Generating for problem #{problem_id}...", end=" ", flush=True)

        result = generate_one(problem, starter_code_info, api_key)

        if result["status"] == "success":
            successes.append(result)
            print("✅")
        else:
            failures.append(result)
            print(f"❌ {result['error']}")

    return successes, failures


def save_generated(results: List[Dict], path: str):
    """
    Append successful generated code to generated_solutions.jsonl

    Args:
        results: List of generation results
        path: Output file path
    """
    with open(path, "a") as f:
        for r in results:
            if r["status"] == "success" and r["generated_code"]:
                json.dump(
                    {
                        "problem_id": r["problem_id"],
                        "generated_code": r["generated_code"],
                    },
                    f,
                )
                f.write("\n")


def save_debug(results: List[Dict], path: str):
    """
    Append raw API responses to code_gen_debug.jsonl

    Args:
        results: List of generation results
        path: Output file path
    """
    with open(path, "a") as f:
        for r in results:
            json.dump(
                {
                    "problem_id": r["problem_id"],
                    "status": r["status"],
                    "error": r["error"],
                    "raw_output": r["raw_output"],
                },
                f,
            )
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4: Generate Rust solution code via Lightning AI API"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Lightning AI API key (or use LIGHTNING_API_KEY env var)",
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=50,
        help="Number of problems to process (default: 50)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="complexity_reasoning_data/dataset.jsonl",
        help="Path to dataset.jsonl file",
    )
    parser.add_argument(
        "--starter-codes-path",
        type=str,
        default="complexity_reasoning_data/starter_codes.jsonl",
        help="Path to starter_codes.jsonl file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="complexity_reasoning_data",
        help="Output directory for generated code",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="API request timeout in seconds (default: 120)",
    )

    args = parser.parse_args()

    # Get API key from CLI arg or environment variable
    api_key = args.api_key or os.getenv("LIGHTNING_API_KEY")

    if not api_key:
        print("❌ Error: Lightning AI API key not found!")
        print("   Set LIGHTNING_API_KEY environment variable or use --api-key argument")
        exit(1)

    # Prepare paths
    dataset_path = args.dataset_path
    starter_codes_path = args.starter_codes_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_output = output_dir / "generated_solutions.jsonl"
    debug_output = output_dir / "code_gen_debug.jsonl"

    print(f"Phase 4: Code Generation Pipeline")
    print(f"=" * 70)
    print(f"API Key: {api_key[:20]}...{api_key[-30:]}")
    print(f"Num problems: {args.num_problems}")
    print(f"Dataset path: {dataset_path}")
    print(f"Starter codes path: {starter_codes_path}")
    print(f"Output directory: {output_dir}")
    print(f"Timeout: {args.timeout}s")
    print()

    # Load starter codes
    print("Loading starter codes...")
    starter_codes_map = load_starter_codes(starter_codes_path)
    print(f"✅ Loaded {len(starter_codes_map)} starter codes")

    # Load dataset
    print("Loading dataset...")
    # problems = []

    # with open(dataset_path) as f:
    #     for i, line in enumerate(f):
    #         if i >= args.num_problems:
    #             break
    #         problem = json.loads(line)
    #         problems.append(problem)

    # print(
    #     f"✅ Loaded {len(problems)} problems from dataset (first {args.num_problems})"
    # )

    problems = []

    with open(dataset_path) as f:
        for line in f:
            problem = json.loads(line)
            problem_id = problem.get("id") or problem.get(
                "problem_id"
            )  # adjust key if needed

            if problem_id is not None and 141 <= problem_id <= 145:
                problems.append(problem)

            # Optional: early stop if we already have all 6 problems
            if len(problems) >= 6:
                break

    print(f"Loaded {len(problems)} problems (141 to 145)")

    # Run generation pipeline
    print()
    successes, failures = generate_batch(problems, starter_codes_map, api_key)

    # Save results
    print()
    print(f"Saving results...")
    save_generated(successes, str(generated_output))
    save_debug(successes + failures, str(debug_output))

    # Summary
    print()
    print("=" * 70)
    print(f"✅ Generation complete!")
    print(f"  Successful: {len(successes)}")
    print(f"  Failed: {len(failures)}")
    print(f"  Output:")
    print(f"    - {generated_output} ({len(successes)} solutions)")
    print(
        f"    - {debug_output} (debug info for {len(successes) + len(failures)} generations)"
    )
    print("=" * 70)
