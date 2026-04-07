"""
Pipeline Orchestrator: run_pipeline.py

Runs the complete pipeline with unified generation + testing per problem:
1. For each problem: generate code → test → if fail, retry up to 3 times
2. Assemble code with header structs based on tags
3. Compute rewards and save results

Usage:
    python pipeline/run_pipeline.py --num-problems 50
    python pipeline/run_pipeline.py --num-problems 100 --output results.jsonl

    Requires LIGHTNING_API_KEY env var:
    export LIGHTNING_API_KEY=your_key
    python pipeline/run_pipeline.py --num-problems 50
"""

import json
import os
import re
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI

# Import from pipeline modules
from assembler import (
    assemble_rust_code_v2,
    transform_harness_to_test_format,
    extract_time_complexity,
    extract_raw_time_complexity,
    extract_reasoning_steps,
)
from llm_judge import LLMJudge
from config import (
    LIGHTNING_BASE_URL,
    LLM_MODEL,
    GENERATION_TIMEOUT,
    get_api_key,
)


# ==================== Data Loading ====================


def load_jsonl(path: str) -> Dict[int, Dict]:
    """Load a JSONL file into a dict keyed by problem_id."""
    result = {}
    if not os.path.isfile(path):
        return result
    with open(path) as f:
        for line in f:
            try:
                data = json.loads(line)
                pid = data.get("problem_id")
                if pid is not None:
                    result[pid] = data
            except json.JSONDecodeError:
                pass
    return result


def load_test_harness(path: str) -> Dict[int, str]:
    """Load test harness from test_harness.jsonl."""
    data = load_jsonl(path)
    return {pid: entry["harness"] for pid, entry in data.items()}


def load_dataset(path: str) -> Dict[int, Dict]:
    """Load dataset from dataset.jsonl."""
    return load_jsonl(path)


def load_starter_codes(path: str) -> Dict[int, Dict]:
    """Load starter codes from starter_codes.jsonl."""
    data = load_jsonl(path)
    return {
        pid: {
            "function_name": entry["function_name"],
            "starter_code": entry["starter_code"],
        }
        for pid, entry in data.items()
    }


# ==================== LLM Code Generation ====================


def generate_initial_code(
    problem: Dict,
    starter_code_info: Dict,
    api_key: str,
) -> Optional[str]:
    """
    Generate initial code using the LLM.

    Returns the generated impl Solution block, or None if failed.
    """
    function_name = starter_code_info["function_name"]
    starter_code = starter_code_info["starter_code"]
    problem_description = problem.get("problem_description", "")
    task_id = problem.get("task_id", "unknown")

    prompt = f"""You are solving a LeetCode problem in Rust.
Below is the starter code with the exact function signature.
Fill in the implementation.

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
1. Output ONLY the complete impl Solution block
2. Keep the exact function signature as given
3. Do NOT add helper functions outside impl Solution
4. Your code must compile with rustc
5. After the impl block, add reasoning comments if desired

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════
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
    try:
        client = OpenAI(
            base_url=LIGHTNING_BASE_URL,
            api_key=api_key,
        )

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            timeout=GENERATION_TIMEOUT,
        )
        raw = response.choices[0].message.content

        # Extract code block
        pattern = r"```rust\n(.*?)\n```"
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: try to find impl Solution
        if "impl Solution" in raw:
            start = raw.find("impl Solution")
            brace_count = 0
            end = start
            for i, char in enumerate(raw[start:], start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            if end > start:
                return raw[start:end].strip()

        return raw.strip()
    except Exception:
        return None


def fix_code_with_error(
    problem: Dict,
    starter_code_info: Dict,
    failed_code: str,
    error_message: str,
    api_key: str,
    error_type: str = "compile",
    test_output: Optional[str] = None,
) -> Optional[str]:
    """
    Send a prompt to the model to fix the code based on the error.

    Args:
        error_type: "compile" for compilation errors, "test" for test failures
        test_output: Full test output (only for error_type="test")

    Returns the corrected impl Solution block, or None if failed.
    """
    function_name = starter_code_info["function_name"]
    problem_description = problem.get("problem_description", "")
    task_id = problem.get("task_id", "unknown")

    if error_type == "compile":
        prompt = f"""You have a LeetCode problem that is failing to compile.
Fix the implementation to make it compile correctly.

═══════════════════════════════════════
PROBLEM
═══════════════════════════════════════
{task_id} (LeetCode #{problem.get("problem_id", "?")})
{problem_description}

═══════════════════════════════════════
STARTER CODE (function signature)
═══════════════════════════════════════
{starter_code_info["starter_code"]}

═══════════════════════════════════════
CURRENT IMPLEMENTATION (that failed)
═══════════════════════════════════════
{failed_code}

═══════════════════════════════════════
COMPILER ERROR
═══════════════════════════════════════
{error_message}

═══════════════════════════════════════
RULES
═══════════════════════════════════════
1. Output ONLY the corrected impl Solution block
2. Keep the exact function signature as given
3. Fix the compilation errors
4. Do NOT add any explanations outside the code block

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════
```rust
impl Solution {{
    pub fn {function_name}(...) -> ... {{
        // fixed implementation
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
    else:
        prompt = f"""You have a LeetCode problem that compiles but tests are failing.
Fix the implementation to make all tests pass.

═══════════════════════════════════════
PROBLEM
═══════════════════════════════════════
{task_id} (LeetCode #{problem.get("problem_id", "?")})
{problem_description}

═══════════════════════════════════════
STARTER CODE (function signature)
═══════════════════════════════════════
{starter_code_info["starter_code"]}

═══════════════════════════════════════
CURRENT IMPLEMENTATION (that failed tests)
═══════════════════════════════════════
{failed_code}

═══════════════════════════════════════
TEST OUTPUT (which tests passed/failed)
═══════════════════════════════════════
{test_output}

═══════════════════════════════════════
RULES
═══════════════════════════════════════
1. Output ONLY the corrected impl Solution block
2. Keep the exact function signature as given
3. Fix the logic to pass ALL tests (every test case must pass)
4. Do NOT add any explanations outside the code block

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════
```rust
impl Solution {{
    pub fn {function_name}(...) -> ... {{
        // fixed implementation
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
    try:
        client = OpenAI(
            base_url=LIGHTNING_BASE_URL,
            api_key=api_key,
        )

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            timeout=GENERATION_TIMEOUT,
        )
        raw = response.choices[0].message.content

        # Extract the impl block
        pattern = r"```rust\n(.*?)\n```"
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: try to find impl Solution
        if "impl Solution" in raw:
            start = raw.find("impl Solution")
            brace_count = 0
            end = start
            for i, char in enumerate(raw[start:], start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            if end > start:
                return raw[start:end].strip()

        return raw.strip()
    except Exception:
        return None


# ==================== Compilation & Testing ====================


@dataclass
class RolloutResult:
    rollout: int
    code: str
    compile_success: bool
    compile_error: Optional[str] = None
    test_success: Optional[bool] = None
    test_output: Optional[str] = None
    failed_tests: Optional[List[str]] = None


def parse_test_output(output: str) -> Tuple[bool, List[str]]:
    """
    Parse Rust test output to determine pass/fail and which tests failed.

    Returns:
        (all_passed: bool, failed_test_names: List[str])
    """
    failed_tests = []

    # Check for test failures in output
    # Rust test runner outputs: "test tests::test_case_N ... FAILED"
    for line in output.split("\n"):
        if " ... FAILED" in line:
            # Extract test name
            match = re.search(r"test tests::(\S+)", line)
            if match:
                failed_tests.append(match.group(1))

    all_passed = len(failed_tests) == 0 and "test result: ok" in output.lower()

    # Also check for test result line
    if "test result: FAILED" in output or "failures:" in output:
        all_passed = False

    return all_passed, failed_tests


def compile_and_test(
    code: str, timeout_seconds: int = 30
) -> Tuple[bool, Optional[str], Optional[bool], Optional[str], List[str]]:
    """
    Compile and test Rust code using rustc --test mode.

    Returns:
        (compile_success, compile_error, all_tests_passed, test_output, failed_tests)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        source_file = tmpdir / "main.rs"
        binary_path = tmpdir / "test_runner"

        # Write source
        source_file.write_text(code)

        # Compile in test mode
        try:
            compile_result = subprocess.run(
                ["rustc", "--test", "-o", str(binary_path), str(source_file)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return False, "Compilation timeout", None, None, []
        except FileNotFoundError:
            return False, "rustc not found - please install Rust", None, None, []
        except Exception as e:
            return False, str(e), None, None, []

        if compile_result.returncode != 0:
            return False, compile_result.stderr or compile_result.stdout, None, None, []

        # Run tests
        try:
            exec_result = subprocess.run(
                [str(binary_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            test_output = exec_result.stdout + exec_result.stderr

            all_passed, failed_tests = parse_test_output(test_output)

            return True, None, all_passed, test_output, failed_tests
        except subprocess.TimeoutExpired:
            return True, None, False, "Execution timeout", []
        except Exception as e:
            return True, None, False, str(e), []


# ==================== Main Pipeline ====================


def run_pipeline(
    num_problems: int,
    output_path: str = "complexity_reasoning_data/results.jsonl",
    api_key: str = None,
    data_dir: str = "complexity_reasoning_data",
    max_rollouts: int = 3,
):
    """Run the complete unified pipeline (generate + test per problem)."""

    if not api_key:
        print("ERROR: LIGHTNING_API_KEY is required for code generation")
        print("Set it via: export LIGHTNING_API_KEY=your_key")
        return

    # Paths
    test_harness_path = os.path.join(data_dir, "test_harness.jsonl")
    dataset_path = os.path.join(data_dir, "dataset.jsonl")
    starter_codes_path = os.path.join(data_dir, "starter_codes.jsonl")

    # Load data
    print("Loading data files...")
    test_harness_map = load_test_harness(test_harness_path)
    print(f"  test_harness.jsonl: {len(test_harness_map)} entries")

    dataset_map = load_dataset(dataset_path)
    print(f"  dataset.jsonl: {len(dataset_map)} entries")

    starter_codes_map = load_starter_codes(starter_codes_path)
    print(f"  starter_codes.jsonl: {len(starter_codes_map)} entries")

    # Target problem IDs from test_harness (source of truth)
    target_problem_ids = set(test_harness_map.keys())
    print(f"\nTarget problem IDs (from test_harness.jsonl): {len(target_problem_ids)}")

    # Find intersection with dataset and starter_codes
    valid_problem_ids = (
        target_problem_ids & set(dataset_map.keys()) & set(starter_codes_map.keys())
    )
    print(f"Problems with all required data: {len(valid_problem_ids)}")

    # Apply num_problems limit
    problem_ids_to_process = sorted(valid_problem_ids)[:num_problems]
    print(f"Processing first {len(problem_ids_to_process)} problems")
    print(f"Max rollouts per problem: {max_rollouts}")
    print("=" * 70)

    # Run unified pipeline for each problem
    results = []
    for i, pid in enumerate(problem_ids_to_process):
        print(f"\n[{i + 1}/{len(problem_ids_to_process)}] Problem {pid}...", flush=True)

        # Get data
        problem = dataset_map[pid]
        original_harness = test_harness_map[pid]
        # Transform harness to test format
        test_harness = transform_harness_to_test_format(original_harness)
        starter_code_info = starter_codes_map[pid]
        tags = problem.get("tags", [])

        # Initialize variables for the loop
        rollout_results = []
        current_code = None
        final_reward = 0.0

        for rollout in range(1, max_rollouts + 1):
            print(f"  Rollout {rollout}...", end=" ", flush=True)

            # Generate code (initial or retry)
            if rollout == 1:
                # First time: generate from scratch
                current_code = generate_initial_code(
                    problem=problem,
                    starter_code_info=starter_code_info,
                    api_key=api_key,
                )
            else:
                # Retry: try to fix based on previous error
                prev_result = rollout_results[-1]

                if not prev_result.compile_success:
                    # Compile error - retry with compiler error
                    current_code = fix_code_with_error(
                        problem=problem,
                        starter_code_info=starter_code_info,
                        failed_code=prev_result.code,
                        error_message=prev_result.compile_error or "Unknown",
                        api_key=api_key,
                        error_type="compile",
                    )
                else:
                    # Test failure - retry with test output
                    current_code = fix_code_with_error(
                        problem=problem,
                        starter_code_info=starter_code_info,
                        failed_code=prev_result.code,
                        error_message="",
                        api_key=api_key,
                        error_type="test",
                        test_output=prev_result.test_output or "",
                    )

            if current_code is None:
                # Generation/retry failed
                result = RolloutResult(
                    rollout=rollout,
                    code=current_code or "",
                    compile_success=False,
                    compile_error="LLM generation failed",
                )
                rollout_results.append(result)
                print("❌ LLM failed")
                break

            # Add struct Solution; declaration before the impl block
            code_with_struct = f"struct Solution;\n\n{current_code}"

            # Assemble and compile
            assembled = assemble_rust_code_v2(
                generated_code=code_with_struct,
                test_harness=test_harness,
                tags=tags,
            )

            compile_ok, compile_err, all_passed, test_output, failed_tests = (
                compile_and_test(assembled)
            )

            result = RolloutResult(
                rollout=rollout,
                code=current_code,
                compile_success=compile_ok,
                compile_error=compile_err if not compile_ok else None,
                test_success=all_passed if compile_ok else None,
                test_output=test_output if compile_ok else None,
                failed_tests=failed_tests if compile_ok else None,
            )
            rollout_results.append(result)

            # Determine outcome
            if compile_ok and all_passed:
                final_reward = 1.0
                print("✅ (pass)")
                break
            elif compile_ok and not all_passed:
                # Tests failed, continue retrying
                print(f"⚠️ (tests fail, {len(failed_tests)} failed)")
                if rollout < max_rollouts:
                    # Continue to next rollout
                    continue
                else:
                    # Exhausted rollouts, still failing
                    final_reward = 0.5
            else:
                # Compile failed
                print("❌ (compile fail)")
                if rollout < max_rollouts:
                    continue
                else:
                    final_reward = 0.0

        # Get best code (first successful compile, or last attempt)
        best_code = None
        for r in rollout_results:
            if r.compile_success:
                best_code = r.code
                break
        if best_code is None and rollout_results:
            best_code = rollout_results[-1].code if rollout_results[-1].code else ""

        # Get all errors
        errors = [r.compile_error for r in rollout_results if r.compile_error]

        # Initialize LLM judge for evaluation
        judge = LLMJudge(api_key=api_key)

        # Extract reasoning and complexity from best code for evaluation
        reasoning_steps = ""
        predicted_complexity = ""
        reasoning_score = 0.0
        time_complexity_score = 0

        if best_code:
            reasoning_steps = extract_reasoning_steps(best_code)
            predicted_complexity = extract_time_complexity(best_code)
            raw_complexity_text = extract_raw_time_complexity(best_code)

            # Get ground truth from dataset
            ground_truth_explanation = problem.get("explanation", "")
            ground_truth_complexity = problem.get("time_complexity", "")

            # Evaluate reasoning if we have both generated and ground truth
            if reasoning_steps and ground_truth_explanation:
                try:
                    reasoning_score = judge.evaluate_reasoning(
                        reasoning_steps, ground_truth_explanation
                    )
                except Exception as e:
                    print(f"    Warning: Reasoning evaluation failed: {e}")

            # Evaluate time complexity using raw text (judge extracts Big-O itself)
            if raw_complexity_text and ground_truth_complexity:
                try:
                    time_complexity_score = judge.evaluate_time_complexity(
                        raw_complexity_text, ground_truth_complexity
                    )
                except Exception as e:
                    print(f"    Warning: Complexity evaluation failed: {e}")

        result = {
            "problem_id": pid,
            "correctness_reward": final_reward,
            "reasoning_score": reasoning_score,
            "time_complexity_score": time_complexity_score,
            "predicted_complexity": predicted_complexity,
            "ground_truth_complexity": problem.get("time_complexity", ""),
            "generated_steps": reasoning_steps,
            "rollouts": [
                {
                    "rollout": r.rollout,
                    "compile_success": r.compile_success,
                    "test_success": r.test_success,
                    "failed_tests": r.failed_tests,
                    "error": r.compile_error,
                }
                for r in rollout_results
            ],
            "best_code": best_code,
            "errors": errors,
        }
        results.append(result)

    # Save results
    print(f"\n" + "=" * 70)
    print(f"Saving results to {output_path}...")
    with open(output_path, "w") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")

    # Summary
    print("\nPIPELINE COMPLETE")
    print("=" * 70)
    total = len(results)
    print(f"Total problems: {total}")
    print(f"\nCorrectness Rewards:")
    print(
        f"  Reward 1.0 (pass): {sum(1 for r in results if r['correctness_reward'] == 1.0)}"
    )
    print(
        f"  Reward 0.5 (compile-only): {sum(1 for r in results if r['correctness_reward'] == 0.5)}"
    )
    print(
        f"  Reward 0.0 (fail): {sum(1 for r in results if r['correctness_reward'] == 0.0)}"
    )
    if total > 0:
        print(
            f"  Average correctness: {sum(r['correctness_reward'] for r in results) / total:.3f}"
        )

    print(f"\nLLM-Judge Scores:")
    reasoning_scores = [
        r["reasoning_score"] for r in results if r.get("reasoning_score", 0) > 0
    ]
    complexity_scores = [
        r["time_complexity_score"]
        for r in results
        if r.get("time_complexity_score", 0) >= 0
    ]
    if reasoning_scores:
        print(
            f"  Average reasoning score: {sum(reasoning_scores) / len(reasoning_scores):.3f}"
        )
    if complexity_scores:
        print(
            f"  Complexity accuracy: {sum(complexity_scores)}/{len(complexity_scores)} ({100 * sum(complexity_scores) / len(complexity_scores):.1f}%)"
        )
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete Rust pipeline")
    parser.add_argument(
        "--num-problems",
        type=int,
        default=50,
        help="Number of problems to process (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="complexity_reasoning_data/results.jsonl",
        help="Output file path (default: complexity_reasoning_data/results.jsonl)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Lightning AI API key (or use LIGHTNING_API_KEY env var)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="complexity_reasoning_data",
        help="Data directory (default: complexity_reasoning_data)",
    )
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=3,
        help="Maximum retry attempts per problem (default: 3)",
    )

    args = parser.parse_args()

    # Get API key from CLI or environment
    api_key = args.api_key or os.getenv("LIGHTNING_API_KEY")

    run_pipeline(
        num_problems=args.num_problems,
        output_path=args.output,
        api_key=api_key,
        data_dir=args.data_dir,
        max_rollouts=args.max_rollouts,
    )
