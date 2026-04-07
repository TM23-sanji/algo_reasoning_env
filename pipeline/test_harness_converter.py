"""
Phase 3: Test Harness Conversion Pipeline

Converts Python test cases to Rust test harnesses using Lightning AI API (Nemotron model).
Uses starter_codes.jsonl as the source of truth for function signatures.

Main components:
1. build_conversion_prompt() — Create API prompt from problem data
2. convert_one() — Convert single problem's tests via API
3. convert_batch() — Process problems in sequence
4. extract_rust_code() — Extract Rust code from raw API response
5. save_harnesses() — Write test_harness.jsonl
6. save_debug() — Write test_harness_debug.jsonl

CLI Usage:
    python pipeline/test_harness_converter.py --num-problems 50

    Or set LIGHTNING_API_KEY env var and run:
    python pipeline/test_harness_converter.py [--num-problems 50]
"""

import json
import os
import re
import requests
import argparse
from typing import Dict, List, Optional, Tuple
from pathlib import Path


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


def build_conversion_prompt(
    problem: Dict, python_tests: str, starter_code_info: Dict
) -> str:
    """
    Build a prompt for Lightning AI API to convert Python tests to Rust.

    Args:
        problem: Problem dict with problem_id, task_id, problem_description
        python_tests: Python test code (def check(candidate): ...)
        starter_code_info: Dict with function_name and starter_code

    Returns:
        Formatted prompt for API
    """
    function_name = starter_code_info["function_name"]
    starter_code = starter_code_info["starter_code"]

    prompt = f"""You are converting Python test cases into a Rust test harness.
The Rust function signature is already fixed — use it exactly as given.

═══════════════════════════════════════
FUNCTION SIGNATURE (source of truth)
═══════════════════════════════════════
{starter_code}

This tells you:
- The exact function name to call via Solution::{function_name}(...)
- The exact parameter types and order
- The exact return type

═══════════════════════════════════════
PROBLEM CONTEXT
═══════════════════════════════════════
Problem: {problem.get("task_id", "unknown")} (LeetCode #{problem.get("problem_id", "?")})

═══════════════════════════════════════
PYTHON TESTS TO CONVERT
═══════════════════════════════════════
```python
{python_tests}
```

═══════════════════════════════════════
RUST TYPE CONVERSION RULES
═══════════════════════════════════════

INTEGERS:
- Python int → match the signature exactly (i32, i64, usize)
- Do NOT add type suffixes unless the literal would overflow i32 (then use 1i64)

LISTS / VECTORS:
- Python list [1, 2, 3]       → vec![1, 2, 3]
- Python list of lists [[1,2],[3,4]] → vec![vec![1, 2], vec![3, 4]]
- Python empty list []         → vec![]
- Python None when return type is Vec<T>   → vec![]
- Python None when return type is Option<T> → None

STRINGS:
- Python str "hello" as parameter of type String → "hello".to_string()
- Python str "hello" as parameter of type &str   → "hello"
- Python str "hello" as return type of String    → "hello".to_string()

BOOLEANS:
- Python True  → true
- Python False → false

MUTABLE REFERENCES (&mut):
- If a parameter is &mut Vec<T> or &mut [T]:
    let mut var = vec![...];
    Solution::{function_name}(&mut var, ...);
    assert_eq!(var, expected_modified_value);
- The function mutates in place — assert on the mutated variable, not the return value

FLOATING POINT (f64):
- NEVER use assert_eq! for f64
- Use: assert!((result - expected).abs() < 1e-5);

ORDER-AGNOSTIC RESULTS:
- If the problem says "any order" or returns combinations/permutations:
    let mut result = Solution::{function_name}(...);
    result.sort();
    let mut expected = vec![...];
    expected.sort();
    assert_eq!(result, expected);

LINKED LIST (Option<Box<ListNode>>):
- Construct input directly: Some(Box::new(ListNode {{ val: 1, next: Some(Box::new(ListNode {{ val: 2, next: None }})) }}))
- For output comparison, destructure the linked list or compare field-by-field
- Python None for linked list → assert_eq!(result, None)

TREE (Option<Rc<RefCell<TreeNode>>>):
- Construct using: Some(Rc::new(RefCell::new(TreeNode {{ val: 1, left: None, right: None }})))
- For output comparison, use tree_to_vec logic or compare field-by-field
- Python None for tree → assert_eq!(result, None)

═══════════════════════════════════════
HARD RULES — NEVER VIOLATE
═══════════════════════════════════════
1. ALWAYS call Solution::{function_name}(...) — NEVER use candidate(...)
2. Do NOT define struct Solution, impl Solution, ListNode, TreeNode, or any helper
3. Do NOT add any use statements — all types are already in scope
4. Use assert_eq! for all equality checks — NEVER use assert!(a == b)
5. Every assert_eq! must compile — if unsure about types, add explicit type annotations
6. If a Python test has a comment explaining the edge case, preserve it as a Rust comment

═══════════════════════════════════════
EXPECTED OUTPUT FORMAT
═══════════════════════════════════════
Output ONLY a Rust code block — no explanation, no prose:

```rust
fn check() {{
    // test case 1: description
    assert_eq!(Solution::{function_name}(...), ...);

    // test case 2: description
    assert_eq!(Solution::{function_name}(...), ...);

    // ... all test cases converted
}}

fn main() {{
    check();
    println!("✅ All tests passed!");
}}
```
"""
    return prompt


def extract_rust_code(raw_output: str) -> str:
    """
    Extract Rust code block from raw API response.

    Tries multiple patterns:
    1. ```rust ... ``` code fence
    2. fn check() ... fn main() ... patterns
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
        # Check if it looks like Rust
        if "fn check()" in code or "fn main()" in code:
            return code

    # Try extracting fn check() ... fn main() ...
    pattern = r"(fn check\(\).*?fn main\(\).*?\})"
    match = re.search(pattern, raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: if response contains fn check and fn main, extract them
    if "fn check()" in raw_output and "fn main()" in raw_output:
        start = raw_output.find("fn check()")
        end = raw_output.rfind("}") + 1
        if start != -1 and end > start:
            return raw_output[start:end].strip()

    # Last resort: return raw output
    return raw_output.strip()


def convert_one(
    problem: Dict,
    python_tests: str,
    starter_code_info: Dict,
    api_key: str,
    model: str = "lightning-ai/nvidia-nemotron-3-super-120b-a12b",
    timeout: int = 120,
) -> Dict:
    """
    Convert a single problem's Python tests to Rust via Lightning AI API.

    Args:
        problem: Problem dict
        python_tests: Python test code
        starter_code_info: Dict with function_name and starter_code
        api_key: Lightning AI API key
        model: Model to use
        timeout: Request timeout in seconds

    Returns:
        Dict with problem_id, harness, raw_output, status, error
    """
    prompt = build_conversion_prompt(problem, python_tests, starter_code_info)

    try:
        response = requests.post(
            url="https://lightning.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ],
            },
            timeout=timeout,
        )

        if response.status_code == 429:
            return {
                "problem_id": problem["problem_id"],
                "harness": None,
                "raw_output": None,
                "status": "rate_limited",
                "error": "429 Rate Limited",
            }

        if response.status_code != 200:
            return {
                "problem_id": problem["problem_id"],
                "harness": None,
                "raw_output": response.text,
                "status": "http_error",
                "error": f"HTTP {response.status_code}",
            }

        data = response.json()
        raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        harness = extract_rust_code(raw)

        return {
            "problem_id": problem["problem_id"],
            "harness": harness,
            "raw_output": raw,
            "status": "success",
            "error": None,
        }

    except requests.Timeout:
        return {
            "problem_id": problem["problem_id"],
            "harness": None,
            "raw_output": None,
            "status": "timeout",
            "error": f"Request timeout ({timeout}s)",
        }
    except Exception as e:
        return {
            "problem_id": problem["problem_id"],
            "harness": None,
            "raw_output": None,
            "status": "error",
            "error": str(e),
        }


def convert_batch(
    problems: List[Dict],
    python_tests_map: Dict[int, str],
    starter_codes_map: Dict[int, Dict],
    api_key: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Convert problems in sequence.

    Args:
        problems: List of problem dicts
        python_tests_map: Dict mapping problem_id → Python test code
        starter_codes_map: Dict mapping problem_id → {function_name, starter_code}
        api_key: Lightning AI API key

    Returns:
        Tuple of (successes, failures)
    """
    successes = []
    failures = []

    # Filter problems that have both tests and starter codes
    problems_to_convert = [
        p
        for p in problems
        if p["problem_id"] in python_tests_map and p["problem_id"] in starter_codes_map
    ]

    print(f"Converting {len(problems_to_convert)} problems...")
    print()

    for problem in problems_to_convert:
        problem_id = problem["problem_id"]
        python_tests = python_tests_map[problem_id]
        starter_code_info = starter_codes_map[problem_id]

        print(f"Converting problem #{problem_id}...", end=" ", flush=True)

        result = convert_one(problem, python_tests, starter_code_info, api_key)

        if result["status"] == "success":
            successes.append(result)
            print("✅")
        else:
            failures.append(result)
            print(f"❌ {result['error']}")

    return successes, failures


def save_harnesses(results: List[Dict], path: str):
    """
    Append successful harnesses to test_harness.jsonl

    Args:
        results: List of conversion results
        path: Output file path
    """
    with open(path, "a") as f:
        for r in results:
            if r["status"] == "success" and r["harness"]:
                json.dump(
                    {
                        "problem_id": r["problem_id"],
                        "harness": r["harness"],
                    },
                    f,
                )
                f.write("\n")


def save_debug(results: List[Dict], path: str):
    """
    Append raw API responses to test_harness_debug.jsonl

    Args:
        results: List of conversion results
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


def load_python_tests_from_jsonl(tests_path: str) -> Dict[int, str]:
    """
    Load Python test cases from python_tests.jsonl.

    Expected format: {"problem_id": 1, "test": "def check(candidate):\\n    assert ..."}

    Args:
        tests_path: Path to python_tests.jsonl

    Returns:
        Dict mapping problem_id → Python test code
    """
    tests_map = {}

    if not os.path.isfile(tests_path):
        print(f"Warning: Test file not found: {tests_path}")
        return tests_map

    try:
        with open(tests_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    problem_id = data.get("problem_id")
                    test = data.get("test")
                    if problem_id and test:
                        tests_map[problem_id] = test
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num}: {e}")
    except IOError as e:
        print(f"Error reading test file: {e}")

    return tests_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3: Convert Python test cases to Rust test harnesses via Lightning AI API"
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
        "--tests-path",
        type=str,
        default="complexity_reasoning_data/python_tests.jsonl",
        help="Path to python_tests.jsonl file",
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
        help="Output directory for harnesses",
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
    tests_path = args.tests_path
    dataset_path = args.dataset_path
    starter_codes_path = args.starter_codes_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    harness_output = output_dir / "test_harness.jsonl"
    debug_output = output_dir / "test_harness_debug.jsonl"

    print(f"Phase 3: Test Harness Conversion Pipeline")
    print(f"=" * 70)
    print(f"API Key: {api_key[:20]}...{api_key[-30:]}")
    print(f"Num problems: {args.num_problems}")
    print(f"Tests path: {tests_path}")
    print(f"Dataset path: {dataset_path}")
    print(f"Starter codes path: {starter_codes_path}")
    print(f"Output directory: {output_dir}")
    print(f"Timeout: {args.timeout}s")
    print()

    # Load test data
    print("Loading Python test cases...")
    python_tests_map = load_python_tests_from_jsonl(tests_path)
    print(f"✅ Loaded {len(python_tests_map)} test cases")

    # Load starter codes
    print("Loading starter codes...")
    starter_codes_map = load_starter_codes(starter_codes_path)
    print(f"✅ Loaded {len(starter_codes_map)} starter codes")

    # Load dataset
    print("Loading dataset...")
    problems = []

    with open(dataset_path) as f:
        for i, line in enumerate(f):
            # if i >= args.num_problems:
            #     break
            problem = json.loads(line)
            problems.append(problem)

    print(
        f"✅ Loaded {len(problems)} problems from dataset (first {args.num_problems})"
    )

    # problems = []

    # with open(dataset_path) as f:
    #     for line in f:
    #         problem = json.loads(line)
    #         problem_id = problem.get("id") or problem.get("problem_id")  # adjust key if needed
            
    #         if problem_id is not None and 141 <= problem_id <= 145:
    #             problems.append(problem)
        
    #     # Optional: early stop if we already have all 6 problems
    #         if len(problems) >= 6:
    #             break

    # print(f"Loaded {len(problems)} problems (141 to 145)")

    # Run conversion pipeline
    print()
    successes, failures = convert_batch(
        problems, python_tests_map, starter_codes_map, api_key
    )

    # Save results
    print()
    print(f"Saving results...")
    save_harnesses(successes, str(harness_output))
    save_debug(successes + failures, str(debug_output))

    # Summary
    print()
    print("=" * 70)
    print(f"✅ Conversion complete!")
    print(f"  Successful: {len(successes)}")
    print(f"  Failed: {len(failures)}")
    print(f"  Output:")
    print(f"    - {harness_output} ({len(successes)} harnesses)")
    print(
        f"    - {debug_output} (debug info for {len(successes) + len(failures)} conversions)"
    )
    print("=" * 70)
