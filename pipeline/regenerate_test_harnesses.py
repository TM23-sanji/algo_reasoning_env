"""
Regenerate test harnesses for problem IDs that failed or are new.

Target problem IDs:
- 79 new IDs from rust-starter.txt (added to starter_codes.jsonl)
- 303 failed IDs from test_harness_debug.jsonl (http_error + rate_limited)
- Total: 382 unique problem IDs

Usage:
    python pipeline/regenerate_test_harnesses.py --api-key YOUR_KEY

    Or set LIGHTNING_API_KEY env var and run:
    python pipeline/regenerate_test_harnesses.py

Options:
    --max-asserts N   Limit to N assertions per problem (default: 30)
"""

import json
import os
import re
import sys
import argparse
import time
from typing import Dict, Optional, List

# Import from existing converter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_harness_converter import (
    load_starter_codes,
    load_python_tests_from_jsonl as load_python_tests,
    build_conversion_prompt,
    convert_one,
    extract_rust_code,
)
from run_pipeline import load_dataset


def truncate_python_tests(python_tests: str, max_asserts: int = 30) -> str:
    """
    Truncate Python tests to limit the number of assert statements.

    Args:
        python_tests: Python test code (def check(candidate): ...)
        max_asserts: Maximum number of assert statements to keep

    Returns:
        Truncated Python test code
    """
    if max_asserts <= 0:
        return python_tests

    # Split into lines
    lines = python_tests.split("\n")

    # Find the function definition line
    result_lines = []
    assert_count = 0

    for line in lines:
        result_lines.append(line)
        # Count assert statements (not inside strings)
        if line.strip().startswith("assert "):
            assert_count += 1
            if assert_count >= max_asserts:
                # Keep the function definition (first line) and stop at max asserts
                break

    return "\n".join(result_lines)


# Target problem IDs (382 total)
TARGET_IDS = [
    38,
    77,
    89,
    412,
    687,
]


def load_existing_test_harness_ids(test_harness_path: str) -> set:
    """Load existing problem IDs from test_harness.jsonl."""
    existing_ids = set()
    if os.path.exists(test_harness_path):
        with open(test_harness_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "problem_id" in data:
                        existing_ids.add(data["problem_id"])
                except json.JSONDecodeError:
                    continue
    return existing_ids


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate test harnesses for failed/new problem IDs"
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
        "--output",
        type=str,
        default="complexity_reasoning_data/test_harness.jsonl",
        help="Output file path (default: complexity_reasoning_data/test_harness.jsonl)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't make API calls, just show what would be processed",
    )
    parser.add_argument(
        "--max-asserts",
        type=int,
        default=30,
        help="Maximum number of assert statements per problem (default: 30)",
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("LIGHTNING_API_KEY")
    if not api_key:
        print("ERROR: LIGHTNING_API_KEY is required")
        print("Set it via: export LIGHTNING_API_KEY=your_key")
        return

    # Paths
    data_dir = args.data_dir
    starter_codes_path = os.path.join(data_dir, "starter_codes.jsonl")
    dataset_path = os.path.join(data_dir, "dataset.jsonl")
    python_tests_path = os.path.join(data_dir, "python_tests.jsonl")

    # Load data
    print("Loading data files...")
    starter_codes_map = load_starter_codes(starter_codes_path)
    print(f"  starter_codes.jsonl: {len(starter_codes_map)} entries")

    dataset_map = load_dataset(dataset_path)
    print(f"  dataset.jsonl: {len(dataset_map)} entries")

    python_tests_map = load_python_tests(python_tests_path)
    print(f"  python_tests.jsonl: {len(python_tests_map)} entries")

    # Get existing test harness IDs
    existing_ids = load_existing_test_harness_ids(args.output)
    print(f"  Existing test_harness.jsonl: {len(existing_ids)} entries")

    # Filter target IDs to only those not yet processed
    target_ids = [pid for pid in TARGET_IDS if pid not in existing_ids]
    print(f"\nTarget problem IDs to process: {len(target_ids)}")

    if args.dry_run:
        print("\nDRY RUN - would process these problem IDs:")
        print(target_ids[:20], "..." if len(target_ids) > 20 else "")
        return

    # Process each target problem
    successes = 0
    failures = 0

    print(f"\nProcessing {len(target_ids)} problems...")
    print("=" * 70)

    for i, pid in enumerate(target_ids):
        # Check if we have data for this problem
        if pid not in starter_codes_map:
            print(
                f"[{i + 1}/{len(target_ids)}] Problem {pid}: SKIP (not in starter_codes)"
            )
            continue
        if pid not in dataset_map:
            print(f"[{i + 1}/{len(target_ids)}] Problem {pid}: SKIP (not in dataset)")
            continue
        if pid not in python_tests_map:
            print(
                f"[{i + 1}/{len(target_ids)}] Problem {pid}: SKIP (not in python_tests)"
            )
            continue

        print(f"[{i + 1}/{len(target_ids)}] Problem {pid}...", end=" ", flush=True)

        # Get data
        problem = dataset_map[pid]
        starter_code_info = starter_codes_map[pid]
        python_tests = python_tests_map[pid]

        # Truncate to max asserts
        python_tests = truncate_python_tests(python_tests, args.max_asserts)

        # Call API (convert_one builds its own prompt internally)
        result = convert_one(problem, python_tests, starter_code_info, api_key)

        # Check result status
        if result is None or result.get("status") != "success":
            print(
                f"❌ (API failed: {result.get('error', 'unknown') if result else 'no result'})"
            )
            failures += 1
            continue

        # Get harness from result
        harness = result.get("harness") or result.get("raw_output")

        if harness is None:
            print("❌ (no harness extracted)")
            failures += 1
            continue

        # Append to output file
        with open(args.output, "a") as f:
            f.write(json.dumps({"problem_id": pid, "harness": harness}) + "\n")

        print("✅")
        successes += 1

        # Rate limiting delay
        time.sleep(args.delay)

    print("=" * 70)
    print(f"DONE! Successes: {successes}, Failures: {failures}")
    print(f"Results appended to: {args.output}")


if __name__ == "__main__":
    main()
