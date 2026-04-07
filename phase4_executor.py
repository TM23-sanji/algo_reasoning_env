#!/usr/bin/env python3
"""
Phase 4: Assembly, Compilation, and Execution Pipeline

Combines boilerplate + solution code + test harnesses into compilable Rust code,
then compiles and executes to verify correctness.

Usage:
    python phase4_executor.py [--num-problems 50] [--output-dir complexity_reasoning_data]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Set

from pipeline.problem import Problem
from pipeline.category_resolver import resolve_category
from pipeline.boilerplate.registry import get_registry
from pipeline.executor import execute_problem
from pipeline.logger import ExecutionLogger


def load_test_harnesses(harness_path: str) -> Dict[int, str]:
    """
    Load test harnesses from test_harness.jsonl.

    Args:
        harness_path: Path to test_harness.jsonl

    Returns:
        Dict mapping problem_id → Rust test harness code
    """
    harnesses = {}

    if not Path(harness_path).exists():
        print(f"Warning: Test harness file not found: {harness_path}")
        return harnesses

    with open(harness_path) as f:
        for line in f:
            data = json.loads(line)
            problem_id = data.get("problem_id")
            harness = data.get("harness")
            if problem_id and harness:
                harnesses[problem_id] = harness

    return harnesses


def load_problems_from_dataset(dataset_path: str, num_problems: int = 50) -> list:
    """
    Load problems from dataset.jsonl.

    Args:
        dataset_path: Path to dataset.jsonl
        num_problems: Number of problems to load

    Returns:
        List of problem dictionaries
    """
    problems = []

    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if i >= num_problems:
                break
            problems.append(json.loads(line))

    return problems


def build_problem_object(
    dataset_entry: dict, harnesses: Dict[int, str], registry
) -> Problem:
    """
    Build a Problem object from dataset entry, harness, and boilerplate.

    Args:
        dataset_entry: Problem data from dataset.jsonl
        harnesses: Dict of problem_id → harness code
        registry: Boilerplate registry

    Returns:
        Problem instance ready for execution
    """
    problem_id = dataset_entry["problem_id"]

    # Resolve categories from tags
    categories = resolve_category(dataset_entry["tags"])

    # Load boilerplate for categories
    boilerplate = registry.merge_boilerplate(categories)

    # Get test harness (or empty string if not available)
    test_harness = harnesses.get(problem_id, "")

    # Create Problem object
    problem = Problem.from_dataset_entry(
        dataset_entry=dataset_entry,
        categories=categories,
        boilerplate=boilerplate,
        test_harness=test_harness,
        solution_code="    // TODO: Implement solution\n",
    )

    return problem


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Compile and execute Rust problems"
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
        help="Path to dataset.jsonl",
    )
    parser.add_argument(
        "--harness-path",
        type=str,
        default="complexity_reasoning_data/test_harness.jsonl",
        help="Path to test_harness.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="complexity_reasoning_data",
        help="Output directory for results",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Execution timeout in seconds (default: 10)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 4: Assembly, Compilation, and Execution Pipeline")
    print("=" * 70)
    print(f"Num problems: {args.num_problems}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Harness path: {args.harness_path}")
    print(f"Output directory: {output_dir}")
    print(f"Timeout: {args.timeout}s")
    print()

    # Load data
    print("Loading dataset...")
    problems_data = load_problems_from_dataset(args.dataset_path, args.num_problems)
    print(f"✅ Loaded {len(problems_data)} problems")

    print("Loading test harnesses...")
    harnesses = load_test_harnesses(args.harness_path)
    print(f"✅ Loaded {len(harnesses)} test harnesses")

    print("Initializing boilerplate registry...")
    registry = get_registry()
    print(f"✅ Registry ready with {len(registry.get_all_categories())} categories")

    # Initialize logger
    results_path = output_dir / "execution_results.jsonl"
    logger = ExecutionLogger(str(results_path))

    # Process problems
    print()
    print(f"Processing {len(problems_data)} problems...")
    print("-" * 70)

    for i, problem_data in enumerate(problems_data, 1):
        problem_id = problem_data["problem_id"]

        # Build problem object
        problem = build_problem_object(problem_data, harnesses, registry)

        # Execute
        print(
            f"[{i:3d}/{len(problems_data)}] Problem #{problem_id:4d}... ",
            end="",
            flush=True,
        )
        problem = execute_problem(problem, args.timeout)

        # Log result
        logger.log_problem(problem)

        # Print status
        if problem.compilation_success and problem.execution_success:
            print("✅ Success")
        elif problem.compilation_success:
            print(f"⚠️  Compiled but execution failed")
        else:
            print(f"❌ Compilation failed")

    print()
    print(f"✅ All problems processed!")
    print(f"Results saved to: {results_path}")

    # Print summary
    logger.print_summary()


if __name__ == "__main__":
    main()
