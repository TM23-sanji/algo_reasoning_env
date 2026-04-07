"""
Phase 4: Logger

Structured logging of execution results to JSONL format.
"""

import json
from pathlib import Path
from typing import List
from .problem import Problem


class ExecutionLogger:
    """Log execution results to JSONL file."""

    def __init__(self, output_path: str):
        """
        Initialize logger with output file path.

        Args:
            output_path: Path to output JSONL file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log_problem(self, problem: Problem):
        """
        Log a single problem's execution results.

        Args:
            problem: Problem instance with execution results
        """
        with open(self.output_path, "a") as f:
            json.dump(problem.to_dict(), f)
            f.write("\n")

    def log_problems(self, problems: List[Problem]):
        """
        Log multiple problems' execution results.

        Args:
            problems: List of Problem instances
        """
        for problem in problems:
            self.log_problem(problem)

    def read_summary(self) -> dict:
        """
        Read summary statistics from log file.

        Returns:
            Dictionary with counts of successes, failures, etc.
        """
        if not self.output_path.exists():
            return {
                "total": 0,
                "compilation_success": 0,
                "execution_success": 0,
                "compilation_failed": 0,
                "execution_failed": 0,
            }

        total = 0
        compilation_success = 0
        execution_success = 0

        with open(self.output_path) as f:
            for line in f:
                data = json.loads(line)
                total += 1

                if data.get("compilation_success"):
                    compilation_success += 1

                if data.get("execution_success"):
                    execution_success += 1

        return {
            "total": total,
            "compilation_success": compilation_success,
            "execution_success": execution_success,
            "compilation_failed": total - compilation_success,
            "execution_failed": compilation_success - execution_success,
        }

    def print_summary(self):
        """Print summary statistics to console."""
        summary = self.read_summary()
        print()
        print("=" * 70)
        print("EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Total problems processed: {summary['total']}")
        print(f"Compilation success: {summary['compilation_success']}")
        print(f"Execution success: {summary['execution_success']}")
        print(f"Compilation failed: {summary['compilation_failed']}")
        print(f"Execution failed: {summary['execution_failed']}")
        if summary["total"] > 0:
            compilation_rate = (summary["compilation_success"] / summary["total"]) * 100
            execution_rate = (summary["execution_success"] / summary["total"]) * 100
            print(f"Compilation success rate: {compilation_rate:.1f}%")
            print(f"Execution success rate: {execution_rate:.1f}%")
        print("=" * 70)
