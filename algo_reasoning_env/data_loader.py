"""
Data loader for the Algo Reasoning Environment.

Loads problems from the embedded dataset files in order (0, 1, 2, ...).
The dataset should be embedded in the Docker image at /data/.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class Problem:
    """Represents a single problem from the dataset."""

    def __init__(self, data: Dict):
        self.problem_id = data["problem_id"]
        self.task_id = data["task_id"]
        self.difficulty = data["difficulty"]
        self.tags = data.get("tags", [])
        self.problem_description = data["problem_description"]
        self.time_complexity = data["time_complexity"]
        self.explanation = data["explanation"]


class StarterCode:
    """Represents starter code for a problem."""

    def __init__(self, data: Dict):
        self.problem_id = data["problem_id"]
        self.function_name = data["function_name"]
        self.starter_code = data["starter_code"]


class TestHarness:
    """Represents test harness for a problem."""

    def __init__(self, data: Dict):
        self.problem_id = data["problem_id"]
        self.harness = data["harness"]


class DataLoader:
    """
    Loads and provides access to problems in order.

    Problems are sorted by problem_id and returned sequentially.
    When all problems are exhausted, cycles back to the beginning.
    """

    def __init__(self, data_dir: str = "/data"):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing dataset files
        """
        self.data_dir = Path(data_dir)
        self._problems: List[Problem] = []
        self._problems_by_id: Dict[int, Problem] = {}
        self._starter_codes: Dict[int, StarterCode] = {}
        self._test_harnesses: Dict[int, TestHarness] = {}
        self._current_index = 0
        # Per-difficulty tracking
        self._problems_by_difficulty: Dict[str, List[Problem]] = {}
        self._current_index_by_difficulty: Dict[str, int] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load all data files."""
        # Load dataset
        dataset_path = self.data_dir / "dataset.jsonl"
        if dataset_path.exists():
            with open(dataset_path) as f:
                for line in f:
                    data = json.loads(line)
                    problem = Problem(data)
                    self._problems.append(problem)
                    self._problems_by_id[problem.problem_id] = problem
            # Sort by problem_id for consistent ordering
            self._problems.sort(key=lambda p: p.problem_id)

            # Build per-difficulty index
            for problem in self._problems:
                diff = problem.difficulty
                if diff not in self._problems_by_difficulty:
                    self._problems_by_difficulty[diff] = []
                    self._current_index_by_difficulty[diff] = 0
                self._problems_by_difficulty[diff].append(problem)

        # Load starter codes
        starter_path = self.data_dir / "starter_codes.jsonl"
        if starter_path.exists():
            with open(starter_path) as f:
                for line in f:
                    data = json.loads(line)
                    sc = StarterCode(data)
                    self._starter_codes[sc.problem_id] = sc

        # Load test harnesses
        harness_path = self.data_dir / "test_harness.jsonl"
        if harness_path.exists():
            with open(harness_path) as f:
                for line in f:
                    data = json.loads(line)
                    th = TestHarness(data)
                    self._test_harnesses[th.problem_id] = th

    def get_problem(self, index: int) -> Optional[Problem]:
        """
        Get problem at specific index.

        Args:
            index: Index into the sorted problems list

        Returns:
            Problem at index, or None if index out of range
        """
        if 0 <= index < len(self._problems):
            return self._problems[index]
        return None

    def get_problem_by_id(self, problem_id: int) -> Optional[Problem]:
        """
        Get problem by its problem_id.

        Args:
            problem_id: The problem ID to look up

        Returns:
            Problem with matching ID, or None if not found
        """
        return self._problems_by_id.get(problem_id)

    def get_next(self) -> Optional[Problem]:
        """
        Get next problem in sequence and advance index.

        Returns:
            Next problem, or None if no problems available
        """
        if not self._problems:
            return None

        problem = self._problems[self._current_index]
        self._current_index = (self._current_index + 1) % len(self._problems)
        return problem

    def get_starter_code(self, problem_id: int) -> Optional[str]:
        """
        Get starter code for a problem.

        Args:
            problem_id: Problem ID

        Returns:
            Starter code string, or None if not found
        """
        if problem_id in self._starter_codes:
            return self._starter_codes[problem_id].starter_code
        return None

    def get_function_name(self, problem_id: int) -> Optional[str]:
        """
        Get function name for a problem.

        Args:
            problem_id: Problem ID

        Returns:
            Function name, or None if not found
        """
        if problem_id in self._starter_codes:
            return self._starter_codes[problem_id].function_name
        return None

    def get_test_harness(self, problem_id: int) -> Optional[str]:
        """
        Get test harness for a problem.

        Args:
            problem_id: Problem ID

        Returns:
            Test harness string, or None if not found
        """
        if problem_id in self._test_harnesses:
            return self._test_harnesses[problem_id].harness
        return None

    def reset(self) -> None:
        """Reset index to beginning."""
        self._current_index = 0

    def get_next_by_difficulty(self, difficulty: str) -> Optional[Problem]:
        """
        Get next problem matching the given difficulty, cycling through.

        Args:
            difficulty: "Easy", "Medium", or "Hard"

        Returns:
            Next problem of that difficulty, or None if no problems match
        """
        problems = self._problems_by_difficulty.get(difficulty)
        if not problems:
            return None

        idx = self._current_index_by_difficulty.get(difficulty, 0)
        problem = problems[idx]
        self._current_index_by_difficulty[difficulty] = (idx + 1) % len(problems)
        return problem

    def reset_by_difficulty(self, difficulty: str) -> None:
        """Reset the cycle index for a specific difficulty."""
        if difficulty in self._current_index_by_difficulty:
            self._current_index_by_difficulty[difficulty] = 0

    def __len__(self) -> int:
        """Return number of problems."""
        return len(self._problems)
