"""
Phase 4: Problem Dataclass

Combines dataset problem + test harness + boilerplate + solution code
into a single entity that can be compiled and executed.
"""

from dataclasses import dataclass
from typing import Optional, Set
import json


@dataclass
class Problem:
    """Represents a LeetCode problem with all necessary components for compilation and testing."""

    # Problem metadata
    problem_id: int
    task_id: str
    title: str
    difficulty: str
    tags: list
    problem_description: str
    time_complexity: str
    explanation: str

    # Categories (resolved from tags)
    categories: Set[str]

    # Rust code components
    boilerplate: str  # Helper functions (linked_list, tree, graph, heap)
    solution_code: str  # User's solution implementation
    test_harness: str  # fn check() { ... } fn main() { ... }

    # Execution results
    compilation_success: Optional[bool] = None
    execution_success: Optional[bool] = None
    compilation_error: Optional[str] = None
    execution_error: Optional[str] = None
    execution_output: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "problem_id": self.problem_id,
            "task_id": self.task_id,
            "title": self.title,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "problem_description": self.problem_description,
            "time_complexity": self.time_complexity,
            "explanation": self.explanation,
            "categories": list(self.categories),
            "boilerplate_size": len(self.boilerplate),
            "solution_code_size": len(self.solution_code),
            "test_harness_size": len(self.test_harness),
            "compilation_success": self.compilation_success,
            "execution_success": self.execution_success,
            "compilation_error": self.compilation_error,
            "execution_error": self.execution_error,
            "execution_output": self.execution_output,
        }

    @staticmethod
    def from_dataset_entry(
        dataset_entry: dict,
        categories: Set[str],
        boilerplate: str,
        test_harness: str,
        solution_code: str = "// TODO: Implement solution\n",
    ) -> "Problem":
        """Create a Problem from dataset entry, categories, boilerplate, and test harness."""
        return Problem(
            problem_id=dataset_entry["problem_id"],
            task_id=dataset_entry["task_id"],
            title=dataset_entry.get("title", f"Problem #{dataset_entry['problem_id']}"),
            difficulty=dataset_entry["difficulty"],
            tags=dataset_entry["tags"],
            problem_description=dataset_entry["problem_description"],
            time_complexity=dataset_entry["time_complexity"],
            explanation=dataset_entry["explanation"],
            categories=categories,
            boilerplate=boilerplate,
            solution_code=solution_code,
            test_harness=test_harness,
        )
