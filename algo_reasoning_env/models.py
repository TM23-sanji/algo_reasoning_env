"""
Data models for the Algo Reasoning Environment.

This environment evaluates AI agents on their ability to:
1. Write correct Rust code that compiles and passes tests
2. Provide accurate step-by-step reasoning
3. State the correct time complexity
"""

from typing import Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Evaluation result containing all scoring components."""

    reasoning_score: float = Field(..., description="Reasoning quality score (0.0-1.0)")
    complexity_score: int = Field(
        ..., description="Time complexity correctness (0 or 1)"
    )
    correctness_reward: float = Field(
        ..., description="Code correctness: 0.0 (fail), 0.3 (compile only), 1.0 (pass)"
    )
    predicted_complexity: str = Field(
        ..., description="Extracted time complexity from response"
    )
    compilation_error: Optional[str] = Field(
        default=None, description="Compilation error message if any"
    )
    test_output: Optional[str] = Field(
        default=None, description="Test execution output"
    )


class AlgoReasoningAction(Action):
    """Action for the Algo Reasoning environment."""

    solution_code: str = Field(..., description="Rust impl Solution block")
    reasoning_steps: str = Field(
        ..., description="Step-by-step reasoning (step-1, step-2, etc.)"
    )
    time_complexity: str = Field(
        ...,
        description="Time complexity (raw text, e.g., 'O(n)' or 'O(n^2) in worst case')",
    )


class AlgoReasoningObservation(Observation):
    """Observation from the Algo Reasoning environment."""

    problem_id: int = Field(..., description="Problem ID from dataset")
    task_id: str = Field(..., description="Task identifier (e.g., 'two-sum')")
    difficulty: str = Field(
        ..., description="Difficulty level: 'Easy', 'Medium', or 'Hard'"
    )
    problem_description: str = Field(..., description="Full problem description")
    starter_code: str = Field(..., description="Starter code template")
    expected_complexity: str = Field(
        ..., description="Expected time complexity (ground truth)"
    )
    ground_truth_explanation: str = Field(
        ..., description="Expert explanation for reasoning evaluation"
    )
    tags: list[str] = Field(
        default_factory=list, description="Problem tags for code assembly"
    )
    test_harness: str = Field(default="", description="Test harness code")

    # Evaluation results (populated after step())
    reward: Optional[float] = Field(
        default=None,
        description="Combined reward (0.5*correctness + 0.3*reasoning + 0.2*complexity)",
    )
    reasoning_score: Optional[float] = Field(
        default=None, description="Reasoning quality score (0.0-1.0)"
    )
    complexity_score: Optional[int] = Field(
        default=None, description="Time complexity correctness (0 or 1)"
    )
    correctness_reward: Optional[float] = Field(
        default=None, description="Code correctness: 0.0, 0.3, or 1.0"
    )
    evaluation: Optional[EvaluationResult] = Field(
        default=None, description="Full evaluation result"
    )
