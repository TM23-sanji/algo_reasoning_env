"""
Algo Reasoning Environment.

An OpenEnv-compatible environment that evaluates AI agents on:
1. Writing correct Rust code that compiles and passes tests
2. Providing accurate step-by-step reasoning
3. Stating the correct time complexity

Implements the OpenEnv Environment interface with reset()/step()/state().
"""

import os
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .data_loader import DataLoader
from .models import (
    AlgoReasoningAction,
    AlgoReasoningObservation,
    EvaluationResult,
)
from .rubric import AlgoReasoningRubric


class AlgoReasoningEnvironment(Environment):
    """
    Environment for evaluating algorithmic reasoning in Rust.

    This environment presents LeetCode-style problems and evaluates
    agent submissions on three dimensions:
    1. Code correctness (compilation + test pass)
    2. Reasoning quality (step-by-step explanation)
    3. Time complexity accuracy

    The environment cycles through problems in order (0, 1, 2, ...)
    and wraps around when all problems are exhausted.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(
        self,
        data_dir: str = "/data",
        api_key: Optional[str] = None,
        compile_timeout: int = 30,
        test_timeout: int = 30,
    ):
        """
        Initialize the environment.

        Args:
            data_dir: Directory containing dataset files
            api_key: API key for LLM judge (reads from env if not provided)
            compile_timeout: Compilation timeout in seconds
            test_timeout: Test execution timeout in seconds
        """
        self._data_loader = DataLoader(data_dir=data_dir)
        self._state = State(
            episode_id=str(uuid.uuid4()),
            step_count=0,
        )
        self._current_observation: Optional[AlgoReasoningObservation] = None
        self._rubric: Optional[AlgoReasoningRubric] = None

        # Initialize rubric
        self._rubric = AlgoReasoningRubric(
            api_key=api_key or os.getenv("LIGHTNING_API_KEY"),
            compile_timeout=compile_timeout,
            test_timeout=test_timeout,
        )

    def set_rubric(self, rubric: AlgoReasoningRubric) -> None:
        """
        Set or replace the rubric used for evaluation.

        Args:
            rubric: The rubric to use for evaluation
        """
        self._rubric = rubric

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AlgoReasoningObservation:
        """
        Reset the environment and return the next problem.

        Args:
            seed: Optional seed for reproducibility (not used)
            episode_id: Optional episode ID for tracking
            **kwargs: Additional arguments (ignored)

        Returns:
            Observation containing the problem description
        """
        # Generate new episode ID if not provided
        if episode_id is None:
            episode_id = str(uuid.uuid4())

        self._state = State(
            episode_id=episode_id,
            step_count=0,
        )

        # Get next problem
        problem = self._data_loader.get_next()

        if problem is None:
            raise RuntimeError("No problems available in dataset")

        # Get starter code and test harness
        starter_code = self._data_loader.get_starter_code(problem.problem_id) or ""
        test_harness = self._data_loader.get_test_harness(problem.problem_id) or ""

        # Create observation with problem data
        self._current_observation = AlgoReasoningObservation(
            problem_id=problem.problem_id,
            task_id=problem.task_id,
            difficulty=problem.difficulty,
            problem_description=problem.problem_description,
            starter_code=starter_code,
            expected_complexity=problem.time_complexity,
            ground_truth_explanation=problem.explanation,
            tags=problem.tags,
            test_harness=test_harness,
            done=False,
            reward=None,
        )

        return self._current_observation

    def step(
        self,
        action: AlgoReasoningAction,
        **kwargs: Any,
    ) -> AlgoReasoningObservation:
        """
        Evaluate an agent's submission.

        The action contains the solution code, reasoning steps, and time complexity.
        The rubric evaluates all three components and computes the combined reward.

        Args:
            action: The agent's submission
            **kwargs: Additional arguments (ignored)

        Returns:
            Observation with evaluation results
        """
        if self._current_observation is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        if self._rubric is None:
            raise RuntimeError(
                "No rubric set. Call set_rubric() or ensure API key is available."
            )

        self._state.step_count += 1

        # Evaluate the action
        evaluation = self._rubric(action, self._current_observation)

        # Compute combined reward: 0.5*correctness + 0.3*reasoning + 0.2*complexity
        combined_reward = (
            0.5 * evaluation.correctness_reward
            + 0.3 * evaluation.reasoning_score
            + 0.2 * evaluation.complexity_score
        )

        # Update observation with results
        self._current_observation.reward = combined_reward
        self._current_observation.reasoning_score = evaluation.reasoning_score
        self._current_observation.complexity_score = evaluation.complexity_score
        self._current_observation.correctness_reward = evaluation.correctness_reward
        self._current_observation.evaluation = evaluation
        self._current_observation.done = True  # Single-step episode

        return self._current_observation

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            State object with episode_id and step_count
        """
        return self._state

    def get_metadata(self) -> dict:
        """
        Get environment metadata.

        Returns:
            Dictionary with environment information
        """
        return {
            "name": "algo_reasoning_env",
            "version": "1.0.0",
            "num_problems": len(self._data_loader),
            "description": "Evaluates AI agents on Rust code correctness, reasoning, and time complexity",
        }
