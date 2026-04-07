"""
Algo Reasoning Environment.

An OpenEnv-compatible environment for evaluating AI agents on:
1. Writing correct Rust code that compiles and passes tests
2. Providing accurate step-by-step reasoning
3. Stating the correct time complexity

Usage:
    from algo_reasoning_env import AlgoReasoningEnvironment

    env = AlgoReasoningEnvironment(data_dir="/data")
    observation = env.reset()

    # Agent submits solution
    action = AlgoReasoningAction(
        solution_code="impl Solution { pub fn two_sum(...) }",
        reasoning_steps="step-1: Use hashmap...",
        time_complexity="O(n)",
    )

    result = env.step(action)
    print(f"Reward: {result.reward}")
"""

from .models import (
    AlgoReasoningAction,
    AlgoReasoningObservation,
    EvaluationResult,
)
from .environment import AlgoReasoningEnvironment
from .rubric import AlgoReasoningRubric

__all__ = [
    "AlgoReasoningAction",
    "AlgoReasoningObservation",
    "AlgoReasoningEnvironment",
    "AlgoReasoningRubric",
    "EvaluationResult",
]
