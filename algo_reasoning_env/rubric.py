"""
Rubrics for the Algo Reasoning Environment.

Implements reward computation for:
1. Code correctness (compilation + test execution)
2. Reasoning quality (LLM judge)
3. Time complexity correctness (LLM judge)
"""

import os
import re
from typing import Optional

from openai import OpenAI

from openenv.core.rubrics.base import Rubric

from .compiler import evaluate_code
from .models import (
    AlgoReasoningAction,
    AlgoReasoningObservation,
    EvaluationResult,
)


# Default configuration
DEFAULT_API_BASE_URL = "https://lightning.ai/api/v1/"
DEFAULT_JUDGE_MODEL = "lightning-ai/gpt-oss-20b"
JUDGE_TIMEOUT = 30
REASONING_TEMPERATURE = 0.0
COMPLEXITY_TEMPERATURE = 0.0


class LLMJudgeRubric(Rubric):
    """
    LLM-based rubric for evaluating reasoning and time complexity.

    Uses an LLM to semantically evaluate generated content against
    ground truth from the dataset.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: str = DEFAULT_API_BASE_URL,
        model: str = DEFAULT_JUDGE_MODEL,
    ):
        """
        Initialize the LLM judge rubric.

        Args:
            api_key: API key for LLM calls. If None, reads from LIGHTNING_API_KEY env.
            api_base_url: Base URL for API endpoint.
            model: Model identifier for LLM calls.
        """
        super().__init__()
        self.api_key = api_key or os.getenv("LIGHTNING_API_KEY")
        if not self.api_key:
            raise ValueError("LIGHTNING_API_KEY environment variable is required")

        self.api_base_url = api_base_url
        self.model = model
        self.client = OpenAI(
            base_url=self.api_base_url,
            api_key=self.api_key,
        )

    def forward(
        self,
        action: AlgoReasoningAction,
        observation: AlgoReasoningObservation,
    ) -> tuple[float, int]:
        """
        Evaluate reasoning and time complexity.

        Args:
            action: The action containing solution and reasoning
            observation: The observation containing ground truth

        Returns:
            Tuple of (reasoning_score, complexity_score)
        """
        reasoning_score = self._evaluate_reasoning(
            action.reasoning_steps,
            observation.ground_truth_explanation,
        )

        complexity_score = self._evaluate_complexity(
            action.time_complexity,
            observation.expected_complexity,
        )

        return reasoning_score, complexity_score

    def _evaluate_reasoning(
        self,
        generated_steps: str,
        ground_truth_explanation: str,
    ) -> float:
        """
        Evaluate the quality of generated reasoning steps.

        Args:
            generated_steps: The generated reasoning steps
            ground_truth_explanation: Expert explanation from dataset

        Returns:
            Score between 0.0 and 1.0
        """
        prompt = f"""You are an expert computer science evaluator. Evaluate the quality of the generated reasoning steps compared to the expert explanation.

Rate the reasoning on a scale from 0.0 to 1.0, where:
- 1.0 = Excellent: All key algorithmic insights are correctly identified
- 0.7-0.9 = Good: Most key insights present, minor omissions or inaccuracies
- 0.4-0.6 = Fair: Some correct elements but missing important insights
- 0.1-0.3 = Poor: Major gaps or significant inaccuracies
- 0.0 = Completely incorrect or irrelevant

═══════════════════════════════════════
GENERATED REASONING STEPS
═══════════════════════════════════════
{generated_steps}

═══════════════════════════════════════
EXPERT EXPLANATION (Ground Truth)
═══════════════════════════════════════
{ground_truth_explanation}

═══════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════
Compare the generated reasoning to the expert explanation. Focus on:
1. Does it correctly identify the algorithmic approach?
2. Are the key data structures mentioned?
3. Is the logic flow correctly described?
4. Are there any misconceptions or errors?

Respond with ONLY a single float number between 0.0 and 1.0. No explanation needed.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                temperature=REASONING_TEMPERATURE,
                timeout=JUDGE_TIMEOUT,
            )

            content = response.choices[0].message.content.strip()

            match = re.search(r"(0?\.\d+|1\.0+|0|1)", content)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            else:
                return 0.0

        except Exception as e:
            print(f"Warning: Reasoning evaluation failed: {e}")
            return 0.0

    def _evaluate_complexity(
        self,
        predicted_complexity: str,
        ground_truth_complexity: str,
    ) -> int:
        """
        Evaluate if the predicted time complexity is semantically correct.

        Args:
            predicted_complexity: Full raw text after // time complexity
            ground_truth_complexity: Ground truth from dataset

        Returns:
            1 if semantically equivalent, 0 otherwise
        """
        prompt = f"""You are an expert in algorithm analysis.

COMPARE THE TIME COMPLEXITY of the predicted output against the ground truth.
Determine if they are semantically equivalent.

RULES:
- Focus on the Big-O notation in the predicted output, ignore extra explanations
- O(m*n) and O(m*l) are equivalent (both represent multiplication of two input sizes)
- O(n) and O(m) are equivalent (both linear in input size)
- O(n^2) and O(m^2) are equivalent (both quadratic)
- O(max(m,n)) and O(m+n) are considered equivalent for this evaluation
- O(log n) and O(log m) are equivalent (both logarithmic)

═══════════════════════════════════════
PREDICTED TIME COMPLEXITY (raw output)
═══════════════════════════════════════
{predicted_complexity}

═══════════════════════════════════════
GROUND TRUTH TIME COMPLEXITY
═══════════════════════════════════════
{ground_truth_complexity}

═══════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════
Extract the Big-O notation from the predicted output and compare it with the ground truth.
Are they semantically equivalent?

Respond with ONLY "1" for yes or "0" for no.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                temperature=COMPLEXITY_TEMPERATURE,
                timeout=JUDGE_TIMEOUT,
            )

            content = response.choices[0].message.content.strip()

            if "1" in content and "0" not in content:
                return 1
            elif content == "1":
                return 1
            else:
                return 0

        except Exception as e:
            print(f"Warning: Complexity evaluation failed: {e}")
            return 0


class AlgoReasoningRubric(Rubric):
    """
    Combined rubric for the Algo Reasoning environment.

    Evaluates:
    - Code correctness (compilation + tests)
    - Reasoning quality (LLM judge)
    - Time complexity correctness (LLM judge)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        compile_timeout: int = 30,
        test_timeout: int = 30,
    ):
        """
        Initialize the rubric.

        Args:
            api_key: API key for LLM judge
            compile_timeout: Compilation timeout in seconds
            test_timeout: Test execution timeout in seconds
        """
        super().__init__()

        self.compile_timeout = compile_timeout
        self.test_timeout = test_timeout

        # LLM Judge for reasoning and complexity
        self.llm_judge = LLMJudgeRubric(api_key=api_key)

    def forward(
        self,
        action: AlgoReasoningAction,
        observation: AlgoReasoningObservation,
    ) -> EvaluationResult:
        """
        Evaluate the complete action.

        Args:
            action: The submitted action with solution, reasoning, complexity
            observation: The observation with ground truth

        Returns:
            EvaluationResult with all scores
        """
        # Evaluate code correctness (compilation + tests)
        correctness_reward, compile_error, test_output = evaluate_code(
            solution_code=action.solution_code,
            starter_code=observation.starter_code,
            test_harness=observation.test_harness,
            tags=observation.tags,
            compile_timeout=self.compile_timeout,
            test_timeout=self.test_timeout,
        )

        # Evaluate reasoning and complexity with LLM
        reasoning_score, complexity_score = self.llm_judge(action, observation)

        # Extract clean complexity for storage
        predicted_complexity = extract_time_complexity(action.time_complexity)

        return EvaluationResult(
            reasoning_score=reasoning_score,
            complexity_score=complexity_score,
            correctness_reward=correctness_reward,
            predicted_complexity=predicted_complexity,
            compilation_error=compile_error,
            test_output=test_output,
        )


def extract_time_complexity(raw_text: str) -> str:
    """
    Extract clean Big-O notation from raw text.

    Args:
        raw_text: Raw text that may contain "O(n^2) in worst case..."

    Returns:
        Clean O notation like "O(n^2)"
    """
    pattern = r"(O\([^)]*(?:\([^)]*\)[^)]*)*\))"
    match = re.search(pattern, raw_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw_text.strip()
