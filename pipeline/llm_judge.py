"""
LLM-as-Judge Module

Provides evaluation capabilities using an LLM to judge:
1. Reasoning quality (comparing generated steps to expert explanation)
2. Time complexity correctness (semantic matching)

Uses OpenAI client with Lightning AI API.
"""

import os
from typing import Optional
from openai import OpenAI

from .config import (
    LIGHTNING_BASE_URL,
    JUDGE_MODEL,
    JUDGE_TIMEOUT,
    REASONING_EVALUATION_TEMPERATURE,
    COMPLEXITY_EVALUATION_TEMPERATURE,
    get_api_key,
)


class LLMJudge:
    """
    LLM-as-Judge for evaluating reasoning and time complexity.

    Uses an LLM to semantically evaluate generated content against
    ground truth from the dataset.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the judge with OpenAI client.

        Args:
            api_key: Lightning AI API key. If None, reads from environment.
        """
        self.api_key = api_key or get_api_key()
        self.client = OpenAI(
            base_url=LIGHTNING_BASE_URL,
            api_key=self.api_key,
        )

    def evaluate_reasoning(
        self,
        generated_steps: str,
        ground_truth_explanation: str,
    ) -> float:
        """
        Evaluate the quality of generated reasoning steps.

        Compares the model's reasoning steps (step-1 through step-5)
        against the expert explanation from the dataset.

        Args:
            generated_steps: The generated reasoning steps (step-1 to step-5).
            ground_truth_explanation: The expert explanation from dataset.

        Returns:
            A score between 0.0 and 1.0 indicating reasoning quality.
            1.0 = perfect reasoning that captures all key insights
            0.0 = completely incorrect or missing reasoning
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
                model=JUDGE_MODEL,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                temperature=REASONING_EVALUATION_TEMPERATURE,
                timeout=JUDGE_TIMEOUT,
            )

            content = response.choices[0].message.content.strip()

            # Extract float from response
            # Handle cases like "0.85", "Score: 0.85", etc.
            import re

            match = re.search(r"(0?\.\d+|1\.0+|0|1)", content)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            else:
                return 0.0

        except Exception as e:
            print(f"Warning: Reasoning evaluation failed: {e}")
            return 0.0

    def evaluate_time_complexity(
        self,
        predicted_complexity: str,
        ground_truth_complexity: str,
    ) -> int:
        """
        Evaluate if the predicted time complexity is semantically correct.

        Uses LLM to judge semantic equivalence, handling variable name
        differences (e.g., O(m*n) vs O(m*l) are equivalent).

        Args:
            predicted_complexity: The full raw text after // time complexity
                (may include explanations like "O(n^2) in the worst case...")
            ground_truth_complexity: The ground truth from dataset (e.g., "O(m + n)")

        Returns:
            1 if the complexities are semantically equivalent, 0 otherwise.
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
                model=JUDGE_MODEL,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                temperature=COMPLEXITY_EVALUATION_TEMPERATURE,
                timeout=JUDGE_TIMEOUT,
            )

            content = response.choices[0].message.content.strip()

            # Extract 0 or 1 from response
            if "1" in content and "0" not in content:
                return 1
            elif content == "1":
                return 1
            else:
                return 0

        except Exception as e:
            print(f"Warning: Complexity evaluation failed: {e}")
            return 0

    def evaluate_complexity_exact(self, predicted: str, ground_truth: str) -> int:
        """
        Exact string match fallback for complexity evaluation.

        Args:
            predicted: Predicted complexity string
            ground_truth: Ground truth complexity string

        Returns:
            1 if exact match, 0 otherwise
        """
        # Normalize both strings
        pred_normalized = predicted.strip().replace(" ", "").replace("\\", "")
        truth_normalized = ground_truth.strip().replace(" ", "").replace("\\", "")

        return 1 if pred_normalized == truth_normalized else 0
