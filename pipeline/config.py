"""
Centralized configuration for the Rust dataset pipeline.

This module contains all configuration settings including:
- API configuration (HuggingFace)
- Model settings
- Timeout configurations
"""

import os
from pathlib import Path


# =============================================================================
# API Configuration
# =============================================================================

HF_BASE_URL = "https://router.huggingface.co/v1"
"""Base URL for HuggingFace Inference API."""

LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"
"""Model to use for code generation."""

JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
"""Model to use for LLM-as-judge evaluation."""


def get_api_key() -> str:
    """
    Get API key from environment variable.

    Returns:
        The API key string.

    Raises:
        ValueError: If HF_TOKEN or API_KEY environment variable is not set.
    """
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "HF_TOKEN or API_KEY environment variable not set. "
            "Please set it via: export HF_TOKEN=your_key"
        )
    return api_key


# =============================================================================
# Timeout Settings
# =============================================================================

GENERATION_TIMEOUT = 120
"""Timeout in seconds for code generation API calls."""

JUDGE_TIMEOUT = 30
"""Timeout in seconds for judge evaluation API calls."""

COMPILATION_TIMEOUT = 30
"""Timeout in seconds for Rust compilation."""

EXECUTION_TIMEOUT = 10
"""Timeout in seconds for test execution."""


# =============================================================================
# Data Paths
# =============================================================================

DEFAULT_DATA_DIR = Path("complexity_reasoning_data")
"""Default directory for data files."""

DATASET_FILENAME = "dataset.jsonl"
"""Filename for the problem dataset."""

STARTER_CODES_FILENAME = "starter_codes.jsonl"
"""Filename for starter codes."""

TEST_HARNESS_FILENAME = "test_harness.jsonl"
"""Filename for test harnesses."""

GENERATED_SOLUTIONS_FILENAME = "generated_solutions.jsonl"
"""Filename for generated solutions."""

RESULTS_FILENAME = "results.jsonl"
"""Filename for pipeline results."""

DEBUG_FILENAME = "code_gen_debug.jsonl"
"""Filename for debug output."""


def get_data_path(filename: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """
    Get full path to a data file.

    Args:
        filename: Name of the file.
        data_dir: Directory containing data files.

    Returns:
        Full path to the file.
    """
    return data_dir / filename


# =============================================================================
# Generation Settings
# =============================================================================

MAX_ROLLOUTS = 3
"""Maximum number of retry attempts per problem."""

DEFAULT_NUM_PROBLEMS = 50
"""Default number of problems to process."""


# =============================================================================
# LLM Judge Settings
# =============================================================================

REASONING_EVALUATION_TEMPERATURE = 0.0
"""Temperature for reasoning evaluation (deterministic)."""

COMPLEXITY_EVALUATION_TEMPERATURE = 0.0
"""Temperature for complexity evaluation (deterministic)."""
