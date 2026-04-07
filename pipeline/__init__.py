"""
Rust LeetCode Problem Compilation Pipeline

Phases:
1. category_resolver.py — Map problem tags to Rust categories
2. boilerplate/registry.py — Load and merge category-specific helpers
3. problem.py — Problem dataclass with test harnesses
4. assembler.py — Combine boilerplate + solution + harness
5. executor.py — Compile and run Rust code
6. logger.py — Log execution results

New Components:
- config.py — Centralized configuration
- llm_judge.py — LLM-as-judge for reasoning and complexity evaluation
- code_generator.py — Code generation using LLM
- run_pipeline.py — Unified pipeline with retry and evaluation
"""
