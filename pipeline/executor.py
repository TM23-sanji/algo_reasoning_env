"""
Phase 4: Executor

Compiles and executes Rust code, captures results and errors.
"""

import subprocess
import tempfile
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional
from .problem import Problem
from .assembler import assemble_rust_code, validate_assembly


class ExecutionResult:
    """Result of code compilation and execution."""

    def __init__(
        self,
        problem_id: int,
        compilation_success: bool,
        execution_success: bool,
        compilation_error: Optional[str] = None,
        execution_error: Optional[str] = None,
        execution_output: Optional[str] = None,
    ):
        self.problem_id = problem_id
        self.compilation_success = compilation_success
        self.execution_success = execution_success
        self.compilation_error = compilation_error
        self.execution_error = execution_error
        self.execution_output = execution_output

    def to_dict(self) -> dict:
        """Convert to dictionary for JSONL serialization."""
        return {
            "problem_id": self.problem_id,
            "compilation_success": self.compilation_success,
            "execution_success": self.execution_success,
            "compilation_error": self.compilation_error,
            "execution_error": self.execution_error,
            "execution_output": self.execution_output,
        }


def compile_rust_code(
    code: str, timeout_seconds: int = 30
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Compile Rust code using rustc in a temporary directory.

    Args:
        code: Rust source code
        timeout_seconds: Compilation timeout

    Returns:
        Tuple of (success, binary_path or error_message, compilation_output)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write source file
        source_file = tmpdir / "main.rs"
        source_file.write_text(code)

        # Compile
        binary_path = tmpdir / "main"
        try:
            result = subprocess.run(
                ["rustc", "-o", str(binary_path), str(source_file)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

            if result.returncode == 0:
                # Copy binary to persistent location (won't work across tmpdir deletion)
                # Instead, we'll need to re-execute within this function
                return True, str(source_file), result.stderr or result.stdout

            else:
                return False, None, result.stderr or result.stdout

        except subprocess.TimeoutExpired:
            return False, None, "Compilation timeout exceeded"
        except FileNotFoundError:
            return (
                False,
                None,
                "rustc not found. Please install Rust (https://rustup.rs/)",
            )
        except Exception as e:
            return False, None, str(e)


def execute_rust_code(code: str, timeout_seconds: int = 10) -> ExecutionResult:
    """
    Compile and execute Rust code, capturing output and errors.

    Args:
        code: Rust source code
        timeout_seconds: Execution timeout

    Returns:
        ExecutionResult with status and output
    """
    # TODO: Extract problem_id from code context - for now use 0
    problem_id = 0

    # Validate assembly first
    is_valid, error_msg = validate_assembly(code)
    if not is_valid:
        return ExecutionResult(
            problem_id=problem_id,
            compilation_success=False,
            execution_success=False,
            compilation_error=f"Validation error: {error_msg}",
        )

    # Compile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write source file
        source_file = tmpdir / "main.rs"
        source_file.write_text(code)

        # Compile
        binary_path = tmpdir / "main"
        try:
            compile_result = subprocess.run(
                ["rustc", "-o", str(binary_path), str(source_file)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

            if compile_result.returncode != 0:
                return ExecutionResult(
                    problem_id=problem_id,
                    compilation_success=False,
                    execution_success=False,
                    compilation_error=compile_result.stderr or compile_result.stdout,
                )

            # Compilation successful, now execute
            try:
                exec_result = subprocess.run(
                    [str(binary_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )

                if exec_result.returncode == 0:
                    return ExecutionResult(
                        problem_id=problem_id,
                        compilation_success=True,
                        execution_success=True,
                        execution_output=exec_result.stdout,
                    )
                else:
                    return ExecutionResult(
                        problem_id=problem_id,
                        compilation_success=True,
                        execution_success=False,
                        execution_error=exec_result.stderr or exec_result.stdout,
                    )

            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    problem_id=problem_id,
                    compilation_success=True,
                    execution_success=False,
                    execution_error="Execution timeout exceeded",
                )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                problem_id=problem_id,
                compilation_success=False,
                execution_success=False,
                compilation_error="Compilation timeout exceeded",
            )
        except FileNotFoundError:
            return ExecutionResult(
                problem_id=problem_id,
                compilation_success=False,
                execution_success=False,
                compilation_error="rustc not found. Please install Rust (https://rustup.rs/)",
            )
        except Exception as e:
            return ExecutionResult(
                problem_id=problem_id,
                compilation_success=False,
                execution_success=False,
                compilation_error=str(e),
            )


def execute_problem(problem: Problem, timeout_seconds: int = 10) -> Problem:
    """
    Execute a problem: assemble code, compile, and run.

    Args:
        problem: Problem instance
        timeout_seconds: Execution timeout

    Returns:
        Updated Problem with execution results
    """
    # Assemble code
    assembled_code = assemble_rust_code(problem)

    # Execute
    result = execute_rust_code(assembled_code, timeout_seconds)

    # Update problem with results
    problem.compilation_success = result.compilation_success
    problem.execution_success = result.execution_success
    problem.compilation_error = result.compilation_error
    problem.execution_error = result.execution_error
    problem.execution_output = result.execution_output

    return problem
