"""
Rust Compiler for the Algo Reasoning Environment.

Handles compilation of Rust code and execution of test cases.
Uses rustc for compilation and test execution.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


LISTNODE_HEADER = """#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode {
            next: None,
            val,
        }
    }
}

"""

TREENODE_HEADER = """use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

"""


def get_headers_for_tags(tags: list[str]) -> str:
    """
    Generate header code based on problem tags.

    Args:
        tags: List of tags from dataset

    Returns:
        Header code string with struct definitions
    """
    tags_lower = [t.lower() for t in tags]

    headers = ""

    if "linked list" in tags_lower:
        headers += LISTNODE_HEADER

    if "tree" in tags_lower or "binary tree" in tags_lower:
        headers += TREENODE_HEADER

    return headers


def strip_reasoning_comments(code: str) -> str:
    """Remove reasoning comments from generated code."""
    code = re.sub(r"\n// reasoning\n.*", "", code, flags=re.DOTALL)
    code = re.sub(r"\n// step-\d+:.*", "", code, flags=re.DOTALL)
    code = re.sub(r"\n// time complexity\n.*", "", code, flags=re.DOTALL)
    return code


def _strip_impl_wrapper(code: str) -> str:
    """
    Extract just the function body from a model-generated ``impl Solution { ... }`` block.

    The starter code already provides:
        impl Solution {
            pub fn two_sum(...) {
                // OUR CODE GOES HERE   <-- marker is INSIDE the fn body
            }
        }

    The model outputs a full ``impl Solution { pub fn ... { body } }`` block.
    We need to extract only the innermost function body so it can replace
    the marker without duplicating the fn signature or impl wrapper.

    Example:
        Input:  "impl Solution {\\n    pub fn foo() {\\n        let x = 1;\\n    }\\n}"
        Output: "let x = 1;"
    """
    # First, strip the outer impl Solution { ... }
    match = re.search(r"impl\s+Solution\s*\{", code)
    if not match:
        return code

    start = match.end()
    depth = 1
    i = start
    while i < len(code) and depth > 0:
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
        i += 1

    inner = code[start : i - 1].strip()

    # Now strip the function signature + body wrapper: pub fn name(...) { ... }
    fn_match = re.search(r"(?:pub\s+)?fn\s+\w+\s*\([^)]*\)\s*(?:->\s*\S+)?\s*\{", inner)
    if not fn_match:
        return inner

    fn_start = fn_match.end()
    fn_depth = 1
    j = fn_start
    while j < len(inner) and fn_depth > 0:
        if inner[j] == "{":
            fn_depth += 1
        elif inner[j] == "}":
            fn_depth -= 1
        j += 1

    body = inner[fn_start : j - 1].strip()
    return body


def transform_harness_to_test_format(harness: str) -> str:
    """
    Transform test harness from fn check() format to #[cfg(test)] format.
    """
    if "#[cfg(test)]" in harness or "mod tests" in harness:
        return harness

    check_start = harness.find("fn check()")
    if check_start == -1:
        return harness

    brace_start = harness.find("{", check_start)
    if brace_start == -1:
        return harness

    depth = 0
    block_end = brace_start
    for i in range(brace_start, len(harness)):
        if harness[i] == "{":
            depth += 1
        elif harness[i] == "}":
            depth -= 1
            if depth == 0:
                block_end = i
                break

    check_body = harness[brace_start + 1 : block_end]

    lines = check_body.split("\n")

    helper_functions = []
    assertions = []
    current_assert_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith("//"):
            continue

        if stripped.startswith("fn ") and "assert_eq!" not in stripped:
            if current_assert_lines:
                assertions.append(" ".join(current_assert_lines))
                current_assert_lines = []
            helper_functions.append(stripped)
        elif "assert_eq!" in stripped:
            current_assert_lines.append(stripped)
            if ";" in stripped:
                assertions.append(" ".join(current_assert_lines))
                current_assert_lines = []
        else:
            if helper_functions:
                helper_functions[-1] += " " + stripped
            elif current_assert_lines:
                current_assert_lines[-1] += " " + stripped
                if ";" in stripped:
                    assertions.append(" ".join(current_assert_lines))
                    current_assert_lines = []

    output = []
    output.append("#[cfg(test)]")
    output.append("mod tests {")
    output.append("    use super::*;")
    output.append("")

    for hf in helper_functions:
        for hf_line in hf.split("\n"):
            output.append("    " + hf_line)
    if helper_functions:
        output.append("")

    for idx, assertion in enumerate(assertions):
        output.append(f"    #[test]")
        output.append(f"    fn test_case_{idx + 1}() {{")
        output.append(f"        {assertion}")
        output.append(f"    }}")
        output.append("")

    output.append("}")
    output.append("")
    output.append("fn main() {}")

    return "\n".join(output)


def assemble_code(
    solution_code: str,
    starter_code: str,
    test_harness: str,
    tags: list[str],
) -> str:
    """
    Assemble complete Rust code for compilation and testing.

    Args:
        solution_code: The impl Solution block
        starter_code: The starter code template
        test_harness: The test harness code
        tags: Problem tags for header selection

    Returns:
        Complete Rust source code ready for compilation
    """
    # Add headers based on tags
    headers = get_headers_for_tags(tags)

    # Strip reasoning comments from solution
    code_body = strip_reasoning_comments(solution_code)

    # Combine starter code with solution
    if "// OUR CODE GOES HERE" in starter_code:
        # Starter already provides the impl Solution wrapper.
        # Strip the impl Solution { ... } wrapper from model output if present
        # so we don't nest impl blocks.
        if "impl Solution" in code_body:
            code_body = _strip_impl_wrapper(code_body)
        assembled = starter_code.replace("// OUR CODE GOES HERE", code_body)
        # Ensure struct Solution; is present (almost never in starter codes)
        if "struct Solution" not in assembled:
            assembled = "struct Solution;\n\n" + assembled
    else:
        assembled = f"{headers}struct Solution;\n\n{code_body}"

    # Transform and append test harness
    if test_harness:
        test_code = transform_harness_to_test_format(test_harness)
        assembled = assembled.strip() + "\n\n" + test_code
    else:
        assembled = (
            assembled.strip() + '\n\nfn main() {\n    println!("No tests provided");\n}'
        )

    return headers + assembled


class CompilationResult:
    """Result of Rust compilation."""

    def __init__(
        self,
        success: bool,
        error: Optional[str] = None,
        binary_path: Optional[str] = None,
    ):
        self.success = success
        self.error = error
        self.binary_path = binary_path


class TestResult:
    """Result of test execution."""

    def __init__(
        self,
        all_passed: bool,
        output: str,
        failed_tests: Optional[list[str]] = None,
    ):
        self.all_passed = all_passed
        self.output = output
        self.failed_tests = failed_tests or []


def compile_rust_code(
    code: str,
    timeout_seconds: int = 30,
    tmpdir: Optional[str] = None,
) -> CompilationResult:
    """
    Compile Rust code using rustc.

    Args:
        code: Rust source code
        timeout_seconds: Compilation timeout
        tmpdir: Directory to write source and binary into.
                If None a new temp directory is created (caller must clean up).

    Returns:
        CompilationResult with success status and error if any
    """
    if tmpdir is None:
        tmpdir_path = Path(tempfile.mkdtemp())
    else:
        tmpdir_path = Path(tmpdir)

    source_file = tmpdir_path / "main.rs"
    binary_path = tmpdir_path / "test_runner"

    source_file.write_text(code)

    try:
        result = subprocess.run(
            ["rustc", "--test", "-o", str(binary_path), str(source_file)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        if result.returncode == 0:
            return CompilationResult(success=True, binary_path=str(binary_path))
        else:
            error_msg = result.stderr or result.stdout
            return CompilationResult(success=False, error=error_msg)

    except subprocess.TimeoutExpired:
        return CompilationResult(success=False, error="Compilation timeout exceeded")
    except FileNotFoundError:
        return CompilationResult(
            success=False, error="rustc not found. Please install Rust."
        )
    except Exception as e:
        return CompilationResult(success=False, error=str(e))


def run_tests(
    binary_path: str,
    timeout_seconds: int = 30,
) -> TestResult:
    """
    Run compiled Rust tests.

    Args:
        binary_path: Path to compiled test binary
        timeout_seconds: Execution timeout

    Returns:
        TestResult with pass/fail status and output
    """
    try:
        result = subprocess.run(
            [binary_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = result.stdout + result.stderr

        failed_tests = []
        for line in output.split("\n"):
            if " ... FAILED" in line:
                match = re.search(r"test tests::(\S+)", line)
                if match:
                    failed_tests.append(match.group(1))

        all_passed = len(failed_tests) == 0 and "test result: ok" in output.lower()

        if "test result: FAILED" in output or "failures:" in output:
            all_passed = False

        return TestResult(
            all_passed=all_passed, output=output, failed_tests=failed_tests
        )

    except subprocess.TimeoutExpired:
        return TestResult(all_passed=False, output="Execution timeout exceeded")
    except Exception as e:
        return TestResult(all_passed=False, output=str(e))


def evaluate_code(
    solution_code: str,
    starter_code: str,
    test_harness: str,
    tags: list[str],
    compile_timeout: int = 30,
    test_timeout: int = 30,
) -> Tuple[float, Optional[str], Optional[str]]:
    """
    Evaluate Rust code: compile and run tests.

    Args:
        solution_code: The impl Solution block
        starter_code: The starter code template
        test_harness: The test harness code
        tags: Problem tags for header selection
        compile_timeout: Compilation timeout
        test_timeout: Test execution timeout

    Returns:
        Tuple of (correctness_reward, compilation_error, test_output)
        correctness_reward: 1.0 (pass), 0.3 (compile only), 0.0 (fail)
    """
    # Assemble the complete code
    assembled = assemble_code(solution_code, starter_code, test_harness, tags)

    # Compile and run tests inside a single TemporaryDirectory so the
    # binary stays alive for run_tests.
    with tempfile.TemporaryDirectory() as tmpdir:
        compile_result = compile_rust_code(assembled, compile_timeout, tmpdir=tmpdir)

        if not compile_result.success:
            return 0.0, compile_result.error, None

        # Run tests — binary_path lives inside tmpdir which is still alive
        test_result = run_tests(compile_result.binary_path, test_timeout)

    if test_result.all_passed:
        return 1.0, None, test_result.output
    else:
        return 0.3, None, test_result.output
