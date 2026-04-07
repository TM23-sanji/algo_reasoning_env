"""
Phase 5: Assembler

New simplified flow:
1. Take generated_code (full impl Solution block from model)
2. Add header structs if needed (based on tags: Linked List, Tree)
3. Strip reasoning comments
4. Append test harness

Header structs:
- Linked List: ListNode struct + impl
- Tree/Binary Tree: TreeNode struct + impl + imports
"""

import re
from typing import Dict, List, Optional, Tuple


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


def get_headers_for_tags(tags: List[str]) -> str:
    """
    Generate header code based on problem tags.

    Args:
        tags: List of tags from dataset.jsonl (e.g., ["Array", "Hash Table", "Linked List"])

    Returns:
        Header code string with struct definitions needed for this problem
    """
    tags_lower = [t.lower() for t in tags]

    headers = ""

    # Check for Linked List
    if "linked list" in tags_lower:
        headers += LISTNODE_HEADER

    # Check for Tree/Binary Tree
    if "tree" in tags_lower or "binary tree" in tags_lower:
        headers += TREENODE_HEADER

    return headers


def strip_reasoning_comments(code: str) -> str:
    """Remove reasoning comments from generated code."""
    # Remove reasoning header and everything after it (including time complexity)
    code = re.sub(r"\n// reasoning\n.*", "", code, flags=re.DOTALL)
    # Remove individual step lines
    code = re.sub(r"\n// step-\d+:.*", "", code, flags=re.DOTALL)
    # Remove time complexity section
    code = re.sub(r"\n// time complexity\n.*", "", code, flags=re.DOTALL)
    return code


def extract_raw_time_complexity(code: str) -> str:
    """
    Extract the full raw text after // time complexity for LLM judge evaluation.

    This captures everything the model wrote after the time complexity header,
    including any explanations like "O(n^2) in the worst case...".

    Args:
        code: The generated code with reasoning comments.

    Returns:
        The full text after // time complexity, or empty string if not found.
    """
    # Match everything after "// time complexity" until end of code
    pattern = r"// time complexity\s*\n?(.*)"
    match = re.search(pattern, code, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_time_complexity(code: str) -> str:
    """
    Extract clean Big-O notation from generated code for display in results.

    Handles formats like:
    - O(n)
    - O(n^2)
    - O(m*n)
    - O(log n)
    - O(max(m, n))

    Args:
        code: The generated code with reasoning comments.

    Returns:
        The extracted Big-O notation string (e.g., "O(n)"), or empty string if not found.
    """
    # Get the raw complexity text first
    raw = extract_raw_time_complexity(code)
    if not raw:
        return ""

    # Extract O(...) with support for nested parentheses like O(max(m, n))
    # This pattern handles: O(n), O(n^2), O(m*n), O(log n), O(max(m,n))
    pattern = r"(O\([^)]*(?:\([^)]*\)[^)]*)*\))"
    match = re.search(pattern, raw, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: find any O(...) in the last few lines
    lines = code.split("\n")
    for line in reversed(lines[-10:]):
        match = re.search(r"(O\([^)]*(?:\([^)]*\)[^)]*)*\))", line)
        if match:
            return match.group(1).strip()

    return ""


def extract_reasoning_steps(code: str) -> str:
    """
    Extract reasoning steps from generated code.

    Args:
        code: The generated code with reasoning comments.

    Returns:
        The extracted reasoning steps (step-1 through step-5), or empty string if not found.
    """
    # Find the reasoning section
    reasoning_match = re.search(
        r"// reasoning\s*\n(.*?)(?=// time complexity|\Z)",
        code,
        re.DOTALL | re.IGNORECASE,
    )
    if reasoning_match:
        return reasoning_match.group(1).strip()

    # Fallback: collect all step lines
    step_lines = []
    for line in code.split("\n"):
        if re.match(r"// step-\d+:", line.strip()):
            step_lines.append(line.strip())

    return "\n".join(step_lines) if step_lines else ""


def assemble_rust_code_v2(
    generated_code: str,
    test_harness: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """
    Assemble complete Rust code for a problem.

    Structure:
    1. Header structs (based on tags: Linked List → ListNode, Tree → TreeNode)
    2. Generated code (impl Solution block with reasoning comments stripped)
    3. Test harness (fn check, fn main)

    Args:
        generated_code: The LLM-generated solution code (full impl block)
        test_harness: The test harness (fn check + fn main)
        tags: List of tags from dataset.jsonl to determine needed headers

    Returns:
        Complete, compilable Rust code
    """
    # Part 1: Add header structs based on tags
    headers = get_headers_for_tags(tags or [])

    # Part 2: Strip reasoning comments from generated code
    code_body = strip_reasoning_comments(generated_code)

    # Assemble: headers + generated code + test harness
    assembled = headers + code_body

    # Part 3: Append test harness
    if test_harness:
        assembled = assembled.strip() + "\n\n" + test_harness
    else:
        assembled = (
            assembled.strip() + '\n\nfn main() {\n    println!("No tests provided");\n}'
        )

    return assembled


# Keep old function for backwards compatibility (deprecated)
def assemble_rust_code(
    starter_code: str,
    generated_code: Optional[str],
    test_harness: Optional[str] = None,
) -> str:
    """Deprecated: Use assemble_rust_code_v2 instead."""
    if generated_code:
        generated_code = strip_reasoning_comments(generated_code)
        assembled = starter_code.replace("// OUR CODE GOES HERE", generated_code)
    else:
        assembled = starter_code.replace("// OUR CODE GOES HERE", "todo!()")

    if test_harness:
        assembled = assembled.strip() + "\n\n" + test_harness
    else:
        assembled = (
            assembled.strip() + '\n\nfn main() {\n    println!("No tests provided");\n}'
        )

    return assembled


def assemble_from_jsonl(
    starter_code: str,
    generated_code: Optional[str],
    test_harness: Optional[str],
) -> str:
    """Alias for assemble_rust_code - for compatibility."""
    return assemble_rust_code(starter_code, generated_code, test_harness)


def validate_assembly(assembled_code: str) -> Tuple[bool, str]:
    """
    Perform basic validation of assembled code (syntax checks).

    Args:
        assembled_code: The assembled Rust code

    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []

    if "impl Solution" not in assembled_code:
        errors.append("Missing 'impl Solution' block")

    if "fn main()" not in assembled_code:
        errors.append("Missing 'fn main()' function")

    if not assembled_code.strip().endswith("}"):
        errors.append("Code does not end with closing brace")

    if errors:
        return False, "; ".join(errors)

    return True, ""


def transform_harness_to_test_format(harness: str) -> str:
    """
    Transform test harness from fn check() format to #[cfg(test)] format.

    Original format:
        fn check() {
            fn helper() { ... }
            assert_eq!(...);
            assert_eq!(...);
        }
        fn main() { ... }

    Transformed format:
        #[cfg(test)]
        mod tests {
            use super::*;

            fn helper() { ... }

            #[test]
            fn test_case_1() { assert_eq!(...); }

            #[test]
            fn test_case_2() { assert_eq!(...); }
        }

        fn main() {}

    Args:
        harness: Original test harness string

    Returns:
        Transformed harness string in test format
    """
    # Find fn check() block
    check_match = re.search(r"fn check\(\)\s*\{", harness)
    if not check_match:
        # No fn check() found, return as-is (might be simple harness)
        return harness

    # Find the matching closing brace for fn check()
    start_pos = check_match.end() - 1  # position of opening {
    brace_count = 0
    end_pos = start_pos
    for i, char in enumerate(harness[start_pos:], start_pos):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                end_pos = i
                break

    check_body = harness[start_pos + 1 : end_pos]

    # Find fn main() position (everything after check block is main)
    main_match = re.search(r"fn main\(\)", harness[end_pos:])
    if main_match:
        main_start = end_pos + main_match.start()
        main_section = harness[main_start:]
    else:
        main_section = ""

    # Extract helper functions (fn definitions before first assert_eq)
    helper_lines = []
    assert_lines = []
    lines = check_body.strip().split("\n")

    # Find first assert_eq line
    first_assert_idx = -1
    for idx, line in enumerate(lines):
        if "assert_eq!" in line:
            first_assert_idx = idx
            break

    if first_assert_idx > 0:
        # Everything before first assert_eq is helper functions
        helper_lines = lines[:first_assert_idx]
        assert_lines = lines[first_assert_idx:]
    else:
        # No helpers, all lines are assert_eq
        assert_lines = lines

    # Clean up helper functions (remove leading indentation)
    cleaned_helpers = []
    for line in helper_lines:
        stripped = line.strip()
        if stripped:
            cleaned_helpers.append(stripped)

    # Extract each assert_eq statement (handle multi-line asserts)
    # Also filter out comment lines (lines starting with //)
    test_cases = []
    current_assert = []
    for line in assert_lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip comment-only lines (lines that are ONLY comments)
        if stripped.startswith("//"):
            continue
        # For lines with assert_eq!, strip any trailing comments
        if "assert_eq!" in stripped:
            # Take only the part before any // comment
            assert_part = stripped.split("//")[0].strip()
            if assert_part:
                current_assert.append(assert_part)
        else:
            current_assert.append(stripped)
        # Check if this assert is complete (has semicolon)
        if ";" in stripped:
            test_cases.append(" ".join(current_assert))
            current_assert = []

    # Build transformed harness
    result = []
    result.append("#[cfg(test)]")
    result.append("mod tests {")
    result.append("    use super::*;")
    result.append("")

    # Add helper functions
    for helper in cleaned_helpers:
        # Add proper indentation (4 spaces)
        for h_line in helper.split("\n"):
            result.append("    " + h_line)
    if cleaned_helpers:
        result.append("")

    # Add each assert_eq as a separate test function
    for idx, assert_stmt in enumerate(test_cases):
        result.append(f"    #[test]")
        result.append(f"    fn test_case_{idx + 1}() {{")
        result.append(f"        {assert_stmt}")
        result.append(f"    }}")
        result.append("")

    result.append("}")
    result.append("")
    result.append("fn main() {{}}")

    return "\n".join(result)


def transform_harness_to_test_format_v2(harness: str) -> str:
    """
    Alternative implementation using more robust parsing.
    Handles edge cases like comments, multi-line statements.
    """
    # Check if already in test format
    if "#[cfg(test)]" in harness or "mod tests" in harness:
        return harness

    # Find fn check() block more robustly
    check_start = harness.find("fn check()")
    if check_start == -1:
        return harness

    # Find the block
    brace_start = harness.find("{", check_start)
    if brace_start == -1:
        return harness

    # Count braces to find end
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

    # Find fn main()
    main_pos = harness.find("fn main()", block_end)
    if main_pos == -1:
        main_section = ""
    else:
        main_section = ""

    # Split into lines and process
    lines = check_body.split("\n")

    # Separate helpers from assertions
    # Helpers are function definitions, assertions are assert_eq!
    helper_functions = []
    assertions = []
    in_assert_block = False
    current_assert_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("//"):
            continue

        # Check if this is a helper function definition
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
            # Continuation of previous line (e.g., function body)
            if helper_functions:
                helper_functions[-1] += " " + stripped
            elif current_assert_lines:
                current_assert_lines[-1] += " " + stripped
                if ";" in stripped:
                    assertions.append(" ".join(current_assert_lines))
                    current_assert_lines = []

    # Build output
    output = []
    output.append("#[cfg(test)]")
    output.append("mod tests {")
    output.append("    use super::*;")
    output.append("")

    # Helpers
    for hf in helper_functions:
        # Indent properly
        for hf_line in hf.split("\n"):
            output.append("    " + hf_line)
    if helper_functions:
        output.append("")

    # Test cases
    for idx, assertion in enumerate(assertions):
        output.append(f"    #[test]")
        output.append(f"    fn test_case_{idx + 1}() {{")
        # Clean assertion - remove leading assert_eq!
        # Keep the full assertion
        output.append(f"        {assertion}")
        output.append(f"    }}")
        output.append("")

    output.append("}")
    output.append("")
    output.append("fn main() {}")

    return "\n".join(output)
