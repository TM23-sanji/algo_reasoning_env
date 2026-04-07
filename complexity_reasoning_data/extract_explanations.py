#!/usr/bin/env python3
"""
Extract solution explanations from LeetCode solution README files.

Output format per problem:
{
    "problem_id": 15,
    "explanations": {
        "1": "We sort the array first...",
        "2": "..."
    },
    "is_english": true,
    "time_complexity": "O(n^2)",
    "space_complexity": "O(log n)"
}

Strategy:
1. Read English README_EN.md first
2. If a solution has no explanation text, fall back to Chinese README.md
3. Extract time/space complexity from explanation text via regex
"""

import os
import re
import json
from pathlib import Path

SOLUTION_BASE = "/teamspace/studios/this_studio/dataset/leetcode/solution"


def parse_explanation_from_block(block):
    """Extract explanation text from a single solution block.

    The block starts right after the '### Solution N' header line.
    Explanation is everything before the first code block marker:
    - #### (language header like #### Python3)
    - <!-- tabs:start -->
    """
    lines = block.strip().split("\n")
    explanation_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("####") or stripped.startswith("<!-- tabs:start -->"):
            break
        explanation_lines.append(line)

    explanation = "\n".join(explanation_lines).strip()
    # Clean up HTML comments that sometimes leak in
    explanation = re.sub(r"<!--.*?-->", "", explanation, flags=re.DOTALL).strip()
    return explanation


def _extract_first_o_expr(text):
    r"""Find the first O(...) with balanced parentheses.

    Returns (inner_content, end_pos) or None.
    Validates that the character after the closing paren is an
    expected end-of-match delimiter (e.g. $, comma, period, backtick,
    whitespace, or end of string).
    """
    i = 0
    while i < len(text) - 2:
        if text[i] == "O" and text[i + 1] == "(":
            depth = 1
            j = i + 2
            while j < len(text) and depth > 0:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            if depth == 0:
                after = text[j] if j < len(text) else ""
                if after == "" or after in "$`,); \t\n，。；":
                    return text[i + 2 : j - 1], j
            i += 2
        else:
            i += 1
    return None


def _extract_complexity_after_keyword(text, kw_regex):
    """Find the first O(...) after a complexity keyword match.

    Returns 'O(inner)' or None.
    """
    for m in re.finditer(kw_regex, text):
        result = _extract_first_o_expr(text[m.end() :])
        if result:
            inner = result[0]
            if inner:
                return f"O({inner})"
    return None


def extract_complexities(text):
    r"""Extract time and space complexity from explanation text.

    Handles variations like:
    - 'The time complexity is $O(n^2)$, and the space complexity is $O(log n)$'
    - 'Time complexity is $O(n^2)$, and space complexity is $O(log n)$'
    - 'Time complexity $O(n)$, space complexity $O(n)$'  (no "is")
    - '时间复杂度 $O(n^2)$，空间复杂度 $O(log n)$'
    - '时间复杂度：$O(logN)$'  (with colon)
    - '时间复杂度为 $O(1)'  (with 为)
    - '时间复杂度 O(logn)'  (no $ delimiters)
    - 'O(\max (m, n))'  (nested parens)
    - 'O((n-m) \times m)'  (double parens)
    """
    # Keyword patterns that precede O-notation
    time_kw = r"(?:[Tt]ime\s+[Cc]omplexity\s+(?:is\s*)?|(?:时间复杂度)[：:]?\s*(?:为\s*)?)\s*(?:\\?\$)?\s*"
    space_kw = r"(?:[Ss]pace\s+[Cc]omplexity\s+(?:is\s*)?|(?:空间复杂度)[：:]?\s*(?:为\s*)?)\s*(?:\\?\$)?\s*"

    return {
        "time_complexity": _extract_complexity_after_keyword(text, time_kw),
        "space_complexity": _extract_complexity_after_keyword(text, space_kw),
    }


def parse_solutions_section(content, is_english):
    """Parse the ## Solutions section and extract all solution explanations.

    Returns:
        explanations: dict mapping solution number -> explanation text
        first_time: first non-null time complexity found
        first_space: first non-null space complexity found
        all_from_primary: whether all explanations came from the primary language
    """
    # Find the Solutions section header
    if is_english:
        match = re.search(r"## Solutions\s*\n", content)
    else:
        match = re.search(r"## 解法\s*\n", content)

    if not match:
        return {}, None, None, True

    solutions_section = content[match.end() :]

    # Split into solution blocks on ### Solution N or ### 方法N
    if is_english:
        parts = re.split(r"(?=### Solution \d+)", solutions_section)
    else:
        parts = re.split(r"(?=### 方法[一二三四五六七八九十\d]+)", solutions_section)

    explanations = {}
    first_time = None
    first_space = None
    all_have_text = True

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Extract solution number
        if is_english:
            num_match = re.match(r"### Solution (\d+)", part)
            if not num_match:
                continue
            sol_num = num_match.group(1)
        else:
            num_match = re.match(r"### 方法([一二三四五六七八九十\d]+)", part)
            if not num_match:
                continue
            raw_num = num_match.group(1)
            # Convert Chinese numerals to digits
            cn_map = {
                "一": "1",
                "二": "2",
                "三": "3",
                "四": "4",
                "五": "5",
                "六": "6",
                "七": "7",
                "八": "8",
                "九": "9",
                "十": "10",
            }
            sol_num = cn_map.get(raw_num, raw_num)

        # Get the block after the header line
        header_end = part.find("\n")
        if header_end == -1:
            explanations[sol_num] = ""
            all_have_text = False
            continue

        block = part[header_end + 1 :]
        explanation = parse_explanation_from_block(block)

        if not explanation:
            all_have_text = False

        explanations[sol_num] = explanation

        # Extract complexities from the first solution that has them
        if explanation and (first_time is None or first_space is None):
            complexities = extract_complexities(explanation)
            if complexities["time_complexity"] and first_time is None:
                first_time = complexities["time_complexity"]
            if complexities["space_complexity"] and first_space is None:
                first_space = complexities["space_complexity"]

    return explanations, first_time, first_space, all_have_text


def process_problem(problem_dir):
    """Process a single problem directory and return the explanation entry."""
    folder_name = os.path.basename(problem_dir)

    # Extract problem_id from folder name (e.g., "0015.3Sum" -> 15)
    id_match = re.match(r"(\d+)\.", folder_name)
    if not id_match:
        return None
    problem_id = int(id_match.group(1))

    en_path = os.path.join(problem_dir, "README_EN.md")
    zh_path = os.path.join(problem_dir, "README.md")

    # Read English README
    en_content = ""
    if os.path.exists(en_path):
        with open(en_path, "r", encoding="utf-8") as f:
            en_content = f.read()

    if not en_content:
        return None

    # Parse English solutions
    en_explanations, time_c, space_c, all_en = parse_solutions_section(
        en_content, is_english=True
    )

    # is_english is True if we got any English explanation text
    # Only False if all solutions had to fall back to Chinese
    is_english = (
        any(v.strip() for v in en_explanations.values()) if en_explanations else False
    )

    # If any solution has empty explanation, try Chinese fallback
    if not all_en and os.path.exists(zh_path):
        with open(zh_path, "r", encoding="utf-8") as f:
            zh_content = f.read()

        zh_explanations, zh_time, zh_space, _ = parse_solutions_section(
            zh_content, is_english=False
        )

        # Merge: use Chinese where English is empty
        for sol_num in en_explanations:
            if not en_explanations[sol_num].strip() and sol_num in zh_explanations:
                en_explanations[sol_num] = zh_explanations[sol_num]

        # Fill complexity from Chinese if still missing
        if time_c is None and zh_time:
            time_c = zh_time
        if space_c is None and zh_space:
            space_c = zh_space

    # Handle case where English had no solutions section at all
    if not en_explanations and os.path.exists(zh_path):
        with open(zh_path, "r", encoding="utf-8") as f:
            zh_content = f.read()
        en_explanations, time_c, space_c, _ = parse_solutions_section(
            zh_content, is_english=False
        )
        is_english = False

    # Skip if we found absolutely nothing
    if not en_explanations:
        return None

    return {
        "problem_id": problem_id,
        "explanations": en_explanations,
        "is_english": is_english,
        "time_complexity": time_c,
        "space_complexity": space_c,
    }


def main():
    """Main entry point: iterate all range directories and process every problem."""
    all_entries = []

    # Get all range directories sorted
    range_dirs = sorted(
        [
            d
            for d in os.listdir(SOLUTION_BASE)
            if os.path.isdir(os.path.join(SOLUTION_BASE, d))
            and re.match(r"\d{4}-\d{4}$", d)
        ]
    )

    print(f"Found {len(range_dirs)} range directories")

    for range_dir in range_dirs:
        range_path = os.path.join(SOLUTION_BASE, range_dir)

        problem_dirs = sorted(
            [
                os.path.join(range_path, d)
                for d in os.listdir(range_path)
                if os.path.isdir(os.path.join(range_path, d))
            ]
        )

        print(f"Processing {range_dir}: {len(problem_dirs)} problems")

        for problem_dir in problem_dirs:
            entry = process_problem(problem_dir)
            if entry:
                all_entries.append(entry)

    # Sort by problem_id
    all_entries.sort(key=lambda x: x["problem_id"])

    output_path = "/teamspace/studios/this_studio/dataset/explanations.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Wrote {len(all_entries)} entries to {output_path}")

    # Summary stats
    en_count = sum(1 for e in all_entries if e["is_english"])
    zh_count = len(all_entries) - en_count
    total_solutions = sum(len(e["explanations"]) for e in all_entries)
    empty_count = sum(
        1 for e in all_entries for sol in e["explanations"].values() if not sol.strip()
    )

    print(f"  English explanations: {en_count}")
    print(f"  Chinese fallback:     {zh_count}")
    print(f"  Total solutions:      {total_solutions}")
    print(f"  Empty explanations:   {empty_count}")


if __name__ == "__main__":
    main()
