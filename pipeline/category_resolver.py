"""
Phase 1: Category Resolution System

Maps LeetCode problem tags to Rust data structure categories.
Resolves a list of tags to a set of categories for boilerplate injection.

Categories:
- "pure" — algorithms without specific DS (Array, String, Math, etc.)
- "linked_list" — linked list operations
- "tree" — binary tree operations
- "graph" — graph algorithms
- "heap" — heap/priority queue operations
- "custom" — unknown/uncategorized (default fallback)
"""

from typing import Set, List


# Comprehensive mapping of LeetCode tags to Rust categories
TAG_TO_CATEGORY = {
    # Pure algorithms (no specific DS)
    "Array": "pure",
    "Hash Table": "pure",
    "String": "pure",
    "Math": "pure",
    "Greedy": "pure",
    "Sorting": "pure",
    "Two Pointers": "pure",
    "Sliding Window": "pure",
    "Dynamic Programming": "pure",
    "Bit Manipulation": "pure",
    "Stack": "pure",
    "Queue": "pure",
    "Trie": "pure",
    "Binary Search": "pure",
    "Prefix Sum": "pure",
    "Simulation": "pure",
    "Counting": "pure",
    "Enumeration": "pure",
    "Backtracking": "pure",
    "Memoization": "pure",
    "Number Theory": "pure",
    "Bitmask": "pure",
    "Combinatorics": "pure",
    "Game Theory": "pure",
    "Brainteaser": "pure",
    "Probability and Statistics": "pure",
    "Divide and Conquer": "pure",
    "Recursion": "pure",
    "String Matching": "pure",
    "Hash Function": "pure",
    "Counting Sort": "pure",
    "Merge Sort": "pure",
    "Quickselect": "pure",
    "Bucket Sort": "pure",
    "Suffix Array": "pure",
    "Radix Sort": "pure",
    "Rolling Hash": "pure",
    "Interactive": "pure",
    "Line Sweep": "pure",
    # Linked List
    "Linked List": "linked_list",
    # Tree
    "Tree": "tree",
    "Binary Tree": "tree",
    "Binary Search Tree": "tree",
    "Segment Tree": "tree",
    "Binary Indexed Tree": "tree",
    # Graph
    "Graph": "graph",
    "Topological Sort": "graph",
    "Breadth-First Search": "graph",
    "Depth-First Search": "graph",
    "Union Find": "graph",
    "Monotonic Stack": "graph",
    "Monotonic Queue": "graph",
    "Ordered Set": "graph",
    "Shortest Path": "graph",
    "Minimum Spanning Tree": "graph",
    "Eulerian Circuit": "graph",
    "Strongly Connected Component": "graph",
    "Biconnected Component": "graph",
    # Heap
    "Heap (Priority Queue)": "heap",
    # Matrix (2D Array, treat as pure but could be specialized)
    "Matrix": "pure",
    # Geometry
    "Geometry": "pure",
}


def resolve_category(tags: List[str]) -> Set[str]:
    """
    Resolve a list of problem tags to a set of Rust categories.

    Args:
        tags: List of LeetCode problem tags (e.g., ["Array", "Hash Table"])

    Returns:
        Set of categories (e.g., {"pure"})

    Examples:
        resolve_category(["Array", "Hash Table"]) → {"pure"}
        resolve_category(["Linked List", "Recursion"]) → {"linked_list", "pure"}
        resolve_category(["Tree", "DFS"]) → {"tree", "graph"}
        resolve_category(["Unknown Tag"]) → {"custom"}
    """
    categories = set()

    for tag in tags:
        if tag in TAG_TO_CATEGORY:
            categories.add(TAG_TO_CATEGORY[tag])
        else:
            # Unknown tags default to "custom"
            categories.add("custom")

    return categories


HELPER_CONTEXT = {
    "pure": "",
    "linked_list": """The following helper functions are already defined and available:
- `list_from_vec(vals: Vec<i32>) -> Option<Box<ListNode>>` — builds a linked list from a vec
- `list_to_vec(node: Option<Box<ListNode>>) -> Vec<i32>` — converts linked list back to vec

Use these in your assert_eq! calls. Do NOT call Solution methods directly with ListNode — 
always convert via these helpers. Return values that are linked lists must also be 
compared via list_to_vec.""",
    "tree": """The following helper functions are already defined:
- `tree_from_vec(vals: Vec<Option<i32>>) -> Option<Rc<RefCell<TreeNode>>>` — level-order build
- `tree_to_vec(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Option<i32>>` — level-order serialize

Use these for all TreeNode inputs and outputs in assert_eq! calls.""",
    "graph": """The following helper function is already defined:
- `build_adj_list(n: usize, edges: Vec<Vec<i32>>) -> Vec<Vec<usize>>` — builds undirected adjacency list
- `build_directed_adj_list(n: usize, edges: Vec<Vec<i32>>) -> Vec<Vec<usize>>` — builds directed adjacency list

For problems that take adjacency matrix (Vec<Vec<i32>>), pass directly without helper.""",
    "heap": "",
    "custom": "",
}


def get_helper_context(categories: Set[str]) -> str:
    """
    Merge HELPER_CONTEXT strings for all provided categories.

    Args:
        categories: Set of categories (e.g., {"linked_list", "tree"})

    Returns:
        Combined helper context string with non-empty entries

    Examples:
        get_helper_context({"pure"}) → ""
        get_helper_context({"linked_list"}) → "The following helper functions..."
        get_helper_context({"linked_list", "tree"}) → "...linked list helpers...\n\n...tree helpers..."
    """
    contexts = []

    # Sort for consistent output
    for category in sorted(categories):
        if category in HELPER_CONTEXT:
            ctx = HELPER_CONTEXT[category].strip()
            if ctx:  # Only add non-empty contexts
                contexts.append(ctx)

    if not contexts:
        return ""

    # Join with double newline separator for clarity
    return "\n\n".join(contexts)


def print_category_stats():
    """Debug utility: print statistics about TAG_TO_CATEGORY mapping"""
    print(f"Total tags mapped: {len(TAG_TO_CATEGORY)}")

    categories_count = {}
    for tag, category in TAG_TO_CATEGORY.items():
        categories_count[category] = categories_count.get(category, 0) + 1

    print("\nBreakdown by category:")
    for category in sorted(categories_count.keys()):
        print(f"  {category}: {categories_count[category]} tags")

    print("\nTags by category:")
    for category in sorted(categories_count.keys()):
        tags = [t for t, c in TAG_TO_CATEGORY.items() if c == category]
        print(f"  {category}:")
        for tag in sorted(tags):
            print(f"    - {tag}")


if __name__ == "__main__":
    # Test with example cases
    print("=" * 60)
    print("CATEGORY RESOLVER TEST")
    print("=" * 60)

    test_cases = [
        ["Array", "Hash Table"],
        ["Linked List", "Recursion"],
        ["Tree", "Depth-First Search"],
        ["Linked List", "Tree", "Recursion"],
        ["Graph", "Topological Sort"],
        ["Heap (Priority Queue)"],
        ["Unknown Tag"],
        ["Array", "Unknown Tag"],
    ]

    print("\nTest cases:")
    for tags in test_cases:
        categories = resolve_category(tags)
        context = get_helper_context(categories)
        print(f"\nTags: {tags}")
        print(f"Categories: {categories}")
        if context:
            print(
                f"Helper context: {context[:80]}..."
                if len(context) > 80
                else f"Helper context: {context}"
            )
        else:
            print("Helper context: (empty)")

    print("\n" + "=" * 60)
    print_category_stats()
