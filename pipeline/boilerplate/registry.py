"""
Boilerplate Registry: Load and merge category-specific Rust helpers

This module provides functionality to:
1. Load .rs boilerplate files by category
2. Merge multiple boilerplate blocks for multi-category problems
3. Cache loaded files for performance
"""

import os
from typing import Set, Dict, Optional


class BoilerplateRegistry:
    """Load and cache Rust boilerplate code by category"""

    def __init__(self, boilerplate_dir: Optional[str] = None):
        """
        Initialize registry with boilerplate directory path.

        Args:
            boilerplate_dir: Path to boilerplate/ directory.
                           If None, uses pipeline/boilerplate/ relative to this file.
        """
        if boilerplate_dir is None:
            boilerplate_dir = os.path.dirname(os.path.abspath(__file__))

        self.boilerplate_dir = boilerplate_dir
        self._cache: Dict[str, str] = {}  # Cache loaded files

        # Verify directory exists
        if not os.path.isdir(self.boilerplate_dir):
            raise ValueError(f"Boilerplate directory not found: {self.boilerplate_dir}")

    def load_boilerplate(self, category: str) -> str:
        """
        Load a single boilerplate file for a category.

        Args:
            category: Category name (e.g., "linked_list", "tree", "graph")

        Returns:
            Content of the .rs file, or empty string if file doesn't exist
        """
        # Check cache first
        if category in self._cache:
            return self._cache[category]

        file_path = os.path.join(self.boilerplate_dir, f"{category}.rs")

        if not os.path.isfile(file_path):
            print(
                f"Warning: Boilerplate file not found for category '{category}': {file_path}"
            )
            self._cache[category] = ""
            return ""

        try:
            with open(file_path, "r") as f:
                content = f.read().strip()
                self._cache[category] = content
                return content
        except IOError as e:
            print(f"Error reading boilerplate file {file_path}: {e}")
            self._cache[category] = ""
            return ""

    def merge_boilerplate(self, categories: Set[str]) -> str:
        """
        Load and merge boilerplate for multiple categories.

        Args:
            categories: Set of category names (e.g., {"linked_list", "tree", "pure"})

        Returns:
            Merged boilerplate code with categories in sorted order

        Examples:
            merge_boilerplate({"pure"}) → "" (empty)
            merge_boilerplate({"linked_list"}) → "ListNode code..."
            merge_boilerplate({"linked_list", "tree"}) → "ListNode code\n\nTreeNode code..."
        """
        boilerplate_blocks = []

        # Load in sorted order for consistency
        for category in sorted(categories):
            content = self.load_boilerplate(category)
            if content:  # Only add non-empty blocks
                boilerplate_blocks.append(content)

        # Join with double newline separator
        if not boilerplate_blocks:
            return ""

        return "\n\n".join(boilerplate_blocks)

    def get_all_categories(self) -> Set[str]:
        """
        Get all available categories (based on .rs files in boilerplate_dir).

        Returns:
            Set of category names (without .rs extension)
        """
        categories = set()

        if not os.path.isdir(self.boilerplate_dir):
            return categories

        for file in os.listdir(self.boilerplate_dir):
            if file.endswith(".rs"):
                category = file[:-3]  # Remove .rs extension
                categories.add(category)

        return categories

    def clear_cache(self):
        """Clear the boilerplate cache"""
        self._cache.clear()


# Global registry instance
_registry: Optional[BoilerplateRegistry] = None


def get_registry() -> BoilerplateRegistry:
    """Get or create the global registry instance"""
    global _registry
    if _registry is None:
        _registry = BoilerplateRegistry()
    return _registry


def load_boilerplate(categories: Set[str]) -> str:
    """
    Convenience function: load and merge boilerplate for categories.

    Args:
        categories: Set of category names

    Returns:
        Merged boilerplate code
    """
    return get_registry().merge_boilerplate(categories)


# ============================================================================
# TEST AND DEBUG
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("=" * 70)
    print("BOILERPLATE REGISTRY TEST")
    print("=" * 70)
    print()

    # Initialize registry
    registry = get_registry()
    print(f"Boilerplate directory: {registry.boilerplate_dir}")
    print()

    # Check available categories
    categories = registry.get_all_categories()
    print(f"Available categories: {sorted(categories)}")
    print()

    # Test loading individual boilerplate
    print("Individual category sizes:")
    for cat in sorted(categories):
        content = registry.load_boilerplate(cat)
        lines = len(content.split("\n")) if content else 0
        chars = len(content)
        print(f"  {cat:15s}: {chars:6d} chars ({lines:3d} lines)")
    print()

    # Test merging
    test_cases = [
        {"pure"},
        {"linked_list"},
        {"tree"},
        {"graph"},
        {"linked_list", "pure"},
        {"linked_list", "tree"},
        {"linked_list", "tree", "graph"},
    ]

    print("Merge test results:")
    for cats in test_cases:
        merged = registry.merge_boilerplate(cats)
        lines = len(merged.split("\n")) if merged else 0
        chars = len(merged)
        cat_str = ", ".join(sorted(cats))
        print(f"  {{{cat_str:30s}}}: {chars:6d} chars ({lines:3d} lines)")
    print()

    # Sample output
    print("Sample: Merged boilerplate for {linked_list, tree}:")
    print("-" * 70)
    merged = registry.merge_boilerplate({"linked_list", "tree"})
    lines = merged.split("\n")
    for i, line in enumerate(lines[:15]):
        print(f"{i + 1:3d}: {line}")
    if len(lines) > 15:
        print(f"... ({len(lines) - 15} more lines)")
    print()

    print("=" * 70)
    print("✅ Registry test complete!")
    print("=" * 70)
