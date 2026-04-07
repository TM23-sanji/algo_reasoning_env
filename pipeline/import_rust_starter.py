import json
import re
import os


def parse_rust_starter(filepath):
    """Parse rust-starter.txt and extract problem_id and starter_code."""
    with open(filepath, "r") as f:
        content = f.read()

    # Split by #<number> pattern
    entries = []

    # Find all #<problem_id> blocks
    pattern = r"#(\d+)\s*\n(impl Solution \{[^}]+\})"
    matches = re.findall(pattern, content, re.DOTALL)

    for problem_id_str, impl_block in matches:
        problem_id = int(problem_id_str)

        # Extract function name from the impl block
        fn_match = re.search(r"pub fn (\w+)\s*\(", impl_block)
        if not fn_match:
            print(f"Warning: Could not find function name for problem {problem_id}")
            continue

        function_name = fn_match.group(1)

        # Add // OUR CODE GOES HERE placeholder
        # Replace empty function body with placeholder
        impl_block = re.sub(
            r"(pub fn \w+\([^)]*\)[^{]*\{\s*)\s*(\})",
            r"\1// OUR CODE GOES HERE\n\2",
            impl_block,
        )

        # Format starter_code (add newline at end if not present)
        starter_code = impl_block.strip() + "\n\n}"

        entries.append(
            {
                "problem_id": problem_id,
                "function_name": function_name,
                "starter_code": starter_code,
            }
        )

    return entries


def load_existing_problem_ids(starter_codes_path):
    """Load existing problem IDs from starter_codes.jsonl."""
    existing_ids = set()
    if os.path.exists(starter_codes_path):
        with open(starter_codes_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                existing_ids.add(entry["problem_id"])
    return existing_ids


def main():
    base_dir = "complexity_reasoning_data"
    rust_starter_path = os.path.join(base_dir, "rust-starter.txt")
    starter_codes_path = os.path.join(base_dir, "starter_codes.jsonl")

    print("Parsing rust-starter.txt...")
    entries = parse_rust_starter(rust_starter_path)
    print(f"  Found {len(entries)} entries in rust-starter.txt")

    print(f"Loading existing problem IDs from starter_codes.jsonl...")
    existing_ids = load_existing_problem_ids(starter_codes_path)
    print(f"  Existing problem IDs: {len(existing_ids)}")

    # Filter out duplicates
    new_entries = [e for e in entries if e["problem_id"] not in existing_ids]
    print(f"  New entries (not in existing): {len(new_entries)}")

    if not new_entries:
        print("No new entries to add!")
        return

    # Append to starter_codes.jsonl
    print(f"Appending {len(new_entries)} new entries to starter_codes.jsonl...")
    with open(starter_codes_path, "a") as f:
        for entry in new_entries:
            f.write(json.dumps(entry) + "\n")

    print("Done!")

    # Print summary
    print("\nNew entries added:")
    for entry in new_entries:
        print(f"  Problem {entry['problem_id']}: {entry['function_name']}")


if __name__ == "__main__":
    main()
