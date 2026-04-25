import json
import re
from pathlib import Path

def extract_relations_from_function_list(function_list):
    """Extract relation-like strings wrapped by ' from function_list."""
    relations = set()
    for step in function_list:
        # Find all strings wrapped in single quotes
        matches = re.findall(r"'(.*?)'", step)
        for match in matches:
            # Filter for relation-like strings
            # 1. Must contain at least 2 dots (e.g., people.person.nationality)
            # 2. Must not be all uppercase (exclude ARGMAX, etc.)
            # 3. Must not start with m., g., or base. followed by just numbers/letters (entity IDs)
            if match.count(".") >= 2 and not match.isupper():
                if "^^http" in match:
                    continue  # Exclude datatype URIs
                if match.startswith("(R "):
                    match = match[3:-1].strip()  # Remove (R )
                # Further check to exclude entity IDs that might have 2 dots (rare but possible)
                if not re.match(r"^[mg]\.[0-9a-z_]+$", match):
                    relations.add(match)
    return relations

def main():
    relation_list_path = Path("data/atomic_kbqa/freebase/relation_list.txt")
    test_files = [
        "data/atomic_kbqa/grailqa/processed/grailqa_test.v1.json",
        "data/atomic_kbqa/webqsp/processed/webqsp_test.v1.json",
        "data/atomic_kbqa/graphq/processed/graphq_test.v1.json",
        "data/atomic_kbqa/grailqa/processed/grailqa_train.v1.json",
        "data/atomic_kbqa/webqsp/processed/webqsp_train.v1.json",
        "data/atomic_kbqa/graphq/processed/graphq_train.v1.json",
    ]

    # Load existing relations
    if relation_list_path.exists():
        with open(relation_list_path, "r") as f:
            existing_relations = set(line.strip() for line in f if line.strip())
    else:
        existing_relations = set()

    print(f"Loaded {len(existing_relations)} existing relations.")

    all_test_relations = set()
    for test_file in test_files:
        path = Path(test_file)
        if not path.exists():
            print(f"Warning: Test file not found: {test_file}")
            continue

        print(f"Processing {test_file}...")
        with open(path, "r") as f:
            data = json.load(f)
            for example in data:
                if "function_list" in example:
                    all_test_relations.update(extract_relations_from_function_list(example["function_list"]))

    print(f"Extracted {len(all_test_relations)} unique relations from test files.")

    missing_relations = all_test_relations - existing_relations
    if missing_relations:
        print(f"Found {len(missing_relations)} missing relations. Adding them to the list...")
        for rel in sorted(missing_relations):
            print(f"  - {rel}")

        updated_relations = existing_relations | missing_relations
        with open(relation_list_path, "w") as f:
            for rel in sorted(updated_relations):
                f.write(f"{rel}\n")
        print(f"Successfully updated {relation_list_path}. Total relations: {len(updated_relations)}")
    else:
        print("No missing relations found. All test relations are already in the list.")

if __name__ == "__main__":
    main()
