import re
from pathlib import Path


def extract_relations():
    base_path = Path("data/atomic_kbqa/freebase")
    fb_roles_path = base_path / "fb_roles"
    reverse_properties_path = base_path / "reverse_properties"

    ontology_relations = set()

    # Extract from fb_roles (2nd column)
    if fb_roles_path.exists():
        with open(fb_roles_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    rel = parts[1]
                    # Predicates usually have at least 2 dots
                    if rel.count(".") >= 2:
                        ontology_relations.add(rel)

    # Extract from reverse_properties (both columns)
    if reverse_properties_path.exists():
        with open(reverse_properties_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    parts = line.strip().split()

                if len(parts) == 2:
                    for rel in parts:
                        if rel.count(".") >= 2:
                            ontology_relations.add(rel)

    print(f"Extracted {len(ontology_relations)} unique relations from ontology files.")

    # Load current relation_list.txt
    relation_list_path = base_path / "relation_list.txt"
    current_relations = set()
    if relation_list_path.exists():
        with open(relation_list_path, "r") as f:
            for line in f:
                rel = line.strip()
                # Filter existing relations to remove junk (types, entity IDs)
                if rel.count(".") >= 2 and not rel.isupper():
                    if not re.match(r"^[mg]\.[0-9a-z_]+$", rel):
                        current_relations.add(rel)

    print(f"Current filtered relation_list.txt has {len(current_relations)} relations.")

    missing_in_current = ontology_relations - current_relations
    print(f"{len(missing_in_current)} relations in ontology are NOT in current relation_list.txt.")

    # Check if ontology covers all relations in test files
    # (I'll just print some stats)

    # Merge and save
    all_relations = current_relations | ontology_relations
    print(f"Total unique relations after merging: {len(all_relations)}")

    with open(relation_list_path, "w") as f:
        for rel in sorted(list(all_relations)):
            f.write(f"{rel}\n")

    print(f"Updated {relation_list_path}")


if __name__ == "__main__":
    extract_relations()
