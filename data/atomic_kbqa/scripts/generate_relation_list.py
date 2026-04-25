"""
Generate complete relation_list.txt from Freebase KB.

This script queries the Freebase KB to identify all relations in the Freebase namespace.

Usage:
    uv run python data/atomic_kbqa/scripts/generate_relation_list.py

Output:
    data/atomic_kbqa/freebase/relation_list.txt
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from planning.tools.freebase.sparql_executor import (  # noqa: E402
    _get_thread_connection,
    configure_odbc_connection,
)


def get_all_relations() -> set[str]:
    """
    Query Freebase KB to find all relations in the Freebase namespace.

    Returns:
        Set of relation names
    """
    conn = _get_thread_connection()

    relations = set()

    # Query to find all relations in the Freebase namespace
    # We use a generous limit to ensure we capture all relevant relations
    # Note: We fetch ALL relations and filter in Python to avoid expensive regex in SPARQL
    query = """SPARQL
        SELECT DISTINCT ?relation WHERE {
            ?subject ?relation ?object .
        }
        LIMIT 2000000
    """

    print("Querying Freebase KB for all relations...")
    print("This may take several minutes...")

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

            for row in rows:
                raw_relation = row[0]
                # Filter for Freebase namespace
                if "http://rdf.freebase.com/ns/" in raw_relation:
                    # Remove URI prefix to get relation name
                    relation = raw_relation.replace("http://rdf.freebase.com/ns/", "")
                    relations.add(relation)

        print(f"Found {len(relations)} relations")

    except Exception as e:
        print(f"Query failed: {e}")
        return set()

    return relations


def save_relation_list(relations: set[str], output_path: Path) -> None:
    """
    Save relation list to file.

    Args:
        relations: Set of relation names
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for relation in sorted(relations):
            f.write(f"{relation}\n")

    print(f"Saved {len(relations)} relations to {output_path}")


def main():
    """Main execution."""
    # Configure ODBC connection with a longer timeout for this heavy query
    configure_odbc_connection(timeout=600)  # 10 minutes timeout

    # Get all relations from KB
    relations = get_all_relations()

    if not relations:
        print("No relations found!")
        sys.exit(1)

    # Save to file
    output_path = Path("data/atomic_kbqa/freebase/relation_list.txt")
    save_relation_list(relations, output_path)

    print("\nDone!")
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
