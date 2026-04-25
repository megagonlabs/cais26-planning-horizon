"""
Generate complete literal_relation_list.txt from Freebase KB.

This script queries the Freebase KB to identify all relations that have literal
(non-entity) values, such as numeric values, dates, strings, etc. These are
relations where the object is not an entity ID (doesn't start with 'm.').

Usage:
    uv run python data/atomic_kbqa/scripts/generate_literal_relation_list.py

Output:
    data/atomic_kbqa/freebase/literal_relation_list.txt
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from planning.tools.freebase.sparql_executor import (
    _get_thread_connection,
    configure_odbc_connection,
)


def get_all_literal_relations() -> set[str]:
    """
    Query Freebase KB to find all relations with literal (non-entity) values.

    Returns:
        Set of relation names that have literal values
    """
    conn = _get_thread_connection()

    literal_relations = set()

    # Query to find relations whose objects are literals (not entity URIs)
    # We identify literal relations by checking if objects are not Freebase entity URIs
    query = """SPARQL
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?relation WHERE {
            ?subject ?relation ?object .
            FILTER regex(?relation, "http://rdf.freebase.com/ns/")
            FILTER (isLiteral(?object) || (!isIRI(?object)))
        }
        LIMIT 100000
    """

    print("Querying Freebase KB for literal relations...")
    print("This may take several minutes...")

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

            for row in rows:
                # Remove URI prefix to get relation name
                relation = row[0].replace("http://rdf.freebase.com/ns/", "")
                literal_relations.add(relation)

        print(f"Found {len(literal_relations)} literal relations")

    except Exception as e:
        print(f"Query failed: {e}")
        return set()

    return literal_relations


def save_literal_relation_list(relations: set[str], output_path: Path) -> None:
    """
    Save literal relation list to file.

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
    # Configure ODBC connection (uses defaults from config)
    configure_odbc_connection()

    # Get literal relations from KB
    literal_relations = get_all_literal_relations()

    if not literal_relations:
        print("No literal relations found!")
        sys.exit(1)

    # Save to file
    output_path = Path("data/atomic_kbqa/freebase/literal_relation_list.txt")
    save_literal_relation_list(literal_relations, output_path)

    print("\nDone!")
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
