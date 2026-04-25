"""Validate s-expression to SPARQL conversion for GrailQA, WebQSP, and GraphQ datasets.

This script tests whether `lisp_to_sparql` correctly converts s-expressions
to executable SPARQL queries by:
1. Loading preprocessed data (GrailQA, WebQSP, or GraphQ)
2. Converting sexpr to SPARQL using lisp_to_sparql
3. Executing against Freebase via ODBC
4. Comparing results with ground truth answers

Usage:
    # WebQSP
    uv run python data/atomic_kbqa/scripts/validate_webqsp_sparql.py \
        --input data/atomic_kbqa/webqsp/processed/webqsp_train.v1.json \
        --num-samples 100

    # GrailQA
    uv run python data/atomic_kbqa/scripts/validate_webqsp_sparql.py \
        --input data/atomic_kbqa/grailqa/processed/grailqa_train.v1.json \
        --num-samples 100

    # GraphQ
    uv run python data/atomic_kbqa/scripts/validate_webqsp_sparql.py \
        --input data/atomic_kbqa/graphq/processed/graphq_train.v1.json \
        --num-samples 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path to import local modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import project modules (after path setup)  # noqa: E402
from planning.tools.freebase.logic_form_utils import lisp_to_sparql  # noqa: E402
from planning.tools.freebase.sparql_executor import (  # noqa: E402
    execute_query_with_odbc,
    configure_odbc_connection,
)
from planning.tools.freebase.default_config import (  # noqa: E402
    FREEBASE_ODBC_PORT,
    VIRTODBC_DRIVER_PATH,
)


def load_webqsp_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load preprocessed WebQSP data.

    Args:
        file_path: Path to preprocessed JSON file

    Returns:
        List of WebQSP instances
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def validate_single_instance(instance: Dict[str, Any], dataset_name: str = "webqsp") -> Dict[str, Any]:
    """Validate a single instance.

    Args:
        instance: Instance with sexpr, sparql, and answer fields
        dataset_name: Dataset name ("grailqa", "webqsp", or "graphq")
                     Note: GraphQ uses same format as GrailQA (nested subquery)

    Returns:
        Validation result dictionary with status and details
    """
    result = {
        "id": instance["id"],
        "question": instance["question"],
        "status": "unknown",
        "error": None,
        "generated_sparql": None,
        "ground_truth_sparql": instance.get("sparql", None),
        "sexpr": instance.get("sexpr", None),
        "predicted_answers": [],
        "ground_truth_answers": instance.get("answer", []),
        "match": False,
    }

    # Check if sexpr exists
    if not instance.get("sexpr"):
        result["status"] = "no_sexpr"
        result["error"] = "No s-expression found in instance"
        return result

    # Convert sexpr to SPARQL
    try:
        sparql_query = lisp_to_sparql(instance["sexpr"], dataset_name=dataset_name)
        result["generated_sparql"] = sparql_query
    except Exception as e:
        result["status"] = "conversion_error"
        result["error"] = f"Failed to convert sexpr to SPARQL: {str(e)}"
        return result

    # Execute SPARQL query
    try:
        execute_results = execute_query_with_odbc(sparql_query)
        # Clean results (remove URI prefix)
        # Convert to string first to handle COUNT results (which return integers)
        predicted_answers = [
            str(res).replace("http://rdf.freebase.com/ns/", "") for res in execute_results
        ]
        result["predicted_answers"] = sorted(predicted_answers)
    except Exception as e:
        result["status"] = "execution_error"
        result["error"] = f"Failed to execute SPARQL query: {str(e)}"
        return result

    # Compare with ground truth
    ground_truth_answers = sorted(instance.get("answer", []))
    result["ground_truth_answers"] = ground_truth_answers

    if set(result["predicted_answers"]) == set(ground_truth_answers):
        result["status"] = "success"
        result["match"] = True
    else:
        result["status"] = "mismatch"
        result["match"] = False

    return result


def main(args):
    """Main validation function."""
    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Detect dataset from path
    path_str = str(input_path).lower()
    if "grailqa" in path_str:
        dataset_name = "grailqa"
    elif "graphq" in path_str:
        dataset_name = "graphq"
    else:
        dataset_name = "webqsp"

    print(f"Loading data from: {input_path}")
    print(f"Detected dataset: {dataset_name}")
    data = load_webqsp_data(input_path)
    print(f"Loaded {len(data)} instances")

    # Limit number of samples if specified
    if args.num_samples and args.num_samples < len(data):
        data = data[: args.num_samples]
        print(f"Processing first {args.num_samples} samples")

    # Configure ODBC connection
    print(
        f"Configuring Freebase ODBC connection (port: {FREEBASE_ODBC_PORT}, driver: {VIRTODBC_DRIVER_PATH})"
    )
    try:
        configure_odbc_connection()
    except Exception as e:
        print(f"Failed to initialize ODBC connection: {e}")
        print("Make sure Virtuoso is running and the driver path is correct")
        return

    # Validate instances
    results = []
    status_counts = {
        "success": 0,
        "mismatch": 0,
        "conversion_error": 0,
        "execution_error": 0,
        "no_sexpr": 0,
    }

    print(f"\nValidating {len(data)} instances...")
    for i, instance in enumerate(data):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(data)}")

        result = validate_single_instance(instance, dataset_name=dataset_name)
        results.append(result)
        status_counts[result["status"]] += 1

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total instances: {len(data)}")
    print(f"Success (exact match): {status_counts['success']}")
    print(f"Mismatch (wrong answers): {status_counts['mismatch']}")
    print(f"Conversion errors: {status_counts['conversion_error']}")
    print(f"Execution errors: {status_counts['execution_error']}")
    print(f"No s-expression: {status_counts['no_sexpr']}")
    print(
        f"\nAccuracy: {status_counts['success'] / len(data) * 100:.2f}% ({status_counts['success']}/{len(data)})"
    )

    # Show examples of failures if requested
    if args.show_failures:
        print("\n" + "=" * 80)
        print("FAILURE EXAMPLES")
        print("=" * 80)

        # Show conversion errors
        conversion_errors = [r for r in results if r["status"] == "conversion_error"]
        if conversion_errors:
            print(f"\nConversion Errors ({len(conversion_errors)}):")
            for r in conversion_errors[: args.max_examples]:
                print(f"\nID: {r['id']}")
                print(f"Question: {r['question']}")
                print(f"S-expr: {r['sexpr']}")
                print(f"Error: {r['error']}")

        # Show mismatches
        mismatches = [r for r in results if r["status"] == "mismatch"]
        if mismatches:
            print(f"\nAnswer Mismatches ({len(mismatches)}):")
            for r in mismatches[: args.max_examples]:
                print(f"\nID: {r['id']}")
                print(f"Question: {r['question']}")
                print(f"S-expr: {r['sexpr']}")
                print(f"Predicted: {r['predicted_answers']}")
                print(f"Ground Truth: {r['ground_truth_answers']}")

    # Save detailed results if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {"summary": status_counts, "results": results}, f, indent=2, ensure_ascii=False
            )
        print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate s-expression to SPARQL conversion (GrailQA/WebQSP/GraphQ)"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to preprocessed JSON file (GrailQA, WebQSP, or GraphQ)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to save detailed validation results (optional)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to validate (default: all)",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Show examples of failed validations",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum number of failure examples to show (default: 5)",
    )

    args = parser.parse_args()
    main(args)
