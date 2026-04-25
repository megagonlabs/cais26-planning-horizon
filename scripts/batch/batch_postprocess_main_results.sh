#!/usr/bin/env bash

set -euo pipefail

DEFAULT_SPLIT="test"
DEFAULT_OUTPUT_DIR="tables/reproduction"
SKIP_EVAL=false

declare -a DATASETS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"


usage() {
    cat <<EOF
Usage: bash scripts/batch/batch_postprocess_main_results.sh [options] [dataset ...]

Run the evaluator-facing post-processing workflow for the main paper tables.

This wrapper optionally runs LLM-based soft evaluation over the selected result
directories and then aggregates metrics into one CSV per dataset.

Options:
  --split NAME         Result split to process.
                       Default: ${DEFAULT_SPLIT}
  --output-dir DIR     Directory for aggregated CSV files.
                       Default: ${DEFAULT_OUTPUT_DIR}
  --skip-eval          Skip the soft-evaluation pass and only collect metrics.
                       Useful when the results were already evaluated.
  -h, --help           Show this message.

Datasets:
  If no dataset arguments are given, the script processes the main paper set:
    - kopl_kbqa/kqa_pro
    - atomic_kbqa/grailqa
    - atomic_kbqa/webqsp
    - atomic_kbqa/graphq
    - multiobj/hotpotqa

Examples:
  bash scripts/batch/batch_postprocess_main_results.sh
  bash scripts/batch/batch_postprocess_main_results.sh --skip-eval kopl_kbqa/kqa_pro
EOF
}


set_default_datasets() {
    DATASETS=(
        "kopl_kbqa/kqa_pro"
        "atomic_kbqa/grailqa"
        "atomic_kbqa/webqsp"
        "atomic_kbqa/graphq"
        "multiobj/hotpotqa"
    )
}


parse_args() {
    SPLIT="${DEFAULT_SPLIT}"
    OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"

    while (($# > 0)); do
        case "$1" in
            --split)
                if (($# < 2)); then
                    echo "Error: --split requires a value." >&2
                    exit 1
                fi
                SPLIT="$2"
                shift 2
                ;;
            --output-dir)
                if (($# < 2)); then
                    echo "Error: --output-dir requires a value." >&2
                    exit 1
                fi
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --skip-eval)
                SKIP_EVAL=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            --)
                shift
                while (($# > 0)); do
                    DATASETS+=("$1")
                    shift
                done
                ;;
            -*)
                echo "Error: unknown option '$1'." >&2
                usage >&2
                exit 1
                ;;
            *)
                DATASETS+=("$1")
                shift
                ;;
        esac
    done

    if ((${#DATASETS[@]} == 0)); then
        set_default_datasets
    fi
}


dataset_output_path() {
    local dataset="$1"
    local safe_name="${dataset//\//__}"
    echo "${OUTPUT_DIR}/${safe_name}.${SPLIT}.csv"
}


process_dataset() {
    local dataset="$1"
    local results_dir="results/${dataset}/${SPLIT}"
    local output_path
    output_path="$(dataset_output_path "$dataset")"

    if [[ ! -d "$results_dir" ]]; then
        echo "Error: results directory '$results_dir' does not exist." >&2
        exit 1
    fi

    echo "== ${dataset} =="

    if [[ "$SKIP_EVAL" == false ]]; then
        bash scripts/evaluate_results.sh "$results_dir"
    else
        echo "Skipping soft evaluation (--skip-eval)."
    fi

    uv run python scripts/collect_metrics.py "$dataset" --split "$SPLIT" -o "$output_path"
    echo "Saved aggregated CSV to ${output_path}"
    echo
}


main() {
    cd "$REPO_ROOT"
    parse_args "$@"

    mkdir -p "$OUTPUT_DIR"

    echo "== Main results post-processing =="
    echo "Split: ${SPLIT}"
    echo "Output directory: ${OUTPUT_DIR}"
    if [[ "$SKIP_EVAL" == true ]]; then
        echo "Mode: collect metrics only"
    else
        echo "Mode: soft evaluation + metric collection"
    fi
    printf 'Datasets:\n'
    printf '  - %s\n' "${DATASETS[@]}"
    echo

    local dataset
    for dataset in "${DATASETS[@]}"; do
        process_dataset "$dataset"
    done
}


main "$@"
