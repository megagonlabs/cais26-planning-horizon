#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/hydra_batch_common.sh"


usage() {
    cat <<EOF
Usage: bash scripts/batch/batch_exp_atomic_kbqa_hydra.sh [options]

Run the Atomic KBQA paper batch for GrailQA, WebQSP, and GraphQ
with both SH and FH planners.

$(print_common_options)
EOF
}


append_worker_provider_overrides() {
    local -n cmd_ref="$1"
    cmd_ref+=(
        "environment.agents.atomic_kb_query_agents.llm_provider=${WORKER_PROVIDER}"
    )
}


main() {
    parse_common_args "$@"
    resolve_llm_config
    prepare_gpu_devices
    print_run_header "Atomic KBQA"

    local strict_suffix=""
    if [[ "$STRICT" == true ]]; then
        strict_suffix=".strict"
    fi

    local dataset
    local planner
    local -a run_specs=()

    for dataset in grailqa webqsp graphq; do
        for planner in sh fh; do
            run_specs+=(
                "atomic_kbqa/${planner}.v1${strict_suffix}|atomic_kbqa/${dataset}|${DEFAULT_SPLIT}"
            )
        done
    done

    run_specs_in_batches run_specs
}


main "$@"
