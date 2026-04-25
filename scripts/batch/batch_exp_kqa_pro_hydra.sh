#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/hydra_batch_common.sh"


usage() {
    cat <<EOF
Usage: bash scripts/batch/batch_exp_kqa_pro_hydra.sh [options]

Run the KQA Pro paper batch for both SH and FH planners.

$(print_common_options)
EOF
}


append_worker_provider_overrides() {
    local -n cmd_ref="$1"
    cmd_ref+=(
        "environment.agents.kopl_schema_free_agents.llm_provider=${WORKER_PROVIDER}"
        "environment.agents.kopl_find_and_filter_concept_agents.llm_provider=${WORKER_PROVIDER}"
        "environment.agents.kopl_key_only_agents.llm_provider=${WORKER_PROVIDER}"
        "environment.agents.kopl_key_and_value_agents.llm_provider=${WORKER_PROVIDER}"
    )
}


main() {
    parse_common_args "$@"
    resolve_llm_config
    prepare_gpu_devices
    print_run_header "KQA Pro"

    local strict_suffix=""
    if [[ "$STRICT" == true ]]; then
        strict_suffix=".strict"
    fi

    local -a run_specs=(
        "kopl_kbqa/sh.v1${strict_suffix}|kopl_kbqa/kqa_pro|${DEFAULT_SPLIT}"
        "kopl_kbqa/fh.v1${strict_suffix}|kopl_kbqa/kqa_pro|${DEFAULT_SPLIT}"
    )

    run_specs_in_batches run_specs
}


main "$@"