#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/hydra_batch_common.sh"


usage() {
    cat <<EOF
Usage: bash scripts/batch/batch_exp_multiobj_hotpotqa_hydra.sh [options]

Run the multi-objective HotpotQA paper batch for both SH and FH planners.

$(print_common_options)
EOF
}


append_worker_provider_overrides() {
    local -n cmd_ref="$1"
    cmd_ref+=(
        "environment.agents.reasoning.llm_provider=${WORKER_PROVIDER}"
        "environment.agents.search.llm_provider=${WORKER_PROVIDER}"
    )
}


main() {
    parse_common_args "$@"
    resolve_llm_config
    prepare_gpu_devices
    print_run_header "Multi-objective HotpotQA"

    local strict_suffix=""
    if [[ "$STRICT" == true ]]; then
        strict_suffix=".strict"
    fi

    local -a run_specs=(
        "multiobj/hotpotqa/sh.v1${strict_suffix}|multiobj/hotpotqa|${DEFAULT_SPLIT}"
        "multiobj/hotpotqa/fh.v1${strict_suffix}|multiobj/hotpotqa|${DEFAULT_SPLIT}"
    )

    run_specs_in_batches run_specs
}


main "$@"
