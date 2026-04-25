#!/usr/bin/env bash

set -euo pipefail

DEFAULT_LLM="gpt-4.1-mini"
DEFAULT_PARALLELISM=1
DEFAULT_WORKERS=10
DEFAULT_SPLIT="test"

STRICT=false
LLM_ID="$DEFAULT_LLM"
NUM_EPISODES=""
PARALLELISM="$DEFAULT_PARALLELISM"
WORKERS="$DEFAULT_WORKERS"

MODEL_CONFIG=""
META_PROVIDER=""
WORKER_PROVIDER=""

declare -a GPU_DEVICES=()


require_positive_integer() {
    local option_name="$1"
    local value="$2"

    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: ${option_name} must be a positive integer." >&2
        exit 1
    fi
}


print_common_options() {
    cat <<EOF
Common options:
  --llm NAME           Model alias for both meta and worker agents.
                       Default: ${DEFAULT_LLM}
  --strict             Run the strict config variant.
  --num-episodes N     Limit episodes for a sanity check.
                       Use "full" (or omit) to run the full split.
  --parallelism N      Number of experiment processes to run at once.
                       Default: ${DEFAULT_PARALLELISM}
                       When N > 1, set CUDA_VISIBLE_DEVICES to a comma-
                       separated GPU list with at least N entries.
  --workers N          Hydra worker count per experiment.
                       Default: ${DEFAULT_WORKERS}
  -h, --help           Show this message.
EOF
}


parse_common_args() {
    while (($# > 0)); do
        case "$1" in
            --llm)
                if (($# < 2)); then
                    echo "Error: --llm requires a value." >&2
                    exit 1
                fi
                LLM_ID="$2"
                shift 2
                ;;
            --strict)
                STRICT=true
                shift
                ;;
            --num-episodes)
                if (($# < 2)); then
                    echo "Error: --num-episodes requires a value." >&2
                    exit 1
                fi
                if [[ "$2" == "full" ]]; then
                    NUM_EPISODES=""
                else
                    require_positive_integer "--num-episodes" "$2"
                    NUM_EPISODES="$2"
                fi
                shift 2
                ;;
            --parallelism)
                if (($# < 2)); then
                    echo "Error: --parallelism requires a value." >&2
                    exit 1
                fi
                require_positive_integer "--parallelism" "$2"
                PARALLELISM="$2"
                shift 2
                ;;
            --workers)
                if (($# < 2)); then
                    echo "Error: --workers requires a value." >&2
                    exit 1
                fi
                require_positive_integer "--workers" "$2"
                WORKERS="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Error: unknown option '$1'." >&2
                usage >&2
                exit 1
                ;;
        esac
    done
}


resolve_llm_config() {
    case "$LLM_ID" in
        gpt-4.1-mini)
            MODEL_CONFIG="gpt-4p1-mini.v1"
            META_PROVIDER="openai"
            WORKER_PROVIDER="openai"
            ;;
        gpt-5-mini)
            MODEL_CONFIG="gpt-5-mini.v1"
            META_PROVIDER="openai"
            WORKER_PROVIDER="openai"
            ;;
        qwen3-235b-instruct)
            MODEL_CONFIG="qwen3-235b-instruct"
            META_PROVIDER="fireworks"
            WORKER_PROVIDER="fireworks"
            ;;
        gemini-flash-preview)
            MODEL_CONFIG="gemini-flash-preview.v1"
            META_PROVIDER="vertexai-openai"
            WORKER_PROVIDER="vertexai-openai"
            ;;
        *)
            echo "Error: unsupported --llm value '$LLM_ID'." >&2
            echo "Supported values: gpt-4.1-mini, gpt-5-mini, qwen3-235b-instruct, gemini-flash-preview" >&2
            exit 1
            ;;
    esac
}


prepare_gpu_devices() {
    if ((PARALLELISM == 1)); then
        return
    fi

    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "Error: --parallelism > 1 requires CUDA_VISIBLE_DEVICES to be set." >&2
        exit 1
    fi

    IFS=',' read -r -a GPU_DEVICES <<<"${CUDA_VISIBLE_DEVICES}"

    local i
    for i in "${!GPU_DEVICES[@]}"; do
        GPU_DEVICES[$i]="${GPU_DEVICES[$i]//[[:space:]]/}"
    done

    if ((${#GPU_DEVICES[@]} < PARALLELISM)); then
        echo "Error: CUDA_VISIBLE_DEVICES provides ${#GPU_DEVICES[@]} device(s), but --parallelism=${PARALLELISM}." >&2
        exit 1
    fi
}


print_run_header() {
    local dataset_label="$1"
    local strict_label="non-strict"

    if [[ "$STRICT" == true ]]; then
        strict_label="strict"
    fi

    echo "== ${dataset_label} Hydra batch run =="
    echo "LLM: ${LLM_ID}"
    echo "Strictness: ${strict_label}"
    if [[ -n "$NUM_EPISODES" ]]; then
        echo "Episodes: ${NUM_EPISODES}"
    else
        echo "Episodes: full split"
    fi
    echo "Parallelism: ${PARALLELISM}"
    echo "Workers per experiment: ${WORKERS}"
    if ((PARALLELISM > 1)); then
        echo "GPU devices: ${GPU_DEVICES[*]}"
    fi
    echo
}


run_experiment() {
    local experiment_name="$1"
    local dataset_id="$2"
    local split="$3"
    local cuda_device="$4"
    local -a cmd=(
        uv run python scripts/run.py
        "experiment=${experiment_name}"
        "model=${MODEL_CONFIG}"
        "worker_model=${MODEL_CONFIG}"
        "meta_agent.llm_provider=${META_PROVIDER}"
        "workers=${WORKERS}"
        "experiment.dataset_id=${dataset_id}"
        "experiment.split=${split}"
    )

    if [[ -n "$NUM_EPISODES" ]]; then
        cmd+=("num_episodes=${NUM_EPISODES}")
    fi

    append_worker_provider_overrides cmd

    if [[ -n "$cuda_device" ]]; then
        echo "[GPU ${cuda_device}] ${experiment_name} on ${dataset_id}"
        CUDA_VISIBLE_DEVICES="$cuda_device" "${cmd[@]}"
    else
        echo "${experiment_name} on ${dataset_id}"
        "${cmd[@]}"
    fi
}


run_specs_in_batches() {
    local -n specs_ref="$1"
    local total_specs="${#specs_ref[@]}"
    local next_index=0

    while ((next_index < total_specs)); do
        local -a pids=()
        local batch_failed=0
        local slot=0

        while ((slot < PARALLELISM && next_index < total_specs)); do
            local spec="${specs_ref[$next_index]}"
            local experiment_name
            local dataset_id
            local split
            local cuda_device=""

            IFS='|' read -r experiment_name dataset_id split <<<"$spec"

            if ((PARALLELISM > 1)); then
                cuda_device="${GPU_DEVICES[$slot]}"
            fi

            run_experiment "$experiment_name" "$dataset_id" "$split" "$cuda_device" &
            pids+=("$!")

            slot=$((slot + 1))
            next_index=$((next_index + 1))
        done

        local pid
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                batch_failed=1
            fi
        done

        if ((batch_failed != 0)); then
            echo "Error: one or more experiments failed." >&2
            exit 1
        fi
    done
}