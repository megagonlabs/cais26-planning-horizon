"""
Single-problem demo runner for KQA Pro Meta Agents.

This script demonstrates how to:
1. Load the default KQA Pro experiment configuration
2. Set up Worker Agents and environment
3. Initialize and run a Meta Agent (supports 'sh' and 'fh')
4. Display the execution trajectory

Usage:
    uv run python scripts/example_run_kqa_pro.py --problem "Who is the spouse of the actor who played Jack in Titanic?"
    uv run python scripts/example_run_kqa_pro.py conf/experiment/kopl_kbqa/fh.v1.yaml --problem "Who is the spouse of the actor who played Jack in Titanic?"
    uv run python scripts/example_run_kqa_pro.py --model-config vllm_qwen3-0p6b --llm-provider vllm-local
"""

from pathlib import Path
from typing import Any, Optional
import argparse
import json
import logging
import os
import sys
import threading
import time

from dotenv import load_dotenv

load_dotenv(override=True)

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from scripts.utils.agent_utils import (  # noqa: E402
    build_llm_registry_from_config,
    create_meta_agent,
    get_meta_agent_llm_call,
    load_config,
    setup_environment,
)
from planning.agents.step import Step, StepStatus  # noqa: E402


DEFAULT_CONFIG_PATH = project_root / "conf" / "experiment" / "kopl_kbqa" / "sh.v1.yaml"
DEFAULT_PROBLEM = "Who is the spouse of the actor who played Jack in Titanic?"
DEFAULT_NUM_DEMONSTRATIONS = 0
STREAM_POLL_INTERVAL_SECONDS = 0.1
KQA_PRO_WORKER_AGENT_IDS = (
    "kopl_schema_free_agents",
    "kopl_find_and_filter_concept_agents",
    "kopl_key_only_agents",
    "kopl_key_and_value_agents",
)


def _get_model_name(config: dict[str, Any]) -> str:
    """Extract a readable model name from the meta-agent config."""
    model_config = config.get("meta_agent", {}).get("model", {})
    if isinstance(model_config, dict):
        return str(model_config.get("model", "N/A"))
    return str(model_config)


def _set_example_num_demonstrations(
    config: dict[str, Any],
    num_demonstrations: int,
) -> None:
    """Override demonstration count for the self-contained example runner."""
    config["num_demonstrations"] = num_demonstrations

    meta_generation = config.get("meta_agent", {}).get("generation", {})
    for generation_key in ("plan_step", "plan"):
        generation_config = meta_generation.get(generation_key)
        if isinstance(generation_config, dict):
            generation_config["num_demonstrations"] = num_demonstrations


def _build_config_overrides(args: argparse.Namespace) -> list[str]:
    """Build Hydra override expressions from CLI convenience flags.

    Args:
        args: Parsed command-line arguments

    Returns:
        list[str]: Hydra override expressions
    """
    overrides = list(args.override)

    if args.model_config:
        overrides.append(f"model={args.model_config}")
        worker_model_config = args.worker_model_config or args.model_config
        overrides.append(f"model@worker_model={worker_model_config}")
    elif args.worker_model_config:
        overrides.append(f"model@worker_model={args.worker_model_config}")

    if args.llm_provider:
        overrides.append(f"meta_agent.llm_provider={args.llm_provider}")
        for agent_id in KQA_PRO_WORKER_AGENT_IDS:
            overrides.append(
                f"environment.agents.{agent_id}.llm_provider={args.llm_provider}"
            )

    return overrides


def _collect_missing_example_prerequisites(config: dict[str, Any]) -> list[str]:
    """Collect missing files or environment variables required by the KQA Pro demo."""
    missing_items: list[str] = []

    llm_providers = config.get("llm_providers", {})
    meta_agent_config = config.get("meta_agent", {})
    meta_provider_id = meta_agent_config.get("llm_provider", "default")
    provider_config = llm_providers.get(meta_provider_id, {})
    api_key_env = provider_config.get("api_key_env")

    if api_key_env and not os.getenv(api_key_env):
        missing_items.append(
            f"Environment variable `{api_key_env}` is not set for provider `{meta_provider_id}`."
        )

    env_config = config.get("environment", {})
    kb_path = env_config.get("tools", {}).get("kopl_schema_free_tools", {}).get(
        "parameters", {}
    ).get("kb_path")
    if kb_path:
        kb_file = project_root / kb_path
        if not kb_file.exists():
            missing_items.append(f"Knowledge base file is missing: `{kb_file}`")

    embeddings_dirs: set[Path] = set()
    for agent_config in env_config.get("agents", {}).values():
        common_parameters = agent_config.get("common_parameters", {})
        embeddings_dir = common_parameters.get("embeddings_dir")
        if embeddings_dir:
            embeddings_dirs.add(project_root / embeddings_dir)

    required_embedding_files = (
        "entity_embeddings.pkl",
        "key_embeddings.pkl",
        "value_embeddings.pkl",
    )
    for embeddings_dir in sorted(embeddings_dirs):
        for filename in required_embedding_files:
            embeddings_file = embeddings_dir / filename
            if not embeddings_file.exists():
                missing_items.append(
                    f"Required embedding file is missing: `{embeddings_file}`"
                )

    return missing_items


def _validate_example_prerequisites(config: dict[str, Any]) -> None:
    """Raise a friendly error if the walkthrough prerequisites are not ready."""
    missing_items = _collect_missing_example_prerequisites(config)
    if not missing_items:
        return

    details = "\n".join(f"- {item}" for item in missing_items)
    raise RuntimeError(
        "The KQA Pro demo is not ready to run yet.\n"
        f"Missing prerequisites:\n{details}\n\n"
        "Prepare the KQA Pro assets first with:\n"
        "- `bash data/kopl_kbqa/kqa_pro/download.sh`\n"
        "- `uv run python data/kopl_kbqa/kqa_pro/scripts/embed_kb.py --kb_path data/kopl_kbqa/kqa_pro/kb.json`\n\n"
        "The full benchmark preprocessing pipeline is documented separately in "
        "`data/kopl_kbqa/kqa_pro/README.md` and `docs/preprocessing/kqa_pro.md`."
    )


def _find_meta_memory(environment) -> Optional[Any]:
    """Return the Meta Agent memory object for the currently running example."""
    for session_id, memory in environment.memory.items():
        if session_id.startswith(("meta_sh_", "meta_fh_")):
            return memory
    return None


def _format_console_value(value: Any) -> str:
    """Format values for readable console output."""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _print_labeled_value(label: str, value: Any) -> None:
    """Print a possibly multiline labeled value with stable indentation."""
    formatted_value = _format_console_value(value)
    lines = formatted_value.splitlines() or [""]
    print(f"   {label}: {lines[0]}", flush=True)
    for line in lines[1:]:
        print(f"      {line}", flush=True)


def _print_step_update(step: Step) -> None:
    """Print a single step update for live or recap output."""
    print(f"Step {step.step_num + 1} [{step.status.value}]", flush=True)
    _print_labeled_value("Action", step.get("action", "N/A"))
    _print_labeled_value("Observation", step.get("observation", "N/A"))
    print(flush=True)


def _stream_execution_steps(stop_event: threading.Event, environment) -> None:
    """Stream completed or failed steps while the agent is running."""
    printed_signatures: dict[int, tuple[str, str]] = {}
    printed_header = False

    while True:
        meta_memory = _find_meta_memory(environment)

        if meta_memory is not None:
            for step in list(meta_memory.step_history):
                observation = step.get("observation")
                if step.status == StepStatus.PLANNED and observation is None:
                    continue

                signature = (step.status.value, str(observation))
                if printed_signatures.get(step.step_num) == signature:
                    continue

                if not printed_header:
                    print("\nLIVE TRAJECTORY:")
                    print("=" * 60)
                    printed_header = True

                _print_step_update(step)
                printed_signatures[step.step_num] = signature

        if stop_event.is_set():
            return

        time.sleep(STREAM_POLL_INTERVAL_SECONDS)


def run_example_problem(
    config: dict[str, Any],
    problem: Optional[str] = None,
    debug: bool = False,
    stream_steps: bool = True,
):
    """Run the Meta Agent on a single example problem."""
    if problem is None:
        problem = DEFAULT_PROBLEM

    print("PROBLEM:")
    print(problem)
    print("\n")

    llm_registry = build_llm_registry_from_config(config)
    environment = setup_environment(config, llm_registry, debug=debug)
    stop_event = threading.Event()
    stream_thread: Optional[threading.Thread] = None

    try:
        meta_llm_call = get_meta_agent_llm_call(config, llm_registry)
        agent_type = config["meta_agent"].get("type", "sh")
        agent = create_meta_agent(agent_type, config, meta_llm_call)

        print(f"RUNNING {agent_type.upper()} META AGENT...")
        print("=" * 60)

        if stream_steps:
            stream_thread = threading.Thread(
                target=_stream_execution_steps,
                args=(stop_event, environment),
                daemon=True,
            )
            stream_thread.start()

        result = agent.run_episode(
            query_or_params=problem,
            environment=environment,
        )

        stop_event.set()
        if stream_thread is not None:
            stream_thread.join()

        print("\nEPISODE RESULTS:")
        print("=" * 60)
        print(f"Success: {result.success}")
        print(f"Final Answer: {result.result_data}")
        print(f"Steps Taken: {result.step_count}")
        print(f"Max Steps Reached: {result.metadata.get('max_steps_reached', False)}")

        trajectory_heading = "FULL TRAJECTORY RECAP" if stream_steps else "FULL TRAJECTORY"
        print(f"\n{trajectory_heading}:")
        print("=" * 60)
        steps_data = result.metadata.get("steps", [])
        if not steps_data:
            print("No steps were recorded.")
        for i, step_dict in enumerate(steps_data, 1):
            step = Step.from_dict(step_dict)
            if step.step_num != i - 1:
                step.step_num = i - 1
            _print_step_update(step)

        return result
    finally:
        stop_event.set()
        if stream_thread is not None and stream_thread.is_alive():
            stream_thread.join()
        environment.close()
        llm_registry.close_all()


def main(args) -> None:
    """Main function to run the KQA Pro example."""
    print("KQA Pro Example Runner")
    print("=" * 40)

    print(f"Loading configuration from: {args.config_path}")
    config_overrides = _build_config_overrides(args)
    if config_overrides:
        print("Applying overrides:")
        for override in config_overrides:
            print(f"- {override}")
        print()

    config = load_config(args.config_path, overrides=config_overrides)
    agent_type = config["meta_agent"].get("type", "sh")
    print(f"Model: {_get_model_name(config)}")
    print(f"Agent: {agent_type}")
    print(f"Max Steps: {config['meta_agent']['max_steps']}")
    print(f"Demonstrations: {args.num_demonstrations}")
    print(f"Live step streaming: {not args.no_stream}")
    print()

    _set_example_num_demonstrations(config, args.num_demonstrations)

    _validate_example_prerequisites(config)

    result = run_example_problem(
        config,
        problem=args.problem,
        debug=args.debug,
        stream_steps=not args.no_stream,
    )

    print("\nExample completed!")
    print(f"Agent provided answer: {result.result_data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an SH or FH Meta Agent on a single KQA Pro question",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the experiment configuration YAML file",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help=(
            "Override the Hydra `model` group for the Meta Agent. When "
            "`--worker-model-config` is omitted, this also updates the "
            "worker model group."
        ),
    )
    parser.add_argument(
        "--worker-model-config",
        type=str,
        help="Override the Hydra `worker_model` group for KQA Pro worker agents",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        help=(
            "Override the Meta Agent and KQA Pro worker agents to use the "
            "specified LLM provider ID"
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Additional Hydra override expression. Repeat this flag to pass "
            "multiple overrides."
        ),
    )
    parser.add_argument(
        "--problem", type=str, help="Custom problem to solve (overrides default)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose planning logs",
    )
    parser.add_argument(
        "--num-demonstrations",
        type=int,
        default=DEFAULT_NUM_DEMONSTRATIONS,
        help=(
            "Override the number of in-context demonstrations used by the "
            "example runner"
        ),
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable live step-by-step trajectory output during execution",
    )

    args = parser.parse_args()

    if args.num_demonstrations < 0:
        raise ValueError("--num-demonstrations must be non-negative")

    logging.getLogger("planning").setLevel(
        logging.DEBUG if args.debug else logging.WARNING
    )

    main(args)
