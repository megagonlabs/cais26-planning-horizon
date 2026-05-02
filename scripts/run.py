"""
Generic Meta Agent experiment script for dataset evaluation.

This script provides functionality to run comprehensive experiments with Meta Agents
(ReAct, SH, FH, etc.) on various datasets from Hugging Face Hub, collecting detailed
traces and performance metrics for in-depth analysis.

The script performs the following operations:
1. Load dataset from Hugging Face Hub
2. Parse YAML configuration files (model parameters, prompts, etc.)
3. Initialize environment with Worker Agents for problem solving
4. Create and configure the Meta Agent (based on type in config)
5. Run episodes and collect detailed traces
6. Generate evaluation metrics and save all outputs

Output Structure:
- Creates timestamped run directory (YYYY-MM-DD-HH-MM-SS)
- config.yaml: Copy of the configuration used
- result.jsonl: Detailed episode traces and outcomes
- memory.pkl: Serialized per-episode environment memory mapping
- metrics.jsonl: Performance metrics (accuracy, success rate, token usage)

Usage:
    uv run python scripts/run.py experiment=kopl_kbqa/sh.v1 \
        -o results/ --dataset-id dataset/name --subset subset --split train --num-episodes 10

Examples:
    # Run SH Meta Agent on KoPL dataset
    uv run python scripts/run.py experiment=kopl_kbqa/sh.v1 \
        -o results/ --num-episodes 5

    # Run ReAct Meta Agent on GSM8K
    uv run python scripts/run.py configs/gsm8k/meta-agent-react.v1.yaml \
        -o results/ --dataset-id Solaris99/AgentBank --subset gsm8k --num-episodes 10
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
import json
import os
import pickle
import random
import sys
import textwrap
import threading
import time
import traceback

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import hydra

import yaml

# Register custom OmegaConf resolver for filename/basename extraction
OmegaConf.register_new_resolver("basename", lambda p: p.split("/")[-1] if p else "", replace=True)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from planning.environment import Environment  # noqa: E402
from scripts.utils.agent_utils import (  # noqa: E402
    build_llm_registry_from_config,
    create_meta_agent,
    get_meta_agent_llm_call,
    load_config,
    setup_environment,
)
from scripts.utils.data_utils import load_dataset_router  # noqa: E402
from scripts.utils.evaluation_utils import calculate_evaluation_metrics  # noqa: E402

# Global lock for sequential environment initialization
# This prevents race conditions when multiple threads initialize PyTorch models simultaneously
_env_init_lock = threading.Lock()


def save_config_copy(config: dict[str, Any], run_dir: Path) -> None:
    """
    Save a copy of the configuration file to the run directory.

    Args:
        config: Configuration dictionary
        run_dir: Run directory to save config to
    """
    config_copy_path = run_dir / "config.yaml"
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Configuration saved to: {config_copy_path}")


def serialize_memory_for_pickle(memory_dict: dict) -> dict:
    """
    Convert memory objects to serializable dictionaries.

    Args:
        memory_dict: Dictionary of ContextMemory objects

    Returns:
        dict: Serializable dictionary representation
    """
    serializable = {}
    for session_id, memory in memory_dict.items():
        serializable[session_id] = {
            "session_id": memory.session_id,
            "agent_name": memory.agent_name,
            "query": memory.query,
            "parent_session_id": memory.parent_session_id,
            "step_history": [step.to_dict() for step in memory.step_history],
            "metadata": memory.metadata,
        }
    return serializable


def format_exception_for_json(e: Exception) -> dict[str, Any]:
    """
    Create a JSON-serializable summary of an exception.

    Args:
        e: Exception to format

    Returns:
        dict: Contains 'type', 'message', 'traceback', and 'frames'.
    """
    tb_list = traceback.extract_tb(e.__traceback__)
    frames = [
        {
            "file": f.filename,
            "line": f.lineno,
            "function": f.name,
            "code": f.line,
        }
        for f in tb_list
    ]
    tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    return {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": tb_str,
        "frames": frames,
    }


def summarize_config(cfg: dict[str, Any]) -> str:
    """
    Generate a formatted summary of the configuration dictionary.

    Args:
        cfg: Configuration dictionary from YAML file

    Returns:
        str: Formatted configuration summary string
    """
    meta_agent = cfg.get("meta_agent", {})
    model = meta_agent.get("model", {})
    environment = cfg.get("environment", {})

    return textwrap.dedent(
        f"""
        Config Summary
        --------------
        Meta Agent: {meta_agent.get("type", "N/A")}
        Model: {model.get("model", "N/A")}
        Max Steps: {meta_agent.get("max_steps", "N/A")}
        Max Retries: {meta_agent.get("max_retries", "N/A")}
        Agents: {len(environment.get("agents", {}))} configured
        """
    ).strip()


def run_episodes(
    agent,
    environment: Environment,
    records: list[dict[str, Any]],
    num_episodes: Optional[int] = None,
    worker_idx: Optional[int] = 0,
) -> tuple[list[dict[str, Any]], dict[str, dict]]:
    """
    Run Meta Agent episodes on dataset problems.

    Args:
        agent: The initialized Meta Agent
        environment: The environment with Worker Agents
        records: Dataset records to run episodes on
        num_episodes: Number of episodes to run (defaults to all records)
        worker_idx: Index of the worker (for distributed training)

    Returns:
        Tuple of (episode results, per-episode memory mapping)
    """
    if num_episodes is None:
        num_episodes = len(records)
    else:
        num_episodes = min(num_episodes, len(records))

    results = []
    episode_memories = {}

    sys.stderr.write(
        f"[{worker_idx}] Running {num_episodes} episodes with {agent.name}...\n"
    )

    start_all = time.time()
    log_interval = num_episodes // 10 if num_episodes > 10 else 1

    for i in range(num_episodes):
        record = records[i]
        query = record["query"]
        answer = record["answer"]

        if not query:
            sys.stderr.write(
                f"[{worker_idx}] Episode {record['id']}: Skipping - no query found\n"
            )
            continue

        try:
            # Reset environment memory before each episode to isolate it
            environment.memory = {}

            start_time = time.time()
            # Pass demonstration_candidates and task_config if present in the record
            demonstration_candidates = record.get("demonstration_candidates")
            task_config = record.get("task_config")
            episode_result = agent.run_episode(
                query_or_params=query,
                environment=environment,
                demonstration_candidates=demonstration_candidates,
                task_config=task_config,
            )
            end_time = time.time()

            # Record demonstrations metadata
            demonstrations = {}
            if demonstration_candidates:
                # Extract num_demonstrations from agent config if available
                num_demos = 0
                if hasattr(agent, "plan_step_params"):
                    num_demos = agent.plan_step_params.get("num_demonstrations", 0)
                elif hasattr(agent, "plan_params"):
                    num_demos = agent.plan_params.get("num_demonstrations", 0)
                if num_demos > 0:
                    # Record the indices and similarities of selected demonstrations
                    demonstrations = [
                        {"text": d["text"], "similarity": d["similarity"]}
                        for d in demonstration_candidates[:num_demos]
                    ]

            # Convert ExecutionResult to dict for compatibility
            result_dict = {
                "episode_index": i,
                "id": record["id"],
                "query": query,
                "expected_answer": answer,
                "expected_answer_label": record.get("answer_label"),
                "output": episode_result.result_data,
                "metadata": {
                    "success": episode_result.success,
                    "runtime": end_time - start_time,
                    "execution_time": episode_result.execution_time,
                    "token_usage": episode_result.token_usage,
                    "demonstrations": demonstrations,
                    "step_count": episode_result.step_count,
                    **episode_result.metadata,
                },
            }

            if "_original_index" in record:
                result_dict["_original_index"] = record["_original_index"]

            if hasattr(episode_result, "steps") or "steps" in episode_result.metadata:
                result_dict["steps"] = episode_result.metadata.get("steps", [])

            results.append(result_dict)

            # Serialize and map the memory
            serialized_memory = serialize_memory_for_pickle(environment.memory)

            # Add plan_history if it exists in the result metadata
            if "plan_history" in episode_result.metadata:
                serialized_memory["plan_history"] = episode_result.metadata[
                    "plan_history"
                ]

            episode_memories[record["id"]] = serialized_memory

            completed = len(results)
            if completed % log_interval == 0:
                elapsed = time.time() - start_all
                eta = None
                if completed > 0:
                    avg_time = elapsed / completed
                    remaining = num_episodes - completed
                    eta = timedelta(seconds=int(avg_time * remaining))
                progress_str = (
                    f"Completed {completed}/{num_episodes} | "
                    f"Elapsed: {timedelta(seconds=int(elapsed))}"
                )
                if eta is not None:
                    progress_str += f" | ETA: {eta}"
                sys.stderr.write(f"[{worker_idx}] {progress_str}\n")

        except KeyboardInterrupt:
            sys.stderr.write(f"[{worker_idx}] Episode {record['id']}: Interrupted\n")
            raise
        except Exception as e:
            tb_list = traceback.extract_tb(e.__traceback__)
            filename, lineno, func, text = tb_list[-1]
            sys.stderr.write(f"-- Error: {e}\n")
            sys.stderr.write(f"-- File: {filename}, Line: {lineno}, Function: {func}\n")
            sys.stderr.write(f"-- Code: {text}\n")
            sys.stderr.write(f"[{worker_idx}] Episode {record['id']}: ERROR - {e}\n")
            exc_json = format_exception_for_json(e)
            result = {
                "episode_index": i,
                "id": record["id"],
                "query": query,
                "expected_answer": answer,
                "output": "ERROR",
                "metadata": {
                    "success": False,
                    "error": exc_json["message"],
                    "exception": exc_json["type"],
                    "traceback": exc_json["traceback"],
                    "frames": exc_json["frames"],
                    "runtime": 0,
                },
            }
            if "_original_index" in record:
                result["_original_index"] = record["_original_index"]

            if len(environment.memory) > 0:
                meta_agent_memory = None
                for key, mem in environment.memory.items():
                    lowered_key = key.lower()
                    if lowered_key.startswith("react_meta") or lowered_key.startswith("meta_sh") or lowered_key.startswith("meta_fh"):
                        meta_agent_memory = mem
                        break
                if meta_agent_memory:
                    steps_list = [
                        step.to_dict(exclude_large_data=True)
                        for step in meta_agent_memory.step_history
                    ]
                    result["steps"] = steps_list
                    result["metadata"]["session_id"] = meta_agent_memory.session_id  # type: ignore
                episode_memories[record["id"]] = serialize_memory_for_pickle(
                    environment.memory
                )
            results.append(result)

    return results, episode_memories


def run(
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    worker_idx: int = 0,
    debug: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict]]:
    """
    Worker function to initialize Meta Agent/environment and run episodes on a subset of records.

    Args:
        records: List of dataset records for this worker
        cfg: Experiment configuration dictionary
        worker_idx: Worker index (for logging)
        debug: Enable debug mode

    Returns:
        Tuple of (episode results, per-episode memory mapping)
    """
    # Build LLM registry
    llm_registry = build_llm_registry_from_config(cfg)
    env = None

    try:
        # Use global lock to ensure env.load_from_config is executed sequentially
        sys.stderr.write(f"[{worker_idx}] Waiting for environment initialization lock...\n")
        with _env_init_lock:
            try:
                sys.stderr.write(f"[{worker_idx}] Initializing environment...\n")
                env = setup_environment(cfg, llm_registry, debug=debug)
                sys.stderr.write(f"[{worker_idx}] Environment initialization completed\n")
            except Exception as e:
                sys.stderr.write(f"[{worker_idx}] Environment initialization FAILED: {e}\n")
                raise RuntimeError(
                    f"Worker {worker_idx} environment initialization failed: {e}"
                ) from e

        # Get Meta Agent LLM call and create agent
        meta_llm_call = get_meta_agent_llm_call(cfg, llm_registry)
        agent_type = cfg.get("meta_agent", {}).get("type", "react")
        agent = create_meta_agent(agent_type, cfg, meta_llm_call)

        results, episode_memories = run_episodes(
            agent, env, records, num_episodes=len(records), worker_idx=worker_idx
        )

        return results, episode_memories

    finally:
        # Cleanup LLM registry
        if llm_registry:
            llm_registry.close_all()

        # Cleanup environment resources (e.g., Pyserini searchers)
        if env:
            env.close()


def print_experiment_summary(results: list[dict[str, Any]]) -> None:
    """Print a summary of experiment results."""
    if not results:
        print("No results to summarize.")
        return

    total_episodes = len(results)
    successful_episodes = sum(1 for r in results if r["metadata"].get("success", False))
    success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0

    total_runtime = sum(r["metadata"].get("runtime", 0) for r in results)
    avg_runtime = total_runtime / total_episodes if total_episodes > 0 else 0

    total_steps_successful = sum(
        r["metadata"].get("step_count", 0)
        for r in results
        if r["metadata"].get("success", False) and r["metadata"].get("step_count") is not None
    )
    total_steps_all = sum(
        r["metadata"].get("step_count", 0)
        for r in results
        if r["metadata"].get("step_count") is not None
    )
    avg_steps_successful = total_steps_successful / successful_episodes if successful_episodes > 0 else 0
    avg_steps_all = total_steps_all / total_episodes if total_episodes > 0 else 0

    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"Total Episodes: {total_episodes}")
    print(f"Successful Episodes: {successful_episodes}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Runtime: {avg_runtime:.2f} seconds")
    print(f"Average Steps (successful episodes): {avg_steps_successful:.1f}")
    print(f"Average Steps (all episodes): {avg_steps_all:.1f}")
    print(f"Total Runtime: {timedelta(seconds=int(total_runtime))}")
    print("=" * 50)


def write_experiment_outputs(
    results: list[dict[str, Any]],
    episode_memories: dict[str, dict],
    config: dict[str, Any],
    run_dir: Path,
) -> None:
    """Write all experiment outputs to the run directory."""
    # Save configuration copy
    save_config_copy(config, run_dir)

    # Write episode results
    results_path = run_dir / "result.jsonl"
    with open(results_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Episode results written to: {results_path}")

    # Write serialized per-episode memories
    memory_path = run_dir / "memory.pkl"
    with open(memory_path, "wb") as f:
        pickle.dump(episode_memories, f)
    print(f"Per-episode environment memory saved to: {memory_path}")

    # Calculate and write metrics
    metrics = calculate_evaluation_metrics(results)
    metrics_path = run_dir / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        f.write(json.dumps(metrics) + "\n")
    print(f"Evaluation metrics saved to: {metrics_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main function to run the Meta Agent experiment."""
    # Resolve configuration
    config: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    exp_cfg = config.get("experiment", {})

    print("Configuration loaded:")
    print(summarize_config(config))
    print()

    # Access Hydra's managed run directory
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"Run directory: {run_dir}")
    print()

    # Load dataset
    dataset_id = exp_cfg["dataset_id"]
    subset = exp_cfg.get("subset")
    split = exp_cfg["split"]
    print(
        f"Loading dataset: {dataset_id} subset: {subset} ({split} split)"
    )
    records = load_dataset_router(dataset_id, subset, split)
    print(f"Loaded {len(records)} records")

    num_episodes = config.get("num_episodes")
    if num_episodes:
        records = records[:num_episodes]
        print(f"Limited to {len(records)} episodes")
    print()

    # Add original index and shuffle to distribute workload
    for idx, record in enumerate(records):
        record["_original_index"] = idx

    random.seed(42)
    random.shuffle(records)
    print("Shuffled records for better workload distribution")

    # Run experiments
    print("Starting experiments...")
    start_time = time.time()

    results = []
    all_memory = {}

    workers = config.get("workers", 1)
    debug = config.get("debug", False)

    if workers == 1:
        # Single worker
        worker_results, episode_memories = run(
            records, config, worker_idx=0, debug=debug
        )
        results.extend(worker_results)
        all_memory = episode_memories
    else:
        # Multiple workers
        chunk_size = len(records) // workers
        if len(records) % workers != 0:
            chunk_size += 1
        chunks = [
            records[i : i + chunk_size] for i in range(0, len(records), chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(run, chunk, config, idx, debug): idx
                for idx, chunk in enumerate(chunks)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    chunk_results, chunk_memories = future.result()
                    results.extend(chunk_results)
                    all_memory.update(chunk_memories)
                    print(f"Worker {idx} completed {len(chunk_results)} episodes")
                except KeyboardInterrupt:
                    print(f"Worker {idx} interrupted")
                    raise
                except Exception as e:
                    print(f"Worker {idx} failed: {e}")
                    if "initialization failed" in str(e):
                        print(
                            f"Stopping program due to worker {idx} initialization failure"
                        )
                        for f in future_to_idx:
                            if not f.done():
                                f.cancel()
                        raise RuntimeError(
                            f"Worker {idx} initialization failed, stopping all workers"
                        ) from e

    # Restore original order
    results.sort(key=lambda x: x.get("_original_index", -1))
    # Remove temporary index
    for r in results:
        r.pop("_original_index", None)

    end_time = time.time()
    print(f"Experiments completed in {timedelta(seconds=int(end_time - start_time))}")
    print()

    # Print summary
    print_experiment_summary(results)

    # Write all outputs to run directory
    write_experiment_outputs(results, all_memory, config, run_dir)

    print(f"\nAll experiment outputs saved to: {run_dir}")


if __name__ == "__main__":
    import logging
    # Set logging level
    logging.getLogger("planning").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    main()
