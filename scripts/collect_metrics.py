"""Collect experiment metrics across result directories and emit a CSV summary.

This utility searches under `results/<dataset>/<split>/` for method subdirectories
and aggregates metrics from all timestamped run subdirectories within each method.
For each core metric field, it computes mean and standard deviation across all runs.

Usage:
    uv run python scripts/collect_metrics.py <dataset> [--split test] -o out.csv

Example:
    uv run python scripts/collect_metrics.py atomic_kbqa/graphq --split test -o tables/graphq.csv
"""

from pathlib import Path
from typing import Any
import argparse
import json
import statistics
import sys

import pandas as pd

def get_all_metrics(method_dir: Path) -> list[dict[str, Any]]:
    """Read all metrics.jsonl files from all timestamped run subdirectories.

    The method directory is expected to contain timestamped run subdirectories
    (e.g. 2025-11-26-13-16-37). This function finds all such directories, opens
    `metrics.jsonl` inside each, and returns a list of parsed JSON objects from
    the last non-empty line of each file.

    Args:
        method_dir: Path to a method directory under `results/<dataset>/<split>/`.

    Returns:
        List of parsed metrics dictionaries. Empty list if no metrics found.
    """
    metrics_list: list[dict[str, Any]] = []

    # Collect timestamp directories
    timestamps: list[str] = [d.name for d in method_dir.iterdir() if d.is_dir()]

    if not timestamps:
        return metrics_list

    # Sort timestamps for stable ordering
    timestamps.sort()

    for timestamp in timestamps:
        metrics_file = method_dir / timestamp / "metrics.jsonl"

        if not metrics_file.exists():
            continue

        # Read metrics.jsonl and extract the last non-empty JSON object
        try:
            with metrics_file.open("r") as fh:
                lines = fh.readlines()
        except Exception as exc:  # pragma: no cover - IO error
            print(f"Error reading {metrics_file}: {exc}", file=sys.stderr)
            continue

        if not lines:
            continue

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                metrics = json.loads(line)
                metrics_list.append(metrics)
                break
            except json.JSONDecodeError:
                # Skip malformed lines and continue searching upwards
                continue

    return metrics_list

def main() -> None:
    """CLI entrypoint for the collect_metrics utility.

    Command line arguments:
        dataset: Path-like string under `results/` identifying dataset and optionally
            a subfolder (e.g. "atomic_kbqa/graphq").
        --split: Which split folder to inspect (default: "test").
        -o/--output: Output CSV filename. If omitted the CSV will not be written but
            the markdown table will still be printed.
    """
    parser = argparse.ArgumentParser(description="Collect metrics from experimental results.")
    parser.add_argument("dataset", help="Dataset name (e.g., atomic_kbqa/graphq)")
    parser.add_argument("--split", default="test", help="Split name (e.g., train, test)")
    parser.add_argument("--eval-version", default="2.0", help="Evaluation version to consider")
    parser.add_argument("-o", "--output", help="Output CSV filename")

    args = parser.parse_args()

    base_path = Path("results") / args.dataset / args.split

    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    records: list[dict[str, Any]] = []

    # Iterate over method directories and collect metrics from all runs
    for method_dir in sorted(base_path.iterdir()):
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name
        all_metrics = get_all_metrics(method_dir)

        if not all_metrics:
            continue

        # Validate that total_episodes is consistent across all runs
        total_episodes_set = {m["total_episodes"] for m in all_metrics}
        if len(total_episodes_set) > 1:
            raise ValueError(
                f"Method {method_name}: total_episodes mismatch across runs. "
                f"Found: {total_episodes_set}"
            )

        total_episodes = all_metrics[0]["total_episodes"]
        num_trials = len(all_metrics)

        # Normalize metrics: add accuracy and correct_answers if not present
        for metrics in all_metrics:
            for key in metrics.keys():
                if not key.startswith("correct_answers_"):
                    continue
                # e.g., correct_answers_{model}_{PROMPT_VERSION}
                if not key.endswith(f"_{args.eval_version}"):
                    continue

                metrics["correct_answers"] = metrics[key]
                metrics["accuracy"] = metrics[key] / total_episodes
                break
            else:
                raise ValueError(
                    f"Method {method_name}: No correct_answers found for eval version {args.eval_version}"
                    f" in metrics: {metrics.keys()}"
                )

        # Collect values for core metric fields to compute mean and std
        metric_fields = [
            "success_rate",
            "successful_episodes",
            "accuracy",
            "correct_answers",
            "avg_steps",
            "avg_runtime",
        ]

        record: dict[str, Any] = {
            "Method": method_name,
            "total_episodes": total_episodes,
            "num_trials": num_trials,
        }

        # Compute mean and std for each metric field
        for field in metric_fields:
            values = [m.get(field) for m in all_metrics if field in m]

            if values:
                mean_val = statistics.mean(values)
                record[field] = mean_val

                # Compute std only if there are 2+ values
                if len(values) > 1:
                    std_val = statistics.stdev(values)
                    record[f"{field}_std"] = std_val
                else:
                    record[f"{field}_std"] = 0.0

        # Token usage aggregation: compute mean across all runs
        token_input_values = []
        token_output_values = []

        for metrics in all_metrics:
            token_usage = metrics.get("total_token_usage", {})
            run_input = 0
            run_output = 0

            if isinstance(token_usage, dict):
                for usage in token_usage.values():
                    if isinstance(usage, dict):
                        run_input += int(
                            usage.get("prompt_tokens", usage.get("input_tokens", 0))
                        )
                        run_output += int(
                            usage.get("completion_tokens", usage.get("output_tokens", 0))
                        )

            token_input_values.append(run_input)
            token_output_values.append(run_output)

        record["total_token_usage.input"] = (
            statistics.mean(token_input_values) if token_input_values else 0
        )
        record["total_token_usage.output"] = (
            statistics.mean(token_output_values) if token_output_values else 0
        )

        records.append(record)

    if not records:
        print("No metrics found.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)

    # Desired column order
    cols = [
        "Method",
        "total_episodes",
        "num_trials",
        "success_rate",
        "success_rate_std",
        "successful_episodes",
        "successful_episodes_std",
        "accuracy",
        "accuracy_std",
        "correct_answers",
        "correct_answers_std",
        "avg_steps",
        "avg_steps_std",
        "avg_runtime",
        "avg_runtime_std",
        "total_token_usage.input",
        "total_token_usage.output",
    ]

    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # Sort rows by method name for stable output
    df = df.sort_values("Method")

    # Format display to 4 decimal places
    df_display = df.copy()
    float_cols = df_display.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        df_display[col] = df_display[col].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else x
        )

    # Write CSV if requested (with full precision)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved metrics to {out_path}")

    # Print a Markdown table preview (formatted display)
    print(df_display.to_markdown(index=False))


if __name__ == "__main__":
    main()
