"""Notebook-local helpers for the Section 4.5 repetitive-call analysis.

This module powers `sec4p5_repetitive-tool-calls.ipynb`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from notebooks.posthoc_results import load_episode_results


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate repetitive tool-call rates from the released result bundles used by the Section 4.5 notebook."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=(ROOT / "results" / "sec4p5_repetitive_tool_calls" / "summary.csv"),
        help="Path to the output summary CSV file.",
    )
    parser.add_argument(
        "--episodes-output",
        type=Path,
        default=None,
        help=("Optional path to write the per-episode rows used to build the summary."),
    )
    return parser.parse_args()


def summarize_runs(episodes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-episode rows into one row per run."""
    run_summary = (
        episodes.groupby(["dataset", "planner", "model", "robustness", "run_id"])
        .agg(
            total_episodes=("question_id", "count"),
            episodes_with_repetition=(
                "repetitive_count",
                lambda series: int((series > 0).sum()),
            ),
            avg_repetition_when_present=(
                "repetitive_count",
                lambda series: float(series[series > 0].mean()) if (series > 0).any() else 0.0,
            ),
            accuracy=("is_correct", "mean"),
        )
        .reset_index()
    )
    run_summary["episodes_with_repetition_pct"] = (
        run_summary["episodes_with_repetition"] / run_summary["total_episodes"] * 100.0
    )
    return run_summary


def summarize_across_runs(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Average the run-level statistics for the paper-facing tables."""
    summary = (
        run_summary.groupby(["dataset", "planner", "model", "robustness"])
        .agg(
            total_episodes=("total_episodes", "mean"),
            episodes_with_repetition=("episodes_with_repetition", "mean"),
            episodes_with_repetition_pct=(
                "episodes_with_repetition_pct",
                "mean",
            ),
            avg_repetition_when_present=(
                "avg_repetition_when_present",
                "mean",
            ),
            accuracy=("accuracy", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(by=["robustness", "dataset", "model", "planner"])


def print_rate_table(summary: pd.DataFrame, robustness: str) -> None:
    """Print one repetition-rate table for the requested robustness level."""
    subset = summary[summary["robustness"] == robustness].copy()
    rate_table = subset.pivot_table(
        index=["dataset", "model"],
        columns="planner",
        values="episodes_with_repetition_pct",
    )
    print(f"\n=== {robustness.title()} robustness repetition rates (%) ===")
    print(rate_table.round(1).to_string())


def main() -> None:
    """Run the Section 4.5 CSV export."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    episodes = load_episode_results(include_steps=False)
    run_summary = summarize_runs(episodes)
    summary = summarize_across_runs(run_summary)

    summary.to_csv(args.output, index=False)
    print(f"Saved Section 4.5 summary to {args.output}")

    if args.episodes_output is not None:
        args.episodes_output.parent.mkdir(parents=True, exist_ok=True)
        episodes.to_csv(args.episodes_output, index=False)
        print(f"Saved per-episode rows to {args.episodes_output}")

    print_rate_table(summary, "high")
    print_rate_table(summary, "low")


if __name__ == "__main__":
    main()
