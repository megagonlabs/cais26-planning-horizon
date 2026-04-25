"""Shared loaders for the Section 4.4 and 4.5 notebook artifacts.

These helpers live in `notebooks/` because they exist to support the
notebook-based post-hoc analyses. The CLI wrapper for Section 4.5 imports this
module indirectly via `notebooks/sec4p5_repetitive_tool_calls.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATTERNS: dict[str, str] = {
    "GPT-4.1-mini": "gpt-4p1-mini",
    "GPT-5-mini": "gpt-5-mini",
    "Qwen3-235B-A22B": "qwen3-235b-a22b-instruct-2507",
    "Gemini-3-Flash": "gemini-3-flash-preview",
}
MODEL_ORDER = list(MODEL_PATTERNS)

PLANNER_ALIASES: dict[str, tuple[str, ...]] = {
    "FH": ("pne", "fh"),
    "SH": ("itr", "sh"),
}
PLANNER_ORDER = list(PLANNER_ALIASES)

DATASET_ROOTS: dict[str, Path] = {
    "kqa_pro": ROOT / "results" / "kopl_kbqa" / "kqa_pro" / "test",
    "grailqa": ROOT / "results" / "atomic_kbqa" / "grailqa" / "test",
    "webqsp": ROOT / "results" / "atomic_kbqa" / "webqsp" / "test",
    "graphq": ROOT / "results" / "atomic_kbqa" / "graphq" / "test",
    "multiobj_hotpotqa": ROOT / "results" / "multiobj" / "hotpotqa" / "test",
}
DATASET_DISPLAY_NAMES: dict[str, str] = {
    "kqa_pro": "KQA Pro",
    "grailqa": "GrailQA",
    "webqsp": "WebQSP",
    "graphq": "GraphQ",
    "multiobj_hotpotqa": "Mul. HotpotQA",
}
DATASET_ORDER = list(DATASET_ROOTS)
DATASET_LABEL_ORDER = [DATASET_DISPLAY_NAMES[name] for name in DATASET_ORDER]

ROBUSTNESS_LABELS: dict[str, str] = {
    "high": "High robustness",
    "low": "Low robustness",
}
ROBUSTNESS_ORDER = list(ROBUSTNESS_LABELS)


def planner_dir_matches(name: str, planner: str) -> bool:
    """Return whether a released result directory matches the planner."""
    aliases = PLANNER_ALIASES[planner]
    return any(re.match(rf"^{alias}(?:[._]|$)", name) for alias in aliases)


def get_robustness_filters(
    dataset: str,
    model_name: str,
    robustness: str,
) -> tuple[list[str], list[str]]:
    """Return include and exclude substrings for released result bundles."""
    if robustness not in ROBUSTNESS_LABELS:
        raise ValueError(f"Unknown robustness setting: {robustness}")

    if robustness == "high":
        return ["retries-8"], ["strict", "topk-1"]

    is_gpt_family = model_name.startswith("GPT")

    if dataset != "multiobj_hotpotqa":
        if is_gpt_family:
            return ["retries-8", "strict_topk-1"], []
        return ["retries-8", "strict"], ["topk-1"]

    if is_gpt_family:
        return ["retries-8", "topk-1"], ["strict"]
    return ["retries-8", "strict"], ["topk-1"]


def select_result_dir(
    dataset: str,
    planner: str,
    model_name: str,
    robustness: str,
) -> Path:
    """Find the unique released result directory for one analysis slice."""
    base_dir = DATASET_ROOTS[dataset]
    include_tags, exclude_tags = get_robustness_filters(
        dataset=dataset,
        model_name=model_name,
        robustness=robustness,
    )

    matches = [
        child
        for child in base_dir.iterdir()
        if child.is_dir()
        and planner_dir_matches(child.name, planner)
        and MODEL_PATTERNS[model_name] in child.name
        and all(tag in child.name for tag in include_tags)
        and all(tag not in child.name for tag in exclude_tags)
    ]

    if len(matches) != 1:
        match_names = [match.name for match in matches]
        raise ValueError(
            "Expected exactly one released result directory for "
            f"{dataset=}, {planner=}, {model_name=}, {robustness=}; "
            f"found {match_names}"
        )

    return matches[0]


def count_repetitive_calls(steps: list[dict[str, Any]]) -> int:
    """Count consecutive repeated tool calls with identical arguments."""
    repetitive_count = 0
    previous_call: tuple[str | None, str] | None = None

    for step in steps:
        action = step.get("data", {}).get("action")
        if not action:
            continue

        current_call = (
            action.get("name"),
            json.dumps(action.get("arguments", {}), sort_keys=True),
        )
        if previous_call is not None and current_call == previous_call:
            repetitive_count += 1
        previous_call = current_call

    return repetitive_count


def extract_question_text(item: dict[str, Any]) -> str | None:
    """Return the first available question-like field from a result record."""
    for key in ["question", "query", "problem", "input"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def load_episode_results(include_steps: bool = False) -> pd.DataFrame:
    """Load released per-episode rows for all Section 4.4/4.5 slices."""
    rows: list[dict[str, Any]] = []

    for dataset in DATASET_ORDER:
        for planner in PLANNER_ORDER:
            for model_name in MODEL_ORDER:
                for robustness in ROBUSTNESS_ORDER:
                    result_dir = select_result_dir(
                        dataset=dataset,
                        planner=planner,
                        model_name=model_name,
                        robustness=robustness,
                    )
                    timestamp_dirs = sorted(
                        path for path in result_dir.iterdir() if path.is_dir()
                    )
                    for timestamp_dir in timestamp_dirs:
                        result_file = timestamp_dir / "result.jsonl"
                        with result_file.open() as handle:
                            for line in handle:
                                item = json.loads(line)
                                steps = item.get("steps", [])
                                row: dict[str, Any] = {
                                    "dataset": dataset,
                                    "dataset_label": DATASET_DISPLAY_NAMES[
                                        dataset
                                    ],
                                    "planner": planner,
                                    "model": model_name,
                                    "robustness": robustness,
                                    "robustness_label": ROBUSTNESS_LABELS[
                                        robustness
                                    ],
                                    "run_id": timestamp_dir.name,
                                    "result_dir": result_dir.name,
                                    "question_id": item["id"],
                                    "question_text": extract_question_text(item),
                                    "final_output": item.get("output"),
                                    "success": bool(
                                        item.get("metadata", {}).get(
                                            "success", False
                                        )
                                    ),
                                    "evaluation_label": item.get(
                                        "evaluation", {}
                                    ).get("is_correct"),
                                    "is_correct": item.get(
                                        "evaluation", {}
                                    ).get("is_correct")
                                    == "correct",
                                    "total_steps": len(steps),
                                    "repetitive_count": count_repetitive_calls(
                                        steps
                                    ),
                                }
                                if include_steps:
                                    row["steps"] = steps
                                rows.append(row)

    return pd.DataFrame(rows)


def load_question_runs(
    dataset: str,
    planner: str,
    model_name: str,
    robustness: str,
    question_id: str,
) -> list[dict[str, Any]]:
    """Load all runs for one question within one analysis slice."""
    result_dir = select_result_dir(
        dataset=dataset,
        planner=planner,
        model_name=model_name,
        robustness=robustness,
    )
    rows: list[dict[str, Any]] = []

    for timestamp_dir in sorted(path for path in result_dir.iterdir() if path.is_dir()):
        result_file = timestamp_dir / "result.jsonl"
        with result_file.open() as handle:
            for line in handle:
                item = json.loads(line)
                if item["id"] != question_id:
                    continue
                steps = item.get("steps", [])
                rows.append(
                    {
                        "dataset": dataset,
                        "planner": planner,
                        "model": model_name,
                        "robustness": robustness,
                        "run_id": timestamp_dir.name,
                        "question_id": question_id,
                        "question_text": extract_question_text(item),
                        "final_output": item.get("output"),
                        "evaluation_label": item.get("evaluation", {}).get(
                            "is_correct"
                        ),
                        "is_correct": item.get("evaluation", {}).get(
                            "is_correct"
                        )
                        == "correct",
                        "repetitive_count": count_repetitive_calls(steps),
                        "total_steps": len(steps),
                        "steps": steps,
                    }
                )
                break

    return rows
