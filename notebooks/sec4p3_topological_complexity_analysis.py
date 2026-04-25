"""Notebook-local helpers for the Section 4.3 topological analysis.

This module contains the reusable computation that powers
`sec4p3_topological-complexity-analysis.ipynb`. The CLI wrapper in
`scripts/reproduce_sec4p3_table3.py` imports `main()` from here so the
command-line workflow stays available without keeping notebook-specific logic
inside `scripts/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATTERNS: dict[str, str] = {
    "GPT-4.1-mini": "gpt-4p1-mini",
    "GPT-5-mini": "gpt-5-mini",
    "Qwen3-235B-A22B": "qwen3-235b-a22b-instruct-2507",
    "Gemini-3-Flash": "gemini-3-flash-preview",
}

DATASET_ORDER = [
    "KQA Pro",
    "Atomic KBQA",
    "Mul. HotpotQA",
]

MODEL_ORDER = [
    "GPT-4.1-mini",
    "GPT-5-mini",
    "Qwen3-235B-A22B",
    "Gemini-3-Flash",
]

TERM_ORDER = [
    "critical_path_len",
    "avg_parallelism",
    "is_sh",
    "critical_path_len:is_sh",
    "avg_parallelism:is_sh",
]

TERM_LABELS: dict[str, str] = {
    "critical_path_len": "Depth ($\\beta_d$)",
    "avg_parallelism": "Breadth ($\\beta_b$)",
    "is_sh": "Is SH? ($\\beta_\\text{SH}$)",
    "critical_path_len:is_sh": "$\\beta_{d:\\text{SH}}$",
    "avg_parallelism:is_sh": "$\\beta_{b:\\text{SH}}$",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the Section 4.3 topological-complexity GEE models "
            "from the released result bundles."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "sec4p3_table3",
        help="Directory where CSV and JSON outputs will be written.",
    )
    return parser.parse_args()


def compute_paper_topology_features(
    frame: pd.DataFrame,
    *,
    has_finish_step: bool,
) -> pd.DataFrame:
    """Derive the paper-facing depth and breadth features from raw CSV metrics.

    The processed CSV files store `workflow_len` as the number of DAG nodes and
    `max_depth` as the longest root-to-node path length measured in edges.

    For Atomic KBQA and multi-objective HotpotQA, the stored DAG includes a
    synthetic `finish` node. That means:

    - `workflow_len` counts the `finish` node
    - `max_depth` includes the final edge into `finish`

    The Section 4.3 analysis in the paper excludes that terminal bookkeeping
    step because it carries little reasoning content. KQA Pro DAGs do not have
    a `finish` node, so their raw metrics already match the paper-facing
    semantics.

    Args:
        frame: Raw feature frame containing `workflow_len` and `max_depth`
        has_finish_step: Whether the raw DAG metrics include a synthetic finish
            node that should be excluded from the paper-facing features

    Returns:
        DataFrame with added raw and paper-facing topology columns
    """
    enriched = frame.copy()
    finish_offset = 1 if has_finish_step else 0

    enriched["workflow_len_raw"] = enriched["workflow_len"]
    enriched["max_depth_raw"] = enriched["max_depth"]
    enriched["critical_path_len"] = enriched["max_depth_raw"] - finish_offset
    enriched["workflow_len"] = enriched["workflow_len_raw"] - finish_offset

    if (enriched["critical_path_len"] <= 0).any():
        bad_rows = enriched.loc[
            enriched["critical_path_len"] <= 0,
            ["id", "workflow_len_raw", "max_depth_raw", "critical_path_len"],
        ]
        raise ValueError(
            "Paper-facing critical path length must stay positive after "
            f"finish-step adjustment. Problem rows: {bad_rows.to_dict('records')}"
        )

    enriched["avg_parallelism"] = (
        enriched["workflow_len"] / enriched["critical_path_len"]
    )
    return enriched


def load_atomic_features() -> pd.DataFrame:
    """Load Atomic KBQA depth/breadth features for the released test split."""
    frames: list[pd.DataFrame] = []
    for dataset in ["grailqa", "webqsp", "graphq"]:
        path = (
            ROOT
            / "data"
            / "atomic_kbqa"
            / dataset
            / "processed"
            / f"{dataset}_values.v1.csv"
        )
        frame = pd.read_csv(path)
        frame = frame[frame["id"].str.startswith("test_")].copy()
        frame["id"] = frame["dataset"] + "_" + frame["id"]
        frame = compute_paper_topology_features(
            frame,
            has_finish_step=True,
        )
        frames.append(
            frame[
                [
                    "id",
                    "dataset",
                    "last_step",
                    "workflow_len_raw",
                    "max_depth_raw",
                    "workflow_len",
                    "critical_path_len",
                    "avg_parallelism",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def load_kqa_features() -> pd.DataFrame:
    """Load KQA Pro depth/breadth features for the released evaluation split."""
    path = (
        ROOT
        / "data"
        / "kopl_kbqa"
        / "kqa_pro"
        / "processed"
        / "kqa_pro_values.v1.csv"
    )
    frame = pd.read_csv(path)
    frame = frame[frame["id"].str.startswith("val_")].copy()

    # The released KQA Pro results still use val_* ids, but the evaluator-facing
    # docs treat this split as the paper's test set.
    frame["id"] = "kqa_pro_" + frame["id"].str.replace(
        "val_", "test_", regex=False
    )
    frame = compute_paper_topology_features(
        frame,
        has_finish_step=False,
    )
    return frame[
        [
            "id",
            "dataset",
            "last_step",
            "workflow_len_raw",
            "max_depth_raw",
            "workflow_len",
            "critical_path_len",
            "avg_parallelism",
        ]
    ]


def load_hotpot_features() -> pd.DataFrame:
    """Load multi-objective HotpotQA depth/breadth features."""
    path = ROOT / "data" / "multiobj_hotpotqa" / "processed" / "hotpotqa_values.v1.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["id"].str.startswith("test_")].copy()
    frame["id"] = "multiobj_hotpotqa_" + frame["id"]
    frame = compute_paper_topology_features(
        frame,
        has_finish_step=True,
    )
    frame["has_bridge"] = frame["component_types"].fillna("").str.contains(
        "bridge"
    )
    frame["has_comparison"] = frame["component_types"].fillna("").str.contains(
        "comparison"
    )
    return frame[
        [
            "id",
            "dataset",
            "component_types",
            "has_bridge",
            "has_comparison",
            "workflow_len_raw",
            "max_depth_raw",
            "workflow_len",
            "critical_path_len",
            "avg_parallelism",
        ]
    ]


def normalize_result_id(dataset_group: str, raw_id: str) -> str:
    """Convert a raw result id into the merged analysis id."""
    if dataset_group == "kqa_pro":
        if raw_id.startswith("val_"):
            raw_id = raw_id.replace("val_", "test_", 1)
        return f"kqa_pro_{raw_id}"
    if dataset_group == "hotpotqa":
        return f"multiobj_hotpotqa_{raw_id}"
    return f"{dataset_group}_{raw_id}"


def pick_result_dir(base_dir: Path, method: str, model_pattern: str) -> Path:
    """Find the unique released result directory for one method/model pair."""
    matches: list[Path] = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if method not in name:
            continue
        if model_pattern not in name:
            continue
        if "strict" in name or "topk-1" in name:
            continue
        if "retries-8" not in name:
            continue
        matches.append(child)

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one released result directory for "
            f"{base_dir}, {method=}, {model_pattern=}; found "
            f"{[match.name for match in matches]}"
        )

    return matches[0]


def load_results(dataset_group: str, result_root: Path) -> pd.DataFrame:
    """Load released result rows for one dataset family."""
    rows: list[dict[str, object]] = []
    split_dir = result_root / dataset_group / "test"

    for model_name, model_pattern in MODEL_PATTERNS.items():
        for method in ["itr", "pne"]:
            run_dir = pick_result_dir(split_dir, method, model_pattern)
            timestamp_dirs = sorted(
                path for path in run_dir.iterdir() if path.is_dir()
            )
            for timestamp_dir in timestamp_dirs:
                result_file = timestamp_dir / "result.jsonl"
                with result_file.open() as handle:
                    for line in handle:
                        item = json.loads(line)
                        question_id = normalize_result_id(dataset_group, item["id"])
                        rows.append(
                            {
                                "id": question_id,
                                "question_id": question_id,
                                "run_id": timestamp_dir.name,
                                "planner": "SH" if method == "itr" else "FH",
                                "is_sh": int(method == "itr"),
                                "model": model_name,
                                "is_correct": int(
                                    item.get("evaluation", {})
                                    .get("is_correct", "")
                                    .lower()
                                    == "correct"
                                ),
                            }
                        )

    return pd.DataFrame(rows)


def standardize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Z-score depth and breadth inside one dataset/model slice."""
    standardized = frame.copy()
    for column in ["critical_path_len", "avg_parallelism"]:
        mean = standardized[column].mean()
        std = standardized[column].std(ddof=0)
        standardized[column] = (standardized[column] - mean) / std
    return standardized


def fit_gee_models(
    frame: pd.DataFrame,
    dataset_name: str,
    formula: str,
) -> pd.DataFrame:
    """Fit one GEE model per backbone model for one dataset family."""
    rows: list[dict[str, object]] = []
    for model_name, model_frame in frame.groupby("model"):
        standardized = standardize_features(model_frame)
        result = smf.gee(
            formula=formula,
            groups="question_id",
            data=standardized,
            family=sm.families.Binomial(),
        ).fit()

        for term in TERM_ORDER:
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "term": term,
                    "term_label": TERM_LABELS[term],
                    "coef": result.params[term],
                    "std_err": result.bse[term],
                    "pvalue": result.pvalues[term],
                    "n_obs": int(result.nobs),
                    "n_questions": int(standardized["question_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def significance_marker(pvalue: float) -> str:
    """Return the paper's significance marker for one p-value."""
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def format_summary_value(coef: float, pvalue: float) -> str:
    """Format one Table 3 coefficient cell."""
    sign = "+" if coef >= 0 else ""
    return f"{sign}{coef:.3f}{significance_marker(pvalue)}"


def build_summary_table(detailed: pd.DataFrame) -> pd.DataFrame:
    """Convert detailed coefficients into the compact Table 3 layout."""
    rows: list[dict[str, str]] = []
    for (dataset_name, model_name), subset in detailed.groupby(["dataset", "model"]):
        row: dict[str, str] = {
            "Dataset": dataset_name,
            "Model": model_name,
        }
        for term in TERM_ORDER:
            term_row = subset[subset["term"] == term].iloc[0]
            row[TERM_LABELS[term]] = format_summary_value(
                float(term_row["coef"]),
                float(term_row["pvalue"]),
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary[
        [
            "Dataset",
            "Model",
            TERM_LABELS["critical_path_len"],
            TERM_LABELS["avg_parallelism"],
            TERM_LABELS["is_sh"],
            TERM_LABELS["critical_path_len:is_sh"],
            TERM_LABELS["avg_parallelism:is_sh"],
        ]
    ]
    summary["Dataset"] = pd.Categorical(
        summary["Dataset"],
        categories=DATASET_ORDER,
        ordered=True,
    )
    summary["Model"] = pd.Categorical(
        summary["Model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    summary = summary.sort_values(["Dataset", "Model"]).reset_index(drop=True)
    return summary


def write_metadata(
    output_dir: Path,
    atomic_frame: pd.DataFrame,
    kqa_frame: pd.DataFrame,
    hotpot_frame: pd.DataFrame,
) -> None:
    """Write lightweight metadata for the current released bundle."""
    metadata = {
        "inputs": {
            "atomic_features": "data/atomic_kbqa/*/processed/*_values.v1.csv",
            "kqa_features": "data/kopl_kbqa/kqa_pro/processed/kqa_pro_values.v1.csv",
            "hotpot_features": "data/multiobj_hotpotqa/processed/hotpotqa_values.v1.csv",
            "atomic_results": "results/atomic_kbqa/*/test/*retries-8/*/result.jsonl",
            "kqa_results": "results/kopl_kbqa/kqa_pro/test/*retries-8/*/result.jsonl",
            "hotpot_results": "results/multiobj/hotpotqa/test/*retries-8/*/result.jsonl",
        },
        "question_counts": {
            "KQA Pro": int(kqa_frame["question_id"].nunique()),
            "Atomic KBQA": int(atomic_frame["question_id"].nunique()),
            "Mul. HotpotQA": int(hotpot_frame["question_id"].nunique()),
        },
        "notes": [
            "Planner directories still use the legacy itr/pne names.",
            "This script reports those planners as SH/FH to match the paper.",
            (
                "Atomic KBQA and multi-objective HotpotQA raw CSV metrics include "
                "a synthetic finish node in workflow_len and the terminal edge in "
                "max_depth. The Section 4.3 analysis removes that bookkeeping step "
                "before computing critical_path_len and avg_parallelism."
            ),
            (
                "The legacy notebook bundled stale outputs from an older sample. "
                "This script recomputes all coefficients from the released CSV "
                "and result.jsonl files instead of relying on stored notebook "
                "output."
            ),
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def main() -> None:
    """Run the full Section 4.3 reproduction pipeline."""
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    atomic_features = load_atomic_features()
    kqa_features = load_kqa_features()
    hotpot_features = load_hotpot_features()

    atomic_results = pd.concat(
        [
            load_results("grailqa", ROOT / "results" / "atomic_kbqa"),
            load_results("webqsp", ROOT / "results" / "atomic_kbqa"),
            load_results("graphq", ROOT / "results" / "atomic_kbqa"),
        ],
        ignore_index=True,
    )
    kqa_results = load_results("kqa_pro", ROOT / "results" / "kopl_kbqa")
    hotpot_results = load_results("hotpotqa", ROOT / "results" / "multiobj")

    atomic_frame = atomic_results.merge(
        atomic_features,
        on="id",
        how="inner",
        validate="many_to_one",
    )
    kqa_frame = kqa_results.merge(
        kqa_features,
        on="id",
        how="inner",
        validate="many_to_one",
    )
    hotpot_frame = hotpot_results.merge(
        hotpot_features,
        on="id",
        how="inner",
        validate="many_to_one",
    )

    detailed = pd.concat(
        [
            fit_gee_models(
                atomic_frame,
                dataset_name="Atomic KBQA",
                formula=(
                    "is_correct ~ critical_path_len * is_sh + "
                    "avg_parallelism * is_sh + C(dataset) + C(last_step)"
                ),
            ),
            fit_gee_models(
                kqa_frame,
                dataset_name="KQA Pro",
                formula=(
                    "is_correct ~ critical_path_len * is_sh + "
                    "avg_parallelism * is_sh + C(last_step)"
                ),
            ),
            fit_gee_models(
                hotpot_frame,
                dataset_name="Mul. HotpotQA",
                formula=(
                    "is_correct ~ critical_path_len * is_sh + "
                    "avg_parallelism * is_sh + has_bridge + has_comparison"
                ),
            ),
        ],
        ignore_index=True,
    )
    detailed["dataset"] = pd.Categorical(
        detailed["dataset"],
        categories=DATASET_ORDER,
        ordered=True,
    )
    detailed["model"] = pd.Categorical(
        detailed["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    detailed["term"] = pd.Categorical(
        detailed["term"],
        categories=TERM_ORDER,
        ordered=True,
    )
    detailed = detailed.sort_values(["dataset", "model", "term"]).reset_index(
        drop=True
    )

    summary = build_summary_table(detailed)

    detailed.to_csv(output_dir / "detailed_coefficients.csv", index=False)
    summary.to_csv(output_dir / "summary_table.csv", index=False)
    write_metadata(output_dir, atomic_frame, kqa_frame, hotpot_frame)

    print("Saved Section 4.3 outputs to:", output_dir)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
