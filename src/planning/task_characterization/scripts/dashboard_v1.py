"""
Streamlit dashboard for v1 task-space characterization.

Features:
- Loads GSM8K and HotpotQA CSVs from data/agentbank/processed
- Loads DAG JSON per instance from data/agentbank/processed/dags/auto.v2 (fallback to auto.v1)
- Analysis types: KDE (1D) and 2D scatter
- Modes: Cross-dataset (overlay) and In-dataset (single dataset)
- On point click (scatter), renders full DAG JSON for that instance

Run:
    uv run streamlit run src/planning/task_characterization/scripts/dashboard_v1.py

Notes:
- Point click for scatter requires the optional dependency `streamlit-plotly-events` for best UX.
  Install with: `uv add streamlit-plotly-events`
  Without it, a fallback selector is shown to view a selected ID's DAG.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import json

import graphviz
import matplotlib.pyplot as plt  # Optional plotting libs
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

# Optional component for capturing plotly click events
try:
    from streamlit_plotly_events import plotly_events  # type: ignore
    HAVE_PLOTLY_EVENTS = True
except Exception:
    HAVE_PLOTLY_EVENTS = False

DATASETS = ["gsm8k", "hotpotqa", "kqapro"]
CANDIDATE_FEATURES = [
    # DAG metrics
    "num_nodes",
    "num_edges",
    "num_roots",
    "num_leaves",
    "num_nonfinish_leaves",
    "max_gap_including_input",
    "max_gap_excluding_input",
    "avg_gap_including_input",
    "avg_gap_excluding_input",
    "max_depth",
    "max_width",
    "merging_ratio",
    "branching_ratio",
    "avg_headroom",
    "max_headroom",
    "min_headroom",
    "max_best_join_span",
    "max_worst_join_span",
    "avg_best_join_span",
    "avg_worst_join_span",
    # Non-DAG axes
    "input_len",
    "workflow_len",
    # Will be computed from DAG JSONs if available
    "max_in_degree",
    "max_out_degree",
]

SUBSET_COLORS = {
    "gsm8k": "#4E79A7",
    "hotpotqa": "#E15759",
    "kqapro": "#59A14F",
}


@st.cache_data(show_spinner=False)
def find_project_root() -> Path:
    """Find project root by looking for a pyproject.toml upwards from this file."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    # Fallback to repo root heuristic: four levels up
    return here.parents[4]


@st.cache_data(show_spinner=False)
def load_subset_csv(root: Path, subset: str) -> pd.DataFrame:
    """Load a subset CSV and return a DataFrame with a `subset` column."""
    if subset == "kqapro":
        csv_path = root / "data" / "kqa_pro" / "processed" / "kqapro_values.v1.csv"
    else:
        csv_path = root / "data" / "agentbank" / "processed" / f"{subset}_values.v1.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df["subset"] = subset
    return df


def _dag_candidate_paths(root: Path, subset: str, example_id: str) -> list[Path]:
    base = root / "data" / "agentbank" / "processed" / "dags"
    return [
        base / "auto.v2" / subset / f"{example_id}.json",  # preferred
        base / "auto.v1" / subset / f"{example_id}.json",  # fallback
    ]


@st.cache_data(show_spinner=False)
def load_kqa_pro_data(root: Path) -> dict[str, Any]:
    """Load KQA Pro train and val data."""
    data_path = root / "data" / "kqa_pro"
    train_path = data_path / "train.json"
    val_path = data_path / "val.json"

    data = []
    if train_path.exists():
        with train_path.open("r", encoding="utf-8") as f:
            data.extend(json.load(f))
    if val_path.exists():
        with val_path.open("r", encoding="utf-8") as f:
            data.extend(json.load(f))

    # Create a mapping from index to data item
    return {str(i): item for i, item in enumerate(data)}


def get_kqa_pro_data(root: Path) -> dict[str, Any]:
    """Get KQA Pro data from session state or load it."""
    if "kqa_pro_data" not in st.session_state:
        st.session_state.kqa_pro_data = load_kqa_pro_data(root)
    return st.session_state.kqa_pro_data


@st.cache_data(show_spinner=False)
def load_dag_json(root: Path, subset: str, example_id: str) -> Optional[dict[str, Any]]:
    """Load DAG JSON for an example, trying auto.v2 then auto.v1 for AgentBank, or converting from KQA Pro program.

    Returns the parsed JSON or None if not found/parseable.
    """
    if subset == "kqapro":
        # Load KQA Pro data (cached) and convert program to DAG format
        kqa_data = get_kqa_pro_data(root)

        if example_id not in kqa_data:
            return None

        item = kqa_data[example_id]
        program = item["program"]

        # Convert program to DAG format
        nodes = []
        for j, step in enumerate(program):
            node = {
                "index": j,
                "parents": step["dependencies"],
                "children": [],
                "action": step["function"],  # Store function in action field
                "args": str(step["inputs"]),  # Store inputs in args field
                "thought": "",  # KQA Pro doesn't have thought
                "observation": "",  # KQA Pro doesn't have observation
            }
            nodes.append(node)

        # Compute children
        for node in nodes:
            for p in node["parents"]:
                if p >= 0 and p < len(nodes):
                    nodes[p]["children"].append(node["index"])

        return {
            "id": example_id,
            "query": item["question"],
            "dag": nodes,
        }
    else:
        # AgentBank datasets - use existing file-based loading
        for p in _dag_candidate_paths(root, subset, example_id):
            if p.exists():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return None
        return None
def get_max_in_degree(dag_list: list[dict[str, Any]]) -> int:
    """Compute maximum in-degree across nodes of a DAG.

    Assumes each node is a dict with key `parents` as a list of indices.
    """
    if not dag_list:
        return 0
    return max((len(n.get("parents", [])) for n in dag_list), default=0)


def get_max_out_degree(dag_list: list[dict[str, Any]]) -> int:
    """Compute maximum out-degree across nodes of a DAG.

    Assumes each node is a dict with key `children` as a list of indices.
    """
    if not dag_list:
        return 0
    return max((len(n.get("children", [])) for n in dag_list), default=0)


@st.cache_data(show_spinner=False)
def attach_degree_features(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Attach `max_in_degree` and `max_out_degree` columns by reading DAG files.

    The function is cached; it returns a new DataFrame.
    """
    out = df.copy()
    max_in: list[Optional[int]] = []
    max_out: list[Optional[int]] = []

    # Create progress bar for the main app
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_rows = len(out)

    for i, (_, row) in enumerate(out.iterrows()):
        # Update progress
        progress = (i + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Processing example {i + 1}/{total_rows} ({progress:.1%})")

        subset = str(row["subset"]) if "subset" in row else None
        ex_id = str(row["id"]) if "id" in row else None
        if not subset or not ex_id:
            max_in.append(None)
            max_out.append(None)
            continue
        dag_json = load_dag_json(root, subset, ex_id)
        dag_list = []
        if isinstance(dag_json, dict):
            # Expect {"id": ..., "dag": [...]} or {"dag": [...]} formats
            dag_list = dag_json.get("dag") or dag_json.get("nodes") or []
        elif isinstance(dag_json, list):
            dag_list = dag_json
        max_in.append(get_max_in_degree(dag_list))
        max_out.append(get_max_out_degree(dag_list))

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    out["max_in_degree"] = max_in
    out["max_out_degree"] = max_out
    return out


def feature_options(df: pd.DataFrame) -> list[str]:
    cols = [c for c in CANDIDATE_FEATURES if c in df.columns]
    # Ensure numeric features
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols


def kde_plot(df: pd.DataFrame, features: list[str], by_subset: Optional[list[str]] = None) -> None:
    n = max(1, len(features))
    cols = 3
    rows = (n + cols - 1) // cols

    sns.set_theme()
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    # Normalize axes to a flat list of Axes
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    try:
        # Try KDE plotting first
        for ax, feat in zip(axes, features):
            if by_subset:
                for s in by_subset:
                    sub = df[(df["subset"] == s) & (df[feat].notna())]
                    if len(sub) < 2:
                        continue
                    sns.kdeplot(
                        data=sub,
                        x=feat,
                        fill=True,
                        common_norm=False,
                        alpha=0.3,
                        ax=ax,
                        label=s,
                        color=SUBSET_COLORS.get(s),
                    )
                ax.legend()
                ax.set_title(f"KDE: {feat}")
                ax.grid(True, alpha=0.2)
            else:
                # Single dataset KDE
                sub = df[df[feat].notna()]
                if len(sub) >= 2:
                    sns.kdeplot(
                        data=sub,
                        x=feat,
                        fill=True,
                        alpha=0.5,
                        ax=ax,
                    )
                ax.set_title(f"KDE: {feat}")
                ax.grid(True, alpha=0.2)

        # Hide leftover axes
        for j in range(len(features), len(axes)):
            axes[j].axis("off")
        st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.warning(f"KDE plotting failed: {e}. Falling back to histograms.")

        # Histogram fallback
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if hasattr(axes, "flatten"):
            axes = axes.flatten()
        else:
            axes = [axes]

        for ax, feat in zip(axes, features):
            if by_subset:
                for s in by_subset:
                    sub = df[(df["subset"] == s) & (df[feat].notna())]
                    if len(sub) > 0:
                        ax.hist(sub[feat], bins=20, alpha=0.4, label=s, color=SUBSET_COLORS.get(s))
                ax.legend()
            else:
                sub = df[df[feat].notna()]
                if len(sub) > 0:
                    ax.hist(sub[feat], bins=20, alpha=0.6)
            ax.set_title(f"Histogram: {feat}")
            ax.grid(True, alpha=0.2)

        # Hide leftover axes
        for j in range(len(features), len(axes)):
            axes[j].axis("off")
        st.pyplot(fig, clear_figure=True)


def scatter_plot_interactive(df: pd.DataFrame, x: str, y: str, cross_dataset: bool):
    """Render interactive scatter using Plotly with marker size based on point density."""
    # Create aggregated data with counts for overlapping points
    if cross_dataset:
        # Group by x, y, and subset to maintain colors
        agg_df = df.groupby([x, y, "subset"]).agg({
            "id": ["count", "first"],  # count for size, first for hover info
        }).reset_index()
        agg_df.columns = [x, y, "subset", "count", "sample_id"]

        # Calculate dataset sizes for normalization
        dataset_sizes = df.groupby("subset").size().to_dict()

        # Add normalized proportion (percentage within each dataset)
        agg_df["proportion"] = agg_df.apply(
            lambda row: (row["count"] / dataset_sizes[row["subset"]]) * 100,
            axis=1
        )

        color = "subset"
        hover_data = {"count": True, "proportion": ":.3f", "sample_id": True}
        symbol = "subset"
        size_col = "proportion"
        size_label = "Proportion (%)"
    else:
        # Group by x, y only for single dataset
        agg_df = df.groupby([x, y]).agg({
            "id": ["count", "first"],
            "subset": "first",  # Keep subset info for consistency
        }).reset_index()
        agg_df.columns = [x, y, "count", "sample_id", "subset"]

        # For single dataset, calculate proportion of total dataset
        total_size = len(df)
        agg_df["proportion"] = (agg_df["count"] / total_size) * 100

        color = None
        hover_data = {"count": True, "proportion": ":.3f", "sample_id": True}
        symbol = None
        size_col = "proportion"
        size_label = "Proportion (%)"

    fig = px.scatter(
        agg_df,
        x=x,
        y=y,
        color=color,
        color_discrete_map=SUBSET_COLORS if color else None,
        hover_data=hover_data,
        opacity=0.7,
        symbol=symbol,
        size=size_col,
        size_max=30,
        labels={
            "count": "Point Count",
            "proportion": size_label,
            "sample_id": "Sample ID"
        }
    )

    # Update hover template to show both count and proportion
    if cross_dataset:
        hover_template = (
            f"<b>%{{fullData.name}}</b><br>"
            f"{x}: %{{x}}<br>"
            f"{y}: %{{y}}<br>"
            f"Count: %{{customdata[0]}}<br>"
            f"Proportion: %{{customdata[1]:.3f}}%<br>"
            f"Sample ID: %{{customdata[2]}}<extra></extra>"
        )
    else:
        hover_template = (
            f"{x}: %{{x}}<br>"
            f"{y}: %{{y}}<br>"
            f"Count: %{{customdata[0]}}<br>"
            f"Proportion: %{{customdata[1]:.3f}}%<br>"
            f"Sample ID: %{{customdata[2]}}<extra></extra>"
        )

    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="black")),
        hovertemplate=hover_template
    )

    title_suffix = "relative proportion within each dataset" if cross_dataset else "proportion of dataset"
    fig.update_layout(
        height=600,
        legend_title_text="Subset" if cross_dataset else None,
        title=f"Scatter Plot: {x} vs {y}<br><sub>Marker size indicates {title_suffix}</sub>"
    )

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # Show aggregation statistics
    with st.expander("Point Density Statistics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Points", len(df))
        with col2:
            st.metric("Unique Coordinates", len(agg_df))
        with col3:
            max_count = agg_df["count"].max()
            st.metric("Max Points per Coordinate", max_count)

        if cross_dataset:
            # Show dataset-specific statistics
            st.markdown("**Dataset Breakdown:**")
            for dataset in df["subset"].unique():
                subset_df = df[df["subset"] == dataset]
                subset_agg = agg_df[agg_df["subset"] == dataset]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**{dataset.upper()}**")
                with col2:
                    st.write(f"Total: {len(subset_df)}")
                with col3:
                    st.write(f"Unique: {len(subset_agg)}")
                with col4:
                    max_prop = subset_agg["proportion"].max() if len(subset_agg) > 0 else 0
                    st.write(f"Max prop: {max_prop:.3f}%")

        if max_count > 1:
            norm_text = "normalized by dataset size" if cross_dataset else "as percentage of total dataset"
            st.info(f"Data aggregated from {len(df)} points to {len(agg_df)} unique coordinates. "
                   f"Marker size represents the proportion of overlapping points {norm_text}.")


def ui_render_example(root: Path, subset: str, example_id: str, features_row: pd.Series) -> None:
    """Render a single example's DAG and metadata in the dashboard."""
    st.subheader(f"DAG for {example_id}")
    dag_json = load_dag_json(root, subset, example_id)
    if not dag_json:
        st.warning("DAG JSON not found.")
        return

    # Display query text if present
    query = dag_json.get("query") or dag_json.get("prompt") or "(no query)"
    st.markdown("**Query:**")
    st.text(query)

    # Extract dag list
    if isinstance(dag_json, dict):
        dag_list = dag_json.get("dag") or dag_json.get("nodes") or []
    elif isinstance(dag_json, list):
        dag_list = dag_json
    else:
        dag_list = []

    # Top-level metadata
    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.metric("Nodes", len(dag_list))
    with meta_cols[1]:
        st.metric("Max in-degree", get_max_in_degree(dag_list))
    with meta_cols[2]:
        st.metric("Max out-degree", get_max_out_degree(dag_list))

    # Render Graphviz DAG using helper (adapted from dag_visualizer.create_dag_graph)
    def _create_dag_graph(dag_data: list[dict[str, Any]]) -> graphviz.Digraph:
        dot = graphviz.Digraph(comment="Reasoning DAG")
        dot.attr(rankdir="TB")
        dot.attr("node", shape="box", style="filled", fillcolor="lightblue")

        # Check if this is KQA Pro data (no thought/observation fields)
        is_kqa_pro = dag_data and not any(node.get("thought", "").strip() for node in dag_data)

        if not is_kqa_pro:
            # Input node for AgentBank datasets
            dot.node("input", "Input Text", shape="ellipse", fillcolor="lightyellow")

        # Nodes
        for node in dag_data:
            index = node.get("index")
            thought = str(node.get("thought", "")).strip()
            action = str(node.get("action", "")).strip()
            args = str(node.get("args", "")).strip()
            observation = str(node.get("observation", "")).strip()

            label_parts = [f"[{index}]"]

            if is_kqa_pro:
                # KQA Pro format: only function and inputs
                if action:
                    label_parts.append(f"Function: {action}")
                if args:
                    if len(args) > 60:
                        args = args[:57] + "..."
                    label_parts.append(f"Inputs: {args}")
            else:
                # AgentBank format: thought, action, args, observation
                if thought:
                    if len(thought) > 120:
                        thought = thought[:117] + "..."
                    label_parts.append(f"Thought: {thought}")
                if action:
                    label_parts.append(f"Action: {action}")
                if args:
                    if len(args) > 100:
                        args = args[:97] + "..."
                    label_parts.append(f"Args: {args}")
                if observation:
                    if len(observation) > 100:
                        observation = observation[:97] + "..."
                    label_parts.append(f"Observation: {observation}")

            label = "\n".join(label_parts)
            if action == "finish":
                dot.node(str(index), label, fillcolor="lightgreen")
            else:
                dot.node(str(index), label)

        if not is_kqa_pro:
            # Edges from input (parents == [-1]) for AgentBank datasets
            try:
                input_children = [n["index"] for n in dag_data if -1 in n.get("parents", [])]
            except Exception:
                input_children = []
            for c in input_children:
                dot.edge("input", str(c))

        # Add edges between nodes
        for node in dag_data:
            idx = node.get("index")
            for child in node.get("children", []) or []:
                dot.edge(str(idx), str(child))

        return dot

    # Show graph
    try:
        dot = _create_dag_graph(dag_list)
        st.graphviz_chart(dot)
    except Exception as e:
        st.warning(f"Graphviz rendering failed: {e}")
        st.text("Falling back to raw JSON view below.")

    # Show per-example axes/feature values
    st.markdown("**Feature values:**")
    st.dataframe(features_row.astype(str))

    # Full JSON collapsible view for debugging
    with st.expander("Raw DAG JSON"):
        st.json(dag_json)


def main() -> None:
    st.set_page_config(page_title="Task-Space v1 Dashboard", layout="wide")
    st.title("Task-Space Characterization (v1)")

    root = find_project_root()

    # Sidebar controls
    st.sidebar.header("Controls")
    analysis_type = st.sidebar.selectbox("Analysis Type", ["KDE", "2D Scatter"], index=1)
    mode = st.sidebar.radio("Mode", ["Cross-dataset", "In-dataset"], index=0)
    in_dataset = None
    if mode == "In-dataset":
        in_dataset = st.sidebar.selectbox("Dataset", DATASETS, index=0)

    # Load data
    dataframes = []

    for dataset in DATASETS:
        try:
            df = load_subset_csv(root, dataset)
            dataframes.append(df)
        except Exception as e:
            st.error(f"Failed to load {dataset.upper()} CSV: {e}")
            return

    df_all = pd.concat(dataframes, ignore_index=True)

    # Pre-load KQA Pro data if needed for better progress tracking
    if "kqapro" in [df["subset"].iloc[0] for df in dataframes if len(df) > 0]:
        st.subheader("Loading KQA Pro data")
        st.info("Pre-loading KQA Pro dataset for efficient processing...")
        with st.spinner("Loading KQA Pro train and val files..."):
            kqa_data = get_kqa_pro_data(root)
        st.success(f"Loaded {len(kqa_data)} KQA Pro examples")

    # Attach on-the-fly degree features (cached)
    st.subheader("Computing degree features from DAGs")
    st.info("This step computes max in-degree and out-degree for each example. Results are cached for subsequent runs.")
    df_all = attach_degree_features(df_all, root)

    # Determine working subset of data
    if mode == "Cross-dataset":
        df_work = df_all.copy()
        available_subsets = DATASETS
    else:
        df_work = df_all[df_all["subset"] == in_dataset].copy()
        available_subsets = [in_dataset] if in_dataset else []

    # Feature selection
    numeric_features = feature_options(df_work)

    if analysis_type == "KDE":
        chosen_feats = st.multiselect(
            "Features (1D)",
            options=numeric_features,
            default=[f for f in [
                "branching_ratio", "merging_ratio", "max_width", "max_depth",
                "avg_gap_excluding_input", "avg_headroom",
            ] if f in numeric_features][:3],
        )
        if not chosen_feats:
            st.info("Select at least one feature.")
            return
        kde_plot(df_work, chosen_feats, by_subset=available_subsets if mode == "Cross-dataset" else None)

    else:
        # 2D Scatter
        left, right = st.columns(2)
        with left:
            x_feat = st.selectbox("X feature", options=numeric_features, index=max(0, numeric_features.index("input_len")) if "input_len" in numeric_features else 0)
        with right:
            y_feat = st.selectbox("Y feature", options=numeric_features, index=max(0, numeric_features.index("workflow_len")) if "workflow_len" in numeric_features else (numeric_features.index("max_in_degree") if "max_in_degree" in numeric_features else 0))

        scatter_plot_interactive(df_work, x_feat, y_feat, cross_dataset=(mode == "Cross-dataset"))

        # Provide a manual selector
        with st.expander("Select an instance to view DAG"):
            # Build choices from current view
            choices = df_work["id"].astype(str).tolist() if "id" in df_work.columns else []
            if choices:
                sel_id = st.selectbox("Instance ID", options=choices)
                if sel_id:
                    sel_row = df_work[df_work["id"].astype(str) == sel_id].head(1)
                    if not sel_row.empty:
                        sel_subset = str(sel_row.iloc[0]["subset"]) if "subset" in sel_row.columns else "gsm8k"
                        ui_render_example(root, sel_subset, sel_id, sel_row.iloc[0])
            else:
                st.info("No instances available in current view.")

    # Data preview and help
    with st.expander("Data preview"):
        st.dataframe(df_work.head(50))
    with st.expander("Help"):
        st.markdown(
            "- **AgentBank CSVs**: data/agentbank/processed/{gsm8k,hotpotqa}_values.v1.csv\n"
            "- **AgentBank DAGs**: data/agentbank/processed/dags/auto.v2/<subset>/<id>.json (fallback to auto.v1)\n"
            "- **KQA Pro CSV**: data/kqa_pro/processed/kqapro_values.v1.csv\n"
            "- **KQA Pro Data**: data/kqa_pro/{train,val}.json (converted to DAG format in-memory)\n"
            "- Point-click on scatter uses 'streamlit-plotly-events' if installed."
        )


if __name__ == "__main__":
    main()
