# Walkthrough

## Overview

This walkthrough shows how to obtain the main results in the paper using the scripts in `scripts/batch/`.

> [!TIP]
> Before launching a longer batch, use `scripts/example_run_kqa_pro.py` as a quick sanity check that your environment, API key, and `KQA Pro` assets are in working order. See "Quick sanity check before the main run" below for details.

> [!IMPORTANT]
> The backbone LLM APIs used in these experiments are not fully deterministic in practice. Even with the same prompts and nominal model names, provider-side model updates, backend routing, and sampling behavior can shift results slightly over time. You should expect reproduced numbers to be close to the paper, but obtaining exactly the same scores for every run can be difficult.


## Quick sanity check before the main run

If you want a quick sanity check before the longer batch jobs, run one `KQA Pro` question through `scripts/example_run_kqa_pro.py`.

### Prerequisites

Complete the following setup first:

- [Code setup](setup/code.md)
- [KQA Pro data setup](../data/kopl_kbqa/kqa_pro/README.md)

If you haven't already, run the following commands from the repository root:

```bash
uv sync

bash data/kopl_kbqa/kqa_pro/download.sh
bash data/kopl_kbqa/kqa_pro/scripts/run_preprocessing.sh
```

### Run the example question

Run the default SH config on a sample question:

```bash
uv run python scripts/example_run_kqa_pro.py --problem "What is the street address of the California Institute of the Arts?"
# Answer: 24700 W McBean Pky, Valencia, CA, 91355-2397
```

To compare against the FH planner, pass the FH experiment config explicitly:

```bash
uv run python scripts/example_run_kqa_pro.py conf/experiment/kopl_kbqa/fh.v1.yaml --problem "What is the street address of the California Institute of the Arts?"
```

Note: The main experiments use a few in-context demonstrations, but `scripts/example_run_kqa_pro.py` disables demonstrations by default (i.e., `--num-demonstrations 0`).

### What the script prints

By default, the script prints five sections in order:

| Section | Contents |
| --- | --- |
| `PROBLEM` | The user question passed on the command line |
| `RUNNING SH META AGENT...` or `RUNNING FH META AGENT...` | Planner startup banner |
| `LIVE TRAJECTORY` | Step-by-step action/observation updates as each step finishes |
| `EPISODE RESULTS` | Success flag, final answer, step count, and max-step status |
| `FULL TRAJECTORY RECAP` | One action/observation pair per recorded step |


### Common setup issues

`scripts/example_run_kqa_pro.py` checks prerequisites before the planner starts. If you hit one of the following errors, rerun the setup commands above and verify that the files exist on disk.

- Missing `OPENAI_API_KEY`
- Missing `data/kopl_kbqa/kqa_pro/kb.json`
- Missing embedding pickle files under
  `data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5/`


## Main experiment scripts

`scripts/run.py` is the main entry point for running the experiments in the paper. It uses Hydra to manage experiment configurations and parallelization.

For convenience, we provide wrapper scripts in `scripts/batch/batch_exp_*.sh` that call `scripts/run.py` with the appropriate configs for each dataset and backbone LLM. These wrappers also expose a few common parameters as command-line flags (e.g., `--llm` to select the backbone LLM across all experiments).

**High-level workflow:** For the main paper results, the workflow is:

1. run the three batch wrappers for the backbone model you want to reproduce
2. run one post-processing wrapper to apply soft evaluation and aggregate the resulting runs into dataset-level CSV files. See "Details on result post-processing" below for more on this step.

**Example:** For example, to start the non-strict `KQA Pro` batch for `Table 2`:

```bash
bash scripts/batch/batch_exp_kqa_pro_hydra.sh --llm gpt-4.1-mini
```

After the experiment batches finish, run the post-processing wrapper:

```bash
bash scripts/batch/batch_postprocess_main_results.sh
```

## Details on the main batch wrappers

Each wrapper corresponds to one dataset (or group of datasets) in the paper. They all call `scripts/run.py` under the hood, but with different Hydra configs.

| Script | Runs | Required setup |
| --- | --- | --- |
| `scripts/batch/batch_exp_kqa_pro_hydra.sh` | `KQA Pro` with `SH` and `FH` | [Code setup](setup/code.md), [`KQA Pro` data setup](setup/data.md) |
| `scripts/batch/batch_exp_atomic_kbqa_hydra.sh` | `Atomic KBQA` (`grailqa`, `webqsp`, `graphq`) with `SH` and `FH` | [Code setup](setup/code.md), [`Atomic KBQA` data setup](setup/data.md) |
| `scripts/batch/batch_exp_multiobj_hotpotqa_hydra.sh` | `Multi-objective HotpotQA` with `SH` and `FH` | [Code setup](setup/code.md), [`Multi-objective HotpotQA` data setup](setup/data.md) |

All three wrappers default to the OpenAI `gpt-4.1-mini`, non-strict configs, the full `test` split, and `10` Hydra workers per experiment.


**Parameters:**

| Parameter | Default | Accepted values | Notes |
| --- | --- | --- | --- |
| `--llm` | `gpt-4.1-mini` | `gpt-4.1-mini`, `gpt-5-mini`, `qwen3-235b-instruct`, `gemini-flash-preview` | Applies to both the meta agent and worker agents |
| `--strict` | Off | flag | Uses the `*.strict` experiment config. Run without this flag for `Table 2` and with this flag for `Table 4`. |
| `--num-episodes` | full | positive integer, `full` | Useful for sanity-check runs |
| `--parallelism` | `1` | positive integer | Runs that many experiment processes at once |
| `--workers` | `10` | positive integer | Hydra worker count per experiment |

When `--parallelism` is greater than `1`, set `CUDA_VISIBLE_DEVICES` to a comma-separated GPU list before running the wrapper. Each batch process is assigned to one visible GPU slot. Example:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/batch/batch_exp_kqa_pro_hydra.sh --llm gpt-4.1-mini --parallelism 4
```


## Details on result post-processing

The main experiment runner `scripts/run.py` writes a `result.jsonl` file and a sibling `metrics.jsonl` file into each timestamped run directory under `results/`. The wrapper `scripts/batch/batch_postprocess_main_results.sh` is the entry point for post-processing these raw results into the final paper tables.

It performs the two post-processing stages in order:

1. soft evaluation of the generated answers
2. aggregation of timestamped runs into one CSV per dataset

Under the hood, it uses `scripts/evaluate_results.sh` and `scripts/collect_metrics.py`.

When you run it without `--skip-eval`, make sure the repository root `.env` file contains a valid `OPENAI_API_KEY`.

**What the wrapper updates:**

- `result.jsonl`: updated in place with per-example `evaluation` fields
- `metrics.jsonl`: updated with `correct_answers_<evaluation-model>_2.0`
- `tables/reproduction/*.csv`: one aggregated CSV per dataset by default

**Default usage:**

Run the wrapper with no dataset arguments to process the main paper datasets:

```bash
bash scripts/batch/batch_postprocess_main_results.sh
```

If you only want a subset, pass dataset IDs explicitly:

```bash
bash scripts/batch/batch_postprocess_main_results.sh kopl_kbqa/kqa_pro atomic_kbqa/grailqa
```

If the runs were already soft-evaluated and you only want to rebuild the CSVs, use `--skip-eval` to avoid repeating API-backed judging:

```bash
bash scripts/batch/batch_postprocess_main_results.sh --skip-eval
```

**Reading the aggregated output:** The generated CSVs contain one row per method directory under `results/<dataset>/<split>/`, averaged across timestamped trials.

- use the non-strict rows for `Table 2`
- use the strict rows for `Table 4`


## Paper table mapping

The main batch wrappers and the shared post-processing step map to the paper tables as follows.

| Paper result | What to run | Scope |
| --- | --- | --- |
| `Table 2` | Run each dataset wrapper without `--strict`, then run `scripts/batch/batch_postprocess_main_results.sh` | Non-strict `SH` vs `FH` comparison across datasets and backbone LLMs |
| `Table 4` | Run each dataset wrapper with `--strict`, then run `scripts/batch/batch_postprocess_main_results.sh` again | Strict `SH` vs `FH` comparison across datasets and backbone LLMs |

## Post-hoc analyses

The notebooks and scripts for Sections `4.3`, `4.4`, and `4.5` are independent of the main batch wrappers, but they rely on the output of the main experiments.  The output can be processed by the Jupyter notebooks in `notebooks/` to reproduce the analyses and figures in those sections.

| Entry point | Paper target |
| --- | --- |
| `notebooks/sec4p3_topological-complexity-analysis.ipynb` | `Table 3` logistic-regression coefficients |
| `notebooks/sec4p4_tool-robustness.ipynb` | Section `4.4` tool-robustness discussion |
| `notebooks/sec4p5_repetitive-tool-calls.ipynb` | Section `4.5` repetitive-call discussion and case study |

### Section 4.3 reproduction

Use the Section 4.3 notebook when you want the logistic-regression coefficients for the topological-complexity analysis.

1. Open `notebooks/sec4p3_topological-complexity-analysis.ipynb` in VS Code.
2. Run all cells.
3. Inspect the Figure 3 task-space plot and the inline regression tables.

### Section 4.4 reproduction

Use `notebooks/sec4p4_tool-robustness.ipynb` when you want the low-vs-high robustness accuracy deltas discussed in Section 4.4.

1. Open `notebooks/sec4p4_tool-robustness.ipynb` in VS Code.
2. Run all cells.
3. Inspect the notebook outputs:
   - a paper-style table of `FH` and `SH` accuracy deltas
   - a heatmap showing how much each planner drops when robustness is reduced
   - a compact comparison of where `SH` drops more than `FH`

### Section 4.5 reproduction

Use `notebooks/sec4p5_repetitive-tool-calls.ipynb` when you want the repeated tool-call analysis discussed in Section 4.5.

1. Open `notebooks/sec4p5_repetitive-tool-calls.ipynb` in VS Code.
2. Run all cells.
3. Inspect the notebook outputs:
   - the high-robustness repetition-rate table
   - the low-robustness repetition-rate table
   - heatmaps of the `SH` minus `FH` repetition gap
   - an automatically selected released case study showing an `SH` loop next to
     an `FH` run of the same question
