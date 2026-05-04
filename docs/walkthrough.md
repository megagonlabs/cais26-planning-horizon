# Walkthrough

## Overview

This walkthrough shows how to obtain the main results in the paper using the scripts in `scripts/batch/`.

> [!TIP]
> Before launching a longer batch, use `scripts/example_run_kqa_pro.py` as a quick sanity check that your environment, API key, and `KQA Pro` assets are in working order. See "Quick sanity check before the main run" below for details.

> [!IMPORTANT]
> The backbone LLM APIs used in these experiments are not fully deterministic in practice. Even with the same prompts and nominal model names, provider-side model updates, backend routing, and sampling behavior can shift results slightly over time. You should expect reproduced numbers to be close to the paper, but obtaining exactly the same scores for every run can be difficult.

## Main batch entry points

Choose the wrapper that matches the experiment track you want to run:

| Script | Runs | Required setup |
| --- | --- | --- |
| `scripts/batch/batch_exp_kqa_pro_hydra.sh` | `KQA Pro` with `SH` and `FH` | [Code setup](setup/code.md), [`KQA Pro` data setup](setup/data.md) |
| `scripts/batch/batch_exp_atomic_kbqa_hydra.sh` | `Atomic KBQA` (`grailqa`, `webqsp`, `graphq`) with `SH` and `FH` | [Code setup](setup/code.md), [`Atomic KBQA` data setup](setup/data.md) |
| `scripts/batch/batch_exp_multiobj_hotpotqa_hydra.sh` | `Multi-objective HotpotQA` with `SH` and `FH` | [Code setup](setup/code.md), [`Multi-objective HotpotQA` data setup](setup/data.md) |

All three wrappers default to the OpenAI-backed paper path:
`gpt-4.1-mini`, non-strict configs, the full `test` split, and `10` Hydra
workers per experiment.

## High-level workflow

For the main paper results, the workflow is:

1. optionally run the single-question `KQA Pro` sanity check
2. run the three batch wrappers for the backbone model you want to reproduce
3. run one post-processing wrapper to apply soft evaluation and aggregate the
  resulting runs into dataset-level CSV files
4. copy the relevant `SH` / `FH` rows into the final paper table

For `Table 2`, run the batch wrappers without `--strict`. For `Table 4`, rerun
them with `--strict` and then repeat the same post-processing step.

## Batch parameters

In addition to the four parameters you listed, the batch wrappers expose
`--workers`, because the Hydra runner parallelizes episodes within each
experiment.

| Parameter | Default | Accepted values | Notes |
| --- | --- | --- | --- |
| `--llm` | `gpt-4.1-mini` | `gpt-4.1-mini`, `gpt-5-mini`, `qwen3-235b-instruct`, `gemini-flash-preview` | Applies to both the meta agent and worker agents |
| `--strict` | Off | flag | Uses the `*.strict` experiment config |
| `--num-episodes` | full | positive integer, `full` | Useful for sanity-check runs |
| `--parallelism` | `1` | positive integer | Runs that many experiment processes at once |
| `--workers` | `10` | positive integer | Hydra worker count per experiment |

When `--parallelism` is greater than `1`, set `CUDA_VISIBLE_DEVICES` to a
comma-separated GPU list before running the wrapper. Each batch process is
assigned to one visible GPU slot.

## Representative commands

For the optional one-question preflight, use the "Quick sanity check before the
main run" section below.

Then launch the dataset wrappers that correspond to the result family you want.
For example, this starts the non-strict `KQA Pro` batch for `Table 2`:

```bash
bash scripts/batch/batch_exp_kqa_pro_hydra.sh --llm gpt-4.1-mini
```

After the experiment batches finish, run the post-processing wrapper:

```bash
bash scripts/batch/batch_postprocess_main_results.sh
```

## Paper table mapping

The main batch wrappers and the shared post-processing step map to the paper tables as follows.

| Paper result | What to run | Scope |
| --- | --- | --- |
| `Table 2` | Run each dataset wrapper without `--strict`, then run `scripts/batch/batch_postprocess_main_results.sh` | Non-strict `SH` vs `FH` comparison across datasets and backbone LLMs |
| `Table 4` | Run each dataset wrapper with `--strict`, then run `scripts/batch/batch_postprocess_main_results.sh` again | Strict `SH` vs `FH` comparison across datasets and backbone LLMs |

## LLM and provider mapping

| `--llm` value | Hydra model override | Provider override |
| --- | --- | --- |
| `gpt-4.1-mini` | `gpt-4p1-mini.v1` | `openai` |
| `gpt-5-mini` | `gpt-5-mini.v1` | `openai` |
| `qwen3-235b-instruct` | `qwen3-235b-instruct` | `fireworks` |
| `gemini-flash-preview` | `gemini-flash-preview.v1` | `vertexai-openai` |

## Result post-processing

The main experiment runner writes a `result.jsonl` file and a sibling
`metrics.jsonl` file into each timestamped run directory under `results/`.
The new wrapper `scripts/batch/batch_postprocess_main_results.sh` is the
entry point for the next step.

It performs the two post-processing stages in order:

1. soft evaluation of the generated answers
2. aggregation of timestamped runs into one CSV per dataset

Under the hood, it uses `scripts/evaluate_results.sh` and
`scripts/collect_metrics.py`, but you usually do not need to call those
directly when reproducing the main paper tables.

When you run it without `--skip-eval`, make sure the repository root `.env`
file contains a valid `OPENAI_API_KEY`.

### What the wrapper updates

- `result.jsonl`: updated in place with per-example `evaluation` fields
- `metrics.jsonl`: updated with `correct_answers_<evaluation-model>_2.0`
- `tables/reproduction/*.csv`: one aggregated CSV per dataset by default

### Default usage

Run the wrapper with no dataset arguments to process the main paper datasets:

```bash
bash scripts/batch/batch_postprocess_main_results.sh
```

If you only want a subset, pass dataset IDs explicitly:

```bash
bash scripts/batch/batch_postprocess_main_results.sh kopl_kbqa/kqa_pro atomic_kbqa/grailqa
```

If the runs were already soft-evaluated and you only want to rebuild the CSVs,
use `--skip-eval` to avoid repeating API-backed judging:

```bash
bash scripts/batch/batch_postprocess_main_results.sh --skip-eval
```

### Reading the aggregated output

The generated CSVs contain one row per method directory under
`results/<dataset>/<split>/`, averaged across timestamped trials.

For the paper-facing tables:

- use the non-strict rows for `Table 2`
- use the strict rows for `Table 4`

## Post-hoc analyses

The notebooks and scripts for Sections `4.3`, `4.4`, and `4.5` are ready to run.

| Entry point | Paper target | Status |
| --- | --- | --- |
| `notebooks/sec4p3_topological-complexity-analysis.ipynb` | `Table 3` logistic-regression coefficients | Ready |
| `notebooks/sec4p4_tool-robustness.ipynb` | Section `4.4` tool-robustness discussion | Ready |
| `notebooks/sec4p5_repetitive-tool-calls.ipynb` | Section `4.5` repetitive-call discussion and case study | Ready |
| `scripts/analyze_repetitive_tool_calls.py` | Optional CSV export for the Section `4.5` notebook | Ready |

### Section 4.3 reproduction

Use the Section 4.3 notebook when you want the logistic-regression
coefficients for the topological-complexity analysis.

1. Open `notebooks/sec4p3_topological-complexity-analysis.ipynb` in VS Code.
2. Run all cells.
3. Inspect the Figure 3 task-space plot and the inline regression tables.

### Section 4.4 reproduction

Use `notebooks/sec4p4_tool-robustness.ipynb` when you want the low-vs-high
robustness accuracy deltas discussed in Section 4.4.

1. Open `notebooks/sec4p4_tool-robustness.ipynb` in VS Code.
2. Run all cells.
3. Inspect the notebook outputs:
   - a paper-style table of `FH` and `SH` accuracy deltas
   - a heatmap showing how much each planner drops when robustness is reduced
   - a compact comparison of where `SH` drops more than `FH`

This notebook reads the released `results/` directories directly, so no
precomputed CSV files are required.

### Section 4.5 reproduction

Use `notebooks/sec4p5_repetitive-tool-calls.ipynb` when you want the repeated
tool-call analysis discussed in Section 4.5.

1. Open `notebooks/sec4p5_repetitive-tool-calls.ipynb` in VS Code.
2. Run all cells.
3. Inspect the notebook outputs:
   - the high-robustness repetition-rate table
   - the low-robustness repetition-rate table
   - heatmaps of the `SH` minus `FH` repetition gap
   - an automatically selected released case study showing an `SH` loop next to
     an `FH` run of the same question

Like the Section 4.4 notebook, this notebook reads the released trajectories
directly from `results/`.


## Quick sanity check before the main run

If you want a fast preflight before the longer batch jobs, run one `KQA Pro`
question through `scripts/example_run_kqa_pro.py`.

### Prerequisites

Complete the following setup first:

- [Code setup](setup/code.md)
- [KQA Pro data setup](../data/kopl_kbqa/kqa_pro/README.md)

If you haven't already, run the following commands from the repository root:

```bash
bash data/kopl_kbqa/kqa_pro/download.sh
bash data/kopl_kbqa/kqa_pro/scripts/run_preprocessing.sh
```

### Run the example question

`scripts/example_run_kqa_pro.py` disables demonstrations by default for this
preflight (`--num-demonstrations 0`), so it does not require
`demonstration_candidates`. By default, it also streams each completed step as
soon as it finishes. Pass `--no-stream` if you prefer only the final recap.

Run the default SH preflight:

```bash
uv run python scripts/example_run_kqa_pro.py --problem "What is the street address of the California Institute of the Arts?"
# Answer: 24700 W McBean Pky, Valencia, CA, 91355-2397
```

To compare against the FH planner, pass the FH experiment config explicitly:

```bash
uv run python scripts/example_run_kqa_pro.py conf/experiment/kopl_kbqa/fh.v1.yaml --problem "What is the street address of the California Institute of the Arts?"
```

### What the script prints

By default, the script prints five sections in order:

| Section | Contents |
| --- | --- |
| `PROBLEM` | The user question passed on the command line |
| `RUNNING SH META AGENT...` or `RUNNING FH META AGENT...` | Planner startup banner |
| `LIVE TRAJECTORY` | Step-by-step action/observation updates as each step finishes |
| `EPISODE RESULTS` | Success flag, final answer, step count, and max-step status |
| `FULL TRAJECTORY RECAP` | One action/observation pair per recorded step |

The exact number of steps and the text of each observation depend on the model
output, but the overall structure should stay the same. If you pass
`--no-stream`, the live section is skipped and only the final trajectory recap
is printed.

### Expected failure modes

`scripts/example_run_kqa_pro.py` checks prerequisites before the planner starts.
Common setup issues produce direct errors instead of a deep stack trace:

- Missing `OPENAI_API_KEY`
- Missing `data/kopl_kbqa/kqa_pro/kb.json`
- Missing embedding pickle files under
  `data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5/`

If you hit one of those errors, rerun the setup commands above and verify that
the files exist on disk before starting the batch run.

## Next step

After the quick preflight works, launch the appropriate wrapper in
`scripts/batch/`. If you need finer-grained control than the wrappers expose,
move on to the raw Hydra runner in `scripts/run.py`.
