# Do Agents Need to Plan Step-by-Step? Rethinking Planning Horizon in Data-Centric Tool Calling

Official codebase for the paper *Do Agents Need to Plan Step-by-Step? Rethinking Planning Horizon in Data-Centric Tool Calling*.

The project compares single-step horizon (SH) planning and full-horizon (FH) planning for data-centric tool-calling tasks such as KBQA and multi-hop QA. The core finding is that, for well-defined data-centric tasks, FH planning with lazy replanning can match SH planning while using substantially fewer tokens.

Codebase DOI: [10.5281/zenodo.20116460](https://doi.org/10.5281/zenodo.20116460)

## Quick start

See [`docs/walkthrough.md`](docs/walkthrough.md). This walkthrough includes:

1. Environment setup instructions ([Code setup](docs/setup/code.md) and [Data setup](docs/setup/data.md))
2. A quick demo of running one KQA Pro question against the KoPL worker-agent (recommended for quick sanity check before the main run)
3. A walkthrough of the main experiment run, including mappings between the paper and the codebase.

## Repository guide

See [`docs/system-design.md`](docs/system-design.md) for an overview of the system architecture and design. The codebase is organized as follows:

| Path | What it contains |
| --- | --- |
| `src/planning/agents/` | Meta agents, worker agents, shared agent abstractions |
| `planning/environment/` | Environment orchestration and executable registries |
| `planning/services/` | LLM provider integrations and routing |
| `planning/tools/` | Task-specific tool implementations and wrappers |
| `conf/` | [Hydra](https://hydra.cc/) configurations by dataset and planner setup |
| `scripts/` | Entry points for single runs and batch experiments |

## Environment

The experiments in the paper were run on the following environment.

| Component | Specification |
| --- | --- |
| OS | Ubuntu 22.04.2 LTS (GNU/Linux 5.15.0-78-generic x86\_64) |
| GPU | NVIDIA A100-SXM4-80GB (≥5 GB GPU memory required for sentence embedding) |
| Python | 3.11 (3.11+ expected to work; not tested on earlier versions) |
| Virtuoso (for Atomic KBQA experiments) | [Open Source Edition](https://vos.openlinksw.com/owiki/wiki/VOS) v7.2.5 |

**RAM and disk requirements** vary by experiment:

- **KoPL/KQA-Pro and atomic KBQA**: ≥10 GB RAM, ≥20 GB disk
- **Multi-hop HotpotQA**: ≥10 GB RAM, ≥20 GB disk (Wikipedia dump and [Pyserini prebuilt indexes](https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md))
- **Freebase-backed experiments**: ≥100 GB RAM recommended (see [Freebase-Setup](https://github.com/dki-lab/Freebase-Setup))
    - Note: The Freebase SPARQL executor uses `pyodbc`, which requires the `unixodbc` system library. On Linux (Ubuntu), this is bundled inside the `pyodbc` PyPI wheel and no extra step is needed. On macOS, install it separately by `brew install unixodbc`.

Python dependencies are listed in `pyproject.toml` and can be installed with `uv` (See [Code setup](docs/setup/code.md) for instructions).

**Approximate per-run usage** for the representative OpenAI settings is shown below. Values are averaged over three runs. `Atomic KBQA` averages over `GrailQA`, `GraphQ`, and `WebQSP`. The reported runtime is the total time summed across episodes, so elapsed wall-clock time can be lower when executed in parallel (for example, `10` workers can approach roughly `1/10` of the listed time).

| Experiment | Planning Horizon | Model | Runtime | Prompt tokens | Completion tokens |
| --- | --- | --- | --- | --- | --- |
| `KoPL/KQA-Pro` | `SH` | `gpt-4p1-mini` | `2h 50m` | `111M` | `184k` |
| `KoPL/KQA-Pro` | `SH` | `gpt-5-mini` | `18h 28m` | `90M` | `3M` |
| `KoPL/KQA-Pro` | `FH` | `gpt-4p1-mini` | `2h 17m` | `41M` | `321k` |
| `KoPL/KQA-Pro` | `FH` | `gpt-5-mini` | `7h 37m` | `20M` | `2M` |
| `Atomic KBQA` | `SH` | `gpt-4p1-mini` | `3h 33m` | `10M` | `58k` |
| `Atomic KBQA` | `SH` | `gpt-5-mini` | `6h 37m` | `9M` | `1M` |
| `Atomic KBQA` | `FH` | `gpt-4p1-mini` | `3h 25m` | `5M` | `112k` |
| `Atomic KBQA` | `FH` | `gpt-5-mini` | `4h 7m` | `4M` | `752k` |
| `Multi-objective HotpotQA` | `SH` | `gpt-4p1-mini` | `9h 48m` | `32M` | `223k` |
| `Multi-objective HotpotQA` | `SH` | `gpt-5-mini` | `64h 12m` | `51M` | `10M` |
| `Multi-objective HotpotQA` | `FH` | `gpt-4p1-mini` | `11h 1m` | `22M` | `1M` |
| `Multi-objective HotpotQA` | `FH` | `gpt-5-mini` | `27h 0m` | `29M` | `5M` |

Using the current list prices (as of April 2026) for OpenAI models, the estimated cost per run is:

| Experiment | `gpt-4.1-mini` (I/O = $0.40/$1.60 [per 1M tokens]) | `gpt-5-mini` (I/O = $0.25/$2.00 [per 1M tokens]) |
| --- | --- | --- |
| `KoPL/KQA-Pro` | about `$17`/`$45` per run (`FH`/`SH`) | about `$9`/`$23` per run (`FH`/`SH`) |
| `Atomic KBQA` | about `$2`/`$4` per run (`FH`/`SH`) | about `$3`/`$4` per run (`FH`/`SH`) |
| `Multi-objective HotpotQA` | about `$10`/`$13` per run (`FH`/`SH`) | about `$17`/`$33` per run (`FH`/`SH`) |

Actual costs can be lower because of input token caching.

## Cautions and Troubleshooting

- **Minor reproducibility drift:** Exact numbers can vary slightly over time because of non-determinism in the underlying LLMs and external services.
- **API connection errors / rate limits:** When you encounter problems like connection, first confirm that the relevant key in `.env` is valid, then retry after a short wait. If you are running multiple experiments in parallel, reduce the worker count or overall concurrency.
- **KoPL compatibility (KQA Pro):** the `KoPL` track requires the fork at [`notani/KoPL`](https://github.com/notani/KoPL), not the upstream `THU-KEG/KoPL` repository. `uv sync` already installs the correct fork from `pyproject.toml`.
- **Freebase / Virtuoso latency (Atomic KBQA):** If your endpoint is not responsive, follow the sanity checks in [`docs/setup/code.md`](docs/setup/code.md) and the external [`Freebase-Setup`](https://github.com/dki-lab/Freebase-Setup) guide before rerunning the experiments.
- **Pyserini first-run failures (Multi-objective HotpotQA):** Make sure the prebuilt indexes downloaded successfully under `~/.cache/pyserini/indexes/`. Missing or partially downloaded caches cause first-run errors. See the official [`Pyserini` prebuilt-index guide](https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md) for the expected cache layout and available indexes.

 the expected cache layout and available indexes.

## Citation

If you use this repository, please cite the paper:

```bibtex
@inproceedings{otani-etal-2026-do,
  title     = {{Do Agents Need to Plan Step-by-Step? Rethinking Planning Horizon in Data-Centric Tool Calling}},
  author    = {Otani, Naoki and Bhutani, Nikita and Kim, Hannah and Zhang, Dan and Hruschka, Estevam},
  booktitle = {Proceedings of the First ACM Conference on AI and Agentic Systems (CAIS)},
  year      = {2026},
}
```

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE).

## Used Datasets

All datasets used in this repository are listed below (including their copyright holders and the license information). We modify the datasets to run with our system but do not redistribute the datasets themselves.

| ID  | OSS Component Name | Modified | Copyright Holder | Upstream Link | License  |
|-----|----------------------------------|----------|------------------|-----------------------------------------------------------------------------------------------------------|--------------------|
| 1 | KQA Pro | Yes | Cao et al. (Tsinghua University) | [link](https://huggingface.co/datasets/drt/kqa_pro) | MIT License |
| 2 | GrailQA | Yes | Gu et al. (Ohio State University) | [link](https://dki-lab.github.io/GrailQA/) | CC BY-SA 4.0 License |
| 3 | WebQSP | Yes | Matt Richardson, Scott Wen-tau Yih (Microsoft) | [link](https://dki-lab.github.io/GrailQA/) | Unspecified |
| 4 | GraphQ | Yes | Su et al. (UC Santa Barbara) | [link](https://github.com/dki-lab/GrailQA/tree/main/data) | CC BY-SA 4.0 License |
| 5 | HotpotQA | Yes | Yang et al. (Stanford University) | [link](https://hotpotqa.github.io/) | CC BY-SA 4.0 License |
| 6 | Wikipedia | No |  | [link](https://hotpotqa.github.io/wiki-readme.html) | CC BY-SA 4.0 License |
| 7 | Freebase | No |  | [link](https://github.com/dki-lab/Freebase-Setup) | CC BY 2.5 License |

## Contact

For questions about the code release, please open an issue in this repository.

## Disclosure

Embedded in, or bundled with, this product are open source software (OSS) components, datasets and other third party components identified below. The license terms respectively governing the datasets and third-party components continue to govern those portions, and you agree to those license terms, which, when applicable, specifically limit any distribution. You may receive a copy of, distribute and/or modify any open source code for the OSS component under the terms of their respective licenses, which may be CC license and Apache 2.0 license. In the event of conflicts between Megagon Labs, Inc., license conditions and the Open Source Software license conditions, the Open Source Software conditions shall prevail with respect to the Open Source Software portions of the software. You agree not to, and are not permitted to, distribute actual datasets used with the OSS components listed below. You agree and are limited to distribute only links to datasets from known sources by listing them in the datasets overview table below. You are permitted to distribute derived datasets of data sets from known sources by including links to original dataset source in the datasets overview table below. You agree that any right to modify datasets originating from parties other than Megagon Labs, Inc. are governed by the respective third party's license conditions. All OSS components and datasets are distributed WITHOUT ANY WARRANTY, without even implied warranty such as for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE, and without any liability to or claim against any Megagon Labs, Inc. entity other than as explicitly documented in this README document. You agree to cease using any part of the provided materials if you do not agree with the terms or the lack of any warranty herein. While Megagon Labs, Inc., makes commercially reasonable efforts to ensure that citations in this document are complete and accurate, errors may occur. If you see any error or omission, please help us improve this document by sending information to contact_oss@megagon.ai.
