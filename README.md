# Do Agents Need to Plan Step-by-Step? Rethinking Planning Horizon in Data-Centric Tool Calling

Official codebase for the paper *Do Agents Need to Plan Step-by-Step?
Rethinking Planning Horizon in Data-Centric Tool Calling*.

The project compares single-step horizon (SH) planning and full-horizon (FH)
planning for data-centric tool-calling tasks such as KBQA and multi-hop QA.
The core finding is that, for well-defined data-centric tasks, FH planning
with lazy replanning can match SH planning while using substantially fewer
tokens.

## Repository guide

| Path | What it contains | Pointers |
| --- | --- | --- |
| `src/planning/agents/` | Meta agents, worker agents, shared agent abstractions | `agents/base_agent.py`, `agents/meta_agents/meta_sh.py`, `agents/meta_agents/meta_fh.py`, `agents/worker_agents/` |
| `planning/environment/` | Environment orchestration and executable registries | `environment/environment.py`, `environment/agent_registry.py`, `environment/tool_registry.py` |
| `planning/services/` | LLM provider integrations and routing | `services/llm_registry.py`, `services/openai.py`, `services/vllm.py`, `services/vertexai_openai.py` |
| `planning/tools/` | Task-specific tool implementations and wrappers | See source package and tool docs linked from `docs/system-design.md` |
| `conf/` | Hydra configurations by dataset and planner setup | Example: `conf/experiment/kopl_kbqa/` |
| `scripts/` | Entry points for single runs and batch experiments | `scripts/example_run_kqa_pro.py`, `scripts/run.py` |
| `docs/system-design.md` | Main entry point for system documentation | Start here for architecture and follow-on docs |

## Setup

- Code setup: [`docs/setup/code.md`](docs/setup/code.md)
- Data setup: [`docs/setup/data.md`](docs/setup/data.md)
- Environment variables: copy [`.env.example`](.env.example) to `.env` and
  fill in only the providers and tools you plan to use.

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
*System dependencies (atomic KBQA only):** The Freebase SPARQL executor uses `pyodbc`, which requires the `unixodbc` system library. On Linux (Ubuntu), this is bundled inside the `pyodbc` PyPI wheel and no extra step is needed. On macOS, install it separately before running `uv sync`:

```bash
brew install unixodbc
```

Python dependencies are listed in `pyproject.toml` and can be installed with `uv` (See `docs/setup/code.md` for instructions).

## Quick walkthrough

Use `scripts/example_run_kqa_pro.py` as the primary single-question demo.

Before running it, complete:

- [Code setup](docs/setup/code.md)
- [Data setup](docs/setup/data.md) for the `KQA Pro` track

Then run one KQA Pro question against the KoPL worker-agent stack:

```bash
uv run python scripts/example_run_kqa_pro.py --problem "Who is the spouse of the actor who played Jack in Titanic?"
```

For the fuller walkthrough, including the optional FH variant, output
structure, and troubleshooting notes, see [`docs/walkthrough.md`](docs/walkthrough.md).

## Citation

If you use this repository, please cite the paper:

```bibtex
# TODO
```

## Contact

For questions about the code release, please open an issue in this repository.

## Disclosure

Embedded in, or bundled with, this product are open source software (OSS) components, datasets and other third party components identified below. The license terms respectively governing the datasets and third-party components continue to govern those portions, and you agree to those license terms, which, when applicable, specifically limit any distribution. You may receive a copy of, distribute and/or modify any open source code for the OSS component under the terms of their respective licenses, which may be CC license and Apache 2.0 license. In the event of conflicts between Megagon Labs, Inc., license conditions and the Open Source Software license conditions, the Open Source Software conditions shall prevail with respect to the Open Source Software portions of the software. You agree not to, and are not permitted to, distribute actual datasets used with the OSS components listed below. You agree and are limited to distribute only links to datasets from known sources by listing them in the datasets overview table below. You are permitted to distribute derived datasets of data sets from known sources by including links to original dataset source in the datasets overview table below. You agree that any right to modify datasets originating from parties other than Megagon Labs, Inc. are governed by the respective third party's license conditions. All OSS components and datasets are distributed WITHOUT ANY WARRANTY, without even implied warranty such as for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE, and without any liability to or claim against any Megagon Labs, Inc. entity other than as explicitly documented in this README document. You agree to cease using any part of the provided materials if you do not agree with the terms or the lack of any warranty herein. While Megagon Labs, Inc., makes commercially reasonable efforts to ensure that citations in this document are complete and accurate, errors may occur. If you see any error or omission, please help us improve this document by sending information to contact_oss@megagon.ai.
