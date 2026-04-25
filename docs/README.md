# Documentation

This directory contains documentation for setting up the project,
reproducing the paper results, and understanding the system design.

## Setup and data guides

- [`walkthrough.md`](walkthrough.md): Quick walkthrough for running a representative example with `scripts/example_run_kqa_pro.py`.
- [`setup/data.md`](setup/data.md): Starting point for dataset setup and preprocessing, with `KQA Pro` as the recommended walkthrough.
- [`preprocessing/kqa_pro.md`](preprocessing/kqa_pro.md): Canonical `KQA Pro` preprocessing guide.
- [`preprocessing/atomic_kbqa.md`](preprocessing/atomic_kbqa.md): Canonical `Atomic KBQA` preprocessing guide.
- [`preprocessing/multiobj_hotpotqa.md`](preprocessing/multiobj_hotpotqa.md): Canonical `Multi-objective HotpotQA` preprocessing guide.

## System Documentation

- [`system-design.md`](system-design.md): An overview of the system architecture and design decisions for the agent implementations in this codebase.
- [`planner.md`](planner.md): Design details of the SH and FH meta-agent planners, including shared architecture, planning loops, finish handling, and a comparison table.
- [`worker-agents-and-tools.md`](worker-agents-and-tools.md): Design details of the worker-agent families (KoPL, AtomicKBQuery, Husky), including the shared preprocess → execute → postprocess pattern, family-specific execution details, and configuration entry points.
- [`environment.md`](environment.md): Design details of the environment used for executing tool calls and how agents interact with it.
- [`configuration.md`](configuration.md): Consolidated reference for Hydra composition, planner config, environment config, worker-agent fragments, tool entries, and provider wiring.
