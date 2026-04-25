# Configuration tree

The `conf/` directory contains the Hydra configuration groups used to compose
experiment runs.

## Layout

| Path | Purpose |
| --- | --- |
| `config.yaml` | Root entry point used by `scripts/run.py` |
| `model/` | Reusable planner and worker model option sets |
| `meta_agent/` | SH and FH planner defaults |
| `env/` | Environment composition: tools, worker agents, and environment flags |
| `worker_agent/` | Reusable worker-agent config fragments injected into environments |
| `experiment/` | Dataset-specific experiment entry points |

## How it fits together

1. `config.yaml` selects a default experiment and default model groups.
2. `experiment/**` chooses one planner config from `meta_agent/` and one
   environment config from `env/`.
3. `env/**` uses Hydra package targeting to inject worker-agent fragments from
   `worker_agent/**` into `environment.agents.*`.
4. `scripts/run.py` resolves the composed config and passes the resolved
   `environment` section into `Environment.load_from_config()`.

## Where to read next

- [`docs/configuration.md`](../docs/configuration.md) for the full configuration reference
- [`docs/planner.md`](../docs/planner.md) for SH/FH behavior
- [`docs/environment.md`](../docs/environment.md) for runtime loading and dispatch
- [`docs/worker-agents-and-tools.md`](../docs/worker-agents-and-tools.md) for worker-family behavior
