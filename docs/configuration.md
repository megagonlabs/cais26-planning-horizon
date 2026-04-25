# Configuration

This project uses Hydra to compose experiment configs from reusable pieces
under `conf/`. The main runtime path is:

1. `conf/config.yaml` selects the root model, worker model, and experiment.
2. `conf/experiment/**` chooses a planner config from `conf/meta_agent/`
   and an environment config from `conf/env/`.
3. `scripts/run.py` resolves the composed Hydra config.
4. `scripts/utils/agent_utils.py` builds the `LLMProviderRegistry`,
   `Environment`, and SH/FH Meta Agent from the resolved config.

This document is the single reference for the current configuration system.
Use it together with:

- `docs/planner.md` for SH/FH runtime behavior
- `docs/environment.md` for registry loading and execution flow
- `docs/worker-agents-and-tools.md` for worker-family behavior

## Configuration tree

| Location | Role | Consumed by |
| --- | --- | --- |
| `conf/config.yaml` | Root entry point, runtime flags, provider definitions, default experiment selection | `scripts/run.py`, `scripts/utils/agent_utils.py` |
| `conf/model/*.yaml` | Planner and worker model option sets | `create_meta_agent()`, worker-agent loaders |
| `conf/meta_agent/*.yaml` | SH/FH planner defaults and output schemas | `create_meta_agent()` |
| `conf/env/**/*.yaml` | Environment-level tool and worker-agent composition | `Environment.load_from_config()` |
| `conf/worker_agent/**/*.yaml` | Reusable worker-agent config fragments injected into environments | Hydra composition into `environment.agents.*` |
| `conf/experiment/**/*.yaml` | Dataset-specific experiment bindings and planner prompts | Hydra composition |

## How composition works

### Root entry point: `conf/config.yaml`

`conf/config.yaml` is the Hydra entry point used by `scripts/run.py`.
It defines:

- default config-group selections in `defaults`
- runtime flags such as `num_episodes`, `workers`, and `debug`
- shared planner prompt settings such as `num_demonstrations`
- the `llm_providers` map
- Hydra output-directory naming under `hydra.run.dir`

Current default composition:

- `model: gpt-4p1-mini.v1`
- `model@worker_model: gpt-4p1-mini.v1`
- `experiment: multiobj/hotpotqa/sh.v1`

### Experiment configs: `conf/experiment/**`

Experiment configs use `# @package _global_` and bind together:

- one planner config from `conf/meta_agent/`
- one environment config from `conf/env/`
- dataset-specific planner prompts
- dataset-selection metadata under `experiment`

For example:

- `conf/experiment/kopl_kbqa/sh.v1.yaml`
- `conf/experiment/atomic_kbqa/sh.v1.yaml`
- `conf/experiment/multiobj/hotpotqa/sh.v1.yaml`

These files are the main entry points for reproducible experiment runs.

### Environment configs: `conf/env/**`

Environment configs use `# @package environment` and define the planner's
runtime action space.

They provide:

- environment-level flags such as `list_finish` and
  `substitute_memory_reference`
- `tools` entries, loaded by `ToolRegistry`
- `agents` entries, loaded by `AgentRegistry`
- Hydra `defaults` entries that inject worker-agent fragments from
  `conf/worker_agent/**`

### Worker-agent fragments: `conf/worker_agent/**`

Worker-agent files are reusable fragments, not standalone runnable configs.
Hydra injects them into `environment.agents.<agent_id>` with package
targeting such as:

- `/worker_agent/husky/search.v1@agents.search`
- `/worker_agent/kopl/base.v1@agents.kopl_schema_free_agents`

In practice, the environment config contributes most of the runtime wiring
(`llm_provider`, `model`, `available_tools`, `common_parameters`), while the
worker-agent fragment contributes the family-specific `type`, prompts, or
other family-local defaults.

## Runtime loading path

### `scripts/run.py`

`scripts/run.py` is the Hydra entry point:

- resolves the composed config with `OmegaConf.to_container(..., resolve=True)`
- loads dataset records from `config["experiment"]`
- builds the runtime with helpers from `scripts/utils/agent_utils.py`
- writes the resolved `config.yaml` into the result directory

### `scripts/utils/agent_utils.py`

This module is the main bridge between the resolved config and the runtime.

| Function | Role |
| --- | --- |
| `load_config()` | Loads a raw YAML file or composes a Hydra experiment when the input path is under `conf/experiment/` |
| `build_llm_registry_from_config()` | Builds `LLMProviderRegistry` from `llm_providers` |
| `setup_environment()` | Calls `Environment.load_from_config(config["environment"])` and assigns worker LLM providers |
| `create_meta_agent()` | Instantiates `SHMetaAgent` or `FHMetaAgent` from `config["meta_agent"]` |

### `Environment.load_from_config()`

The environment consumes only `config["environment"]` and loads it in two
phases:

1. `ToolRegistry.load_from_config(config["tools"])`
2. `AgentRegistry.load_from_config(config["agents"], tool_registry)`

The order matters because some worker families reuse tool-factory instances
that were created during the tools phase.

## Root-level fields

### Runtime flags in `conf/config.yaml`

| Field | Meaning |
| --- | --- |
| `num_episodes` | Optional cap on the number of loaded dataset records |
| `workers` | Number of experiment workers used by `scripts/run.py` |
| `debug` | Enables debug-mode behavior in the environment |
| `num_demonstrations` | Shared default interpolated into planner prompt configs |

### `llm_providers`

`llm_providers` is a provider-id-to-spec map consumed by
`LLMProviderRegistry.from_config()`.

Supported provider `type` values in the current codebase:

- `openai`
- `fireworks`
- `vertexai-openai`
- `vllm`

Common provider fields:

| Field | Meaning |
| --- | --- |
| `type` | Backend type |
| `api_key` | Inline API key |
| `api_key_env` | Environment variable containing the API key |
| `base_url` | Custom endpoint URL |
| `organization` | Optional OpenAI organization |
| `timeout` | Request timeout |
| `max_retries` | Client retry count |
| `project_id_env` | Vertex AI project variable |
| `location_env` | Vertex AI location variable |

## Model configs: `conf/model/*.yaml`

Model configs are plain option dictionaries that get interpolated into planner
or worker-agent configs.

Current examples:

| File | Notable fields |
| --- | --- |
| `conf/model/gpt-4p1-mini.v1.yaml` | `api_type: chat`, deterministic temperature, `seed` |
| `conf/model/gpt-5-mini.v1.yaml` | `api_type: responses`, `reasoning_effort` |

Common model fields used in the current code path:

| Field | Meaning |
| --- | --- |
| `model` | Model identifier passed to the provider client |
| `api_type` | `chat` or `responses` |
| `temperature` | Sampling temperature |
| `max_completion_tokens` | Completion budget |
| `seed` | Optional deterministic seed |
| `reasoning_effort` | Optional reasoning level for supported models |
| `store` | Optional Responses API storage flag |

The planner model usually comes from root `model`, while worker agents often
receive `${worker_model}`.

## Planner configs: `conf/meta_agent/*.yaml`

Planner configs use `# @package meta_agent` and are consumed by
`create_meta_agent()`.

### Common planner fields

| Field | Meaning |
| --- | --- |
| `type` | Canonical planner ID: `sh` or `fh` |
| `max_steps` | Step budget for the episode |
| `max_retries` | Retry budget |
| `llm_provider` | Provider ID looked up in `llm_providers` |
| `model` | Planner model options, usually `${model}` |
| `use_builtin_tool_input` | If `true`, pass worker schemas to the model as native tool definitions |
| `tool_formatter_id` | Formatter used when tool definitions are embedded into prompts |
| `generation` | Prompt templates and structured-output settings |

### SH planner config

`conf/meta_agent/sh.json.v1.yaml` defines `generation.plan_step`, which is the
single-turn planning prompt used on each SH iteration.

Important fields inside `generation.plan_step`:

| Field | Meaning |
| --- | --- |
| `system` | Planner system prompt template |
| `user` | Planner user prompt template |
| `response_format` | Structured-output schema for one action |
| `tool_schema_path` | Path to the action schema inside `response_format` |
| `num_demonstrations` | Number of demonstrations to insert into the prompt |

`use_builtin_tool_output` is currently only relevant to the SH creation path.

### FH planner config

`conf/meta_agent/fh.json.v1.yaml` defines:

- `generation.plan` for initial full-plan generation
- `generation.revision` for replanning after failure

Important FH-only fields inside `generation.plan`:

| Field | Meaning |
| --- | --- |
| `step_index_path` | Path to the step-index property inside the structured plan schema |

## Environment config surface: `conf/env/**`

The environment config is the part of the resolved config that
`Environment.load_from_config()` consumes directly.

### Environment-level fields

| Field | Meaning |
| --- | --- |
| `list_finish` | Whether the synthetic `finish` action accepts a list instead of a single string |
| `substitute_memory_reference` | Whether `$i` references are resolved before worker-agent execution |
| `tools` | Tool or tool-set entries loaded by `ToolRegistry` |
| `agents` | Worker-agent entries loaded by `AgentRegistry` |

### Tool entries

Each `environment.tools.<tool_id>` entry has this shape:

| Field | Meaning |
| --- | --- |
| `type` | Tool class or tool-factory ID |
| `parameters` | Shared parameters passed to the tool constructor or tool factory |

Current environment examples:

| Environment | Tool entry examples |
| --- | --- |
| `conf/env/kopl_kbqa/v1.yaml` | KoPL tool factories such as `kopl_schema_free_tools` |
| `conf/env/atomic_kbqa/v1.yaml` | `atomic_kb_query_tools` with Virtuoso connection parameters |
| `conf/env/multiobj/hotpotqa/v1.yaml` | `search` with Pyserini FAISS/Lucene settings |

### Agent entries

Each `environment.agents.<agent_id>` entry is the final merged worker-agent
config after Hydra applies any worker-agent fragment.

Common fields consumed by `AgentRegistry`:

| Field | Meaning |
| --- | --- |
| `type` | Worker-agent family or family-specific type |
| `llm_provider` | Provider ID for that worker agent or group |
| `model` | Worker model options |
| `available_tools` | Tool-set IDs or tool IDs the worker family should use |
| `agent_per_tool` | Bulk-instantiation mode for KoPL and Atomic KB Query families |
| `enable_schema_update_from_tool` | Whether to copy planner-facing schemas from linked tools |
| `common_parameters` | Shared family-specific parameters |
| `prompt` | Prompt templates for Husky-style workers |

## Worker-agent fragments by family

### KoPL: `conf/worker_agent/kopl/*.yaml`

These files mainly provide a family-specific `type`:

| File | Type |
| --- | --- |
| `base.v1.yaml` | `kopl_agent` |
| `find_and_filter_concept.v1.yaml` | `kopl_find_and_filter_concept_agent` |
| `key_only.v1.yaml` | `kopl_key_only_agent` |
| `key_and_value.v1.yaml` | `kopl_key_and_value_agent` |

The environment config supplies the rest of the runtime wiring:

- `llm_provider`
- `model`
- `available_tools`
- `agent_per_tool`
- `enable_schema_update_from_tool`
- `common_parameters.embeddings_dir`
- `common_parameters.encoder_model_name`
- optional `common_parameters.strict_mode`
- optional `common_parameters.retrieval_topk`

### Atomic KB Query: `conf/worker_agent/atomic_kb_query.v1.yaml`

This fragment defines:

- `type: atomic_kb_query_agents`
- shared `common_parameters` such as:
  - `schema_resources_path`
  - `embedding_model_name`
  - optional `retrieval_topk`
  - optional `strict_mode`

The environment config supplies:

- `llm_provider`
- `model`
- `available_tools`
- `agent_per_tool`
- `enable_schema_update_from_tool`

### Husky: `conf/worker_agent/husky/*.yaml`

These fragments define the worker type plus prompt templates.

| File | Type |
| --- | --- |
| `commonsense.v1.yaml` | `husky_commonsense_agent` |
| `search.v1.yaml` | `husky_search_agent` |

The environment config supplies the worker IDs, model selection, provider
selection, and linked tools.

## Experiment-specific patterns

### KoPL KBQA

Main files:

- `conf/experiment/kopl_kbqa/sh.v1.yaml`
- `conf/experiment/kopl_kbqa/fh.v1.yaml`
- `conf/env/kopl_kbqa/v1.yaml`

Pattern:

- one worker agent per KoPL operator
- one tool set per KoPL operator family
- worker agents receive embeddings/grounding resources through
  `common_parameters`

### Atomic KBQA

Main files:

- `conf/experiment/atomic_kbqa/sh.v1.yaml`
- `conf/experiment/atomic_kbqa/fh.v1.yaml`
- `conf/env/atomic_kbqa/v1.yaml`

Pattern:

- one worker agent per atomic query operator
- `substitute_memory_reference: false` to avoid eagerly expanding large
  intermediate states before worker execution
- Virtuoso connection parameters live in the tool entry

### Multi-objective HotpotQA

Main files:

- `conf/experiment/multiobj/hotpotqa/sh.v1.yaml`
- `conf/experiment/multiobj/hotpotqa/fh.v1.yaml`
- `conf/env/multiobj/hotpotqa/v1.yaml`

Pattern:

- two worker entries (`reasoning`, `search`)
- `list_finish: true` so the planner can return multiple answer strings
- search-tool parameters live directly in the environment config

## Overrides and direct loading

### Hydra CLI overrides

Typical override patterns:

- switch planner experiment: `experiment=kopl_kbqa/fh.v1`
- switch planner model: `model=gpt-5-mini.v1`
- switch worker model separately: `model@worker_model=gpt-4p1-mini.v1`
- override nested values: `meta_agent.max_retries=5`

### Direct config loading in helper scripts

`scripts/utils/agent_utils.py:load_config()` supports two modes:

| Input path | Behavior |
| --- | --- |
| Under `conf/experiment/` | Compose the full Hydra config rooted at `conf/config.yaml` |
| Any other YAML file | Load the file directly with `yaml.safe_load()` |

This is why scripts such as `scripts/example_run_kqa_pro.py`
can accept an experiment YAML path and still get the full resolved config.

## Practical conventions

- Use `sh` and `fh` as the only planner IDs.
- Treat `conf/worker_agent/**` as reusable fragments, not runnable configs.
- Put dataset-specific planner prompts in `conf/experiment/**`, not in
  `conf/meta_agent/**`.
- Put task-environment composition in `conf/env/**`.
- Put provider wiring in `conf/config.yaml`.

## Related files

| File | Role |
| --- | --- |
| `scripts/run.py` | Hydra experiment entry point |
| `scripts/utils/agent_utils.py` | Config-to-runtime bridge |
| `src/planning/services/llm_registry.py` | LLM provider loading |
| `src/planning/environment/environment.py` | Environment config consumption |
| `src/planning/environment/tool_registry.py` | Tool config loading |
| `src/planning/environment/agent_registry.py` | Worker-agent config loading |
