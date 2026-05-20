# Local LLM Serving with vLLM

## Step 0: Create a separate project for vLLM

Create a separate project for vLLM to avoid dependency conflicts with the main project. For example:

```bash
mkdir ~/vllm-serving
cd ~/vllm-serving
```

## Step 1: Install vLLM

Follow [the installation instructions for vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart/#installation). The official guide recommends installing it via uv.

```bash
# In a separate project directory
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

You may need to downgrade vllm if your CUDA version is not compatible with the latest vLLM.

## Step 2: Start an OpenAI-Compatible Server

Follow [the official guide](https://docs.vllm.ai/en/stable/getting_started/quickstart/#openai-compatible-server) to start an OpenAI-compatible server. For example:

```bash
vllm serve Qwen/Qwen3-0.6B
```

If you encounter `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended. Did you mean: 'num_special_tokens_to_add'?`, downgrade `transformers` to `v4.57.6` following [this issue](https://github.com/QwenLM/Qwen3-VL/issues/2058).

**Sanity Check**: You can test the server using `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```

## Step 3: Run `scripts/example_run_kqa_pro.py`

Now you can run `scripts/example_run_kqa_pro.py` to test the integration with vLLM.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/example_run_kqa_pro.py --model-config vllm_qwen3-0p6b --llm-provider vllm-local
```

(Change `CUDA_VISIBLE_DEVICES` to the GPU you want to use for SentenceTransformer in the KoPL agents. The KoPL agents will work without GPU, but it will be slower.
)

Because Qwen3 0.6B's tool calling capability is very limited, the agent will fail to call tools in the right format and exhaust the max retries. You will see output like this:

```
KQA Pro Example Runner
========================================
Loading configuration from: /zfs1/users/notani/dev/llm-planning-comparison/conf/experiment/kopl_kbqa/sh.v1.yaml
Applying overrides:
- model=vllm_qwen3-0p6b
- model@worker_model=vllm_qwen3-0p6b
- meta_agent.llm_provider=vllm-local
- environment.agents.kopl_schema_free_agents.llm_provider=vllm-local
- environment.agents.kopl_find_and_filter_concept_agents.llm_provider=vllm-local
- environment.agents.kopl_key_only_agents.llm_provider=vllm-local
- environment.agents.kopl_key_and_value_agents.llm_provider=vllm-local

Model: Qwen/Qwen3-0.6B
Agent: sh
Max Steps: 30
Demonstrations: 0
Live step streaming: True

PROBLEM:
Who is the spouse of the actor who played Jack in Titanic?


RUNNING SH META AGENT...
============================================================

LIVE TRAJECTORY:
============================================================
Step 1 [completed]
   Action: {"name": "find_all", "arguments": {}, "updated_arguments": {}}
   Observation: # Found 17754 entities.
      # - Q7270 (republic)
      # - Q130232 (drama film)
      # - Q280658 (forward)
      # - Q8355 (violin)
      # - Q8445 (marriage)
      # - Q1640319 (experimental music)
      # - Q5043 (Christianity)
      # - Q289 (television)
      # - Q7168625 (historical period drama)
      # - Q912985 (running back)
      # ... [truncated]
      $0 = find_all[{}]

Agent error during step generation 1: Max retries exceeded while generating plan step

EPISODE RESULTS:
============================================================
Success: False
Final Answer: Episode terminated without success
Steps Taken: 1
Max Steps Reached: False

FULL TRAJECTORY RECAP:
============================================================
Step 1 [completed]
   Action: {"name": "find_all", "arguments": {}, "updated_arguments": {}}
   Observation: # Found 17754 entities.
      # - Q7270 (republic)
      # - Q130232 (drama film)
      # - Q280658 (forward)
      # - Q8355 (violin)
      # - Q8445 (marriage)
      # - Q1640319 (experimental music)
      # - Q5043 (Christianity)
      # - Q289 (television)
      # - Q7168625 (historical period drama)
      # - Q912985 (running back)
      # ... [truncated]
      $0 = find_all[{}]


Example completed!
Agent provided answer: Episode terminated without success
```

To make the example work, use a stronger model like [`Qwen/Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) using [`../../conf/model/vllm_qwen3-30b-instruct`](../../conf/model/vllm_qwen3-30b-instruct). Note that you need to start a vLLM server with that model as well by `cd ~/vllm-serving && vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507`.



## Step 4: Run full experiments

The batch wrapper scripts support vLLM directly. Two aliases are built in:

| `--llm` value | Model config | Hugging Face model |
| --- | --- | --- |
| `vllm-qwen3-0p6b` | `conf/model/vllm_qwen3-0p6b.yaml` | `Qwen/Qwen3-0.6B` |
| `vllm-qwen3-30b` | `conf/model/vllm_qwen3-30b-instruct.yaml` | `Qwen/Qwen3-30B-A3B-Instruct-2507` |

For example, to run both SH and FH on KQA Pro with the 0.6B model:

```bash
bash scripts/batch/batch_exp_kqa_pro_hydra.sh --llm vllm-qwen3-0p6b --num-episodes 5
```

Both aliases assume the vLLM server is at `http://localhost:8000` (the default). To use a different host or port, update the `vllm-local` entry in `conf/config.yaml`.

### Adding a custom model

To add an alias for a model not listed above:

**1. Create a model config** in `conf/model/`, e.g. `conf/model/vllm_my_model.yaml`:

```yaml
api_type: chat
model: org/MyModel
temperature: 0
max_completion_tokens: 10000
```

The `model` field must match the model name served by your vLLM instance (the value you passed to `vllm serve`).

**2. Add a case** to `resolve_llm_config()` in `scripts/batch/hydra_batch_common.sh`:

```bash
vllm-my-model)
    MODEL_CONFIG="vllm_my_model"
    META_PROVIDER="vllm-local"
    WORKER_PROVIDER="vllm-local"
    ;;
```

After that you can pass `--llm vllm-my-model` to any batch wrapper.

### Manual invocation

If you prefer to call `scripts/run.py` directly instead of using the batch wrappers:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run.py \
  experiment=kopl_kbqa/sh.v1 \
  model=vllm_qwen3-0p6b \
  model@worker_model=vllm_qwen3-0p6b \
  meta_agent.llm_provider=vllm-local \
  environment.agents.kopl_schema_free_agents.llm_provider=vllm-local \
  environment.agents.kopl_find_and_filter_concept_agents.llm_provider=vllm-local \
  environment.agents.kopl_key_only_agents.llm_provider=vllm-local \
  environment.agents.kopl_key_and_value_agents.llm_provider=vllm-local \
  num_episodes=5 \
  workers=1
```

## Step 5: Evaluate results with a local model

Pass `--base-url` to `scripts/evaluate_result.py` to use the local vLLM server instead of the OpenAI API:

```bash
uv run python scripts/evaluate_result.py \
  -i results/<experiment_dir>/result.jsonl \
  -m Qwen/Qwen3-0.6B \
  --base-url http://localhost:8000/v1 \
  --max-tokens 1024
```

To evaluate all runs under a directory at once, use `scripts/evaluate_results.sh` and pass the same flags after the directory:

```bash
scripts/evaluate_results.sh results/kopl_kbqa/kqa_pro/test \
  -m Qwen/Qwen3-0.6B \
  --base-url http://localhost:8000/v1 \
  --max-tokens 1024
```

Replace:

- `results/<experiment_dir>/result.jsonl` with the actual path to the experiment output file
- `Qwen/Qwen3-0.6B` with the model name served by your vLLM instance.
- `http://localhost:8000/v1` with the base URL of your vLLM server if it's different.

Cost reporting is skipped for local models since token pricing is not defined in `TOKEN_COSTS`.

## Step 6: Collect metrics

After evaluation, aggregate metrics across runs using `scripts/collect_metrics.py`. Pass `--eval-model` to match the evaluator model used in Step 5:

```bash
uv run python scripts/collect_metrics.py kopl_kbqa/kqa_pro \
  --eval-model Qwen/Qwen3-0.6B \
  --eval-version 2.0 \
  -o tables/kqa_pro.csv
```

**Note:** If any `metrics.jsonl` file is missing the `correct_answers_<eval-model>_<eval-version>` key (e.g., because that run was evaluated with a different model), `collect_metrics.py` will raise a `ValueError` and list the available keys. Re-run `evaluate_result.py` on those runs with the same `--model` to populate the missing key before collecting metrics.
