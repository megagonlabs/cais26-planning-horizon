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

You can also run the full experiments with vLLM. For example, to run SH agent on KQA Pro:

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

Essentially, you need to set the `model` and `model@worker_model` to the vLLM model config, and set the `llm_provider` for all agents to `vllm-local`. You can also change the number of episodes and workers as needed. See [`../../scripts/batch/`](../../scripts/batch/) for base batch scripts.
