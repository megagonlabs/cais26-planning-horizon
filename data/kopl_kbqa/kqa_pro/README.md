# KQA Pro dataset

This directory contains the local files for the `KQA Pro` benchmark
(Cao et al., 2022) used in the KoPL experiments.

For the canonical preprocessing commands and evaluator guidance, see:

- [`docs/setup/data.md`](../../../docs/setup/data.md)
- [`docs/preprocessing/kqa_pro.md`](../../../docs/preprocessing/kqa_pro.md)

## Directory layout

| Path | Purpose |
| --- | --- |
| `download.sh` | Downloads `train.json`, `val.json`, and `kb.json` into this directory |
| `embeddings/` | Generated KoPL grounding embeddings used by the demo and worker agents |
| `scripts/` | KQA Pro-specific preprocessing, retrieval, and inspection scripts |
| `processed/` | Generated benchmark splits, held-out retrieval pool, and derived analysis outputs |

## Expected local files

After running `bash data/kopl_kbqa/kqa_pro/download.sh`, this directory should
contain:

- `train.json` (84 MB)
- `val.json` (11 MB)
- `kb.json` (75 MB)


After running the preprocessing walkthrough,

- `embeddings/BAAI___bge-base-en-v1.5/` should contain embedding files used for soft schema matching
    - `entity_embeddings.pkl`
    - `key_embeddings.pkl`
    - `value_embeddings.pkl`
- `processed/` should contain the benchmark JSON files


## Quick start

```shell
bash data/kopl_kbqa/kqa_pro/download.sh
bash data/kopl_kbqa/kqa_pro/scripts/run_preprocessing.sh
```

`download.sh` downloads the three raw files listed above into this directory.

`run_preprocessing.sh` then:

1. generates the KB embedding pickle files used by the KoPL worker agents
2. validates, filters, and converts the raw data
3. retrieves similar examples for the benchmark questions; this step
   downloads the `BAAI/bge-base-en-v1.5` model weights (`~400 MB`), and takes
   a few minutes to perform the retrieval.
4. computes DAG features *for the paper's post-hoc topological analysis*;
   this output is not required to run the benchmark itself

See [the detailed guide](../../../docs/preprocessing/kqa_pro.md) for stage-by-stage commands, expected outputs, sanity checks, and notes on runtime/storage.

## Reference

Shulin Cao, Jiaxin Shi, Liangming Pan, Lunyiu Nie, Yutong Xiang, Lei Hou, Juanzi Li, Bin He, and Hanwang Zhang. 2022. [KQA Pro: A Dataset with Explicit Compositional Programs for Complex Question Answering over Knowledge Base.](https://aclanthology.org/2022.acl-long.422/) In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6101–6119, Dublin, Ireland. Association for Computational Linguistics.
