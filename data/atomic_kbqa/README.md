# Atomic KBQA datasets

This directory contains the local data and helper resources for the
`Atomic KBQA` experiments over Freebase.

For the canonical preprocessing workflow and evaluator guidance, see:

- [`docs/setup/data.md`](../../docs/setup/data.md)
- [`docs/preprocessing/atomic_kbqa.md`](../../docs/preprocessing/atomic_kbqa.md)

## Directory layout

| Path | Purpose |
| --- | --- |
| `freebase/` | Local ontology files and relation inventories used by the atomic KBQA tooling |
| `grailqa/` | GrailQA raw files, scripts, and processed outputs |
| `webqsp/` | WebQSP raw files, scripts, and processed outputs |
| `graphq/` | GraphQ raw files, scripts, and processed outputs |
| `scripts/` | Shared wrapper, relation-list helpers, and optional SPARQL validation |

## Expected local files

The shared preprocessing guide assumes the following are already present:

- `grailqa/GrailQA_train.json`
- `grailqa/GrailQA_test.json`
- `webqsp/WebQSP_train.json`
- `webqsp/WebQSP_test.json`
- `graphq/GraphQ_train.json`
- `graphq/GraphQ_test.json`
- `freebase/fb_roles`
- `freebase/fb_types`
- `freebase/reverse_properties`

The generated benchmark files are written under each dataset's `processed/`
directory.

## Quick start

```shell
bash data/atomic_kbqa/scripts/run_preprocessing.sh
```

`run_preprocessing.sh` runs the reproducible preprocessing stages for all
three datasets:

1. balanced split creation with DAG conversion
2. held-out example retrieval
3. task-characterization metrics

Optional SPARQL validation is documented separately in
[`data/atomic_kbqa/scripts/README.md`](scripts/README.md) and in the
canonical preprocessing guide.

## References

- Haoran Luo, Haihong E, Yikai Guo, Qika Lin, Xiaobao Wu, Xinyu Mu,
	Wenhao Liu, Meina Song, Yifan Zhu, and Anh Tuan Luu. 2025.
	`KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search`.
	In Proceedings of the 42nd International Conference on Machine Learning,
	pages 41177–41199. PMLR.
- Wen-tau Yih, Matthew Richardson, Chris Meek, Ming-Wei Chang, and
	Jina Suh. 2016.
	[The Value of Semantic Parse Labeling for Knowledge Base Question Answering.](https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering-2/)
- Yu Gu, Sue Kase, Michelle Vanni, Brian Sadler, Percy Liang,
	Xifeng Yan, and Yu Su. 2021.
	`Beyond I.I.D.: Three Levels of Generalization for Question Answering on Knowledge Bases`.
- Yu Su, Huan Sun, Brian Sadler, Mudhakar Srivatsa, Izzeddin Gür,
	Zenghui Yan, and Xifeng Yan. 2016.
	`On Generating Characteristic-rich Question Sets for QA Evaluation`.
