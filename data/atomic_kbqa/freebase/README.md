# Freebase resources for Atomic KBQA

This directory contains the local ontology files and relation inventories
used by the `Atomic KBQA` tooling.

For the canonical preprocessing workflow and evaluator guidance, see:

- [`docs/setup/data.md`](../../../docs/setup/data.md)
- [`docs/preprocessing/atomic_kbqa.md`](../../../docs/preprocessing/atomic_kbqa.md)

## Directory layout

| Path | Purpose |
| --- | --- |
| `fb_roles` | Ontology roles file downloaded from the GrailQA ontology repository |
| `fb_types` | Ontology types file downloaded from the GrailQA ontology repository |
| `reverse_properties` | Reverse-relation mapping downloaded from the GrailQA ontology repository |
| `relation_list.txt` | Relation inventory used by the atomic KBQA tooling |
| `literal_relation_list.txt` | Literal-valued relation inventory |
| `join_ban_relation_list.txt` | Auxiliary relation filter list used by the worker tooling |

## Expected local files

At minimum, this directory should contain:

- `fb_roles`
- `fb_types`
- `reverse_properties`

The relation-list files shipped in this repository are already populated and
do not need to be regenerated for normal artifact evaluation.

## Quick start

```shell
uv run python data/atomic_kbqa/scripts/extract_ontology_relations.py
uv run python data/atomic_kbqa/scripts/verify_and_update_relation_list.py
uv run python data/atomic_kbqa/scripts/generate_literal_relation_list.py
```

`extract_ontology_relations.py` merges ontology relations into the local
`relation_list.txt`.

`verify_and_update_relation_list.py` checks that every relation appearing in
the processed benchmark files is covered.

`generate_literal_relation_list.py` rebuilds the literal-relation inventory
by querying the live Freebase endpoint.
