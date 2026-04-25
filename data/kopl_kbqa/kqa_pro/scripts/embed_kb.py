"""Generate KoPL grounding embeddings for the KQA Pro knowledge base.

This script embeds entity names, attribute/relation keys, and string values
from the KQA Pro KoPL knowledge base and stores them as pickle files under the
dataset-local ``embeddings/`` directory.
"""

from pathlib import Path
import argparse
import pickle

from kopl.kopl import KoPLEngine
from sentence_transformers import SentenceTransformer


DEFAULT_KB_PATH = Path("data/kopl_kbqa/kqa_pro/kb.json")


def main(args: argparse.Namespace) -> None:
    """Generate embeddings for the configured KQA Pro knowledge base.

    Args:
        args: Parsed command-line arguments.
    """
    print(f"Output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model_name}")
    model = SentenceTransformer(args.model_name)

    print(f"Loading KB from {args.kb_path}")
    engine = KoPLEngine.from_json(str(args.kb_path))

    names: list[str] = list(engine.kb.name_to_id.keys())  # type: ignore
    attrs: list[str] = engine.kb.attribute_keys  # type: ignore
    rels: list[str] = engine.kb.relations  # type: ignore
    keys = list(set(attrs + rels))
    vals: list[str] = list(
        set(
            [
                str(value_obj)
                for value_list in engine.kb.key_values.values()
                for value_obj in value_list
                if value_obj.type == "string"
            ]
        )
    )  # type: ignore
    print(
        f"{len(names)} names, {len(keys)} keys ({len(attrs)} attributes, "
        f"{len(rels)} relations), and {len(vals)} values"
    )

    path_entity_emb = args.output_dir / "entity_embeddings.pkl"
    if not path_entity_emb.exists() or args.force_recompute:
        model = SentenceTransformer(args.model_name)
        print(f"Embedding entities and saving to {path_entity_emb}")
        names = sorted(names, key=len)
        vecs = model.encode(
            names,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        entity_embeddings = {name: vec for name, vec in zip(names, vecs)}
        with open(path_entity_emb, "wb") as file_obj:
            pickle.dump(entity_embeddings, file_obj)
    else:
        print(
            f"Entity embeddings already exist at {path_entity_emb}, skipping. "
            "Use --force_recompute to recompute."
        )

    path_key_emb = args.output_dir / "key_embeddings.pkl"
    if not path_key_emb.exists() or args.force_recompute:
        model = SentenceTransformer(args.model_name)
        print(f"Embedding keys and saving to {path_key_emb}")
        keys = sorted(keys, key=len)
        vecs = model.encode(
            keys,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        key_embeddings = {key: vec for key, vec in zip(keys, vecs)}
        with open(path_key_emb, "wb") as file_obj:
            pickle.dump(key_embeddings, file_obj)
    else:
        print(
            f"Key embeddings already exist at {path_key_emb}, skipping. "
            "Use --force_recompute to recompute."
        )

    path_value_emb = args.output_dir / "value_embeddings.pkl"
    if not path_value_emb.exists() or args.force_recompute:
        model = SentenceTransformer(args.model_name)
        print(f"Embedding values and saving to {path_value_emb}")
        vals = sorted(vals, key=len)
        vecs = model.encode(
            vals,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        value_embeddings = {val: vec for val, vec in zip(vals, vecs)}
        with open(path_value_emb, "wb") as file_obj:
            pickle.dump(value_embeddings, file_obj)
    else:
        print(
            f"Value embeddings already exist at {path_value_emb}, skipping. "
            "Use --force_recompute to recompute."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate grounding embeddings for the KQA Pro KoPL KB"
    )
    parser.add_argument(
        "--kb_path",
        type=Path,
        default=DEFAULT_KB_PATH,
        help="Path to the KQA Pro KB in KoPL JSON format",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Sentence Transformer model name",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help=(
            "Directory to save the embeddings. Defaults to "
            "<kb_path.parent>/embeddings/<model_name>."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for embedding",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Recompute embeddings even if the pickle files already exist",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            args.kb_path.parent / "embeddings" / args.model_name.replace("/", "___")
        )

    main(args)
