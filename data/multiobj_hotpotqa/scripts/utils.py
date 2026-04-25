"""
Utility functions for HotpotQA structure validation.

Provides shared functionality for both batch validation (via Batch API) and
direct validation (via ThreadPoolExecutor).
"""

from pathlib import Path
from typing import Any, Optional
import json

import orjson


INPUT_COST = {
    "gpt-4.1-2025-04-14": 2.00 / 1_000_000,
    "gpt-4.1-mini-2025-04-14": 0.40 / 1_000_000,
    "gpt-5-mini-2025-08-07": 0.25 / 1_000_000,
    "gpt-5.2-2025-12-11": 1.75 / 1_000_000,
}
OUTPUT_COST = {
    "gpt-4.1-2025-04-14": 8.00 / 1_000_000,
    "gpt-4.1-mini-2025-04-14": 1.60 / 1_000_000,
    "gpt-5-mini-2025-08-07": 2.00 / 1_000_000,
    "gpt-5.2-2025-12-11": 14.00 / 1_000_000,
}


VALIDATION_SYS_PROMPT = """You are validating whether a HotpotQA bridge question follows a valid linear reasoning structure. Judge only from the wording of the question; do not use outside knowledge or supporting facts. Decompose the question into steps, state if any step yields multiple candidates (a set), and then output a single JSON object.

Core definitions
- Linear chain (valid): A strictly sequential chain of 2+ steps where each step yields exactly one intermediate entity/property that directly feeds the next step. Direct property lookup or a single yes/no verification on that single entity is fine.
- Branching/set filtering (invalid): Any step produces multiple candidates that must be checked/filtered individually (set operations), or the question asks for intersections/commonalities/comparisons across multiple entities.

Conservative uniqueness policy (do not assume):
- Do not infer uniqueness from plausibility or typical world facts. If the wording does not make uniqueness explicit, treat the step as set-producing → invalid.
- Roles that vary over time (e.g., "the president of [org]", "the coach of [team]") are non-unique unless the question specifies a time/tenure (e.g., a year/season/ordinal/current).
- Combining multiple descriptors (e.g., "Austrian forest caretaker, naturalist, pseudoscientist") does not guarantee uniqueness; still treat as potentially multiple unless uniquely pinned.

Valid patterns (usually unique by wording)
- Specific titled work → role/property (e.g., "the director of [titled work]", "the vocalist on '[song]'").
- Definite, singular roles tied to a specific proper noun with an explicit time/ordinal (e.g., "the head coach of [team] in [year]", "the 42nd president of [country]").
- Definitional superlatives that denote a single item by definition (e.g., "the capital of [country]", "the largest city in [county]").

Invalid patterns (must label invalid)
- Set-producing first step (guests, stars, cast members, contestants, authors, films featuring X, etc.) followed by filtering.
- Parallel/compound questions seeking two independent facts or commonalities (e.g., "What do X and Y have in common?", "Which star of A was also in B?").
- Ambiguous cardinality or vague qualifiers (e.g., "former", "long-time") without time bounds; treat as set-producing.
- Any step where uniqueness across time is not fixed by the wording.

Output requirements
- Return only a single JSON object with keys:
    - "reasoning": 2–4 concise sentences that enumerate the steps (e.g., "Step 1... Step 2...") and explicitly state where (if anywhere) branching occurs.
    - "is_valid": true if linear (single-path), false if branching/set/parallel or if only a single-hop lookup.
- Do not answer the original question. Do not include extra keys, disclaimers, or formatting.

Always think step by step based solely on the question text, err on the side of treating ambiguous steps as branching, indicate whether branching occurs and where, and then provide the final JSON.
""".strip()

VALIDATION_USR_PROMPT = """Question: {question}

Is the reasoning structure of this question valid according to the criteria provided? Decompose the question into stepwise reasoning (label Step 1, Step 2, etc.), indicate if and where branching (set operations or multiple candidate entities) is required, and return only the required JSON object.
""".strip()


def load_json(file_path: Path) -> list[dict]:
    """
    Load data from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        list[dict]: Loaded data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return orjson.loads(f.read())


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """
    Load data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        list[dict]: Loaded data
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(orjson.loads(line.strip()))
    return data


def save_json(data, file_path):
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def filter_by_type(
    examples: list[dict[str, Any]], question_type: str
) -> list[dict[str, Any]]:
    """
    Filter examples by question type.

    Args:
        examples: List of examples
        question_type: Type to filter by (e.g., "bridge", "comparison")

    Returns:
        list[dict]: Filtered examples
    """
    return [ex for ex in examples if ex["type"] == question_type]


def format_supporting_sentences(example: dict) -> str:
    """
    Format supporting sentences from a HotpotQA example.

    Extracts and formats the supporting sentences mentioned in supporting_facts.

    Args:
        example: HotpotQA example with supporting_facts and context fields

    Returns:
        str: Formatted supporting sentences (one per line with title in brackets)
    """
    supporting_facts = example["supporting_facts"]
    context = {title: sentences for title, sentences in example["context"]}

    sentences = []
    for title, sent_id in supporting_facts:
        try:
            sentence_text = context[title][sent_id]
            sentences.append(f"- [{title}] {sentence_text}")
        except (KeyError, IndexError) as e:
            print(
                f"Error retrieving sentence for title '{title}' and sent_id '{sent_id}': {e}"
            )
            continue

    return "\n".join(sentences)


def create_validation_prompt(example: dict) -> tuple[str, str]:
    """
    Create system and user prompts for validation.

    Args:
        example: HotpotQA example

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    sys_msg = VALIDATION_SYS_PROMPT
    usr_msg = VALIDATION_USR_PROMPT.format(question=example["question"])
    return sys_msg, usr_msg


def get_validation_response_format() -> dict:
    """
    Get the JSON schema for structured validation output.

    Returns:
        dict: Response format configuration with schema for OpenAI API
    """
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "2–4 concise sentences that enumerate the steps and explicitly state where (if anywhere) branching occurs.",
            },
            "is_valid": {
                "type": "boolean",
                "description": "True if linear (single-path), false if branching/set/parallel or if only a single-hop lookup.",
            },
        },
        "required": ["reasoning", "is_valid"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "validation_result",
            "schema": schema,
            "strict": True,
        },
    }


def get_dag_annotation_response_format() -> dict:
    """
    Get the JSON schema for structured DAG annotation format.

    Returns:
        dict: Response format configuration with schema for OpenAI API
    """
    schema = {
        "type": "object",
        "properties": {
            "rephrased_question": {
                "type": "string",
                "description": "Rephrased question that preserves the original meaning but is clearer and more explicit.",
            },
            "dag": {
                "type": "array",
                "description": "List of nodes in the reasoning DAG.",
                "items": {
                    "type": "object",
                    "properties": {
                        "function": {
                            "type": "string",
                            "description": "The function name representing the reasoning step.",
                            "enum": ["search", "reasoning"],
                        },
                        "dependencies": {
                            "type": "array",
                            "description": "List of node indices that this node depends on.",
                            "items": {"type": "integer"},
                        },
                        "input": {
                            "type": "string",
                            "description": "Input to the function for this reasoning step.",
                        },
                    },
                    "required": ["function", "dependencies", "input"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["rephrased_question", "dag"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "dag_annotation",
            "schema": schema,
            "strict": True,
        },
    }


def get_api_params(
    model: str,
    max_completion_tokens: int = 512,
    response_format: Optional[dict] = None,
) -> dict:
    """
    Get API parameters for requests.

    Args:
        model: Model name (e.g., "gpt-4.1-mini-2025-04-14", "gpt-5.2-2025-12-11")

    Returns:
        dict: API parameters for validation
    """
    params = {
        "model": model,
        "temperature": 0.0,
        "max_completion_tokens": max_completion_tokens,
        "response_format": response_format,
    }
    if model.startswith("gpt-5"):
        del params["temperature"]  # gpt-5* does not support temperature
    return params
