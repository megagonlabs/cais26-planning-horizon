"""
Annotate DAG structures for multi-objective HotpotQA examples.

This script processes preprocessed examples and annotates each component question's DAG
using an LLM, then deterministically merges them into a single DAG for the multi-objective
question.

Workflow:
1. Load preprocessed examples (with component_metadata containing supporting_sentences)
2. For each component question, call LLM to annotate its individual DAG
3. Deterministically merge component DAGs (concatenate nodes, adjust dependencies, add finish)
4. Save back with the dag field populated and annotation metadata

Usage:
    # Annotate train set
    uv run python data/multiobj_hotpotqa/scripts/annotate_dag.py \
        --input data/multiobj_hotpotqa/processed/train.v1.json \
        --output data/multiobj_hotpotqa/processed/train.v1.annotated.json

    # Pilot mode (annotate first 10 examples only)
    uv run python data/multiobj_hotpotqa/scripts/annotate_dag.py \
        --input data/multiobj_hotpotqa/processed/train.v1.json \
        --output data/multiobj_hotpotqa/processed/train.v1.annotated.json \
        --limit 10

    # Single-threaded mode (for debugging)
    uv run python data/multiobj_hotpotqa/scripts/annotate_dag.py \
        --input data/multiobj_hotpotqa/processed/train.v1.json \
        --output data/multiobj_hotpotqa/processed/train.v1.annotated.json \
        --workers 1
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import json
import sys

from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from planning.services.openai import OpenAIClient  # noqa: E402
from data.multiobj_hotpotqa.scripts.utils import (  # noqa: E402
    INPUT_COST,
    OUTPUT_COST,
    load_json,
    get_dag_annotation_response_format,
    get_api_params,
)

# Default model for DAG annotation
DEFAULT_MODEL = "gpt-5.2-2025-12-11"

# Shared Templates

SYSTEM_MSG_TEMPLATE = """{introduction}

## Context
- **Goal:** Annotate multi-hop QA examples with ground-truth reasoning plans as a Directed Acyclic Graph (DAG). Your outputs must illustrate the required multi-step pattern.
- **Tools Available:**
    - `search(query)`: Retrieves facts (returns a string).
    - `reasoning(instruction)`: Applies simple logic (comparison, filtering, conditionals) on strings.
- **Annotator Role:** The answer and supporting facts are provided for your annotation. The QA system you annotate for cannot see them; avoid any leakage or hinting in your node inputs.

{definition}

## Process
1. **Rephrase:** Restate the input question in clear, natural English, keeping all constraints.
2. **Construct DAG:** Break down the question into a minimal, strictly ordered DAG using the above rules and the tools provided.

## Node Construction Rules
- **Self-Contained:** Each node’s `input` is a clear, standalone sentence (replace `$i` with the literal value).
- **Natural Embedding:** `$i` appears as though the entity or value is being asked about, not referenced. Never meta-phrase.
- **Literal Use:** Treat `$i` as a value to inquire about—not a doc/source.
- **Conciseness:** Each node contains exactly **one** input sentence.
- **No Redundant Search:** Never search for information already stated in the question; instead, use it in a reasoning node if involved.

{requirements}

## Output Format
Return a single flat JSON object, no Markdown, no comments:
- `rephrased_question` (string)
- `dag`: List of nodes:
    - `function`: "search" or "reasoning"
    - `dependencies`: List of integers (node indices)
    - `input`: String (standalone, with `$i` references as needed)

# Example

{examples}

# Notes
- Do not collapse comparison questions into one search.
- Do not add extra search steps after the two parallel searches.
- Always retrieve the same attribute for both entities with parallel phrasing.
- Always end with exactly one reasoning node that makes the selection and states what to output.
- Never reference the answer or supporting facts in any node input fields.
""".strip()


USER_MSG_TEMPLATE = """Generate the reasoning DAG for this question.

Question: {question}
Answer: {answer}
Supporting Facts: {supporting_facts}

{notes}
""".strip()

# Bridge-specific components

BRG_INTRODUCTION = """
You're tasked with always decomposing multi-hop "bridge" questions into at least two sequential `search` steps—never a single step.

For every question that requires finding a pivot entity to reach the answer, follow this strict pattern:
- **Node 0:** Find the pivot entity with `search`.
- **Node 1:** Use that pivot (`$0`) in a second `search` to get the final answer.
- **Node 2 (optional):** Add a `reasoning` step only if the second `search` does not directly produce the answer.
A bridge question **cannot** be solved by a single `search`. Annotate every example accordingly.
""".strip()

BRG_DEFINITION = """
## Required "Bridge" Pattern
- For all "bridge" questions:
    - The minimum DAG: Two `search` nodes, where the second node *always* depends on the result of the first by embedding `$0` directly and naturally.
    - A third, optional `reasoning` node can be included if and only if the second `search` does not directly match the provided answer.
    - No bridge question can be solved with a single search. Violations will be considered incorrect.
""".strip()

BRG_REQUIREMENTS = """
## Logic & Reasoning
- Use `reasoning` nodes only if the final `search` node does not yield the required answer, per the answer provided.
- When to stop:
    - If your last search yields the answer: **Stop.**
    - If further logic is needed: Add a reasoning node that references both the candidate ($0) and evidence ($1 or more).
- **Reasoning Node Format:** One sentence, referencing all required prior node outputs:
    - Template: "If [entity] $0 satisfies [constraint] (evidence: $1), output $0; otherwise output unknown."
""".strip()

BRG_EXAMPLES = """
### Example 1: Bridge Pattern (No Reasoning Needed)
**INPUT:**
- **Question:** PersonA plays for which team; what city is that team based in?
- **Answer:** City A
- **Supporting Facts:** "PersonA plays for the Tigers." "The Tigers are based in City A."

**OUTPUT:**
{
  "rephrased_question": "In which city is the team that PersonA plays for based?",
  "dag": [
    {
      "function": "search",
      "dependencies": [],
      "input": "Which team does PersonA play for?"
    },
    {
      "function": "search",
      "dependencies": [0],
      "input": "What city is the team $0 based in?"
    }
  ]
}
*In this bridge pattern, the answer requires two sequential search steps: finding a team, then using it to find a city. No single-search solution is possible.*

### Example 2: Bridge Pattern with Reasoning (Verification Needed)
**INPUT:**
- **Question:** Which school, founded in 1901, did PersonB attend?
- **Answer:** School X
- **Supporting Facts:** "PersonB graduated from School X." "School X was founded in 1901."

**OUTPUT:**
{
  "rephrased_question": "Which school, founded in 1901, did PersonB attend?",
  "dag": [
    {
      "function": "search",
      "dependencies": [],
      "input": "Which school did PersonB attend?"
    },
    {
      "function": "search",
      "dependencies": [0],
      "input": "In what year was the school $0 founded?"
    },
    {
      "function": "reasoning",
      "dependencies": [0, 1],
      "input": "If the school $0 was founded in 1901 (founding year: $1), output $0; otherwise output \"unknown\"."
    }
  ]
}
*Here, even though we find a school (step 1), we verify its founding year (step 2), and use a reasoning step to check the constraint. Still, two search steps precede reasoning. Never compress this into a single `search`.*

### Example 3: Bridge Pattern with Reasoning (Condition Verification)
**INPUT:**
- **Question:** What speech did Person A deliver at Place B, during which they declared "Phrase C"?
- **Answer:** Speech X
- **Supporting Facts:** "Person A delivered Speech X at Place B." "During Speech X, Person A declared 'Phrase C'."

**OUTPUT:**
{
  "rephrased_question": "What speech did Person A deliver at Place B, during which they declared \"Phrase C\"?",
  "dag": [
    {
      "function": "search",
      "dependencies": [],
      "input": "What speech did Person A deliver at Place B?"
    },
    {
      "function": "search",
      "dependencies": [0],
      "input": "Did Person A declare \"Phrase C\" during the speech $0?"
    },
    {
      "function": "reasoning",
      "dependencies": [0, 1],
      "input": "If Person A declared \"Phrase C\" during the speech $0 (evidence: $1), output $0; otherwise output unknown."
    }
  ]
}
*Even with conditions, maintain the two-step bridge: find candidate in step 1, gather evidence in step 2, reason in step 3. No merging allowed.*
"""

BRG_NOTES = """
**Notes:**
- Do **not** generate DAGs that answer a bridge question in a single step.
- Every bridge question DAG must have at least two search nodes following the pivot pattern, with an optional reasoning node if required.
- Always embed the output of the first search as `$0` into the second search's query naturally.
- Do not use meta-language, wrappers, or any non-natural phrasing in the input fields.
- Never reference the answer or supporting facts in your node input fields.

Remember, your primary objective is to enforce the required two-step `search` bridge pattern on all relevant questions. Reasoning nodes are only allowed as an addition—never as a replacement for the two-search pattern.
"""

# Comparison-specific components

CMP_INTRODUCTION = """
You're tasked with decomposing multi-hop "comparison" questions into exactly two parallel `search` steps followed by a single `reasoning` step—never fewer.

For every comparison question that requires choosing between Entity A and Entity B based on an attribute, follow this strict pattern:
- **Node 0:** Retrieve Entity A’s attribute with `search`.
- **Node 1:** Retrieve Entity B’s attribute with `search`.
- **Node 2:** Use `reasoning` to compare $0 vs $1 and output the required entity/value.

A comparison question **must** include one reasoning step that explicitly compares the two retrieved values. Annotate every example accordingly.
""".strip()

CMP_DEFINITION = """
## Required “Comparison” Pattern
- For all “comparison” questions:
    - The DAG must contain exactly:
        - Two `search` nodes with no dependencies (parallel retrieval).
        - One `reasoning` node that depends on both searches.
    - The two `search` queries must be parallel in phrasing (same attribute asked for both entities) so outputs are directly comparable.
    - The final decision must be made in the `reasoning` node, not via a third `search`.
""".strip()

CMP_REQUIREMENTS = """
## Reasoning Node Requirements (Comparison)
- Must be exactly one sentence.
- Must explicitly name both entities and reference both retrieved values ($0 and $1).
- Must state what to return (e.g., “output PersonA”).
- Use a direct comparative construction; avoid awkward roles for $i (e.g., do not write “If $0 shows...”).
- Template: "[EntityA] has [attribute] $0, and [EntityB] has [attribute] $1; based on the question’s criterion, output the correct entity."
""".strip()

CMP_EXAMPLES = """
**INPUT:**
- **Question:** Who was born earlier, PersonA or PersonB?
- **Answer:** PersonA
- **Supporting Facts:** "PersonA was born on DateA." "PersonB was born on DateB."

**OUTPUT:**
{
  "rephrased_question": "Which person was born earlier, PersonA or PersonB?",
  "dag": [
    {
      "function": "search",
      "dependencies": [],
      "input": "What is PersonA's date of birth?"
    },
    {
      "function": "search",
      "dependencies": [],
      "input": "What is PersonB's date of birth?"
    },
    {
      "function": "reasoning",
      "dependencies": [0, 1],
      "input": "PersonA was born on $0, and PersonB was born on $1; output the person who was born earlier."
    }
  ]
}
""".strip()

CMP_NOTES = """
**Notes**
- Do not collapse comparison questions into one search.
- Do not add extra search steps after the two parallel searches.
- Always retrieve the same attribute for both entities with parallel phrasing.
- Always end with exactly one reasoning node that makes the selection and states what to output.
- Never reference the answer or supporting facts in any node input fields.
""".strip()


@dataclass
class AnnotationResult:
    dag_nodes: list[dict[str, Any]]
    rephrased_question: str
    annotation_meta: dict[str, Any]


def format_supporting_facts(component_meta: dict[str, Any]) -> str:
    """Format supporting facts for annotation prompt."""
    lines = []
    for sent in component_meta.get("supporting_sentences", []):
        lines.append(f"- [{sent['title']}] {sent['text']}")
    return "\n".join(lines) if lines else "(No supporting facts available)"


def determine_question_type(question: str) -> str:
    """
    Determine if a question is a bridge or comparison type based on question content.

    Args:
        question: The question text

    Returns:
        "bridge" or "comparison"
    """
    question_lower = question.lower()
    # Heuristic: comparison questions often ask about two entities and comparison keywords
    comparison_keywords = [
        "both",
        "compare",
        "same",
        "different",
        "which",
        "who",
        "vs",
        "versus",
        "earlier",
        "later",
        "more",
        "less",
        "older",
        "younger",
    ]
    if any(kw in question_lower for kw in comparison_keywords):
        # Check for two entity patterns ("X and Y", "X or Y")
        if " and " in question_lower or " or " in question_lower:
            return "comparison"
    return "bridge"


def annotate_component_dag(
    question: str,
    answer: str,
    supporting_facts: str,
    model: str,
    client: OpenAIClient,
    question_type: str = "bridge",
) -> AnnotationResult:
    """
    Annotate DAG for a single component question using LLM.

    Args:
        question: The component question text
        answer: The answer to the question
        supporting_facts: Formatted supporting facts string
        model: Model name to use
        client: OpenAI client
        question_type: "bridge" or "comparison"

    Returns:
        AnnotationResult containing dag_nodes, rephrased_question, and annotation_meta
    """
    # Select appropriate template based on question type
    if question_type == "bridge":
        introduction = BRG_INTRODUCTION
        definition = BRG_DEFINITION
        requirements = BRG_REQUIREMENTS
        examples = BRG_EXAMPLES
        notes = BRG_NOTES
    else:  # comparison
        introduction = CMP_INTRODUCTION
        definition = CMP_DEFINITION
        requirements = CMP_REQUIREMENTS
        examples = CMP_EXAMPLES
        notes = CMP_NOTES

    # Construct prompts using the template
    system_prompt = SYSTEM_MSG_TEMPLATE.format(
        introduction=introduction,
        definition=definition,
        requirements=requirements,
        examples=examples,
    )
    user_prompt = USER_MSG_TEMPLATE.format(
        question=question,
        answer=answer,
        supporting_facts=supporting_facts,
        notes=notes,
    )

    params = get_api_params(
        model,
        max_completion_tokens=5000,
        response_format=get_dag_annotation_response_format(),
    )
    response = client.call(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **params,
    )

    content = response.choices[0].message.content
    result = json.loads(content)
    dag_nodes = result["dag"]
    rephrased_question = result["rephrased_question"]

    # Collect annotation metadata
    usage = response.usage
    annotation_meta = {
        "model": model,
        "question_type": question_type,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }

    return AnnotationResult(dag_nodes=dag_nodes, rephrased_question=rephrased_question, annotation_meta=annotation_meta)


def merge_component_dags(
    component_dags: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Merge multiple component DAGs into a single DAG.

    Strategy:
    1. Concatenate all nodes from component DAGs, adjusting indices
    2. Track terminal nodes (nodes not depended on by others) from each component
    3. Add a single finish node that depends on all terminal nodes

    Args:
        component_dags: List of DAGs (each DAG is a list of nodes)

    Returns:
        Merged DAG with a single finish node
    """
    merged_nodes = []
    terminal_indices = []
    current_offset = 0

    for dag in component_dags:
        if not dag:
            # Empty DAG: skip but note that this component has no nodes
            continue

        # Find terminal nodes in this component DAG (nodes not depended on by others)
        all_deps = set()
        for node in dag:
            all_deps.update(node["dependencies"])

        component_terminals = []
        for i, node in enumerate(dag):
            # Adjust dependencies with offset
            adjusted_node = {
                "function": node["function"],
                "dependencies": [
                    d + current_offset for d in node["dependencies"]
                ],
                "inputs": [node["input"]]
            }
            merged_nodes.append(adjusted_node)

            # Check if this is a terminal node
            if i not in all_deps:
                component_terminals.append(current_offset + i)

        # If no terminals found (shouldn't happen), use the last node
        if not component_terminals and dag:
            component_terminals.append(current_offset + len(dag) - 1)

        terminal_indices.extend(component_terminals)
        current_offset += len(dag)

    # Add finish node that depends on all terminal nodes
    finish_node = {
        "function": "finish",
        "dependencies": terminal_indices,
        "inputs": ["$" + str(i) for i in terminal_indices],
    }
    merged_nodes.append(finish_node)

    return merged_nodes


def extract_all_components(
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Extract all component questions from all examples.

    Args:
        examples: List of preprocessed examples

    Returns:
        List of component info dicts with fields:
            - example_id: ID of the source example
            - example_idx: Index in the examples list
            - component_idx: Index of the component within the example
            - question: The component question text
            - answer: The component answer
            - supporting_facts: Formatted supporting facts string
            - question_type: "bridge" or "comparison"
    """
    all_components = []

    for ex_idx, example in enumerate(examples):
        components = example["metadata"]["components"]
        answers = example["answers"]
        combined_question = example["question"]
        k = example["metadata"]["k"]

        for comp_idx, comp_meta in enumerate(components):
            # Extract question text
            if k == 1:
                question_text = combined_question
            else:
                # Parse numbered questions: "1. Q1\n2. Q2\n..."
                lines = combined_question.split("\n")
                question_text = (
                    lines[comp_idx].split(". ", 1)[1] if comp_idx < len(lines) else ""
                )

            answer = answers[comp_idx] if comp_idx < len(answers) else ""
            supporting_facts = format_supporting_facts(comp_meta)
            question_type = comp_meta.get("type", "bridge")

            all_components.append(
                {
                    "example_id": example["id"],
                    "example_idx": ex_idx,
                    "component_idx": comp_idx,
                    "question": question_text,
                    "answer": answer,
                    "supporting_facts": supporting_facts,
                    "question_type": question_type,
                }
            )

    return all_components


def annotate_components_batch(
    components: list[dict[str, Any]],
    model: str,
    client: OpenAIClient,
    num_workers: int = 1,
) -> dict[tuple[int, int], AnnotationResult]:
    """
    Annotate a batch of component questions in parallel.

    Args:
        components: List of component info dicts
        model: Model name for LLM
        client: OpenAI client
        num_workers: Number of parallel workers (default: 1 for sequential)

    Returns:
        Dict mapping (example_idx, component_idx) to AnnotationResult
    """
    results = {}

    if num_workers == 1:
        # Sequential processing
        for i, comp in enumerate(tqdm(components, desc="Annotating components")):
            try:
                annotation_result = annotate_component_dag(
                    question=comp["question"],
                    answer=comp["answer"],
                    supporting_facts=comp["supporting_facts"],
                    model=model,
                    client=client,
                    question_type=comp["question_type"],
                )
                results[(comp["example_idx"], comp["component_idx"])] = annotation_result
            except Exception as e:
                print(
                    f"    Error annotating component {comp['example_id']}-{comp['component_idx']}: {e}"
                )
                # Store empty result on error
                results[(comp["example_idx"], comp["component_idx"])] = AnnotationResult(
                    dag_nodes=[],
                    rephrased_question="",
                    annotation_meta={"error": str(e), "model": model},
                )
    else:
        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    annotate_component_dag,
                    question=comp["question"],
                    answer=comp["answer"],
                    supporting_facts=comp["supporting_facts"],
                    model=model,
                    client=client,
                    question_type=comp["question_type"],
                ): (comp["example_idx"], comp["component_idx"], comp["example_id"])
                for comp in components
            }

            completed = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Annotating components"):
                ex_idx, comp_idx, example_id = futures[future]
                try:
                    annotation_result = future.result()
                    results[(ex_idx, comp_idx)] = annotation_result
                except Exception as e:
                    print(
                        f"    Error annotating component {example_id}-{comp_idx}: {e}"
                    )
                    # Store empty result on error
                    results[(ex_idx, comp_idx)] = AnnotationResult(
                        dag_nodes=[],
                        rephrased_question="",
                        annotation_meta={"error": str(e), "model": model},
                    )

                completed += 1

    return results


def reconstruct_examples(
    examples: list[dict[str, Any]],
    component_results: dict[tuple[int, int], AnnotationResult],
    model: str,
) -> list[dict[str, Any]]:
    """
    Reconstruct annotated examples from component results.

    Args:
        examples: Original list of examples
        component_results: Dict mapping (example_idx, component_idx) to annotation results
        model: Model name used for annotation

    Returns:
        List of annotated examples
    """
    annotated = []

    for ex_idx, example in enumerate(examples):
        num_components = len(example["metadata"]["components"])

        # Collect component DAGs and token usage
        component_dags = []
        component_rephrased_questions = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        has_error = False

        for comp_idx in range(num_components):
            key = (ex_idx, comp_idx)
            if key in component_results:
                annotation_result = component_results[key]
                dag_nodes = annotation_result.dag_nodes
                rephrased_question = annotation_result.rephrased_question
                anno_meta = annotation_result.annotation_meta
                component_dags.append(dag_nodes)
                component_rephrased_questions.append(rephrased_question)
                total_prompt_tokens += anno_meta.get("prompt_tokens", 0)
                total_completion_tokens += anno_meta.get("completion_tokens", 0)
                if "error" in anno_meta:
                    has_error = True
            else:
                # Missing component result (shouldn't happen)
                component_dags.append([])
                component_rephrased_questions.append("")
                has_error = True

        # Merge component DAGs
        merged_dag = merge_component_dags(component_dags)

        # Create updated example
        updated_example = example.copy()
        updated_example["dag"] = merged_dag
        updated_example["metadata"] = example["metadata"].copy()
        updated_example["metadata"]["dag_annotation"] = {
            "model": model,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "num_components": num_components,
        }
        for comp_idx, rephrased_question in enumerate(component_rephrased_questions):
            updated_example["metadata"]["components"][comp_idx]["rephrased_question"] = rephrased_question

        if has_error:
            updated_example["metadata"]["dag_annotation"]["has_component_errors"] = True

        annotated.append(updated_example)

    return annotated


def process_components_by_type(
    examples: list[dict[str, Any]],
    model: str,
    num_workers: int = 1,
) -> list[dict[str, Any]]:
    """
    Process component questions grouped by type for efficient prompt caching.

    This function extracts all components, groups by type (bridge/comparison),
    processes each group sequentially to maximize prompt cache hits, then
    reconstructs the annotated examples.

    Args:
        examples: List of examples to annotate
        model: Model name for LLM
        num_workers: Number of parallel workers (currently uses 1 client)

    Returns:
        List of annotated examples
    """
    # Step 1: Extract all components
    print("  Step 1: Extracting all component questions...")
    all_components = extract_all_components(examples)
    print(f"    Total components: {len(all_components)}")

    # Step 2: Group by question type
    print("  Step 2: Grouping by question type...")
    bridge_components = [c for c in all_components if c["question_type"] == "bridge"]
    comparison_components = [
        c for c in all_components if c["question_type"] == "comparison"
    ]
    other_components = [
        c
        for c in all_components
        if c["question_type"] not in ["bridge", "comparison"]
    ]

    print(f"    Bridge: {len(bridge_components)}")
    print(f"    Comparison: {len(comparison_components)}")
    print(f"    Other: {len(other_components)}")

    # Step 3: Process each group sequentially (for prompt caching)
    print("  Step 3: Annotating components (grouped by type for caching)...")
    client = OpenAIClient()
    component_results: dict[tuple[int, int], AnnotationResult] = {}

    if bridge_components:
        print(f"    Processing {len(bridge_components)} bridge components...")
        bridge_results = annotate_components_batch(
            bridge_components, model, client, num_workers
        )
        component_results.update(bridge_results)

    if comparison_components:
        print(f"    Processing {len(comparison_components)} comparison components...")
        comparison_results = annotate_components_batch(
            comparison_components, model, client, num_workers
        )
        component_results.update(comparison_results)

    if other_components:
        print(f"    Processing {len(other_components)} other components...")
        other_results = annotate_components_batch(
            other_components, model, client, num_workers
        )
        component_results.update(other_results)

    client.close()

    # Step 4: Reconstruct examples
    print("  Step 4: Reconstructing annotated examples...")
    annotated = reconstruct_examples(examples, component_results, model)
    print(f"    Reconstructed {len(annotated)} examples")

    return annotated


def main(args):
    """Main annotation workflow."""
    load_dotenv(override=True)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load preprocessed examples
    print(f"Loading examples from {input_path}...")
    examples = load_json(input_path)
    print(f"Loaded {len(examples)} examples")

    # Apply limit if specified
    if args.limit:
        examples = examples[: args.limit]
        print(f"Limiting to {len(examples)} examples (--limit {args.limit})")

    # Process examples with type-based grouping for prompt caching
    print(f"\nAnnotating DAGs using model: {args.model}")

    annotated = process_components_by_type(examples, args.model, args.workers)

    # Compute statistics
    total_input_tokens = sum(
        ex.get("metadata", {}).get("dag_annotation", {}).get("prompt_tokens", 0)
        for ex in annotated
    )
    total_output_tokens = sum(
        ex.get("metadata", {}).get("dag_annotation", {}).get("completion_tokens", 0)
        for ex in annotated
    )
    total_tokens = total_input_tokens + total_output_tokens

    errors = sum(
        1
        for ex in annotated
        if "error" in ex.get("metadata", {}).get("dag_annotation", {})
        or ex.get("metadata", {})
        .get("dag_annotation", {})
        .get("has_component_errors", False)
    )

    print(f"\n{'=' * 80}")
    print("ANNOTATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total examples: {len(annotated)}")
    print(f"Successfully annotated: {len(annotated) - errors}")
    print(f"With errors: {errors}")

    # Token usage
    print(f"\n{'=' * 80}")
    print("TOKEN USAGE")
    print(f"{'=' * 80}")
    print(f"Model: {args.model}")
    print(f"Input tokens:  {total_input_tokens:,}")
    print(f"Output tokens: {total_output_tokens:,}")
    print(f"Total tokens:  {total_tokens:,}")

    # Cost calculation
    if args.model in INPUT_COST and args.model in OUTPUT_COST:
        input_cost = total_input_tokens * INPUT_COST[args.model]
        output_cost = total_output_tokens * OUTPUT_COST[args.model]
        total_cost = input_cost + output_cost

        print(f"\n{'=' * 80}")
        print("COST BREAKDOWN")
        print(f"{'=' * 80}")
        print(
            f"Input cost:  ${input_cost:.6f} (@${INPUT_COST[args.model] * 1_000_000:.3f}/1M tokens)"
        )
        print(
            f"Output cost: ${output_cost:.6f} (@${OUTPUT_COST[args.model] * 1_000_000:.3f}/1M tokens)"
        )
        print(f"Total cost:  ${total_cost:.6f}")
    else:
        print(f"\nNote: Cost calculation not available for model {args.model}")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotated, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Saved annotated examples to: {output_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate DAG structures for multi-objective HotpotQA examples"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file (preprocessed examples)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file (annotated examples)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use for DAG annotation (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to annotate (for piloting)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Number of parallel workers (default: 50, use 1 for sequential)",
    )

    args = parser.parse_args()
    main(args)
