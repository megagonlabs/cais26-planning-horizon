from pathlib import Path
from typing import Any, Optional, cast

from datasets import load_dataset
import orjson


def load_agentbank(subset: str, split: str = "train") -> list[dict[str, Any]]:
    """
    Load and normalize AgentBank dataset subset.

    Args:
        subset: Subset name (e.g., "gsm8k")
        split: Dataset split (defaults to "train")

    Returns:
        list[dict[str, Any]]: List of normalized records with 'id', 'query', 'answer'
    """

    def extract_query_from_conversation(
        conversation: list[dict[str, str]],
    ) -> Optional[str]:
        """
        Extract the query from the first human turn in a conversation.

        Args:
            conversation: List of conversation turns with 'from' and 'value' keys

        Returns:
            Optional[str]: The query, or None if not found
        """
        for turn in conversation:
            if turn["from"] == "human":
                return turn["value"]
        return None

    def extract_answer_from_conversation(
        conversation: list[dict[str, str]],
    ) -> Optional[str]:
        """
        Extract the answer from the last GPT turn in a conversation.

        Args:
            conversation: List of conversation turns with 'from' and 'value' keys

        Returns:
            Optional[str]: The answer, or None if not found
        """
        for turn in reversed(conversation):
            if turn["from"] == "gpt":
                return turn["value"]
        return None

    ds = load_dataset("Solaris99/AgentBank", subset)
    records = []
    for item in ds[split]:
        item = cast(dict[str, Any], item)
        id_ = item["id"]
        query = extract_query_from_conversation(item["conversations"])
        answer = extract_answer_from_conversation(item["conversations"])
        records.append({"id": id_, "query": query, "answer": answer})
    return records


def load_huskyqa(
    local_path: Path = Path("data/huskyqa/test.json"),
) -> list[dict[str, Any]]:
    """
    Load and normalize HuskyQA dataset.

    HuskyQA only has a test split (292 records).

    Args:
        local_path: Optional local path to JSON file (defaults to "data/huskyqa/test.json")

    Returns:
        list[dict[str, Any]]: List of normalized records with 'id', 'query', 'answer'
    """
    if local_path.exists():
        # Load from local file if it exists
        ds = load_dataset("json", data_files={"test": str(local_path)})
    else:
        # Otherwise load from Hugging Face Hub
        ds = load_dataset("agent-husky/HuskyQA")
    records = []
    for item in ds["test"]:
        item = cast(dict[str, Any], item)
        id_ = item["index"]
        query = item["question"]
        answer = item["answer"]
        records.append({"id": id_, "query": query, "answer": answer})
    return records


def load_drop(
    local_path: Path = Path("data/husky-drop/test.v2025-09-16.jsonl"),
    use_original: bool = False,
) -> list[dict[str, Any]]:
    """
    Load and normalize DROP dataset.

    Returns:
        list[dict[str, Any]]: List of normalized records with 'id', 'query', 'answer'
    """
    if use_original:
        # Use the original DROP eval set
        local_path = Path("data/husky-drop/test.json")
        if local_path.exists():
            # Load from local file if it exists
            ds = load_dataset("json", data_files={"test": str(local_path)})
        else:
            # Otherwise load from Hugging Face Hub
            ds = load_dataset("agent-husky/DROP-eval")
    else:
        # Use the revised DROP eval set
        ds = load_dataset("json", data_files={"test": str(local_path)})
    records = []
    for item in ds["test"]:
        item = cast(dict[str, Any], item)
        id_ = item["index"]
        query = item["question"]
        answer = item["answer"]
        records.append({"id": id_, "query": query, "answer": answer})
    return records


def load_hotpotqa(local_path: Path = Path("data/hotpotqa"), split: str = "train") -> list[dict[str, Any]]:
    """
    Load and normalize HotpotQA dataset.

    Args:
        split: Dataset split (defaults to "train")

    Returns:
        list[dict[str, Any]]: List of normalized records with 'id', 'query', 'answer', 'task_config'
    """
    FILES = {
        "train": "hotpot_train_v1.1.json",
        "dev": "hotpot_dev_fullwiki_v1.json",
        "test": "hotpot_test_fullwiki_v1.json",
    }
    assert split in FILES, f"Invalid split: {split}"
    records = []
    file_path = local_path / FILES[split]
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}. Please download it first.")
    with open(file_path, "r") as f:
        raw_data = orjson.loads(f.read())
        for item in raw_data:
            item = cast(dict[str, Any], item)
            id_ = item["_id"]
            query = item["question"]
            answer = item.get("answer", "")  # test set does not have answers
            record = {
                "id": id_,
                "query": query,
                "answer": answer,
                "task_config": {},  # Empty dict for HotpotQA (no program/dag)
            }
            records.append(record)
    return records


def load_multihoprag(
    local_path: Path = Path("data/multihoprag/MultiHopRAG.json"),
) -> list[dict[str, Any]]:
    """
    Load and normalize MultiHopRAG dataset.

    MultiHopRAG only has a test split (2.56k records).

    Args:
        local_path: Optional local path to JSON file (defaults to "data/multihoprag/MultihopRAG.json")

    Returns:
        list[dict[str, Any]]: List of normalized records with 'id', 'query', 'answer'
    """
    if local_path.exists():
        # Load from local file if it exists
        ds = load_dataset("json", data_files={"train": str(local_path)})
    else:
        # Otherwise load from Hugging Face Hub
        ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
    records = []
    for i, item in enumerate(ds["train"]):
        item = cast(dict[str, Any], item)
        id_ = str(i)
        query = item["query"]
        answer = item["answer"]
        records.append({"id": id_, "query": query, "answer": answer})
    return records


def load_kqa_pro(
    local_path: Path = Path("data/kopl_kbqa/kqa_pro/processed"), split: str = "train"
) -> list[dict[str, Any]]:
    """
    Load and normalize KQA Pro dataset.

    Args:
        local_path: Optional local path to dataset directory (defaults to "data/kopl_kbqa/kqa_pro/")
        split: Dataset split (defaults to "train")

    Returns:
        list[dict[str, Any]]: List of normalized records with 'id', 'query', 'answer'
    """
    FILES = {
        "train-no-demo": "train.v1.json",
        "test-no-demo": "val.v1.json",
        "train": "train.v1.50nn.json",
        "test": "val.v1.50nn.json",
    }
    assert split in FILES, f"Invalid split: {split}"
    records = []
    file_path = local_path / FILES[split]
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}. Please download it first.")
    with open(file_path, "r") as f:
        data = orjson.loads(f.read())
        for i, item in enumerate(data):
            item = cast(dict[str, Any], item)
            query = item["question"]
            answer = item["answer"]
            # Build task_config with groundtruth program and other task information
            task_config = {
                "program": item.get("program", []),
                "dag": item.get("dag", []),
            }
            # Include demonstration_candidates if present (for in-context learning)
            record = {
                "id": item["id"],
                "query": query,
                "answer": answer,
                "task_config": task_config,
            }
            if "demonstration_candidates" in item:
                record["demonstration_candidates"] = item["demonstration_candidates"]
            records.append(record)
    return records


def load_planbench(domain: str, local_path: Path = Path("data/planbench/processed/")) -> list[dict[str, Any]]:
    """
    Load and normalize PlanBench dataset.

    Args:
        domain: Domain name (e.g., "blocksworld_basic", "logistics_randomized")
        local_path: Path to processed PlanBench data directory

    Returns:
        list[dict[str, Any]]: List of normalized records with 'id', 'query', 'answer', 'task_config'
    """
    # Map domain names to processed JSON files
    data_file = local_path / f"{domain}.v1.json"

    if not data_file.exists():
        raise FileNotFoundError(f"PlanBench data file not found: {data_file}. Please run preprocessing first.")

    records = []
    with open(data_file) as f:
        for line in f:
            instance = orjson.loads(line)

            # Extract core fields
            instance_id = instance["id"]
            objects = instance.get("objects", {})
            initial_state = instance.get("initial_state", [])
            goal_state = instance.get("goal_state", [])
            ground_truth_plan = instance.get("plan", [])

            # Format as text query for LLM
            query = (
                "Plan to achieve the goal state from the current state.\n\n"
                f"Objects:\n{orjson.dumps(objects, option=orjson.OPT_INDENT_2).decode()}\n\n"
                "Current State:\n" + "\n".join(f"  {pred}" for pred in sorted(initial_state)) + "\n\n"
                "Goal State:\n" + "\n".join(f"  {pred}" for pred in sorted(goal_state))
            )

            # Ground truth answer is the plan sequence
            answer = orjson.dumps(ground_truth_plan).decode()

            # Pass full instance data as task_config for tools/agents
            task_config = {
                "instance_id": instance_id,
                "objects": objects,
                "initial_state": initial_state,
                "goal_state": goal_state,
                "ground_truth_plan": ground_truth_plan,
            }

            records.append(
                {
                    "id": instance_id,
                    "query": query,
                    "answer": answer,
                    "task_config": task_config,
                }
            )

    return records


def load_grailqa(
    local_path: Path = Path("data/atomic_kbqa/grailqa/processed/"),
    split: str = "train",
) -> list[dict[str, Any]]:
    """
    Load and normalize GrailQA dataset.

    Args:
        local_path: Path to processed GrailQA data directory (defaults to "data/atomic_kbqa/grailqa/processed/")
        split: Dataset split - "train" or "test" (defaults to "train")

    Returns:
        list[dict[str, Any]]: List of normalized records with 'id', 'query', 'answer', 'task_config', optionally 'demonstration_candidates'

    Note:
        - The "train-no-demo" split without demonstrations has 500 examples
        - The "train" split includes 50 nearest neighbor candidates per example
    """
    FILES = {
        "train-no-demo": "grailqa_train.v1.json",
        "test-no-demo": "grailqa_test.v1.json",
        "train": "grailqa_train.v1.50nn.json",
        "test": "grailqa_test.v1.50nn.json",
    }
    if split not in FILES:
        raise ValueError(f"Invalid split: {split}")

    file_path = local_path / FILES[split]
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}. Please run preprocessing first.")

    records = []
    with open(file_path) as f:
        data = orjson.loads(f.read())
        for item in data:
            item = cast(dict[str, Any], item)
            id_ = item["id"]
            query = item["question"]
            answer = item["answer"]
            answer_label = item.get("answer_label", [])

            # Build the record
            record = {
                "id": id_,
                "query": query,
                "answer": answer,
                "answer_label": answer_label,
            }

            ## Metadata
            task_config = {
                "dataset": "grailqa",
                "original_id": item.get("original_id"),
                "entities": item.get("entities", []),
                "s_expression": item.get("s_expression"),
                "function_list": item.get("function_list", []),
                "level": item.get("level", "unknown"),
                "metadata": item.get("metadata", {}),
            }
            record["task_config"] = task_config

            # Include demonstration_candidates if present (for in-context learning)
            if "demonstration_candidates" in item:
                record["demonstration_candidates"] = item["demonstration_candidates"]

            records.append(record)

    return records


def load_webqsp(
    local_path: Path = Path("data/atomic_kbqa/webqsp/processed/"),
    split: str = "train",
) -> list[dict[str, Any]]:
    """
    Load and normalize WebQSP dataset.

    Args:
        local_path: Path to processed WebQSP data directory (defaults to "data/atomic_kbqa/webqsp/processed/")
        split: Dataset split - "train" or "test" (defaults to "train")

    Returns:
        list[dict[str, Any]]: List of normalized records with 'id', 'query', 'answer', 'task_config', optionally 'demonstration_candidates'

    Note:
        - The "train-no-demo" split without demonstrations has 500 examples
        - The "train" split includes 50 nearest neighbor candidates per example
    """
    FILES = {
        "train-no-demo": "webqsp_train.v1.json",
        "test-no-demo": "webqsp_test.v1.json",
        "train": "webqsp_train.v1.50nn.json",
        "test": "webqsp_test.v1.50nn.json",
    }
    if split not in FILES:
        raise ValueError(f"Invalid split: {split}")

    file_path = local_path / FILES[split]
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}. Please run preprocessing first.")

    records = []
    with open(file_path) as f:
        data = orjson.loads(f.read())
        for item in data:
            item = cast(dict[str, Any], item)
            id_ = item["id"]
            query = item["question"]
            answer = item["answer"]
            answer_label = item.get("answer_label", [])

            # Build the record
            record = {
                "id": id_,
                "query": query,
                "answer": answer,
                "answer_label": answer_label,
            }

            ## Metadata
            task_config = {
                "dataset": "webqsp",
                "original_id": item.get("original_id"),
                "entities": item.get("entities", []),
                "s_expression": item.get("s_expression"),
                "function_list": item.get("function_list", []),
                "metadata": item.get("metadata", {}),
            }
            record["task_config"] = task_config

            # Include demonstration_candidates if present (for in-context learning)
            if "demonstration_candidates" in item:
                record["demonstration_candidates"] = item["demonstration_candidates"]

            records.append(record)

    return records


def load_graphq(
    local_path: Path = Path("data/atomic_kbqa/graphq/processed/"),
    split: str = "train",
) -> list[dict[str, Any]]:
    """
    Load and normalize GraphQ dataset.

    Args:
        local_path: Path to processed GraphQ data directory (defaults to "data/atomic_kbqa/graphq/processed/")
        split: Dataset split - "train" or "test" (defaults to "train")

    Returns:
        list[dict[str, Any]]: List of normalized records with 'id', 'query', 'answer', 'task_config', optionally 'demonstration_candidates'

    Note:
        - The "train-no-demo" split without demonstrations has 500 examples
        - The "train" split includes 50 nearest neighbor candidates per example
    """
    FILES = {
        "train-no-demo": "graphq_train.v1.json",
        "test-no-demo": "graphq_test.v1.json",
        "train": "graphq_train.v1.50nn.json",
        "test": "graphq_test.v1.50nn.json",
    }
    if split not in FILES:
        raise ValueError(f"Invalid split: {split}")

    file_path = local_path / FILES[split]
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}. Please run preprocessing first.")

    records = []
    with open(file_path) as f:
        data = orjson.loads(f.read())
        for item in data:
            item = cast(dict[str, Any], item)
            id_ = item["id"]
            query = item["question"]
            answer = item["answer"]
            answer_label = item.get("answer_label", [])

            # Build the record
            record = {
                "id": id_,
                "query": query,
                "answer": answer,
                "answer_label": answer_label,
            }

            ## Metadata
            task_config = {
                "dataset": "graphq",
                "original_id": item.get("original_id"),
                "entities": item.get("entities", []),
                "s_expression": item.get("s_expression"),
                "function_list": item.get("function_list", []),
                "metadata": item.get("metadata", {}),
            }
            record["task_config"] = task_config

            # Include demonstration_candidates if present (for in-context learning)
            if "demonstration_candidates" in item:
                record["demonstration_candidates"] = item["demonstration_candidates"]

            records.append(record)

    return records


def load_multiobj_hotpotqa(
    local_path: Path = Path("data/multiobj_hotpotqa/processed/"),
    split: str = "train",
) -> list[dict[str, Any]]:
    """
    Load and normalize multi-objective HotpotQA dataset.

    Args:
        local_path: Path to processed data directory (defaults to "data/multiobj_hotpotqa/processed/")
        split: Dataset split (defaults to "train")

    Returns:
        list[dict[str, Any]]: List of normalized records with 'id', 'query', 'answer', 'task_config', 'demonstration_candidates'
    """
    FILES = {
        "train": "train.v1.annotated.50nn.json",
        "test": "test.v1.annotated.50nn.json",
        "train-no-demos": "train.v1.annotated.json",
        "test-no-demos": "test.v1.annotated.json",
    }
    if split not in FILES:
        raise ValueError(f"Invalid split: {split}")

    file_path = local_path / FILES[split]
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}. Please run preprocessing first.")

    records = []
    with open(file_path) as f:
        data = orjson.loads(f.read())
        for item in data:
            record = {
                "id": item["id"],
                "query": item["question"],
                "answer": item["answers"],  # Always list[str]
                "task_config": {
                    "k": item["metadata"]["k"],
                    "components": item["metadata"]["components"],
                },
            }
            # Add demonstration candidates if present
            if "demonstration_candidates" in item:
                record["demonstration_candidates"] = item["demonstration_candidates"]
            records.append(record)

    return records


def load_dataset_router(dataset_id: str, subset: Optional[str] = None, split: str = "train") -> list[dict[str, Any]]:
    """
    Router to load and normalize datasets from Hugging Face Hub.

    Args:
        dataset_id: Hugging Face dataset ID (e.g., "Solaris99/AgentBank")
        subset: Optional subset name
        split: Dataset split (defaults to "train")

    Returns:
        list[dict[str, Any]]: List of normalized records with 'id', 'query', 'answer'

    Raises:
        NotImplementedError: If the dataset is not supported

    Notes:
        To add support for a new dataset:
        1. Implement a loader function (e.g., `load_new_dataset`) that loads the dataset
           and returns a list of dicts with normalized fields: 'id', 'query', 'answer'.
        2. Add a condition in this router function to dispatch to the new loader based on dataset_id.
        3. Ensure the loader handles any dataset-specific extraction logic internally.
    """
    if dataset_id == "Solaris99/AgentBank":
        if subset is None:
            raise ValueError("Subset is required for AgentBank dataset")
        return load_agentbank(subset, split)
    elif dataset_id == "agent-husky/HuskyQA":
        return load_huskyqa()
    elif dataset_id == "agent-husky/DROP-eval":
        return load_drop()
    elif dataset_id == "HotpotQA":
        return load_hotpotqa(split=split)
    elif dataset_id == "multiobj/hotpotqa":
        return load_multiobj_hotpotqa(split=split)
    elif dataset_id == "yixuantt/MultiHopRAG":
        return load_multihoprag()
    elif dataset_id == "kopl_kbqa/kqa_pro":
        return load_kqa_pro(split=split)
    elif dataset_id == "atomic_kbqa/grailqa":
        # GrailQA dataset for atomic KBQA
        # Use subset to specify whether to include demonstration candidates
        return load_grailqa(split=split)
    elif dataset_id == "atomic_kbqa/webqsp":
        # WebQSP dataset for atomic KBQA
        # Use subset to specify whether to include demonstration candidates
        return load_webqsp(split=split)
    elif dataset_id == "atomic_kbqa/graphq":
        # GraphQ dataset for atomic KBQA
        # Use subset to specify whether to include demonstration candidates
        return load_graphq(split=split)
    elif dataset_id.startswith("planbench/"):
        # Extract domain from dataset_id (e.g., "planbench/blocksworld_basic")
        domain = dataset_id.split("/", 1)[1]
        return load_planbench(domain)
    else:
        raise NotImplementedError(
            f"Dataset {dataset_id} is not supported yet. Please implement a loader function in data_utils.py."
        )
