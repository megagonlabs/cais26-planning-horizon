"""
Tool module for Worker Agents.
"""

# Common tools
from .base_tools import create_finish_tool

# Specialized standalone tools
from .search_tool import (
    create_ddg_google_search_tool,
    create_google_search_api_tool,
    create_llama_index_search_tool,
    create_pyserini_faiss_search_tool,
    create_pyserini_prebuilt_faiss_search_tool,
    create_weaviate_search_tool,
)

# Import factory classes
from .kopl_tools import (
    KoPLToolFactory,
    KoPLSchemaFreeToolFactory,
    KoPLFindFilterConceptToolFactory,
    KoPLKeyOnlyToolFactory,
    KoPLKeyValueToolFactory,
)
from .atomic_kb_query_tools import (
    AtomicKBQueryToolFactory,
)

# Tool factory registry for tool sets
TOOL_FACTORIES = {
    "kopl_tools": KoPLToolFactory,
    "kopl_schema_free_tools": KoPLSchemaFreeToolFactory,
    "kopl_find_and_filter_concept_tools": KoPLFindFilterConceptToolFactory,
    "kopl_key_only_tools": KoPLKeyOnlyToolFactory,
    "kopl_key_and_value_tools": KoPLKeyValueToolFactory,
    "atomic_kb_query_tools": AtomicKBQueryToolFactory,
}

# Individual tool creators (only standalone tools, not tool set members)
TOOLS = {
    "finish": create_finish_tool,
    "google_ddg_search": create_ddg_google_search_tool,
    "google_search": create_google_search_api_tool,
    "llama_index_search": create_llama_index_search_tool,
    "pyserini_faiss_search": create_pyserini_faiss_search_tool,
    "pyserini_prebuilt_faiss_search": create_pyserini_prebuilt_faiss_search_tool,
    "weaviate_search": create_weaviate_search_tool,
}

__all__ = [
    "create_finish_tool",
    "create_ddg_google_search_tool",
    "create_google_search_api_tool",
    "create_weaviate_search_tool",
    "create_llama_index_search_tool",
    "create_pyserini_faiss_search_tool",
    "create_pyserini_prebuilt_faiss_search_tool",
    "TOOLS",
    "TOOL_FACTORIES",
]
