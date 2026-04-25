"""
Tool wrapper for a search-related tools.

This module provides tools specifically designed for solving
search-related tasks.
"""

from pathlib import Path
from typing import Annotated, Any, Optional, TYPE_CHECKING
import json
import os
import threading
import time

from ddgs import DDGS
from dotenv import load_dotenv
from googleapiclient.discovery import build as google_build

# Lazy imports - only load heavy ML libraries when actually needed
if TYPE_CHECKING:
    from llama_index.core import load_index_from_storage
    from llama_index.core import Settings
    from llama_index.core import StorageContext
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore
    import faiss

import ddgs.exceptions as ddgs_exceptions
import orjson

from .base_tools import FunctionTool, ParamMetadata

# Module-level caches for Pyserini resources
_PYSERINI_CORPUS_CACHE: dict[str, dict[str, dict[str, str]]] = {}
_PYSERINI_CORPUS_CACHE_LOCK = threading.Lock()

# Thread-local storage for FAISS/Lucene searchers to avoid sharing PyTorch models
# and potential thread-safety issues with searcher instances
_PYSERINI_THREAD_LOCAL_CACHE = threading.local()


def close_pyserini_resources() -> None:
    """
    Close thread-local Pyserini resources to prevent JVM crashes on shutdown.

    This should be called at the end of a thread's execution if Pyserini tools
    were used in that thread.
    """
    # Close Lucene searchers
    if hasattr(_PYSERINI_THREAD_LOCAL_CACHE, "lucene_searchers"):
        for searcher in _PYSERINI_THREAD_LOCAL_CACHE.lucene_searchers.values():
            try:
                # LuceneSearcher in Pyserini has a close() method
                if hasattr(searcher, "close"):
                    searcher.close()
            except Exception:
                pass
        _PYSERINI_THREAD_LOCAL_CACHE.lucene_searchers = {}

    # Close FAISS searchers
    if hasattr(_PYSERINI_THREAD_LOCAL_CACHE, "searchers"):
        _PYSERINI_THREAD_LOCAL_CACHE.searchers = {}


class WeaviateClient:
    """
    Client wrapper for querying a Weaviate vector store.

    This class encapsulates a Weaviate client instance and provides a small,
    robust search API used by the tool factory.
    """

    def __init__(
        self,
        collection_name: str,
        topk: int = 1,
        query_prefix: Optional[str] = None,
        port: int = 8080,
        max_retries: int = 5,
    ) -> None:
        """Initialize the Weaviate client.

        Args:
            collection_name (str): The name of the Weaviate collection to search.
            topk (int): The number of top results to retrieve (default: 1).
            query_prefix (Optional[str]): An optional prefix to add to each query.
            port (int): The port number for the Weaviate instance (default: 8080).
            max_retries (int): The maximum number of retry attempts (default: 5).
        """
        import weaviate
        self.client = weaviate.connect_to_local(host="localhost", port=port)
        self.collection = self.client.collections.use(collection_name)

        self.topk = topk
        self.query_prefix = query_prefix.strip() + " " if query_prefix else ""
        self.max_retries = max_retries

    def search(self, query: str) -> str:
        """
        Perform a similarity search against the configured collection.

        The provided query is optionally prefixed and the method retries on
        transient errors using exponential backoff.

        Args:
            query: The user's free-text query.

        Returns:
            A formatted string containing the top results, or an error message
            beginning with "Error(" on failure.
        """
        import weaviate.classes.query as wq
        if self.query_prefix:
            query = self.query_prefix + query

        attempt = 0
        while attempt < self.max_retries:
            try:
                response = self.collection.query.near_text(
                    query=query,
                    limit=self.topk,
                    return_metadata=wq.MetadataQuery(distance=True),
                )

                lines: list[str] = []
                # response.objects may be empty or missing fields; guard access
                for obj in getattr(response, "objects", []):
                    props: dict[str, Any] = getattr(obj, "properties", {}) or {}
                    title = props.get("title", "Untitled")
                    paragraph = props.get("paragraph", "")
                    lines.append(f"# {title}")
                    lines.append(f"{paragraph}")
                    lines.append("-" * 80)
                return "\n".join(lines) if lines else "Error(No results)"
            except Exception as e:
                attempt += 1
                if attempt < self.max_retries:
                    # exponential backoff before next attempt
                    time.sleep(2**attempt)
                    continue
                return f"Error({str(e)})"
        return "Error(Reached maximum retries)"

    def close(self) -> None:
        """Close internal Weaviate client connection if supported."""
        try:
            self.client.close()
        except Exception:
            # Suppress errors during cleanup to avoid noisy finalizers.
            pass

    def __del__(self) -> None:
        # Delegate to close() to centralize cleanup logic.
        try:
            self.close()
        except Exception:
            pass


def create_weaviate_search_tool(
    collection_name: str,
    topk: int = 1,
    query_prefix: Optional[str] = None,
    port: int = 8080,
    max_retries: int = 5,
) -> FunctionTool:
    """
    Factory that returns a FunctionTool wrapping a Weaviate similarity search.

    The tool accepts a single argument `query` (string) and returns a human-
    readable aggregation of top-k matching documents (title + paragraph).

    Args:
        collection_name: Name of the Weaviate collection to query.
        topk: Number of top results to return.
        query_prefix: Optional prefix to prepend to queries.
        port: Port for the local Weaviate instance.
        max_retries: Number of retry attempts for transient failures.

    Returns:
        FunctionTool configured to call the underlying WeaviateClient.search.
    """
    weaviate_client = WeaviateClient(
        collection_name=collection_name,
        topk=topk,
        query_prefix=query_prefix,
        port=port,
        max_retries=max_retries,
    )

    def weaviate_search(
        query: Annotated[
            str,
            ParamMetadata(
                description=("A search query to send to Weaviate."),
                min_length=1,
                max_length=30,
            ),
        ],
    ) -> str:
        return weaviate_client.search(query)

    return FunctionTool(weaviate_search, name="weaviate_search")


def format_ddgs_result(result: list[dict[str, Any]]) -> str:
    """
    Format results returned by DDGS into a readable multi-document string.

    Each result item is expected to contain 'title' and 'body' keys.
    """
    lines: list[str] = []

    for item in result:
        lines.append(f"# {item.get('title', 'Untitled')}")
        lines.append(f"{item.get('body', '')}")
        lines.append("-" * 80 + "\n")
    return "\n".join(lines)


def _perform_search(
    ddgs: DDGS,
    query: str,
    backend: str,
    region: str,
    topk: int,
    timelimit: Optional[str],
    retries: int,
) -> str:
    """
    Perform a search request using DDGS with retries and exponential backoff.

    Args:
        ddgs: DDGS client instance.
        query: Cleaned search query string.
        backend: Backend name to pass to ddgs.text (e.g., "duckduckgo", "google").
        region: Region code for the search.
        topk: Number of top results to fetch.
        timelimit: Optional time filter for results.
        retries: Number of attempts (>=1).

    Returns:
        Formatted search results or an error string beginning with "Error(".
    """
    attempt = 0
    backoff = 1.0
    while attempt < retries:
        attempt += 1
        try:
            texts = ddgs.text(
                query,
                region=region,
                max_results=topk,
                timelimit=timelimit,
                backend=backend,
            )
            return format_ddgs_result(texts)
        except (ddgs_exceptions.TimeoutException, TimeoutError) as e:
            if attempt >= retries:
                return f"Error(Timeout after {attempt} attempts: {str(e)})"
            time.sleep(backoff)
            backoff *= 2
            continue
        except ddgs_exceptions.DDGSException as e:
            return f"Error({str(e)})"
        except Exception as e:
            return f"Error({str(e)})"
    return "Error(Reached maximum retries)"


def create_ddg_search_tool(
    topk: int = 1,
    region: str = "us-en",
    timelimit: Optional[str] = None,
    max_retries: int = 5,
) -> FunctionTool:
    """
    Create a DuckDuckGo search tool for retrieving search results.

    **NOT AVAILABLE YET (08/27/2025)**

    This function returns a tool that performs search queries using the
    DuckDuckGo search engine. It supports configurable options for the
    number of results, region, time range, and retry attempts.

    Parameters: https://duckduckgo.com/duckduckgo-help-pages/settings/params

    Args:
        topk (int): Number of top search results to retrieve (default: 1).
        region (str): Region code for the search (default: "us-en").
        timelimit (Optional[str]): Time filter for results. Accepts "d" (day),
            "w" (week), "m" (month), "y" (year), or a date range in the format
            "YYYY-MM-DD..YYYY-MM-DD". If None, no time filter is applied.
        max_retries (int): Maximum number of retry attempts in case of timeout
            errors (default: 5).

    Returns:
        FunctionTool: A tool instance that can execute DuckDuckGo search queries.
    """
    ddgs = DDGS()

    def ddg_search(
        query: Annotated[
            str,
            ParamMetadata(
                description=("A search query to send to DuckDuckGo."),
                min_length=1,
                max_length=30,
            ),
        ],
    ) -> str:
        """
        Perform a DuckDuckGo search and return the results.

        Args:
            query: A search query to send to DuckDuckGo.

        Returns:
            str: The search results from DuckDuckGo, or an error string on failure.

        Raises:
            ValueError: If the query is invalid.
        """
        # Clean the query
        query = query.replace("\n", " ").strip()

        retries = max(1, max_retries)
        return _perform_search(
            ddgs=ddgs,
            query=query,
            backend="duckduckgo",
            region=region,
            topk=topk,
            timelimit=timelimit,
            retries=retries,
        )

    return FunctionTool(ddg_search, name="ddg_search")


def create_ddg_google_search_tool(
    topk: int = 1,
    region: str = "us-en",
    timelimit: Optional[str] = None,
    max_retries: int = 5,
) -> FunctionTool:
    """
    Create a Google search tool for retrieving search results.

    This function returns a tool that performs search queries using the
    Google search engine. It supports configurable options for the
    number of results, region, time range, and retry attempts.

    Args:
        topk (int): Number of top search results to retrieve (default: 1).
        region (str): Region code for the search (default: "US-en").
        timelimit (Optional[str]): Time filter for results. Accepts "d" (day),
            "w" (week), "m" (month), "y" (year), or a date range in the format
            "YYYY-MM-DD..YYYY-MM-DD". If None, no time filter is applied.
        max_retries (int): Maximum number of retry attempts in case of timeout
            errors (default: 5).

    Returns:
        FunctionTool: A tool instance that can execute Google search queries.
    """
    ddgs = DDGS()

    def google_search(
        query: Annotated[
            str,
            ParamMetadata(
                description=("A search query to send to Google."),
                min_length=1,
                max_length=30,
            ),
        ],
    ) -> str:
        """
        Perform a Google search and return the results.

        Args:
            query: A search query to send to Google.

        Returns:
            str: The search results from Google, or an error string on failure.

        Raises:
            ValueError: If the query is invalid.
        """
        # Clean the query
        query = query.replace("\n", " ").strip()

        retries = max(1, max_retries)
        return _perform_search(
            ddgs=ddgs,
            query=query,
            backend="google",
            region=region,
            topk=topk,
            timelimit=timelimit,
            retries=retries,
        )

    return FunctionTool(google_search, name="ddg_google_search")


def create_google_search_api_tool(
    topk: int = 1,
    language: str = "en",
    timelimit: Optional[str] = None,
    max_retries: int = 5,
    ENV_VAR_API_KEY: str = "GOOGLE_API_KEY",
    ENV_VAR_SEARCH_ENGINE_ID: str = "GOOGLE_SEARCH_ENGINE_ID",
    **unused_kwargs: Any,
) -> FunctionTool:
    """
    Create a Google search tool for retrieving search results via Google API.

    This function returns a tool that performs search queries using the
    Google search engine. It supports configurable options for the
    number of results, region, time range, and retry attempts.

    Set the following environment variables to use this tool:
    - Your Google API key (default variable name: GOOGLE_API_KEY)
    - Your Google Custom Search Engine ID (default variable name: GOOGLE_SEARCH_ENGINE_ID)

    Args:
        topk (int): Number of top search results to retrieve (default: 1).
        language (str): Language code for the search (default: "en").
        max_retries (int): Maximum number of retry attempts in case of timeout
            errors (default: 5).
        ENV_VAR_API_KEY (str): Environment variable name for the API key.
        ENV_VAR_SEARCH_ENGINE_ID (str): Environment variable name for the search engine ID.

    Returns:
        FunctionTool: A tool instance that can execute Google search queries.
    """
    # Get API key and Search Engine ID from environment variables
    load_dotenv(override=True)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    # Set up the Google Custom Search API client
    service = google_build("customsearch", "v1", developerKey=google_api_key)

    start_date, end_date = None, None
    if timelimit is not None:
        start_date, end_date = timelimit.split("..")
        if start_date:
            start_date = start_date.strip().replace("/", "-")
        if end_date:
            end_date = end_date.strip().replace("/", "-")

    def google_search(
        query: Annotated[
            str,
            ParamMetadata(
                description=("A search query to send to Google."),
                min_length=1,
                max_length=30,
            ),
        ],
    ) -> str:
        """
        Perform a Google search and return the results.

        Args:
            query: A search query to send to Google.

        Returns:
            str: The search results from Google, or an error string on failure.

        Raises:
            ValueError: If the query is invalid.
        """
        # Clean the query
        query = query.replace("\n", " ").strip()

        # Add timelimit to the query if specified
        if start_date:
            query += f" after:{start_date}"
        if end_date:
            query += f" before:{end_date}"

        retries = max(1, max_retries)
        attempt = 0
        response = {}
        while attempt < retries:
            try:
                response = (
                    service.cse()
                    .list(q=query, cx=google_search_engine_id, hl=language, num=topk)
                    .execute()
                )
                break
            except Exception as e:
                attempt += 1
                if attempt >= retries:
                    return f"Error(Timeout after {attempt} attempts: {str(e)})"
                time.sleep(2**attempt)
                continue
        # Format the response
        lines: list[str] = []
        for item in response.get("items", []):
            lines.append(f"# {item.get('title', 'Untitled')}")
            lines.append(f"{item.get('snippet', '')}")
            lines.append("-" * 80 + "\n")
        return "\n".join(lines) if lines else "Error(No results)"

    return FunctionTool(google_search, name="google_search")


def create_llama_index_search_tool(
    index_path: Path | str,
    embed_model_name: str = "BAAI/bge-base-en-v1.5",
    topk: int = 1,
) -> FunctionTool:
    # Lazy imports - only load when actually creating this tool
    from llama_index.core import load_index_from_storage, Settings, StorageContext
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore
    import faiss

    Settings.llm = None

    # Set up embedding model (same as in create script)
    embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name, embed_batch_size=8, cache_folder=".cache/"
    )
    Settings.embed_model = embed_model

    # Load the faiss index
    if isinstance(index_path, str):
        index_path = Path(index_path)
    faiss_index_path = index_path / "default__vector_store.json"
    faiss_index = faiss.read_index(str(faiss_index_path))
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    # Load the storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=str(index_path)
    )
    index = load_index_from_storage(storage_context)

    # Create query engine
    retriever = index.as_retriever(similarity_top_k=topk)

    def llama_index_search(
        query: Annotated[
            str,
            ParamMetadata(
                description=("A search query to send to LlamaIndex."),
                min_length=1,
                max_length=30,
            ),
        ],
    ) -> str:
        """
        Perform a LlamaIndex search and return the results.

        Args:
            query: A search query to send to LlamaIndex.

        Returns:
            str: The search results from LlamaIndex, or an error string on failure.

        Raises:
            ValueError: If the query is invalid.
        """
        # Clean the query
        query = query.replace("\n", " ").strip()

        # Perform the query
        response = retriever.retrieve(query)

        lines: list[str] = []
        for i, node in enumerate(response):
            lines.append(f"{node.text}")
            lines.append("-" * 80 + "\n")
        return "\n".join(lines) if lines else "Error(No results)"

    return FunctionTool(llama_index_search, name="llama_index_search")


def create_pyserini_prebuilt_faiss_search_tool(
    faiss_index_name: str,
    lucene_index_name: str,
    embed_model_name: str = "BAAI/bge-base-en-v1.5",
    include_title: bool = True,
    topk: int = 1,
) -> FunctionTool:
    """
    Create a Pyserini FAISS search tool for retrieving search results using a prebuilt index.

    This function returns a tool that performs search queries using the Pyserini library.

    Thread-safe: Each thread gets its own encoder and searcher instances to avoid
    PyTorch model sharing issues. Indices are loaded during tool configuration.

    See https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md for available prebuilt indexes.

    Args:
        faiss_index_name (str): Name of the prebuilt FAISS index.
        lucene_index_name (str): Name of the prebuilt Lucene index for document retrieval.
        embed_model_name (str): Name of the embedding model used (default: "BAAI/bge-base-en-v1.5").
        include_title (bool): Whether to include document titles in the results (default: True).
        topk (int): Number of top search results to retrieve (default: 1).
    """
    from pyserini.encode import AutoQueryEncoder
    from pyserini.search.faiss import FaissSearcher
    from pyserini.search.lucene import LuceneSearcher

    def get_lucene_searcher():
        """Get thread-local Lucene searcher, creating it if it doesn't exist."""
        if not hasattr(_PYSERINI_THREAD_LOCAL_CACHE, "lucene_searchers"):
            _PYSERINI_THREAD_LOCAL_CACHE.lucene_searchers = {}

        if lucene_index_name not in _PYSERINI_THREAD_LOCAL_CACHE.lucene_searchers:
            _PYSERINI_THREAD_LOCAL_CACHE.lucene_searchers[lucene_index_name] = (
                LuceneSearcher.from_prebuilt_index(lucene_index_name)
            )
        return _PYSERINI_THREAD_LOCAL_CACHE.lucene_searchers[lucene_index_name]

    def get_faiss_searcher():
        """Get thread-local FAISS searcher, creating it if it doesn't exist."""
        if not hasattr(_PYSERINI_THREAD_LOCAL_CACHE, "searchers"):
            _PYSERINI_THREAD_LOCAL_CACHE.searchers = {}

        cache_key = f"prebuilt::{faiss_index_name}::{embed_model_name}"
        if cache_key not in _PYSERINI_THREAD_LOCAL_CACHE.searchers:
            # Initialize the FAISS searcher with index and encoder (thread-local)
            query_encoder = AutoQueryEncoder(embed_model_name)
            _PYSERINI_THREAD_LOCAL_CACHE.searchers[cache_key] = (
                FaissSearcher.from_prebuilt_index(faiss_index_name, query_encoder)
            )
        return _PYSERINI_THREAD_LOCAL_CACHE.searchers[cache_key]

    def pyserini_faiss_search(
        query: Annotated[
            str,
            ParamMetadata(
                description="A question to search for relevant documents",
                min_length=1,
                max_length=30,
            ),
        ],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform a search and return the results.

        Args:
            query: A search query

        Returns:
            dict[str, Any]: The search results and metadata.

        Raises:
            ValueError: If the query is invalid.
        """
        # Clean the query
        query = query.replace("\n", " ").strip()

        # Get thread-local FAISS searcher
        faiss_searcher = get_faiss_searcher()
        lucene_searcher = get_lucene_searcher()

        # Perform the query
        hits = faiss_searcher.search(query, k=topk)  # type: ignore

        lines: list[str] = []
        for i, hit in enumerate(hits):
            docid: str = hit.docid  # type: ignore
            doc = json.loads(lucene_searcher.doc(docid).raw())  # type: ignore
            if include_title:
                title = doc.get("title", "Untitled")
                lines.append(f"# {title}\n")
            lines.append(doc["text"])
            lines.append("-" * 80 + "\n")

        results_text = "\n".join(lines) if lines else "Error(No results)"

        return {
            "query": query,
            "results": results_text,
        }

    # Pre-load indexes during tool creation to avoid timing skew on first use
    get_faiss_searcher()
    get_lucene_searcher()

    return FunctionTool(
        pyserini_faiss_search,
        name="pyserini_faiss_search",
        cleanup_func=close_pyserini_resources,
    )


def create_pyserini_faiss_search_tool(
    index_path: Path | str,
    corpus_path: Path | str,
    embed_model_name: str = "BAAI/bge-base-en-v1.5",
    include_title: bool = True,
    topk: int = 1,
) -> FunctionTool:
    """
    Create a Pyserini FAISS search tool for retrieving search results.

    This function returns a tool that performs search queries using the Pyserini library.

    Thread-safe: Each thread gets its own searcher instance to avoid
    PyTorch model sharing issues. Corpus is loaded during tool configuration.

    Args:
        index_path (Path | str): Path to the FAISS index directory.
        corpus_path (Path | str): Path to the corpus JSON file.
        embed_model_name (str): Name of the embedding model used (default: "BAAI/bge-base-en-v1.5").
        include_title (bool): Whether to include document titles in the results (default: True).
        topk (int): Number of top search results to retrieve (default: 1).
    """
    from pyserini.search.faiss import FaissSearcher

    def get_corpus():
        corpus_path_str = str(Path(corpus_path).resolve())
        # Fast path
        corpus = _PYSERINI_CORPUS_CACHE.get(corpus_path_str)
        if corpus is not None:
            return corpus
        with _PYSERINI_CORPUS_CACHE_LOCK:
            corpus = _PYSERINI_CORPUS_CACHE.get(corpus_path_str)
            if corpus is not None:
                return corpus
            with open(corpus_path, "rb") as f:
                corpus = orjson.loads(f.read())
            _PYSERINI_CORPUS_CACHE[corpus_path_str] = corpus
            return corpus

    def get_searcher():
        """Get thread-local searcher, creating it if it doesn't exist."""
        if not hasattr(_PYSERINI_THREAD_LOCAL_CACHE, "searchers"):
            _PYSERINI_THREAD_LOCAL_CACHE.searchers = {}

        index_path_str = str(Path(index_path).resolve())
        cache_key = f"local::{index_path_str}::{embed_model_name}"
        if cache_key not in _PYSERINI_THREAD_LOCAL_CACHE.searchers:
            # Initialize the FAISS searcher with index and encoder (thread-local)
            _PYSERINI_THREAD_LOCAL_CACHE.searchers[cache_key] = FaissSearcher(
                index_path_str, embed_model_name
            )
        return _PYSERINI_THREAD_LOCAL_CACHE.searchers[cache_key]

    def pyserini_faiss_search(
        query: Annotated[
            str,
            ParamMetadata(
                description=("A search query"),
                min_length=1,
                max_length=30,
            ),
        ],
        step_index: Annotated[
            int,
            ParamMetadata(
                description="Current step index in the overall plan (hidden from LLM)",
                visible=False,
            ),
        ] = None,
        function_list: Annotated[
            list[str],
            ParamMetadata(
                description="Current function list (hidden from LLM)",
                visible=False,
            ),
        ] = None,
        dataset_config: Annotated[
            dict,
            ParamMetadata(
                description="Dataset configuration (hidden from LLM)",
                visible=False,
            ),
        ] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform a search and return the results.

        Args:
            query: A search query
            step_index: Current step index (hidden)
            function_list: Current function list (hidden)
            dataset_config: Dataset configuration (hidden)

        Returns:
            dict[str, Any]: The search results and metadata.

        Raises:
            ValueError: If the query is invalid.
        """
        # Clean the query
        query = query.replace("\n", " ").strip()

        # Get thread-local searcher
        searcher = get_searcher()
        corpus = get_corpus()

        # Perform the query
        hits = searcher.search(query, k=topk)

        lines: list[str] = []
        for i, hit in enumerate(hits):
            docid: str = hit.docid  # type: ignore
            if include_title:
                title = corpus[docid].get("title", "Untitled")
                lines.append(f"# {title}\n")
            lines.append(corpus[docid]["text"])
            lines.append("-" * 80 + "\n")

        results_text = "\n".join(lines) if lines else "Error(No results)"

        return {
            "query": query,
            "results": results_text,
        }

    # Pre-load corpus and searcher during tool creation to avoid timing skew on first use
    get_corpus()
    get_searcher()

    return FunctionTool(
        pyserini_faiss_search,
        name="pyserini_faiss_search",
        cleanup_func=close_pyserini_resources,
    )


if __name__ == "__main__":
    # # pyserini example
    # tool = create_pyserini_prebuilt_faiss_search_tool(
    #     faiss_index_name="beir-v1.0.0-hotpotqa.bge-base-en-v1.5",
    #     lucene_index_name="beir-v1.0.0-hotpotqa.flat",
    #     embed_model_name="BAAI/bge-base-en-v1.5",
    #     include_title=True,
    #     topk=3
    # )
    # print("Pyserini FAISS Search Tool")
    # queries = [
    #     "largest planet in our solar system",
    #     "when were Tokyo 2020 Olympic Games held",
    # ]
    # for query in queries:
    #     print(f"Query: {query}")
    #     print(tool.execute(query=query))
    #     print()

    tool = create_llama_index_search_tool(
        index_path=Path("data/drop/llama_index/BAAI___bge-base-en-v1.5"),
        embed_model_name="BAAI/bge-base-en-v1.5",
    )
    print("LlamaIndex Search Tool")
    queries = [
        "largest planet in our solar system",
        "when were Tokyo 2020 Olympic Games held",
    ]
    for query in queries:
        print(f"Query: {query}")
        print(tool.execute(query=query))

    tool = create_google_search_api_tool(topk=3)

    queries = [
        "capital location",
        "largest planet in our solar system",
        "when were Tokyo 2020 Olympic Games held",
    ]
    for query in queries:
        print(f"Query: {query}")
        print(tool.execute(query=query))
    # Tokyo Olympics -> 2021

    print()

    timelimit = "2000-01-01..2020-01-01"
    print(f"# {timelimit=}")
    tool = create_google_search_api_tool(timelimit=timelimit, topk=3)
    query = queries[-1]
    print(f"Query: {query}")
    print(tool.execute(query=query))
    # Tokyo Olympics -> 2020

    region = "JP-ja"
    print(f"# {region=}")
    tool = create_ddg_google_search_tool(region=region)
    print("Query: capital location")
    print(tool.execute(query="capital location"))

    # Custom date range is not supported by the ddgs package yet
    print()
    timelimit = "01/01/2000..01/01/2024"
    print(f"# {timelimit=}")
    tool = create_ddg_google_search_tool(timelimit=timelimit)
    print("Query: who is the current president of the United States")
    print(tool.execute(query="who is the current president of the United States"))

    # Weaviate Example
    tool = create_weaviate_search_tool(
        "WikipediaLeadCollection",
        topk=1,
        query_prefix="Represent this sentence for searching relevant passages: ",
        port=8080,
    )
    queries = [
        "Messi birthdate",
        "largest planet in our solar system",
        "who is the current president of the United States",
    ]
    for query in queries:
        print(f"Query: {query}")
        print(tool.execute(query=query))
