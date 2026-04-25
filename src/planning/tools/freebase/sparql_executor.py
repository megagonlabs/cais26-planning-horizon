"""SPARQL query execution against Virtuoso-hosted Freebase KB.

This module provides utilities for:
- Executing SPARQL queries via ODBC connection to Virtuoso
- Fetching entity labels from Freebase
- Discovering relations connected to entities

Thread-safety: This module uses thread-local storage to provide each thread
with its own ODBC connection. This is necessary because pyodbc connections
are not thread-safe, and concurrent access from multiple threads (e.g., via
ThreadPoolExecutor) could cause race conditions or corrupted results.

Modified from KBQA-o1 codebase (originally based on RNG-KBQA).
Original license: BSD-3-Clause (salesforce.com)
"""

from typing import Optional
import threading
import pyodbc

from planning.tools.freebase.default_config import (
    FREEBASE_ODBC_PORT,
    VIRTODBC_DRIVER_PATH,
    SPARQL_QUERY_TIMEOUT,
)

# Thread-local storage for ODBC connections
# Each thread gets its own connection to avoid race conditions
_thread_local = threading.local()

# Module-level configuration (set once, read by all threads)
_config_lock = threading.Lock()
_default_port: Optional[str] = None
_default_driver_path: Optional[str] = None
_default_timeout: Optional[int] = None


def _get_thread_connection() -> pyodbc.Connection:
    """Get or create the ODBC connection for the current thread.

    Returns:
        pyodbc.Connection: Thread-local ODBC connection

    Raises:
        pyodbc.Error: If connection fails
    """
    # Check if this thread already has a connection
    if hasattr(_thread_local, "conn") and _thread_local.conn is not None:
        return _thread_local.conn

    # Get configuration (use defaults if not explicitly set)
    port = _default_port or FREEBASE_ODBC_PORT
    driver_path = _default_driver_path or VIRTODBC_DRIVER_PATH
    timeout = _default_timeout or SPARQL_QUERY_TIMEOUT

    # Create connection for this thread
    conn_str = f"DRIVER={driver_path};Host=localhost:{port};UID=dba;PWD=dba"
    conn = pyodbc.connect(conn_str)

    # Configure character encoding
    conn.setdecoding(pyodbc.SQL_CHAR, encoding="utf8")
    conn.setdecoding(pyodbc.SQL_WCHAR, encoding="utf8")
    conn.setencoding(encoding="utf8")

    # Set query timeout
    conn.timeout = timeout

    # Store in thread-local storage
    _thread_local.conn = conn

    thread_id = threading.current_thread().name
    print(f"Freebase Virtuoso ODBC connected on port {port} (thread: {thread_id})")

    return conn


def configure_odbc_connection(
    port: Optional[str] = None,
    driver_path: Optional[str] = None,
    timeout: Optional[int] = None,
) -> None:
    """Configure ODBC connection settings for all threads.

    This sets the default port and driver path. Each thread will create
    its own connection lazily on first query execution.

    Args:
        port: Virtuoso ODBC port (default: from config)
        driver_path: Path to Virtuoso ODBC driver (default: from config)
        timeout: Query timeout in seconds (default: from config)

    Note:
        Thread-safe. Call this before any queries to set custom port/driver.
        If not called, default values from config are used.
    """
    global _default_port, _default_driver_path, _default_timeout

    with _config_lock:
        _default_port = port or FREEBASE_ODBC_PORT
        _default_driver_path = driver_path or VIRTODBC_DRIVER_PATH
        _default_timeout = timeout or SPARQL_QUERY_TIMEOUT


def close_thread_connection() -> None:
    """Close the ODBC connection for the current thread.

    This should be called when the thread is done using the connection
    (e.g., at the end of an episode or when the environment is closed).
    """
    if hasattr(_thread_local, "conn") and _thread_local.conn is not None:
        try:
            _thread_local.conn.close()
            # thread_id = threading.current_thread().name
            # print(f"Freebase Virtuoso ODBC connection closed (thread: {thread_id})")
        except Exception as e:
            print(f"Error closing ODBC connection: {e}")
        finally:
            _thread_local.conn = None


def execute_query_with_odbc(sparql_query: str) -> set[str]:
    """Execute SPARQL query against Freebase via ODBC.

    Args:
        sparql_query: SPARQL query string (without "SPARQL" prefix)

    Returns:
        Set of result strings (entity IDs, literals, etc.)
        Empty set if query fails

    Note:
        - Thread-safe: uses thread-local connection
        - Automatically initializes connection on first use per thread
        - Virtuoso requires "SPARQL " prefix before query
        - Results are automatically cleaned (remove URI prefix)
    """
    conn = _get_thread_connection()

    result_set = set()

    # Virtuoso requires "SPARQL " prefix
    query = "SPARQL " + sparql_query

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

            # Extract first column from each row
            for row in rows:
                result_set.add(row[0])

    except Exception:
        # Query execution failed - return empty set
        # Note: In production, consider logging this error
        return result_set

    return result_set


def get_label_with_odbc(entity_id: str) -> str:
    """Get human-readable label for a Freebase entity.

    Args:
        entity_id: Freebase entity ID (e.g., "m.02mjmr")

    Returns:
        English label of the entity, or entity_id if no label found

    Note:
        Thread-safe: uses thread-local connection

    Example:
        >>> get_label_with_odbc("m.02mjmr")
        "Barack Obama"
    """
    conn = _get_thread_connection()

    # SPARQL query to fetch English label
    query = f"""SPARQL
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT (?x0 AS ?label) WHERE {{
            SELECT DISTINCT ?x0 WHERE {{
                ns:{entity_id} rdfs:label ?x0 .
                FILTER (langMatches(lang(?x0), "EN"))
            }}
        }}
    """

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

            if rows and len(rows) > 0:
                return rows[0][0]  # Return first English label

    except Exception:
        # Query failed - return entity ID as fallback
        pass

    return entity_id


def get_in_relations_with_odbc(entity_id: str) -> set[str]:
    """Get all incoming (reverse) relations for an entity.

    Args:
        entity_id: Freebase entity ID (e.g., "m.02mjmr")

    Returns:
        Set of relation names where entity is the object

    Note:
        Thread-safe: uses thread-local connection

    Example:
        For "Barack Obama", might return:
        {'government.politician.party', 'people.person.nationality', ...}
    """
    conn = _get_thread_connection()

    in_relations = set()

    query = f"""SPARQL
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/>
        SELECT (?x0 AS ?value) WHERE {{
            SELECT DISTINCT ?x0 WHERE {{
                ?x1 ?x0 :{entity_id} .
                FILTER regex(?x0, "http://rdf.freebase.com/ns/")
            }}
        }}
    """

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

            for row in rows:
                # Remove URI prefix to get relation name
                relation = row[0].replace("http://rdf.freebase.com/ns/", "")
                in_relations.add(relation)

    except Exception:
        pass

    return in_relations


def get_out_relations_with_odbc(entity_id: str) -> set[str]:
    """Get all outgoing (forward) relations for an entity.

    Args:
        entity_id: Freebase entity ID (e.g., "m.02mjmr")

    Returns:
        Set of relation names where entity is the subject

    Note:
        Thread-safe: uses thread-local connection

    Example:
        For "Barack Obama", might return:
        {'film.actor.film', 'people.person.spouse_s', ...}
    """
    conn = _get_thread_connection()

    out_relations = set()

    query = f"""SPARQL
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/>
        SELECT (?x0 AS ?value) WHERE {{
            SELECT DISTINCT ?x0 WHERE {{
                :{entity_id} ?x0 ?x1 .
                FILTER regex(?x0, "http://rdf.freebase.com/ns/")
            }}
        }}
    """

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

            for row in rows:
                # Remove URI prefix to get relation name
                relation = row[0].replace("http://rdf.freebase.com/ns/", "")
                out_relations.add(relation)

    except Exception:
        pass

    return out_relations


def get_1hop_relations_with_odbc(entity_id: str) -> set[str]:
    """Get all 1-hop relations (both incoming and outgoing) for an entity.

    Args:
        entity_id: Freebase entity ID (e.g., "m.02mjmr")

    Returns:
        Set of all relation names connected to the entity

    Note:
        Thread-safe: uses thread-local connection.
        This is more efficient than calling get_in_relations + get_out_relations
        separately as it combines both directions in a single query.
    """
    conn = _get_thread_connection()

    relations = set()

    # Query combining both incoming and outgoing relations using UNION
    query = f"""SPARQL
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/>
        SELECT (?x0 AS ?value) WHERE {{
            SELECT DISTINCT ?x0 WHERE {{
                {{ ?x1 ?x0 :{entity_id} }}
                UNION
                {{ :{entity_id} ?x0 ?x1 }}
                FILTER regex(?x0, "http://rdf.freebase.com/ns/")
            }}
        }}
    """

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

            for row in rows:
                # Remove URI prefix to get relation name
                relation = row[0].replace("http://rdf.freebase.com/ns/", "")
                relations.add(relation)

    except Exception:
        pass

    return relations


def close_thread_connection() -> None:
    """Close the ODBC connection for the current thread.

    Call this when a thread is about to terminate to clean up resources.
    This is optional - connections will be closed when the thread ends,
    but explicit cleanup is recommended for long-running applications.
    """
    if hasattr(_thread_local, "conn") and _thread_local.conn is not None:
        try:
            _thread_local.conn.close()
        except Exception:
            pass
        _thread_local.conn = None
