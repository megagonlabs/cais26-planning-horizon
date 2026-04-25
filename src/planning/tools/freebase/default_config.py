"""Default configuration for Freebase Virtuoso connection.

These settings can be overridden via environment-level configuration.
"""

# Default Virtuoso ODBC port and SPARQL endpoint URL for Freebase KBQA-o1 instance
FREEBASE_ODBC_PORT = 13002
FREEBASE_SPARQL_ENDPOINT_URL = f"http://localhost:{FREEBASE_ODBC_PORT}/sparql"

# Path to Virtuoso ODBC driver
# This should be updated based on your system installation
VIRTODBC_DRIVER_PATH = "vendor/KBQA-o1/utils/lib/virtodbc.so"

# Timeout for SPARQL queries in seconds
SPARQL_QUERY_TIMEOUT = 60
