# Code Setup

This guide covers all software dependencies needed to run experiments in this repository.
Three experiment tracks exist, each requiring a different subset of setup steps:

| Track                    | Config path                    | Required setup sections                                    |
| ------------------------ | ------------------------------ | ---------------------------------------------------------- |
| KoPL KBQA (primary)      | `conf/experiment/kopl_kbqa/`   | §1 Base environment, §2 LLM API keys, §3 KoPL              |
| Atomic KBQA              | `conf/experiment/atomic_kbqa/` | §1 Base environment, §2 LLM API keys, §4 Freebase/Virtuoso |
| Multi-objective HotpotQA | `conf/experiment/multiobj/`    | §1 Base environment, §2 LLM API keys, §5 Pyserini          |

> [!TIP]
> If you want one reproducible walkthrough, start with `KoPL KBQA`. It has the lightest end-to-end code setup in this repository.

> [!NOTE]
> Dataset-specific setup (downloading and preprocessing data files) is covered separately in [docs/setup/data.md](./data.md). Complete the data setup *after* finishing the code setup in this guide.

---

## 1. Base Environment

### Install `uv`

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency and environment management.
Install it by following [the official guide](https://docs.astral.sh/uv/#installation):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Python dependencies

From the repository root, run:

```bash
uv sync
```

This creates a `.venv` virtual environment and installs all dependencies declared in `pyproject.toml`.
Use `uv run python ...` (instead of `python ...`) for all subsequent commands to ensure the virtualenv is activated.

### Sanity check

```bash
PYTHONPATH=src uv run python -c "import planning; print('OK')"
# Expected output: OK
```

> **Note:** `PYTHONPATH=src` is required for all commands that import `planning`, because the package lives under `src/` and is not installed as an editable package.

---

## 2. LLM API Keys

LLM API keys must be stored in a `.env` file at the repository root. You can copy the provided [`.env.example`](.env.example) and fill in the keys for the providers you plan to use. Only `OPENAI_API_KEY` is required; the others are optional and only needed if you want to run experiments with those backbone LLMs.

```bash
cp .env.example .env # Then edit .env to add your keys
```

| Environment variable   | Provider     | Models used         | Required?    | Where to obtain                                                        |
| ---------------------- | ------------ | ------------------- | ------------ | ---------------------------------------------------------------------- |
| `OPENAI_API_KEY`       | OpenAI       | GPT-4.1, GPT-5      | **Required** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys)   |
| `FIREWORKS_API_KEY`    | Fireworks AI | Qwen3-235B-Instruct | Optional     | [fireworks.ai/account/api-keys](https://fireworks.ai/account/api-keys) |
| `GOOGLE_CLOUD_PROJECT` | Google Cloud | Gemini Flash        | Optional     | [console.cloud.google.com](https://console.cloud.google.com/)          |

Example `.env` file:

```
OPENAI_API_KEY=sk-...
FIREWORKS_API_KEY=fw_...
GOOGLE_CLOUD_PROJECT=my-project-id
```

### Sanity check

```bash
PYTHONPATH=src uv run python -c "import dotenv; dotenv.load_dotenv(); from openai import OpenAI;  print(OpenAI().models.list().data[0].id)"
# Expected output: a model ID string, e.g. "gpt-4-0613"
```

---

## 3. KoPL (KoPL KBQA track only)

The KoPL KBQA experiments use a modified fork of the KoPL library.

- **Use the fork** at <https://github.com/notani/KoPL> — the upstream repository at <https://github.com/THU-KEG/KoPL> is **not** compatible.
- The fork is already declared as a dependency in `pyproject.toml` (`kopl = { git = "https://github.com/notani/KoPL" }`), so **no manual installation is needed** — `uv sync` installs it automatically.

### Sanity check

```bash
PYTHONPATH=src uv run python -c "import kopl; print('OK')"
# Expected output: OK
```

### Data

After verifying KoPL imports, follow [docs/setup/data.md](./data.md) to download and preprocess the KQA Pro dataset that KoPL requires.

---

## 4. Freebase / Virtuoso (Atomic KBQA track only)

Atomic KBQA queries are executed against a local Freebase SPARQL endpoint backed by Virtuoso.

### Installation

#### System packages

`pyodbc` requires `libodbc.so.2` at runtime, which is not bundled in the Python wheel on Linux.
Install it with your system package manager before proceeding:

```bash
# Ubuntu / Debian
sudo apt-get install -y unixodbc unixodbc-dev
```

#### ODBC driver (git submodule)

This codebase connects to Virtuoso via ODBC using a prebuilt driver (`virtodbc.so`) bundled in `vendor/KBQA-o1`.
Initialize the submodule after cloning the repository:

```bash
git submodule update --init vendor/KBQA-o1
```

#### Virtuoso and Freebase data

Follow the full instructions at <https://github.com/dki-lab/Freebase-Setup> to:

1. Install [Virtuoso Open Source](https://vos.openlinksw.com/owiki/wiki/VOS) (v7.2.5 recommended).
2. Download the Freebase data dump provided in that repository.

### Starting the server

> **Note on ports:** `virtuoso.py` exposes two ports from a single `<port>` argument:
> - HTTP/SPARQL endpoint: `http://localhost:<port>/sparql` (used for direct SPARQL queries)
> - ODBC/isql port: `10000 + <port>` (used internally by this codebase via `pyodbc`)
>
> This codebase starts Virtuoso with `<port> = 3002`, so the ODBC port becomes `13002`. This is why `conf/env/atomic_kbqa/v1.yaml` and `src/planning/tools/freebase/default_config.py` reference port `13002` — that is the ODBC port, not the HTTP port. No manual configuration is needed; the port mapping is handled automatically.

From the directory where you cloned `Freebase-Setup`, run:

```bash
python virtuoso.py start 3002 -d <path-to-freebase-data-dump>
```

### Sanity check

Use `curl` to verify the HTTP SPARQL endpoint is live (HTTP port = 3002):

```bash
curl -s "http://localhost:3002/sparql" \
  --data-urlencode "query=SELECT ?s WHERE { ?s ?p ?o } LIMIT 1" \
  -H "Accept: application/sparql-results+json"
# Expected output:
# { "head": { "link": [], "vars": ["s"] },
#   "results": { "distinct": false, "ordered": true, "bindings": [
#     { "s": { "type": "uri", "value": "http://www.openlinksw.com/virtrdf-data-formats#default-iid" }} ] } }
```

Or use Python via ODBC (port 13002), matching how the codebase connects:

```python
import pyodbc

conn_str = "DRIVER=vendor/KBQA-o1/utils/lib/virtodbc.so;Host=localhost:13002;UID=dba;PWD=dba"
conn = pyodbc.connect(conn_str)
conn.setdecoding(pyodbc.SQL_CHAR, encoding="utf8")
conn.setdecoding(pyodbc.SQL_WCHAR, encoding="utf8")
conn.setencoding(encoding="utf8")

cursor = conn.cursor()
cursor.execute("SPARQL SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")  # Virtuoso requires "SPARQL " prefix
print(cursor.fetchone())
conn.close()
# Expected output: ('http://www.openlinksw.com/virtrdf-data-formats#default-iid',)
```

A non-`None` row confirms the ODBC connection is working.

### Data

After verifying KoPL imports, follow [docs/setup/data.md](./data.md) to download and preprocess the KQA Pro dataset that KoPL requires.

---

## 5. Pyserini (Multi-objective HotpotQA track only)

### Installation

[Pyserini](https://github.com/castorini/pyserini/) is already declared in `pyproject.toml` and installed automatically by `uv sync` (no extra steps are needed).

### Prebuilt indexes

The experiments use two prebuilt HotpotQA indexes. They are **downloaded automatically** on first run to `~/.cache/pyserini/indexes/`:

| Index name                              | Type          | Purpose                              |
| --------------------------------------- | ------------- | ------------------------------------ |
| `beir-v1.0.0-hotpotqa.bge-base-en-v1.5` | FAISS (dense) | Semantic retrieval                   |
| `beir-v1.0.0-hotpotqa.flat`             | Lucene BM25   | Fetching full Wikipedia passage text |

These are configured in [`conf/env/multiobj/hotpotqa/v1.yaml`](../../conf/env/multiobj/hotpotqa/v1.yaml).

Verified on this machine, the downloaded index directories occupy approximately:

- `beir-v1.0.0-hotpotqa.bge-base-en-v1.5` (FAISS): **16 GB**
- `beir-v1.0.0-hotpotqa.flat` (Lucene): **2.4 GB**
- Combined: **18.6 GB**

Pyserini stores these under versioned cache directory names inside `~/.cache/pyserini/indexes/`, so the on-disk folder names may include prefixes such as `faiss-flat.` / `lucene-inverted.` and a hash suffix.

### Sanity check

Run the following snippet (`uv run python`) to confirm both the download and query pipeline work (this will trigger the index download on first run):

```python
from pyserini.encode import AutoQueryEncoder
from pyserini.search.faiss import FaissSearcher

encoder = AutoQueryEncoder("BAAI/bge-base-en-v1.5")
searcher = FaissSearcher.from_prebuilt_index(
    "beir-v1.0.0-hotpotqa.bge-base-en-v1.5",
    encoder,
)
hits = searcher.search("Who is the president of France?", k=1)
print(hits[0].docid, hits[0].score)
```

A printed docid and score confirm the index is functional.

### Data

After verifying KoPL imports, follow [docs/setup/data.md](./data.md) to download and preprocess the KQA Pro dataset that KoPL requires.
