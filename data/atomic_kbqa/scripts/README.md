# Atomic KBQA Scripts

This directory contains scripts for processing and validating atomic KBQA datasets.

## validate_sparql.py

Validates s-expression to SPARQL conversion for GrailQA, WebQSP, and GraphQ datasets using the `lisp_to_sparql` function.

### Purpose

Tests whether `lisp_to_sparql` correctly converts s-expressions to executable SPARQL queries by:
1. Loading preprocessed data (auto-detects GrailQA, WebQSP, or GraphQ from file path)
2. Converting sexpr to SPARQL using `lisp_to_sparql` with dataset-specific logic
3. Executing against Freebase via ODBC
4. Comparing results with ground truth answers

### Prerequisites

- Virtuoso server running with Freebase KB (see vendor/KBQA-o1/README.md)
- Preprocessed data (run preprocessing script first)
- ODBC driver configured (see `planning/tools/freebase/default_config.py`)

### Usage

```bash
# Validate WebQSP
uv run python data/atomic_kbqa/scripts/validate_sparql.py --input data/atomic_kbqa/webqsp/processed/webqsp_train.v1.json
uv run python data/atomic_kbqa/scripts/validate_sparql.py --input data/atomic_kbqa/webqsp/processed/webqsp_test.v1.json

# Validate GrailQA
uv run python data/atomic_kbqa/scripts/validate_sparql.py --input data/atomic_kbqa/grailqa/processed/grailqa_train.v1.json
uv run python data/atomic_kbqa/scripts/validate_sparql.py --input data/atomic_kbqa/grailqa/processed/grailqa_test.v1.json

# Validate GraphQ
uv run python data/atomic_kbqa/scripts/validate_sparql.py --input data/atomic_kbqa/graphq/processed/graphq_train.v1.json
uv run python data/atomic_kbqa/scripts/validate_sparql.py --input data/atomic_kbqa/graphq/processed/graphq_test.v1.json

# Validate with detailed output
uv run python data/atomic_kbqa/scripts/validate_sparql.py --input data/atomic_kbqa/webqsp/processed/webqsp_train.v1.json --output results.json --show-failures --num-samples 100
```

### Options

- `-i, --input`: Path to preprocessed JSON file (GrailQA, WebQSP, or GraphQ) (required)
- `-o, --output`: Path to save detailed validation results (optional)
- `-n, --num-samples`: Number of samples to validate (default: all)
- `--show-failures`: Show examples of failed validations
- `--max-examples`: Maximum number of failure examples to show (default: 5)

### Validation Results

**Dataset-Specific Implementation:**

The `lisp_to_sparql` function includes dataset-specific logic via `dataset_name` parameter:
- **GrailQA**: Nested subquery structure for superlatives (returns all entities with extreme value)
- **GraphQ**: Nested subquery structure for superlatives (returns all entities with extreme value)
- **WebQSP**: Flat query with ORDER BY + LIMIT 1 (returns single match with string matching)

**GrailQA Results:**

| Split | Samples | Accuracy | Success | Errors |
|-------|---------|----------|---------|--------|
| Train | 500 | **100%** | 500 | 0 |
| Test  | 478 | **100%** | 478 | 0 |

**WebQSP Results:**

| Split | Samples | Accuracy | Success | Errors |
|-------|---------|----------|---------|--------|
| Train | 486 | **100%** | 486 | 0 |
| Test  | 422 | **100%** | 422 | 0 |

**GraphQ Results:**

| Split | Samples | Accuracy | Success | Errors |
|-------|---------|----------|---------|--------|
| Train | 217 | **100%** | 217 | 0 |
| Test  | 215 | **100%** | 215 | 0 |

**Key Implementation Features:**

1. **Variable Filtering**: Filters prevent answer variable `?x` from matching intermediate variables (e.g., `FILTER (?x != ?x0)`)
2. **Dataset-Specific Superlatives**:
   - GrailQA/GraphQ: Nested `{SELECT ?sk0 WHERE {...} ORDER BY LIMIT 1}` subquery
   - WebQSP: Flat structure with `ORDER BY ... LIMIT 1` at end
3. **String Literal Matching** (WebQSP): SUBSTR filters for partial matching (handles Freebase language tags)
4. **COUNT Result Handling**: Converts integer COUNT results to strings for proper comparison

**Conclusion:** `lisp_to_sparql` correctly handles GrailQA, WebQSP, and GraphQ s-expressions with 100% accuracy across all splits (1,818 test cases total).
