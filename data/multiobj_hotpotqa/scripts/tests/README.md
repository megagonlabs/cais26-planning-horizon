# Multi-objective HotpotQA Tests

This directory contains comprehensive unit tests for the multi-objective HotpotQA preprocessing pipeline.

## Test Files

### Core Logic Tests

**`test_dag_generation.py`** (13 tests)
- DAG structure generation for bridge and comparison questions
- DAG merging with dependency renumbering
- DAG complexity metrics (depth, width)

**`test_sampling.py`** (10 tests)
- Data loading and filtering
- Balanced sampling for k=1, k≥2
- Train/heldout splitting with no overlap
- Question combination and metadata preservation

**`test_integration.py`** (11 tests)
- Integration tests with real HotpotQA data
- End-to-end preprocessing workflows
- Real data validation

**`test_preprocessing.py`** (template)
- High-level test templates for future implementation
- Not executed in test runs

## Running Tests

### Run all tests
```bash
cd data/multiobj_hotpotqa/scripts/tests
uv run python -m unittest discover -s . -p "test_*.py" -v
```

### Run specific test file
```bash
uv run python test_dag_generation.py -v
uv run python test_sampling.py -v
uv run python test_integration.py -v
```

### Run specific test class
```bash
uv run python test_dag_generation.py TestDAGMerging -v
uv run python test_sampling.py TestSamplingK1 -v
```

### Run specific test method
```bash
uv run python -m unittest test_dag_generation.TestDAGMerging.test_merge_bridge_and_comparison -v
```

## Test Results

**Status**: ✅ All 57 tests passing

```
Ran 57 tests in 0.335s - OK
```

## Test Coverage

### DAG Generation (13 tests)
- ✅ Bridge DAG: 3 nodes (search → search → finish)
- ✅ Comparison DAG: 4 nodes (search, search → aggregation → finish)
- ✅ Merging 1, 2, 3+ DAGs with single finish node
- ✅ Dependency renumbering after merge
- ✅ Depth calculation (longest path)
- ✅ Width calculation (maximum parallelism)

### Sampling Strategy (10 tests)
- ✅ Filtering by question type (bridge/comparison)
- ✅ k=1: Balanced 50/50 sampling
- ✅ k≥2: Random type distribution
- ✅ No duplicate samples
- ✅ Train/heldout split with no overlap
- ✅ Question combination with numbered format
- ✅ Metadata preservation

### Integration (11 tests)
- ✅ Loading real HotpotQA files (90K+ train, 7K+ dev)
- ✅ Validating required fields
- ✅ Filtering and sampling real data
- ✅ End-to-end k=1 and k=2 workflows
- ✅ DAG annotation with real questions

## Key Functions Tested

### DAG Operations
```python
generate_bridge_dag() -> List[DagNode]
generate_comparison_dag() -> List[DagNode]
merge_dags(dags: List[List[DagNode]]) -> List[DagNode]
compute_dag_depth(dag: List[DagNode]) -> int
compute_dag_width(dag: List[DagNode]) -> int
```

### Data Operations
```python
load_jsonl(file_path: Path) -> List[Dict]
filter_by_type(examples: List[Dict], type: str) -> List[Dict]
sample_balanced_k1(bridge, comparison, num_samples, seed) -> List[Dict]
sample_balanced_multi(bridge, comparison, k, num_samples, seed) -> List[List[Dict]]
split_train_heldout(examples, num_train, num_heldout, seed) -> Tuple[List, List]
combine_questions(components: List[Dict]) -> Dict
```

## Next Steps

With comprehensive tests in place, implementation can proceed confidently:

1. ✅ **Tests complete** - All core functionality validated
2. ⬜ **Main preprocessing script** - Implement using tested functions
3. ⬜ **Nearest neighbor retrieval** - Embed and find similar examples
4. ⬜ **DAG metrics export** - Compute and save complexity features

## Notes

- Tests use real data files when available: `../train.jsonl`, `../dev.jsonl`
- Integration tests load first 1000 train examples for speed
- All functions handle edge cases (insufficient data, empty inputs, etc.)
- DAG merging correctly renumbers dependencies across multiple DAGs
- Sampling ensures no overlap between train and heldout sets

See `TEST_SUMMARY.md` for detailed test descriptions.
