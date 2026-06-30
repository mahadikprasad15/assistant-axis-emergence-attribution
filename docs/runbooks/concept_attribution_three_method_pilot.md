# Three-Method Concept Attribution Pilot

## Purpose

Run Vector Filter, activation-gradient dot, and layer-12 FOPCI on the same ten
packed sequences from `step256 -> step512`. This is an execution and numerical
pilot, not a population-level attribution result.

## Required Inputs

The pilot deliberately does not invent or silently download inputs. Provide:

1. A real packed-sequence JSONL containing at least ten records from
   `step256_to_step512`.
2. A construction-split `concept_target_bundle.json` with its target vectors
   and held-out `evaluation_records.jsonl` present at the recorded paths or in
   the bundle result directory.
3. GPU access and enough Hugging Face cache space for
   `EleutherAI/pythia-410m-deduped@step256`.

Build the target bundle from complete `step256`, `step512`, and `step143000`
activation runs:

```bash
python scripts/analysis/build_concept_target_bundle.py \
  --native-activation-run-dir <step256-activation-run> \
  --endpoint-activation-run-dir <step512-activation-run> \
  --final-activation-run-dir <step143000-activation-run> \
  --run-id concept-targets-step256-step512-v0
```

## Preflight Only

```bash
python scripts/analysis/run_concept_attribution_pilot.py \
  --sample-jsonl <shared-packed-sequence-sample.jsonl> \
  --target-bundle <target-run>/results/concept_target_bundle.json \
  --pilot-size 10 \
  --dry-run \
  --run-id three-method-pilot-v0
```

Preflight validates shared sample IDs, window identity, required target names,
question-split disjointness, evaluation-record availability, and hashes. It
does not load a model.

## Real GPU Pilot

```bash
python scripts/analysis/run_concept_attribution_pilot.py \
  --sample-jsonl <shared-packed-sequence-sample.jsonl> \
  --target-bundle <target-run>/results/concept_target_bundle.json \
  --pilot-size 10 \
  --vector-filter-batch-size 10 \
  --activation-candidate-batch-size 8 \
  --fopci-parameter-scope layer12_only \
  --hf-cache-dir .cache/huggingface \
  --run-id three-method-pilot-v0
```

The orchestrator uses stable child run IDs and resumes completed stages. It
runs activation-gradient scoring at batch sizes 1 and 8 and requires the raw
dot comparison to pass at maximum absolute delta `1e-6` before FOPCI.

## Proceed Gate

Proceed to the 50-record FOPCI smoke only if:

- all three methods complete on identical records;
- query and sequence gradient norms are finite and nonzero;
- activation-gradient raw dots pass the batch-size comparison;
- no target has systematic NaNs;
- child manifests identify the same target bundle, checkpoint, and sample IDs.

## Optimized FOPCI Validation and 50-Record Run

The optimized FOPCI runner has two independent batching dimensions:

```text
--query-batch-size:
  batches held-out evaluation records while preserving global default/contrast weights

--sequence-batch-size:
  batches candidate packed sequences in directional_jvp mode
```

The sequential implementation remains the reference. Validate query batching
and directional scoring separately before combining them.

### Colab wrapper

On a fresh Colab A100 runtime, clone the revision containing the optimized
runner, install dependencies, set the token without printing it, and invoke the
gated wrapper:

```python
!git clone https://github.com/mahadikprasad15/assistant-axis-emergence-attribution.git
%cd assistant-axis-emergence-attribution
!python -m pip install -q -r requirements-model.txt -r requirements-data.txt

import getpass, os
os.environ["HF_TOKEN"] = getpass.getpass("Hugging Face token: ")
os.environ["QUERY_BATCH_SIZE"] = "16"
os.environ["SEQUENCE_BATCH_SIZE"] = "8"
```

```python
!bash scripts/colab/run_optimized_fopci_50.sh
```

The wrapper stops immediately if either ten-record equivalence gate fails. It
runs 50 records only after both optimized components match the sequential
reference, then packages and uploads the resulting artifact subtree.

### 1. Directional scoring against the existing sequential query

Use the ten-record sequential run as the reference and import its cached query
gradient into a fresh optimized run:

```bash
BASE="artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/pile-deduped-pythia-preshuffled/concept-attribution-256-512-v0"
SAMPLE="$BASE/training-sequence-sample/pilot-sequences-50-v0/results/sampled_sequences.jsonl"
TARGET="$BASE/concept-target-bundle-layer12/concept-targets-step256-step512-v0/results/concept_target_bundle.json"
REFERENCE="$BASE/fopci-layer12_only/pilot-smoke-10-4090-v0-fopci-layer12_only"

python scripts/analysis/score_first_order_concept_influence.py \
  --sample-jsonl "$SAMPLE" \
  --target-bundle "$TARGET" \
  --parameter-scope layer12_only \
  --limit 10 \
  --sequence-score-mode directional_jvp \
  --sequence-batch-size 8 \
  --query-gradient-bundle "$REFERENCE/results/query_gradient_bundle.pt" \
  --query-gradient-summary "$REFERENCE/results/query_gradient_summary.json" \
  --run-id fopci-directional-validation-10-v0
```

Compare primary raw dots:

```bash
CANDIDATE="$BASE/fopci-layer12_only/fopci-directional-validation-10-v0"

python scripts/analysis/compare_fopci_runs.py \
  --reference-run-dir "$REFERENCE" \
  --candidate-run-dir "$CANDIDATE" \
  --absolute-tolerance 1e-6 \
  --relative-tolerance 1e-5 \
  --run-id directional-vs-sequential-10-v0
```

Directional mode preserves `negative_gradient_dot`. It intentionally writes
`null` for `sequence_gradient_norm` and `gradient_cosine` because it does not
materialize full per-example parameter gradients.

### 2. Batched query construction against the sequential reference

Build a fresh query in batches but retain sequential sequence scoring so the
comparison isolates query batching:

```bash
python scripts/analysis/score_first_order_concept_influence.py \
  --sample-jsonl "$SAMPLE" \
  --target-bundle "$TARGET" \
  --parameter-scope layer12_only \
  --limit 10 \
  --query-batch-size 16 \
  --sequence-score-mode sequential_gradient \
  --sequence-batch-size 1 \
  --run-id fopci-query-batch16-validation-10-v0
```

Compare the query-gradient tensors directly:

```bash
QUERY_BATCHED="$BASE/fopci-layer12_only/fopci-query-batch16-validation-10-v0"

python scripts/analysis/compare_fopci_query_gradients.py \
  --reference-bundle "$REFERENCE/results/query_gradient_bundle.pt" \
  --candidate-bundle "$QUERY_BATCHED/results/query_gradient_bundle.pt" \
  --absolute-tolerance 1e-6 \
  --relative-tolerance 1e-5 \
  --run-id query-gradient-batch16-vs-sequential-v0
```

Then compare the downstream ten-sequence raw dots:

```bash
python scripts/analysis/compare_fopci_runs.py \
  --reference-run-dir "$REFERENCE" \
  --candidate-run-dir "$QUERY_BATCHED" \
  --absolute-tolerance 1e-6 \
  --relative-tolerance 1e-5 \
  --run-id query-batch16-vs-sequential-10-v0
```

### 3. Fifty-record optimized run

Proceed only if both comparisons report `"passed": true`. Reuse the validated
batched query and score all 50 records directionally:

```bash
python scripts/analysis/score_first_order_concept_influence.py \
  --sample-jsonl "$SAMPLE" \
  --target-bundle "$TARGET" \
  --parameter-scope layer12_only \
  --limit 50 \
  --sequence-score-mode directional_jvp \
  --sequence-batch-size 8 \
  --query-gradient-bundle "$QUERY_BATCHED/results/query_gradient_bundle.pt" \
  --query-gradient-summary "$QUERY_BATCHED/results/query_gradient_summary.json" \
  --run-id fopci-directional-50-v0
```

For an A100 80 GB, test sequence batch 16 only after batch 8 matches the
sequential ten-record reference. Do not infer numerical validity from the
absence of an out-of-memory error.
