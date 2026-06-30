#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after installing requirements-model.txt and
# requirements-data.txt. Required environment variable: HF_TOKEN.

: "${HF_TOKEN:?Set HF_TOKEN before running this script}"

QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-16}"
SEQUENCE_BATCH_SIZE="${SEQUENCE_BATCH_SIZE:-8}"
HF_REPO="${HF_REPO:-Prasadmahadik/assistant-axis-emergence-attribution}"
ARCHIVE_IN_REPO="${ARCHIVE_IN_REPO:-concept-attribution/pilot-smoke-10-4090-v0.tar.gz}"
HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is required")
props = torch.cuda.get_device_properties(0)
print({
    "gpu": props.name,
    "memory_gib": round(props.total_memory / 1024**3, 2),
    "torch": torch.__version__,
})
PY

mkdir -p artifacts/imports/fopci-pilot artifacts/packages "$HF_HOME"

hf download "$HF_REPO" "$ARCHIVE_IN_REPO" \
  --repo-type dataset \
  --token "$HF_TOKEN" \
  --local-dir artifacts/imports/fopci-pilot

ARCHIVE="artifacts/imports/fopci-pilot/$ARCHIVE_IN_REPO"
test -f "$ARCHIVE"
tar -xzf "$ARCHIVE" -C artifacts/runs

BASE="artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/pile-deduped-pythia-preshuffled/concept-attribution-256-512-v0"
SAMPLE="$BASE/training-sequence-sample/pilot-sequences-50-v0/results/sampled_sequences.jsonl"
TARGET="$BASE/concept-target-bundle-layer12/concept-targets-step256-step512-v0/results/concept_target_bundle.json"
REFERENCE="$BASE/fopci-layer12_only/pilot-smoke-10-4090-v0-fopci-layer12_only"

test "$(wc -l < "$SAMPLE")" -eq 50
test -f "$TARGET"
test -f "$REFERENCE/results/query_gradient_bundle.pt"
test "$(wc -l < "$REFERENCE/results/fopci_scores.jsonl")" -eq 10

# Gate A: directional-JVP scores must match the sequential ten-record reference
# while using the exact same cached query gradient.
python scripts/analysis/score_first_order_concept_influence.py \
  --sample-jsonl "$SAMPLE" \
  --target-bundle "$TARGET" \
  --parameter-scope layer12_only \
  --limit 10 \
  --sequence-score-mode directional_jvp \
  --sequence-batch-size "$SEQUENCE_BATCH_SIZE" \
  --query-gradient-bundle "$REFERENCE/results/query_gradient_bundle.pt" \
  --query-gradient-summary "$REFERENCE/results/query_gradient_summary.json" \
  --hf-cache-dir "$HF_HOME" \
  --run-id fopci-directional-validation-10-v0

DIRECTIONAL="$BASE/fopci-layer12_only/fopci-directional-validation-10-v0"

python scripts/analysis/compare_fopci_runs.py \
  --reference-run-dir "$REFERENCE" \
  --candidate-run-dir "$DIRECTIONAL" \
  --absolute-tolerance 1e-6 \
  --relative-tolerance 1e-5 \
  --run-id directional-vs-sequential-10-v0

# Gate B: a newly batched query must match both the sequential query tensors
# and their downstream sequential FOPCI scores.
python scripts/analysis/score_first_order_concept_influence.py \
  --sample-jsonl "$SAMPLE" \
  --target-bundle "$TARGET" \
  --parameter-scope layer12_only \
  --limit 10 \
  --query-batch-size "$QUERY_BATCH_SIZE" \
  --sequence-score-mode sequential_gradient \
  --sequence-batch-size 1 \
  --hf-cache-dir "$HF_HOME" \
  --run-id "fopci-query-batch${QUERY_BATCH_SIZE}-validation-10-v0"

QUERY_BATCHED="$BASE/fopci-layer12_only/fopci-query-batch${QUERY_BATCH_SIZE}-validation-10-v0"

python scripts/analysis/compare_fopci_query_gradients.py \
  --reference-bundle "$REFERENCE/results/query_gradient_bundle.pt" \
  --candidate-bundle "$QUERY_BATCHED/results/query_gradient_bundle.pt" \
  --absolute-tolerance 1e-6 \
  --relative-tolerance 1e-5 \
  --run-id "query-gradient-batch${QUERY_BATCH_SIZE}-vs-sequential-v0"

python scripts/analysis/compare_fopci_runs.py \
  --reference-run-dir "$REFERENCE" \
  --candidate-run-dir "$QUERY_BATCHED" \
  --absolute-tolerance 1e-6 \
  --relative-tolerance 1e-5 \
  --run-id "query-batch${QUERY_BATCH_SIZE}-vs-sequential-10-v0"

# Both comparison commands return nonzero on failure, so reaching this point is
# the proceed gate for the clean 50-record optimized run.
python scripts/analysis/score_first_order_concept_influence.py \
  --sample-jsonl "$SAMPLE" \
  --target-bundle "$TARGET" \
  --parameter-scope layer12_only \
  --limit 50 \
  --sequence-score-mode directional_jvp \
  --sequence-batch-size "$SEQUENCE_BATCH_SIZE" \
  --query-gradient-bundle "$QUERY_BATCHED/results/query_gradient_bundle.pt" \
  --query-gradient-summary "$QUERY_BATCHED/results/query_gradient_summary.json" \
  --hf-cache-dir "$HF_HOME" \
  --run-id "fopci-directional-50-qb${QUERY_BATCH_SIZE}-sb${SEQUENCE_BATCH_SIZE}-v0"

FINAL_RUN="$BASE/fopci-layer12_only/fopci-directional-50-qb${QUERY_BATCH_SIZE}-sb${SEQUENCE_BATCH_SIZE}-v0"
test "$(wc -l < "$FINAL_RUN/results/fopci_scores.jsonl")" -eq 50

PACKAGE="artifacts/packages/fopci-optimized-50-qb${QUERY_BATCH_SIZE}-sb${SEQUENCE_BATCH_SIZE}-v0.tar.gz"
tar -czf "$PACKAGE" \
  -C artifacts/runs \
  "assistant_axis_attribution/pythia-410m-deduped/pile-deduped-pythia-preshuffled/concept-attribution-256-512-v0"
shasum -a 256 "$PACKAGE"

hf upload "$HF_REPO" "$PACKAGE" \
  "concept-attribution/$(basename "$PACKAGE")" \
  --repo-type dataset \
  --token "$HF_TOKEN"

echo "Optimized FOPCI validation, 50-record run, and upload completed."
