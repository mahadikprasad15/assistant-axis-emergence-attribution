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
