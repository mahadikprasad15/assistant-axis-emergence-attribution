# Training Window Planner Design

## Purpose

The attribution phase needs a precise bridge from checkpoint windows to the
Pythia packed training stream. The planner answers:

```text
For this checkpoint interval, which Parquet shard files and batch_idx filter
should the sampler use?
```

It does not download or sample the dataset. It creates the reproducible plan
that later scripts consume.

## Input

- Dataset config: `configs/datasets/pythia_preshuffled_stream.yaml`
- Selected windows, for example:
  - `step128 -> step256`
  - `step256 -> step512`
  - `step512 -> step1000`
  - `step1000 -> step2000`
  - `step80000 -> step143000`

## Mapping Rule

The practical Parquet stream uses `batch_idx` as the checkpoint-step index:

```text
stepA -> stepB
means
batch_idx >= A and batch_idx < B
```

Shard files are 1000-step blocks:

```text
train-001000.parquet: batch_idx 0..999
train-002000.parquet: batch_idx 1000..1999
...
train-143000.parquet: batch_idx 142000..142999
```

Examples:

```text
step128 -> step256:
  parquet files: train-001000.parquet
  filter: 128 <= batch_idx < 256

step1000 -> step2000:
  parquet files: train-002000.parquet
  filter: 1000 <= batch_idx < 2000

step80000 -> step143000:
  parquet files: train-081000.parquet ... train-143000.parquet
  filter: 80000 <= batch_idx < 143000
```

## Output

The planner writes a canonical run directory:

```text
artifacts/runs/assistant_axis_attribution/
  pythia-410m-deduped/
    pile-deduped-pythia-preshuffled/
      assistant-axis-attribution-v0/
        training-window-plan/
          <run_id>/
            meta/run_manifest.json
            meta/status.json
            checkpoints/progress.json
            results/window_plan.json
            results/window_plan.jsonl
            results/window_plan.csv
            logs/run.log
```

Each window plan records:

```text
window_id
from_revision
to_revision
batch_idx_start
batch_idx_end_exclusive
parquet_files
row_filter
sample_policy
source dataset config
```

## Downstream Use

The sampler will read `window_plan.jsonl`, download/load only the listed Parquet
files, filter rows by `batch_idx`, and save sampled packed sequences.

The gradient scorer will consume sampled `token_ids`, compute next-token loss,
retain the layer-12 activation gradient, and score:

```text
aa_amplification = -cos(dL/dh_layer12, v_AA)
```

## Non-Goals

- No dataset download.
- No token decoding.
- No gradient scoring.
- No raw-document mapping.
