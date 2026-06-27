# Training Sequence Sampler Design

## Purpose

The sampler turns a training-window plan into concrete packed training sequence
records:

```text
window plan -> Parquet shard download/load -> batch_idx filter -> sampled token_ids
```

It is the bridge between checkpoint geometry and gradient attribution.

## Input

- `results/window_plan.jsonl` from `scripts/data/plan_training_window.py`
- Practical dataset: `pietrolesci/pile-deduped-pythia-preshuffled`
- Optional local shard directory if files have already been downloaded

Each plan row tells the sampler:

```text
window_id
batch_idx_start
batch_idx_end_exclusive
parquet_files
sample_size
seed
```

## Output

Canonical run directory:

```text
artifacts/runs/assistant_axis_attribution/
  pythia-410m-deduped/
    pile-deduped-pythia-preshuffled/
      assistant-axis-attribution-v0/
        training-sequence-sample/
          <run_id>/
            meta/run_manifest.json
            meta/status.json
            checkpoints/progress.json
            results/sampled_sequences.jsonl
            results/window_sample_summary.json
            results/window_sample_summary.csv
            logs/run.log
```

Each sampled record contains:

```text
sample_id
window_id
uid
batch_idx
source_file
token_ids
token_count
source metadata
```

## Download/Load Strategy

The sampler supports two modes:

1. `--hf-cache-dir`: use `huggingface_hub.hf_hub_download` for planned files.
2. `--local-data-dir`: read files from a local directory.

The expected HF path is:

```text
data/train-001000.parquet
```

because the dataset config stores `data_dir: data`.

The sampler prints progress at three levels:

```text
window -> Parquet file -> large read/filter completion
```

This matters because one planned window can still require reading a multi-GB
Parquet shard. Hugging Face download progress can finish while the local
`read_parquet` call is still working. The run log records:

```text
parquet_read_start
parquet_read_done
parquet_filter_read_fallback
```

The loader first tries Parquet predicate filters on `batch_idx` so pyarrow can
avoid materializing unrelated rows when supported. If the installed Parquet
engine rejects filters, the sampler logs the fallback and applies the same
filter in pandas after loading the planned columns.

## Sampling Strategy

For each window:

1. Load planned Parquet shards.
2. Filter rows by:

```text
batch_idx >= batch_idx_start
batch_idx < batch_idx_end_exclusive
```

3. Uniformly sample without replacement using the window seed.
4. Append samples to `results/sampled_sequences.jsonl`.

## Resume Behavior

The durable source of truth is `results/sampled_sequences.jsonl`. On restart,
the sampler reads existing records and skips windows that already have enough
records.

## Non-Goals

- No token decoding.
- No gradient scoring.
- No raw-document mapping.
