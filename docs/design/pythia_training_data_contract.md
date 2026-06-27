# Pythia Training-Data Contract

This project uses Pythia because its checkpoints and training-order artifacts make checkpoint-window attribution practical.

## Fixed Model Target

- Model: `EleutherAI/pythia-410m-deduped`
- Dataset listed by model card: `EleutherAI/the_pile_deduplicated`
- Checkpoints: `step0`, `step1`, `step2`, `step4`, ..., `step512`, then `step1000` through `step143000` every 1000 steps
- Final checkpoint: `step143000`, equivalent to `main`
- Total training tokens: 299,892,736,000
- Batch size: 2,097,152 tokens

Sources:

- https://huggingface.co/EleutherAI/pythia-410m-deduped
- https://raw.githubusercontent.com/EleutherAI/pythia/main/README.md

## Training Data Artifacts

### Raw-ish Deduplicated Pile

- Repo: `EleutherAI/the_pile_deduplicated`
- Use for: human-readable text lookup and broad raw corpus inspection
- Do not use as the primary checkpoint-window attribution stream because it does not directly encode the exact packed sequence order seen during training

Source:

- https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated

### Official GPT-NeoX Idxmaps

- Repo: `EleutherAI/pythia_deduped_pile_idxmaps`
- Format: GPT-NeoX/Megatron `.bin` and `.idx` memory-mapped data
- Use for: official training replication and low-level verification
- Tradeoff: large and inconvenient for analysis

Source:

- https://huggingface.co/datasets/EleutherAI/pythia_deduped_pile_idxmaps
- https://raw.githubusercontent.com/EleutherAI/pythia/main/README.md

### Official Preshuffled Packed Stream

- Repo: `EleutherAI/pile-deduped-pythia-preshuffled`
- Format: sharded `document-*.bin` plus `document.idx`
- Use for: official checkpoint-ordered packed sequence stream
- Tradeoff: very large binary format

Source:

- https://huggingface.co/datasets/EleutherAI/pile-deduped-pythia-preshuffled
- https://raw.githubusercontent.com/EleutherAI/pythia/main/README.md

### Practical Analysis Stream

- Repo: `pietrolesci/pile-deduped-pythia-preshuffled`
- Format: Parquet
- Columns: `uid`, `batch_idx`, `token_ids`
- Sequence length: 2049 token ids
- Use for: first attribution experiments

This dataset is a community repackaging of the official EleutherAI preshuffled stream into a more convenient Parquet layout. It should be treated as the practical first path, with official EleutherAI data retained as the verification fallback.

Source:

- https://huggingface.co/datasets/pietrolesci/pile-deduped-pythia-preshuffled

## Attribution Unit

The initial attribution unit is:

```text
packed 2049-token Pythia training sequence
```

not:

```text
original raw document
```

Reason: Pythia trains on packed token sequences. Documents are concatenated and separated by EOD tokens; a training sample may start or end inside a raw document. The model consumes 2049 tokens because 2048-token next-token training uses a one-token-shifted target.

## Checkpoint Window Mapping

In the Parquet repackaging:

- `train-001000.parquet` contains batches `0..999`
- `train-002000.parquet` contains batches `1000..1999`
- ...
- `train-143000.parquet` contains batches `142000..142999`

Therefore, for a checkpoint interval:

```text
step A -> step B
```

where `A` and `B` are multiples of 1000, use:

```text
train-(A+1000).parquet through train-B.parquet
```

Example:

```text
step30000 -> step40000
```

uses:

```text
train-031000.parquet through train-040000.parquet
```

## Raw Text / Source Mapping

Decoded packed sequences are enough for first-pass gradient attribution and inspection. Mapping a packed sequence back to original raw documents or source categories is a separate layer and should be tracked explicitly.

Initial claim:

> These packed Pythia training sequences exert first-order Assistant-Axis-amplifying pressure.

Deferred stronger claim:

> These original raw documents or sources caused the Assistant Axis.

The deferred claim requires a validated mapping from packed sequence to raw document/source.

## Required Run Metadata

Every attribution run must record:

- model repo and checkpoint revision
- tokenizer repo/revision
- training-stream repo and file names
- selected checkpoint interval
- Parquet file list and `batch_idx` range
- sampled `uid` values
- layer and pooling rule
- axis artifact path and checksum or vector id
- gradient score definition and sign convention
- random seed
- output paths under `artifacts/runs/...`

## First Planner Artifact

The first attribution data-stage script is:

```text
scripts/data/plan_training_window.py
```

It reads:

```text
configs/datasets/pythia_preshuffled_stream.yaml
```

and writes:

```text
results/window_plan.json
results/window_plan.jsonl
results/window_plan.csv
```

The plan is consumed by the sampler and records exact Parquet files plus
`batch_idx` filters for each selected attribution window.

The first sampler script is:

```text
scripts/data/sample_training_sequences.py
```

It reads `window_plan.jsonl`, loads the planned Parquet shards either from
Hugging Face or a local directory, filters rows by `batch_idx`, and writes:

```text
results/sampled_sequences.jsonl
results/window_sample_summary.json
results/window_sample_summary.csv
```

The first decoder/inspection script is:

```text
scripts/data/decode_training_sequences.py
```

It reads `sampled_sequences.jsonl`, decodes `token_ids` with the Pythia
tokenizer while preserving special tokens, and writes:

```text
results/decoded_sequences.jsonl
results/decoded_preview.csv
results/decode_summary.json
```

This is still packed-sequence inspection, not raw-document reconstruction.

The first activation-gradient attribution script is:

```text
scripts/analysis/score_training_sequence_gradients.py
```

It reads `sampled_sequences.jsonl`, loads a Pythia checkpoint and Assistant
Axis vector, computes next-token loss on `token_ids[:-1] -> token_ids[1:]`,
retains the gradient at the same hidden-state site used by activation caching,
and writes:

```text
results/attribution_scores.jsonl
results/attribution_scores.csv
results/attribution_summary.json
results/gradient_pressure_vectors/*.pt  # optional
```

The score is:

```text
cosine(-mean_tokens(dL/dh_layer), v_AA)
```

Positive scores mean first-order AA-amplifying pressure under gradient descent.
