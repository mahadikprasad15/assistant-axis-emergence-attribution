# VAST MVP Runbook

This runbook is the first end-to-end VAST path for the Pythia-410M Assistant Axis MVP.

It produces the final-checkpoint geometry report from:

- fixed rollout corpus: `data/rollouts/assistant_axis_rollouts_v0.jsonl`
- response generator: `meta-llama/Llama-3.2-1B-Instruct`
- activation model: `EleutherAI/pythia-410m-deduped`
- checkpoint: `step143000`
- layer: `12`
- pooling: `response_token_mean`

The first successful run should be treated as a final-checkpoint sanity report, not yet the checkpoint sweep or attribution result.

## Expected Runtime

On an A5000, budget conservatively:

| Stage | Expected Time |
| --- | --- |
| Environment and model downloads | 10-30 min |
| Llama 3.2 1B fixed responses, 1040 records | 10-40 min with batched generation, depending sequence length and GPU memory |
| Response import/validation | seconds to minutes |
| Pythia-410M final-checkpoint activations | 10-30 min |
| AA vector, role geometry, report | seconds to minutes |

Fresh VAST total: usually budget `2-3 hours`.

Cached rerun: often under `1 hour`.

## Setup

Run from the repository root.

```bash
pwd
```

Expected:

```text
/workspace/Assistant Axis Emergence and Attribution
```

Create or activate the environment.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-model.txt
```

Authenticate to Hugging Face for gated Llama access.

```bash
huggingface-cli login
```

Use a cache directory with enough disk space. This keeps downloads out of accidental locations and makes the run easier to resume.

```bash
mkdir -p .cache/huggingface
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
```

Check runtime libraries before model work.

```bash
.venv/bin/python scripts/system/check_model_runtime.py
```

## Step 1: Generate Fixed Responses

Primary run: Llama 3.2 1B Instruct.
This shows a `tqdm` progress bar by default.

```bash
.venv/bin/python scripts/rollouts/generate_fixed_responses.py \
  --provider hf_local \
  --hf-model-id meta-llama/Llama-3.2-1B-Instruct \
  --variant llama-3.2-1b-instruct \
  --run-id llama-3.2-1b-full-v0 \
  --hf-cache-dir .cache/huggingface \
  --max-new-tokens 192 \
  --batch-size 20 \
  --temperature 0.0 \
  --save-every 25
```

Expected raw output:

```text
artifacts/runs/assistant_axis_attribution/fixed-response-generator/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/llama-3.2-1b-instruct/llama-3.2-1b-full-v0/results/generated_responses_raw.jsonl
```

Resume the same run if interrupted:

```bash
.venv/bin/python scripts/rollouts/generate_fixed_responses.py \
  --resume-run-dir artifacts/runs/assistant_axis_attribution/fixed-response-generator/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/llama-3.2-1b-instruct/llama-3.2-1b-full-v0 \
  --provider hf_local \
  --hf-model-id meta-llama/Llama-3.2-1B-Instruct \
  --hf-cache-dir .cache/huggingface \
  --batch-size 20
```

Fallback only if Llama access fails:

```bash
.venv/bin/python scripts/rollouts/generate_fixed_responses.py \
  --provider hf_local \
  --hf-model-id Qwen/Qwen2.5-0.5B-Instruct \
  --variant qwen2.5-0.5b-instruct \
  --run-id qwen2.5-0.5b-full-v0 \
  --hf-cache-dir .cache/huggingface \
  --max-new-tokens 192 \
  --batch-size 20 \
  --temperature 0.0 \
  --save-every 25
```

## Step 2: Import And Validate Responses

```bash
.venv/bin/python scripts/rollouts/import_fixed_responses.py \
  --input-jsonl artifacts/runs/assistant_axis_attribution/fixed-response-generator/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/llama-3.2-1b-instruct/llama-3.2-1b-full-v0/results/generated_responses_raw.jsonl \
  --output-jsonl data/rollouts/assistant_axis_rollouts_v0_responses.jsonl \
  --output-manifest data/rollouts/assistant_axis_rollouts_v0_responses_manifest.json \
  --mode full
```

Proceed only if this writes a full validated response file and manifest.

## Step 3: Cache Pythia Final-Checkpoint Activations

This is the first serious Pythia run. It uses fixed Llama responses and does not generate with Pythia.

```bash
.venv/bin/python scripts/activations/cache_rollout_activations.py \
  --response-jsonl data/rollouts/assistant_axis_rollouts_v0_responses.jsonl \
  --revision step143000 \
  --layer 12 \
  --batch-size 8 \
  --hf-cache-dir .cache/huggingface \
  --run-id activation-step143000-layer12-full-v0
```

Expected activation run:

```text
artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/response-token-mean-layer12/activation-step143000-layer12-full-v0
```

If the A5000 has room, try `--batch-size 16` on a fresh run. Keep `8` as the conservative default.

Resume if interrupted:

```bash
.venv/bin/python scripts/activations/cache_rollout_activations.py \
  --resume-run-dir artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/response-token-mean-layer12/activation-step143000-layer12-full-v0 \
  --response-jsonl data/rollouts/assistant_axis_rollouts_v0_responses.jsonl \
  --revision step143000 \
  --layer 12 \
  --batch-size 8 \
  --hf-cache-dir .cache/huggingface
```

## Step 4: Inspect Activation Run

```bash
.venv/bin/python scripts/activations/inspect_activation_run.py \
  --run-dir artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/response-token-mean-layer12/activation-step143000-layer12-full-v0
```

Proceed only if:

- status/progress/index agree,
- tensor files exist for indexed rows,
- response token spans are non-empty,
- no group is unexpectedly missing.

## Step 5: Build Assistant Axis

```bash
.venv/bin/python scripts/analysis/build_assistant_axis.py \
  --activation-run-dir artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/response-token-mean-layer12/activation-step143000-layer12-full-v0 \
  --axis-variant-id aa_main \
  --run-id aa-main-step143000-layer12-full-v0
```

Expected AA run:

```text
artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/aa-main-layer12/aa-main-step143000-layer12-full-v0
```

## Step 6: Build Role Geometry

```bash
.venv/bin/python scripts/analysis/build_role_geometry.py \
  --activation-run-dir artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/response-token-mean-layer12/activation-step143000-layer12-full-v0 \
  --assistant-axis-run-dir artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/aa-main-layer12/aa-main-step143000-layer12-full-v0 \
  --run-id role-geometry-step143000-layer12-full-v0
```

Expected role-geometry run:

```text
artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/role-geometry-layer12/role-geometry-step143000-layer12-full-v0
```

## Step 7: Build Geometry Report

```bash
.venv/bin/python scripts/reporting/report_geometry.py \
  --assistant-axis-run-dir artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/aa-main-layer12/aa-main-step143000-layer12-full-v0 \
  --role-geometry-run-dir artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/role-geometry-layer12/role-geometry-step143000-layer12-full-v0 \
  --run-id geometry-report-step143000-layer12-full-v0
```

Expected report:

```text
artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/geometry-report-layer12/geometry-report-step143000-layer12-full-v0/results/geometry_report.md
```

## Step 8: Fill Upload Manifest

Copy `docs/manifests/vast_mvp_upload_manifest_template.json` into the run output area or edit a project-local copy after the run.

Minimum files to preserve:

- `data/rollouts/assistant_axis_rollouts_v0_responses.jsonl`
- `data/rollouts/assistant_axis_rollouts_v0_responses_manifest.json`
- full Llama response-generation run directory
- final activation run directory
- AA run directory
- role-geometry run directory
- geometry-report run directory

Do not upload `.cache/huggingface` unless explicitly needed.

## Next Decision

Read the generated `geometry_report.md`.

If the gate is `proceed` or defensible `caution`, the next run is the coarse checkpoint sweep from `configs/experiments/pythia_410m_mvp_v0.yaml`.

If the gate is `stop`, inspect:

- role group separability,
- default prompt family behavior,
- response quality,
- whether layer 12 is too early/late for the first readout.
