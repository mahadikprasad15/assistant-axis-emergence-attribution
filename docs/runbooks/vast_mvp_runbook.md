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
Use a fresh run id if a prior run produced role-washed responses.

```bash
.venv/bin/python scripts/rollouts/generate_fixed_responses.py \
  --provider hf_local \
  --hf-model-id meta-llama/Llama-3.2-1B-Instruct \
  --variant llama-3.2-1b-instruct-rolefaithful \
  --run-id llama-3.2-1b-rolefaithful-full-v0 \
  --hf-cache-dir .cache/huggingface \
  --max-new-tokens 192 \
  --batch-size 20 \
  --temperature 0.0 \
  --save-every 25
```

Expected raw output:

```text
artifacts/runs/assistant_axis_attribution/fixed-response-generator/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/llama-3.2-1b-instruct-rolefaithful/llama-3.2-1b-rolefaithful-full-v0/results/generated_responses_raw.jsonl
```

Resume the same run if interrupted:

```bash
.venv/bin/python scripts/rollouts/generate_fixed_responses.py \
  --resume-run-dir artifacts/runs/assistant_axis_attribution/fixed-response-generator/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/llama-3.2-1b-instruct-rolefaithful/llama-3.2-1b-rolefaithful-full-v0 \
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
  --input-jsonl artifacts/runs/assistant_axis_attribution/fixed-response-generator/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/llama-3.2-1b-instruct-rolefaithful/llama-3.2-1b-rolefaithful-full-v0/results/generated_responses_raw.jsonl \
  --output-jsonl data/rollouts/assistant_axis_rollouts_v0_responses.jsonl \
  --output-manifest data/rollouts/assistant_axis_rollouts_v0_responses_manifest.json \
  --mode full
```

Proceed only if this writes a full validated response file and manifest.

## Step 3: Cache Pythia Final-Checkpoint Activations

This is the first serious Pythia run. It uses fixed Llama responses and does not generate with Pythia.
After the model download, this shows an `activation records` progress bar by default.

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

## Coarse Checkpoint Sweep

After a `proceed` final-checkpoint report, run the configured coarse sweep:

```bash
.venv/bin/python scripts/analysis/run_checkpoint_sweep.py \
  --response-jsonl data/rollouts/assistant_axis_rollouts_v0_responses.jsonl \
  --hf-cache-dir .cache/huggingface \
  --activation-batch-size 8 \
  --sweep-run-id coarse8-full-v0
```

This runs, per checkpoint:

```text
activation cache -> activation inspection -> Assistant Axis -> role geometry -> geometry report
```

The configured checkpoints are:

```text
step0, step1000, step5000, step10000, step20000, step40000, step80000, step143000
```

The existing `step143000` artifacts will be skipped by the underlying stage scripts if their status files already say `completed`.

Sweep summary:

```text
artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/checkpoint-sweep-layer12/coarse8-full-v0/results/checkpoint_sweep_summary.json
```

If interrupted, rerun the same command. Completed stage outputs are detected from their run directories and skipped by the stage scripts.

## Axis Trajectory Analysis

After the sweep finishes, build the cross-checkpoint trajectory report:

```bash
.venv/bin/python scripts/analysis/analyze_axis_trajectory.py \
  --sweep-summary artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/checkpoint-sweep-layer12/coarse8-full-v0/results/checkpoint_sweep_summary.json \
  --final-revision step143000 \
  --run-id coarse8-full-v0
```

This writes:

```text
artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-layer12/coarse8-full-v0/results/axis_trajectory.csv
artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-layer12/coarse8-full-v0/results/checkpoint_transitions.csv
artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-layer12/coarse8-full-v0/results/top_moving_roles.csv
artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-layer12/coarse8-full-v0/results/trajectory_report.md
```

Print the report:

```bash
cat artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-layer12/coarse8-full-v0/results/trajectory_report.md
```

## Axis Trajectory Plots

After trajectory analysis, create the plot pack:

```bash
.venv/bin/python scripts/reporting/plot_axis_trajectory.py \
  --trajectory-run-dir artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-layer12/coarse8-full-v0 \
  --run-id coarse8-full-v0
```

This writes:

```text
artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-plots-layer12/coarse8-full-v0/results/plots/cosine_trajectory.png
artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-plots-layer12/coarse8-full-v0/results/plots/geometry_quality.png
artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-plots-layer12/coarse8-full-v0/results/plots/loading_correlations.png
artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-plots-layer12/coarse8-full-v0/results/plots/transition_scores.png
artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-plots-layer12/coarse8-full-v0/results/plots/top_moving_roles.png
```

Plot report:

```bash
cat artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-plots-layer12/coarse8-full-v0/results/plot_report.md
```

## Early Dense 0-1000 Sweep

The coarse trajectory shows the largest unresolved transition between
`step0 -> step1000`. To localize it, run the dense early checkpoint sweep:

```bash
.venv/bin/python scripts/analysis/run_checkpoint_sweep.py \
  --experiment-config configs/experiments/pythia_410m_early_dense_0_1000_v0.yaml \
  --response-jsonl data/rollouts/assistant_axis_rollouts_v0_responses.jsonl \
  --hf-cache-dir .cache/huggingface \
  --activation-batch-size 8 \
  --sweep-run-id early-dense-0-1000-full-v0
```

Then analyze the dense trajectory using `step1000` as the local endpoint:

```bash
.venv/bin/python scripts/analysis/analyze_axis_trajectory.py \
  --sweep-summary artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/checkpoint-sweep-layer12/early-dense-0-1000-full-v0/results/checkpoint_sweep_summary.json \
  --final-revision step1000 \
  --run-id early-dense-0-1000-full-v0
```

Create the dense plot pack:

```bash
.venv/bin/python scripts/reporting/plot_axis_trajectory.py \
  --trajectory-run-dir artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-layer12/early-dense-0-1000-full-v0 \
  --run-id early-dense-0-1000-full-v0
```

## Dense 1000-5000 Sweep

The coarse trajectory also shows substantial continued alignment from
`step1000 -> step5000`. To resolve that second large early transition, run:

```bash
.venv/bin/python scripts/analysis/run_checkpoint_sweep.py \
  --experiment-config configs/experiments/pythia_410m_dense_1000_5000_v0.yaml \
  --response-jsonl data/rollouts/assistant_axis_rollouts_v0_responses.jsonl \
  --hf-cache-dir .cache/huggingface \
  --activation-batch-size 8 \
  --sweep-run-id dense-1000-5000-full-v0
```

Analyze the dense trajectory using `step5000` as the local endpoint:

```bash
.venv/bin/python scripts/analysis/analyze_axis_trajectory.py \
  --sweep-summary artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/checkpoint-sweep-layer12/dense-1000-5000-full-v0/results/checkpoint_sweep_summary.json \
  --final-revision step5000 \
  --run-id dense-1000-5000-full-v0
```

Create the dense plot pack:

```bash
.venv/bin/python scripts/reporting/plot_axis_trajectory.py \
  --trajectory-run-dir artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-layer12/dense-1000-5000-full-v0 \
  --run-id dense-1000-5000-full-v0
```

## Upload Artifacts To Hugging Face

Use `HF_TOKEN`; do not use `huggingface-cli login` on VAST.

Dry run first:

```bash
.venv/bin/python scripts/reporting/upload_artifacts_to_hf.py \
  --repo-id YOUR_HF_USERNAME/assistant-axis-pythia410m-mvp-artifacts \
  --repo-type dataset \
  --path-in-repo pythia410m-mvp-v0 \
  --private \
  --dry-run
```

If the dry run finds all required artifacts, upload:

```bash
.venv/bin/python scripts/reporting/upload_artifacts_to_hf.py \
  --repo-id YOUR_HF_USERNAME/assistant-axis-pythia410m-mvp-artifacts \
  --repo-type dataset \
  --path-in-repo pythia410m-mvp-v0 \
  --private
```

The uploader creates the dataset repo if it does not already exist. It uploads the curated MVP artifact set:

- role-faithful fixed responses and manifest,
- Llama fixed-response generation run,
- final-checkpoint activation, AA, role-geometry, and report runs,
- coarse checkpoint sweep run,
- every activation, Assistant Axis, role-geometry, and geometry-report run referenced by an included checkpoint sweep summary,
- trajectory analysis run,
- trajectory plot run.

After uploading, the uploader lists the remote repository and fails unless every expected local file is present remotely. A sweep upload is therefore incomplete if its summary was uploaded but its referenced stage directories were not.

It does not upload:

- `.cache/huggingface`,
- `.venv`,
- model downloads,
- `__pycache__`.
