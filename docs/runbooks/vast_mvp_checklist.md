# VAST MVP Checklist

Use this checklist while running the first full final-checkpoint report on VAST.

## Preflight

- [ ] Repo is present on VAST.
- [ ] Running from repo root.
- [ ] `.venv` exists.
- [ ] `pip install -r requirements-model.txt` completed.
- [ ] `huggingface-cli login` completed with access to `meta-llama/Llama-3.2-1B-Instruct`.
- [ ] `HF_HOME` points to a repo-local or attached-volume cache with enough disk space.
- [ ] `.venv/bin/python scripts/system/check_model_runtime.py` passes.

## Inputs

- [ ] `configs/rollouts/assistant_axis_roles_v0.yaml` is the rollout config.
- [ ] `configs/experiments/pythia_410m_mvp_v0.yaml` is the experiment config.
- [ ] `data/rollouts/assistant_axis_rollouts_v0.jsonl` exists.
- [ ] `data/rollouts/assistant_axis_rollouts_v0_manifest.json` exists.
- [ ] Expected record count is `1040`.
- [ ] Expected role records are `960`.
- [ ] Expected default records are `80`.

## Fixed Responses

- [ ] Run id: `llama-3.2-1b-full-v0`.
- [ ] Generator model: `meta-llama/Llama-3.2-1B-Instruct`.
- [ ] Raw generated responses exist.
- [ ] Generator `meta/status.json` says `completed`.
- [ ] Generator `checkpoints/progress.json` count matches expected records.
- [ ] Full response import completed.
- [ ] `data/rollouts/assistant_axis_rollouts_v0_responses.jsonl` exists.
- [ ] `data/rollouts/assistant_axis_rollouts_v0_responses_manifest.json` exists.

## Pythia Activations

- [ ] Run id: `activation-step143000-layer12-full-v0`.
- [ ] Model: `EleutherAI/pythia-410m-deduped`.
- [ ] Revision: `step143000`.
- [ ] Layer: `12`.
- [ ] Pooling: response-token mean.
- [ ] Activation run completed.
- [ ] Activation inspector passes.
- [ ] Tensor count agrees with activation index rows.
- [ ] Response spans are non-empty.

## Geometry

- [ ] AA run id: `aa-main-step143000-layer12-full-v0`.
- [ ] AA vector artifact exists.
- [ ] Role geometry run id: `role-geometry-step143000-layer12-full-v0`.
- [ ] Role loadings CSV exists.
- [ ] Role geometry summary JSON exists.
- [ ] Geometry report run id: `geometry-report-step143000-layer12-full-v0`.
- [ ] `geometry_report.md` exists.
- [ ] `geometry_metrics.json` exists.
- [ ] Report gate is recorded as `proceed`, `caution`, or `stop`.

## Preservation

- [ ] Fill `docs/manifests/vast_mvp_upload_manifest_template.json` or a copied run-specific manifest.
- [ ] Preserve generated response JSONL and manifest.
- [ ] Preserve fixed-response generator run directory.
- [ ] Preserve activation run directory.
- [ ] Preserve AA, role-geometry, and report run directories.
- [ ] Do not upload model cache unless there is a specific reason.

## Decision After Report

- [ ] If `proceed`: run coarse checkpoint sweep next.
- [ ] If `caution`: inspect role loadings and response quality before sweep.
- [ ] If `stop`: adjust role/default config, layer, or response-generation policy before spending on sweep.
