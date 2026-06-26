# Project Tracker

This is the canonical running tracker for the Assistant Axis Emergence and Attribution project.

## Fixed Decisions

- Model target: `EleutherAI/pythia-410m-deduped`
- Primary reason: Pythia provides open checkpoints and checkpoint-ordered training stream artifacts.
- First attribution unit: packed 2049-token Pythia training sequence.
- Practical training stream: `pietrolesci/pile-deduped-pythia-preshuffled`
- Official verification fallback: `EleutherAI/pile-deduped-pythia-preshuffled`
- Raw text lookup: `EleutherAI/the_pile_deduplicated`, plus `pietrolesci/pile-deduped` if access and format are useful.
- Output root: `artifacts/runs/...`
- Source reuse: import/adapt Assistant Axis role instructions, selected question pool, and builder/runner patterns from `/Users/prasadmahadik/Documents/Do traits remain consistent across personas?`.
- Old trait-grid shape: 6 traits x 6 roles x 40 questions x 2 variants x 3 conditions = 8640 records.
- New rollout target: 48 roles x 20 questions + 4 default prompt families x 20 questions = 1040 fixed rollout records.
- New rollout semantics: default-vs-role AA construction, not explicit trait-instruction conditions.

## Current Status

| Area | Status | Notes |
| --- | --- | --- |
| Project scaffold | Done | Initial README and design docs created. |
| Pythia data verification | Done | Model, checkpoint, and training-stream claims verified against HF/GitHub sources. |
| Tasklist | Done | `docs/design/tasklist.md` is the operational next-action queue. |
| Repo build map | Done | `docs/design/repo_build_map.md` defines objects, configs, datasets, builders, runners, analyzers, gates, logs, and artifacts. |
| Learning layer | Done | `docs/learning/research_engineering_curriculum.md` explains concepts in build order. |
| Directory scaffold | Done | Initial `configs/`, `data/`, `scripts/`, `src/`, `artifacts/`, and `docs/` layout exists. |
| Assistant Axis source reuse | Decided | Reuse old repo's source-pinned role/question material and builder patterns; do not reuse old 8640 trait-instruction records as the main corpus. |
| Source material config | Done | `configs/rollouts/assistant_axis_source_material_v0.yaml` records old repo paths, hashes, source URLs, old shape, new target, and the imported 40-question pool. |
| Rollout corpus config | Drafted | `configs/rollouts/assistant_axis_roles_v0.yaml` defines 48 roles, 20 shared questions, 4 default prompt families, and expected 1040 records. The contrast group now mixes theatrical and non-neutral/adversarial roles rather than being mostly fantasy. 6 target roles use locally imported instructions; 42 target roles are drafted placeholders needing source verification before final corpus construction. |
| Rollout corpus design | Drafted | New target is 48 roles, 20 shared questions, 4 default prompt families, and fixed texts. |
| Rollout schemas | In progress | Rollout, generated-response, and activation schemas exist; vector/attribution/run schemas remain. |
| Rollout builder design | Done | `docs/design/rollout_corpus_builder_design.md` explains input/output, validation, and non-goals. |
| Rollout learning walkthrough | Done | `docs/learning/rollout_corpus_walkthrough.md` explains configs, schemas, helper functions, artifacts, and the connection to AA construction. |
| Rollout corpus builder | Done | `scripts/rollouts/build_rollout_corpus.py` writes 1040 default-vs-role records and a manifest. |
| Rollout corpus artifact | Done with warnings | `data/rollouts/assistant_axis_rollouts_v0.jsonl` and manifest exist; validation passed with expected warnings for placeholder role instructions. |
| Rollout inspector | Done | `scripts/rollouts/inspect_rollouts.py` prints manifest status, count tables, and sample records; design doc includes flow and helper diagrams. |
| Model config | Done | `configs/models/pythia_410m_deduped.yaml` pins Pythia-410M-deduped model facts, checkpoint naming, and layer 12 as the first middle-layer readout. |
| Experiment config | Done | `configs/experiments/pythia_410m_mvp_v0.yaml` defines the coarse 8-checkpoint sweep, response-token pooling, AA variants, and artifact layout. |
| Activation schema | Done | `configs/schemas/activation_record.schema.yaml` defines activation index records and response-token span requirements. |
| Activation config design | Done | `docs/design/activation_config_design.md` explains the generated-response requirement and activation flow with diagrams. |
| Failure learning log | Done | `docs/learning/failure_learning_log.md` records imported lessons from the earlier trait-geometry repo on generation padding, activation padding, empty responses, span pooling, and resume checks. |
| Fixed response schemas | Done | `configs/schemas/generated_response_record.schema.yaml` and `configs/schemas/generated_response_manifest.schema.yaml` define the validated response artifact contract. |
| Fixed response importer | Done | `scripts/rollouts/import_fixed_responses.py` validates fixture/full response JSONLs against rollout ids, prompt text, non-empty response text, and response provenance. |
| Fixed response fixture | Done | Tiny four-record fixture validates successfully; expected warning only says it is a subset, not the full corpus. |
| Fixed response generator harness | Done | `scripts/rollouts/generate_fixed_responses.py` creates resumable artifact runs and supports a `template_fixture` smoke provider. Smoke run generated and imported 4/4 records successfully. |
| Model runtime preflight | Done in `.venv` | `.venv/bin/python scripts/system/check_model_runtime.py` passes with torch 2.12.1, transformers 5.12.1, and accelerate 1.14.0. Active global miniforge Python remains broken; use `.venv`. |
| Llama fixed response provider | Ready for VAST | `meta-llama/Llama-3.2-1B-Instruct` is gated and local HF auth is unavailable; run on VAST after `huggingface-cli login`. |
| Qwen fixed response provider | Smoke passed | `Qwen/Qwen2.5-0.5B-Instruct` is ungated, Apache-2.0, instruction-tuned, and works through `hf_local`. A 12-record stratified smoke generated and imported successfully. |
| Fixed generated responses | Ready to run on VAST | Full 1040-record Llama response corpus should be generated next and imported with `--mode full`; Qwen remains fallback. |
| VAST MVP runbook | Done | `docs/runbooks/vast_mvp_runbook.md`, `docs/runbooks/vast_mvp_checklist.md`, and `docs/manifests/vast_mvp_upload_manifest_template.json` define the first final-checkpoint report run. |
| Activation extraction | Implemented, waiting on responses | `scripts/activations/cache_rollout_activations.py` implements response-token mean pooling for Pythia checkpoints; run after fixed responses exist. |
| Activation run inspector | Done | `scripts/activations/inspect_activation_run.py` audits status/progress/index rows, tensor file existence, response spans, and shape metadata. |
| Activation cache runbook | Done | `docs/design/activation_cache_runbook.md` records the response generation, import, activation smoke, and inspection commands. |
| AA vector schemas/design | Done | Assistant-axis vector, role-vector, and geometry-manifest schemas exist; `docs/design/assistant_axis_builder_design.md` explains the math and artifact contract. |
| AA construction | Implemented, waiting on activations | `scripts/analysis/build_assistant_axis.py` builds `default_mean`, `contrast_mean`, and normalized AA vector from an activation run. Needs real activation artifacts to execute. |
| Role geometry design | Done | `docs/design/role_geometry_builder_design.md` explains role/default means, PC1, loadings, and AA-PC1 alignment. |
| Role geometry builder | Implemented, waiting on activations | `scripts/analysis/build_role_geometry.py` builds role/default mean vectors, PC1, loadings CSV, and geometry summary from activation + AA runs. |
| Geometry sanity report | Implemented, waiting on geometry artifacts | `scripts/reporting/report_geometry.py` reads AA and role-geometry runs and writes `geometry_report.md`, `geometry_metrics.json`, and a proceed/caution/stop gate. |
| Checkpoint sweep | Runner implemented, ready to run | `scripts/analysis/run_checkpoint_sweep.py` orchestrates activation, inspection, AA, role geometry, and report over the config's coarse 8 checkpoints. Run after the final-checkpoint geometry report is defensible. |
| Axis trajectory analyzer | Implemented, waiting on sweep artifacts | `scripts/analysis/analyze_axis_trajectory.py` computes cosine-to-final, adjacent cosine, AA-PC1, PC1 EVR, loading correlations, moving roles, and candidate transition windows. |
| Axis trajectory plots | Implemented, waiting on trajectory artifacts | `scripts/reporting/plot_axis_trajectory.py` writes cosine, geometry-quality, loading-correlation, transition-score, and moving-role plots. |
| HF artifact upload | Implemented, waiting on VAST artifacts | `scripts/reporting/upload_artifacts_to_hf.py` uploads the curated MVP artifact set to a private HF dataset repo using `HF_TOKEN`. |
| Steering tests | Not started | Need hook implementation and prompt set. |
| Gradient attribution | Not started | Need Parquet loader, sampler, gradient scorer, and resumable run state. |
| Causal validation | Deferred | Start after attribution scores look stable. |

## Next Build Order

1. Run the VAST MVP runbook with Llama fixed responses.
2. Import the generated response JSONL with `--mode full` to create `data/rollouts/assistant_axis_rollouts_v0_responses.jsonl`.
3. Run Pythia final-checkpoint activation caching for `step143000`, layer 12.
4. Inspect the activation run with `scripts/activations/inspect_activation_run.py`.
5. Build final-checkpoint AA and role PC1.
6. Build the geometry report and read the proceed/caution/stop gate.
7. If the report is defensible, run `scripts/analysis/run_checkpoint_sweep.py`.
8. Densify around candidate emergence/refinement windows.
9. Implement Pythia Parquet stream loader with explicit checkpoint-window mapping.
10. Run activation-gradient attribution on a debug sample.
11. Produce top/bottom sequence tables and source/mapping TODOs.
12. Add tiny continued-pretraining validation only after the gradient scorer is stable.

The detailed task queue lives in `docs/design/tasklist.md`; update it together with this tracker.

## Control Docs

| File | Role | Update When |
| --- | --- | --- |
| `README.md` | Front door and fixed project anchor. | Main project target or doc map changes. |
| `docs/design/mvp_scope.md` | Research scope and claim discipline. | Research questions or MVP boundaries change. |
| `docs/design/pythia_training_data_contract.md` | Pythia model/data/checkpoint contract. | Dataset source, mapping rule, or attribution unit changes. |
| `docs/design/tasklist.md` | Operational task queue. | Any task starts, completes, blocks, or changes priority. |
| `docs/design/repo_build_map.md` | System map and object/component glossary. | New objects, configs, scripts, datasets, or artifact contracts are added. |
| `docs/design/activation_cache_runbook.md` | Activation smoke run and inspection procedure. | Activation commands, artifact paths, or proceed gates change. |
| `docs/design/assistant_axis_builder_design.md` | AA vector math and artifact contract. | Axis variants, vector outputs, or selection rules change. |
| `docs/design/geometry_report_design.md` | Final-checkpoint geometry sanity gate. | Gate criteria or report outputs change. |
| `docs/runbooks/vast_mvp_runbook.md` | End-to-end VAST commands for the first final-checkpoint report. | VAST setup, generator model, run ids, or artifact paths change. |
| `docs/runbooks/vast_mvp_checklist.md` | Preflight/progress checklist for the VAST MVP run. | Run criteria, required artifacts, or post-run preservation changes. |
| `docs/manifests/vast_mvp_upload_manifest_template.json` | Template for recording artifacts to preserve/upload after VAST. | Required artifacts or upload targets change. |
| `docs/learning/research_engineering_curriculum.md` | Concept ladder for understanding the repo. | New engineering pattern or concept is introduced. |
| `docs/experiments/` | Decision records and short experiment reports. | A run produces an interpretation or go/no-go decision. |

## Open Decisions

| Decision | Default | Why |
| --- | --- | --- |
| First layer | Middle residual layer | Smallest useful MVP before layer sweep. |
| Pooling | Response-token mean | Avoids role-instruction contamination. |
| Coarse checkpoints | 8 checkpoints | First pass uses `step0`, `step1000`, `step5000`, `step10000`, `step20000`, `step40000`, `step80000`, and `step143000`. |
| Early dense checkpoints | 12 checkpoints | Follow-up sweep localizes the large `step0 -> step1000` transition with `step0`, `step1`, `step2`, `step4`, `step8`, `step16`, `step32`, `step64`, `step128`, `step256`, `step512`, and `step1000`. |
| First attribution sample | 1,000 sequences/window | Debug before 10k+ runs. |
| Raw-source mapping | Deferred | Packed-sequence attribution is the honest first unit. |
| Role count | 48 | Need enough role-space structure for PC1 and contrast; split into 16 assistant-like, 16 non-assistant/non-neutral, 16 neutral/control. |
| Shared question count | 20 | Smaller than old 40-question trait grid; enough to preserve category coverage while keeping checkpoint sweeps manageable. |
| Imported old prompt records | Do not use as main corpus | They encode explicit trait instructions and conditions, which would contaminate default-vs-role AA construction. |
| Planned role instructions | Blocked before final corpus | The config includes local drafted placeholders for roles not present in the old local `core_roles.yaml`; source-verify or replace them before final rollout generation. |

## Source Reuse From Trait-Geometry Repo

| Old Asset | Old Path | Reuse Decision | Change Needed |
| --- | --- | --- | --- |
| Assistant Axis role instructions | `/Users/prasadmahadik/Documents/Do traits remain consistent across personas?/configs/personas/core_roles.yaml` | Reuse/adapt source-pinned instruction variants. | Expand from old selected roles to 48 roles and regroup as assistant-like, non-assistant/non-neutral, neutral/control. |
| Assistant Axis selected questions | `/Users/prasadmahadik/Documents/Do traits remain consistent across personas?/configs/experiments/assistant_axis_6x6_v0.yaml` | Reuse as candidate pool. | Reduce from 40 to 20 while preserving all 8 categories. |
| AssistantAxisGridBuilder | `/Users/prasadmahadik/Documents/Do traits remain consistent across personas?/src/trait_geometry/prompts/build_assistant_axis_grid.py` | Reuse patterns only. | New `RolloutCorpusBuilder` must remove trait axes and instruction conditions, then add default prompt families. |
| Existing generated JSONLs | `/Users/prasadmahadik/Documents/Do traits remain consistent across personas?/data/prompts/assistant_axis_6x6_v001/` | Do not use as main corpus. | Old records are trait-instruction prompts, not default-vs-role fixed rollouts. |

## Artifact Layout

Use:

```text
artifacts/
  runs/
    assistant_axis_attribution/
      pythia-410m-deduped/
        pile-deduped-pythia-preshuffled/
          <probe_set>/
            <variant>/
              <run_id>/
                inputs/
                checkpoints/
                results/
                logs/
                meta/
```

Every run must include:

- `meta/run_manifest.json`
- `meta/status.json`
- `checkpoints/progress.json`
- structured results in `results/`
- `logs/run.log`

## Repo Layout

| Path | Status | Purpose |
| --- | --- | --- |
| `configs/experiments/` | scaffolded | Experiment-level wiring. |
| `configs/models/` | scaffolded | Model and checkpoint configs. |
| `configs/datasets/` | scaffolded | Pythia training-stream configs. |
| `configs/rollouts/` | scaffolded | Fixed rollout corpus configs. |
| `configs/schemas/` | scaffolded | Record and manifest schemas. |
| `data/rollouts/` | scaffolded | Fixed semantic stimuli and manifests. |
| `data/training_windows/` | scaffolded | Sampled packed-sequence ids or decoded snapshots. |
| `scripts/rollouts/` | scaffolded | Rollout builders and inspectors. |
| `scripts/activations/` | scaffolded | Activation caching runners. |
| `scripts/analysis/` | scaffolded | Axis, geometry, and attribution analyzers. |
| `scripts/data/` | scaffolded | Pythia stream planning/loading utilities. |
| `scripts/reporting/` | scaffolded | Plot and report builders. |
| `scripts/steering/` | scaffolded | Steering and validation runners. |
| `src/assistant_axis_attribution/` | scaffolded | Shared package code once scripts need it. |
| `artifacts/runs/` | scaffolded | Generated experiment outputs only. |

## Component Board

### Builders

| Component | Status | Planned Script | Purpose |
| --- | --- | --- | --- |
| `AssistantAxisSourceImporter` | todo | `scripts/rollouts/import_assistant_axis_sources.py` | Import/copy source-pinned roles and question pool from the trait-geometry repo with provenance. |
| `RolloutCorpusBuilder` | done | `scripts/rollouts/build_rollout_corpus.py` | Build 1040 fixed default-vs-role rollout records. |
| `FixedResponseImporter` | done | `scripts/rollouts/import_fixed_responses.py` | Validate and normalize generated/handwritten responses against rollout records. |
| `FixedResponseGeneratorHarness` | done | `scripts/rollouts/generate_fixed_responses.py` | Create resumable fixed-response generation runs; currently smoke-only via `template_fixture`. |
| `AssistantAxisBuilder` | done, final checkpoint run passed | `scripts/analysis/build_assistant_axis.py` | Build checkpoint AA vectors. |
| `RoleGeometryBuilder` | done, final checkpoint run passed | `scripts/analysis/build_role_geometry.py` | Build role vectors, PC1, and loadings. |
| `ReportBuilder` | done, final checkpoint run passed | `scripts/reporting/report_geometry.py` | Build final-checkpoint geometry sanity report. |

### Runners

| Component | Status | Planned Script | Purpose |
| --- | --- | --- | --- |
| `FixedResponseGeneratorProvider` | done, VAST Llama run passed | `scripts/rollouts/generate_fixed_responses.py` | Generate frozen responses for all 1040 rollout records using local Hugging Face model provider. |
| `ActivationCacheRunner` | done, final checkpoint run passed | `scripts/activations/cache_rollout_activations.py` | Cache pooled residual activations. |
| `CheckpointSweepRunner` | done, ready to run | `scripts/analysis/run_checkpoint_sweep.py` | Run activation, inspection, AA, role geometry, and report stages over selected checkpoints. |
| `GradientAttributionRunner` | todo | `scripts/analysis/score_training_sequence_gradients.py` | Score packed training sequences against local/final AA. |
| `SteeringRunner` | later | `scripts/steering/run_axis_steering.py` | Test checkpoint-local causal steering. |

### Analyzers and Gates

| Component | Status | Planned Script | Purpose |
| --- | --- | --- | --- |
| `TrajectoryAnalyzer` | done, waiting on sweep artifacts | `scripts/analysis/analyze_axis_trajectory.py` | Find stabilization or transition windows. |
| `ActivationRunInspector` | done | `scripts/activations/inspect_activation_run.py` | Inspect activation run status, progress, index rows, tensor files, spans, and shapes. |
| `TrainingWindowPlanner` | todo | `scripts/data/plan_training_window.py` | Map checkpoint intervals to Parquet files and batch ranges. |
| `AttributionSummaryAnalyzer` | todo | `scripts/reporting/summarize_attribution.py` | Produce top/bottom tables and aggregate summaries. |
| final-AA sanity gate | todo | decision record | Decide whether the axis is meaningful enough to sweep. |
| checkpoint-transition gate | todo | decision record | Select attribution windows from geometry curves. |
| attribution-debug gate | todo | decision record | Decide whether to scale from 1k to 10k sequences. |

## Claim Discipline

Allowed MVP claim:

> These packed Pythia training sequences exert first-order Assistant-Axis-amplifying pressure at this checkpoint.

Not allowed without later evidence:

> These original raw documents created the Assistant Axis.
