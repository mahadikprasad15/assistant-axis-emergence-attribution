# Tasklist

This is the operational tasklist for progressively building the repo. Update this file before and after each implementation step. The goal is to keep the construction process understandable, resumable, and auditable.

Status labels:

- `done`: complete enough for the current stage.
- `in_progress`: currently being built or reviewed.
- `todo`: planned, not started.
- `blocked`: waiting on a decision, artifact, or environment.
- `later`: intentionally deferred.

## Phase 0: Repo Understanding and Control Surface

| ID | Task | Status | Output |
| --- | --- | --- | --- |
| P0.1 | Define fixed model/data decision for Pythia 410M deduped. | done | `README.md`, `docs/design/pythia_training_data_contract.md` |
| P0.2 | Create canonical tracker and tasklist. | done | `docs/design/project_tracker.md`, this file |
| P0.3 | Define repo build map: objects, configs, datasets, builders, runners, analyzers, gates, logs, artifacts. | done | `docs/design/repo_build_map.md` |
| P0.4 | Define learning notes for how to understand the repo as it is built. | done | `docs/learning/research_engineering_curriculum.md` |
| P0.5 | Create initial directory scaffold without implementation code. | done | `configs/`, `data/`, `scripts/`, `src/`, `artifacts/`, `docs/` |
| P0.6 | Decide how to reuse Assistant Axis assets from the `Do traits remain consistent across personas?` repo. | done | Reuse source-pinned roles/questions and builder patterns; do not reuse old trait-instruction grids as the main corpus. |

Exit criteria:

- A new contributor can read the docs and explain what each major component will do.
- No implementation starts before the relevant config, object, artifact, and logging expectations are recorded.

## Phase 1: Config and Schema Foundation

| ID | Task | Status | Output |
| --- | --- | --- | --- |
| P1.0 | Import or copy source-pinned Assistant Axis role/question material from the trait-geometry repo with provenance. | done | `configs/rollouts/assistant_axis_source_material_v0.yaml` |
| P1.1 | Define experiment config for the first vertical slice. | done | `configs/experiments/pythia_410m_mvp_v0.yaml` |
| P1.2 | Define model config for `EleutherAI/pythia-410m-deduped`. | done | `configs/models/pythia_410m_deduped.yaml` |
| P1.3 | Define dataset/training-stream config. | done | `configs/datasets/pythia_preshuffled_stream.yaml` |
| P1.4 | Define new rollout corpus config: 48 roles, 20 shared questions, 4 default prompts, generation policy. | done | `configs/rollouts/assistant_axis_roles_v0.yaml` |
| P1.5 | Define JSON/JSONL schemas for rollout records, generated response records, activation records, vector records, attribution records, and run manifests. | in_progress | Rollout record, rollout manifest, generated response record/manifest, activation record, assistant-axis vector, role-vector, geometry manifest, and training-window plan schemas exist; attribution/run schemas remain. |
| P1.6 | Add schema validation helpers. | todo | `src/assistant_axis_attribution/schemas/` |

Exit criteria:

- Configs are human-readable and parameterize every first-stage script.
- Schema docs explain required ids and metadata fields before records are generated.
- The imported source-material config records the old repo path, old config hashes if available, Assistant Axis source URLs, and the exact old-vs-new count change.

## Phase 1A: Assistant Axis Source Import and Redesign

The old trait-geometry repo already has useful Assistant Axis material:

```text
/Users/prasadmahadik/Documents/Do traits remain consistent across personas?/configs/personas/core_roles.yaml
/Users/prasadmahadik/Documents/Do traits remain consistent across personas?/configs/experiments/assistant_axis_6x6_v0.yaml
/Users/prasadmahadik/Documents/Do traits remain consistent across personas?/src/trait_geometry/prompts/build_assistant_axis_grid.py
```

Reuse policy:

- Reuse role instruction text and source provenance where appropriate.
- Reuse the old 40-question source-pinned list as the candidate pool.
- Reuse manifest, validation, stable-id, and builder/runner patterns.
- Do not use the existing 8640 trait-instruction prompt records as the main AA construction corpus.
- Do not keep the old `instruction_positive`, `instruction_negative`, `instruction_neutral` condition structure for initial AA construction.

Old trait-geometry grid:

```text
6 traits x 6 roles x 40 questions x 2 role variants x 3 conditions = 8640 records
```

New Pythia AA rollout target:

```text
48 roles x 20 questions = 960 role records
4 default prompt families x 20 questions = 80 default records
total = 1040 fixed rollout records
```

| ID | Task | Status | Output |
| --- | --- | --- | --- |
| P1A.1 | Copy/import source metadata for the old Assistant Axis role config and question config. | done | source-material config with old paths and hashes |
| P1A.2 | Select 48 roles grouped as 16 assistant-like, 16 non-assistant/non-neutral, 16 neutral/control. | done | role group section in rollout config |
| P1A.3 | Select 20 shared questions from the old 40-question pool, preserving all 8 categories. | done | question section in rollout config |
| P1A.4 | Add 4 default prompt families: helpful assistant, large language model, respond as yourself, bare question. | done | default prompt section in rollout config |
| P1A.5 | Document what cannot be imported directly from the old trait-grid builder. | done | notes in rollout config and repo build map |
| P1A.6 | Replace or source-verify planned role instructions not locally pinned in the old repo. | blocked | Needs exact upstream Assistant Axis role instruction files or an explicit decision to use local drafted role instructions. |
| P1A.7 | Audit non-assistant/non-neutral roles for over-fantasy bias and unsafe/actionable wording. | done | Group now mixes theatrical roles with adversarial, manipulative, authoritarian, conspiratorial, and self-interested roles; prompts describe style/worldview rather than actionable harm. |
| P1A.8 | Write rollout corpus builder design note before implementation. | done | `docs/design/rollout_corpus_builder_design.md` |

## Phase 2: Fixed Rollout Corpus

| ID | Task | Status | Output |
| --- | --- | --- | --- |
| P2.1 | Build rollout prompt records from config. | done | `scripts/rollouts/build_rollout_corpus.py` |
| P2.2 | Validate role/default/question balance: 960 role records plus 80 default records. | done | `data/rollouts/assistant_axis_rollouts_v0_manifest.json` |
| P2.3 | Save fixed rollout corpus and manifest. | done | `data/rollouts/assistant_axis_rollouts_v0.jsonl`, `data/rollouts/assistant_axis_rollouts_v0_manifest.json` |
| P2.4 | Add readable inspection command for sampled records. | done | `scripts/rollouts/inspect_rollouts.py`, `docs/design/rollout_inspector_design.md` |
| P2.5 | Define fixed generated response requirement for response-token pooling. | done | `configs/experiments/pythia_410m_mvp_v0.yaml`, `docs/design/activation_config_design.md` |
| P2.5A | Create failure learning log for known generation/activation caching issues from the old repo. | done | `docs/learning/failure_learning_log.md` |
| P2.6 | Define generated response record and manifest schemas. | done | `configs/schemas/generated_response_record.schema.yaml`, `configs/schemas/generated_response_manifest.schema.yaml` |
| P2.7 | Build fixed-response importer/validator. | done | `scripts/rollouts/import_fixed_responses.py`, `docs/design/fixed_response_import_design.md` |
| P2.8 | Create and validate tiny fixed-response fixture. | done | `data/rollouts/fixtures/fixed_responses_tiny_validated.jsonl`, `data/rollouts/fixtures/fixed_responses_tiny_manifest.json` |
| P2.9 | Generate or import fixed responses for all rollout records. | done | Role-faithful Llama run generated 1040 responses and importer wrote `data/rollouts/assistant_axis_rollouts_v0_responses.jsonl` plus manifest. |
| P2.10 | Build resumable fixed-response generator harness with smoke provider. | done | `scripts/rollouts/generate_fixed_responses.py`, `docs/design/fixed_response_generator_design.md`, `artifacts/runs/.../smoke-template-fixture/` |
| P2.11 | Add real local Hugging Face model provider for Llama fixed-response generation. | done | `scripts/rollouts/generate_fixed_responses.py --provider hf_local --hf-model-id meta-llama/Llama-3.2-1B-Instruct`; syntax verified, not model-run yet. |
| P2.12 | Run Llama fixed-response generation for the full 1040 rollout corpus. | done | VAST run used `meta-llama/Llama-3.2-1B-Instruct` role-faithful generation. |
| P2.13 | Persist reproducible model runtime requirements. | done | `requirements-model.txt`, `.venv` preflight passed. |
| P2.14 | Validate ungated Qwen fixed-response fallback. | done | `Qwen/Qwen2.5-0.5B-Instruct`; 12-record stratified smoke generated and imported successfully. |
| P2.15 | Run Qwen fixed-response generation for the full 1040 rollout corpus. | later | Fallback only if VAST Llama access fails. |
| P2.16 | Add VAST MVP runbook, checklist, and upload manifest template. | done | `docs/runbooks/vast_mvp_runbook.md`, `docs/runbooks/vast_mvp_checklist.md`, `docs/manifests/vast_mvp_upload_manifest_template.json` |

Exit criteria:

- Rollout corpus has stable ids.
- Manifest records source config hashes, counts, and validation results.
- Manifest explicitly records that the corpus is default-vs-role, not trait-instruction.
- Tiny response fixture proves the response contract before the full generator exists.

## Phase 3: Activation Extraction

| ID | Task | Status | Output |
| --- | --- | --- | --- |
| P3.1 | Implement activation extraction for one checkpoint, one layer, response-token mean pooling. | done | `scripts/activations/cache_rollout_activations.py` ran successfully for final, coarse, and early-dense checkpoint sweeps. |
| P3.1A | Add span/padding/empty-response/resume safeguards from failure log to activation runner design before coding. | done | `docs/design/activation_cache_runner_design.md` |
| P3.2 | Save activations with index, tensor shape metadata, run manifest, status, and progress. | done | Verified by activation inspector on VAST artifacts. |
| P3.3 | Add resume/skip behavior by rollout id and checkpoint. | done | Verified during checkpoint sweeps; completed stage outputs are reused. |
| P3.4 | Run a tiny local smoke test if dependencies are available. | done | Superseded by full VAST activation runs. |
| P3.5 | Add activation run inspector and smoke runbook. | done | `scripts/activations/inspect_activation_run.py`, `docs/design/activation_cache_runbook.md` |

Exit criteria:

- One checkpoint can produce pooled activations for a small rollout subset.
- Output can be resumed and audited from manifest/status/progress.

## Phase 4: Axis and Geometry Builders

| ID | Task | Status | Output |
| --- | --- | --- | --- |
| P4.1 | Build final-checkpoint Assistant Axis from default-vs-role means. | done | `scripts/analysis/build_assistant_axis.py`; final `step143000` run completed. |
| P4.1A | Define Assistant Axis vector schemas and builder design. | done | `configs/schemas/assistant_axis_vector.schema.yaml`, `configs/schemas/role_vector.schema.yaml`, `configs/schemas/geometry_manifest.schema.yaml`, `docs/design/assistant_axis_builder_design.md` |
| P4.1B | Implement first Assistant Axis builder. | done | `scripts/analysis/build_assistant_axis.py`; final and sweep runs completed. |
| P4.2 | Build role vectors and PC1 for one checkpoint. | done | `scripts/analysis/build_role_geometry.py`; final and sweep runs completed. |
| P4.3 | Compute AA-PC1 alignment and role loadings. | done | Implemented in `scripts/analysis/build_role_geometry.py`; writes `role_geometry_summary.json`, `role_pc1.pt`, and `role_loadings.csv`. |
| P4.4 | Add plot/report builder for final-checkpoint sanity checks. | done | `scripts/reporting/report_geometry.py`, `docs/design/geometry_report_design.md`; syntax verified, needs AA + role geometry run inputs. |

Exit criteria:

- Final checkpoint AA exists as a durable vector artifact.
- Role PC1 and loadings can be inspected before any checkpoint sweep.

## Phase 5: Checkpoint Sweep

| ID | Task | Status | Output |
| --- | --- | --- | --- |
| P5.1 | Define coarse checkpoint list. | done | `configs/experiments/pythia_410m_mvp_v0.yaml` defines the first `coarse_8` sweep. |
| P5.2 | Build checkpoint sweep runner. | done | `scripts/analysis/run_checkpoint_sweep.py` orchestrates per-checkpoint activation, inspection, AA, role geometry, and report stages. |
| P5.3 | Run activation extraction over selected checkpoints. | done | Coarse and early-dense VAST sweep artifacts uploaded to HF. |
| P5.4 | Build AA and PC1 per checkpoint. | done | Coarse and early-dense AA/role-geometry artifacts uploaded to HF. |
| P5.5 | Compute cosine-to-final, adjacent cosine, AA-PC1 alignment, PC1 variance, role-loading correlation, moving roles, and candidate transition windows. | done | `scripts/analysis/analyze_axis_trajectory.py` |
| P5.6 | Run trajectory analyzer on completed sweep. | done | Coarse and early-dense trajectory artifacts uploaded to HF. |
| P5.7 | Add plots for trajectory metrics. | done | `scripts/reporting/plot_axis_trajectory.py` |
| P5.7A | Run trajectory plotter on completed trajectory artifacts. | done | Coarse and early-dense plot artifacts uploaded to HF; plot style improved afterward and should be regenerated. |
| P5.8 | Select candidate emergence/refinement windows. | done | `docs/experiments/chosen_attribution_windows.md` |
| P5.9 | Upload curated MVP artifacts to Hugging Face. | done | HF dataset `Prasadmahadik/assistant-axis-emergence-attribution` contains fixed responses, final checkpoint artifacts, sweeps, trajectories, and plots. |
| P5.10 | Define 1000-to-5000 dense checkpoint sweep. | done | `configs/experiments/pythia_410m_dense_1000_5000_v0.yaml` |
| P5.11 | Run 1000-to-5000 dense checkpoint sweep. | done | `dense-1000-5000-full-v1` sweep, trajectory, and plots; see `docs/experiments/dense_1000_5000_sweep.md`. |

Exit criteria:

- We can justify which checkpoint windows should feed attribution.

## Phase 6: Training-Stream Loader and Attribution

| ID | Task | Status | Output |
| --- | --- | --- | --- |
| P6.0 | Write training-window planner design. | done | `docs/design/training_window_planner_design.md` |
| P6.1 | Implement checkpoint-window to Parquet-file mapping. | done | `scripts/data/plan_training_window.py` |
| P6.2 | Sample packed 2049-token sequences by `uid`/`batch_idx`. | done | `scripts/data/sample_training_sequences.py`, `configs/schemas/training_sequence_sample.schema.yaml`, `docs/design/training_sequence_sampler_design.md`; dry-run verified, real Parquet sampling runs externally. |
| P6.3 | Decode sample sequences for inspection. | done | `scripts/data/decode_training_sequences.py`, `configs/schemas/training_sequence_decoded_preview.schema.yaml`, `docs/design/training_sequence_decoder_design.md` |
| P6.4 | Compute activation-gradient cosine scores against local and final AA. | todo | `scripts/analysis/score_training_sequence_gradients.py` |
| P6.5 | Save optional per-sequence update-pressure vectors for structural analysis. | todo | gradient-pressure tensor/index artifacts |
| P6.6 | Produce top/bottom sequence tables and score summaries. | todo | attribution report |

Exit criteria:

- Debug sample of 1,000 packed sequences can be scored and inspected.
- Score sign convention is verified and documented in the manifest.
- Optional saved gradient-pressure vectors are available for PCA without rerunning backward passes.

## Phase 6B: Gradient-Pressure Structure and Attribution Extensions

| ID | Task | Status | Output |
| --- | --- | --- | --- |
| P6B.1 | Document Shifting-the-Gradient-inspired extensions. | done | `docs/design/gradient_attribution_extensions_design.md` |
| P6B.2 | Analyze PCA/SVD over per-sequence update-pressure vectors. | todo | `scripts/analysis/analyze_gradient_pressure_pca.py` |
| P6B.3 | Compare PC1/top-k gradient-pressure directions with local and final AA. | todo | PCA summary/report |
| P6B.4 | Build top/bottom/random sequence subset manifests for causal validation. | todo | subset manifests |
| P6B.5 | Design gradient-component interventions: neutralize, amplify, attenuate AA component. | later | intervention design + runner stub |

Exit criteria:

- We know whether AA-aligned training pressure is low-dimensional.
- We have ranked and control-matched sequence subsets ready for Phase 7.
- Causal intervention scripts are not started until observational scores and PCA are stable.

## Phase 7: Steering and Causal Validation

| ID | Task | Status | Output |
| --- | --- | --- | --- |
| P7.1 | Build small steering prompt set. | later | config + JSONL |
| P7.2 | Implement checkpoint-local AA steering. | later | `scripts/steering/run_axis_steering.py` |
| P7.3 | Run tiny continued-pretraining validation on Phase 6B subsets. | later | validation run artifacts |
| P7.4 | Run gradient-component intervention variants if justified. | later | neutralize/amplify/attenuate artifacts |
| P7.5 | Recompute geometry and behavior after validation runs. | later | comparison report |

Exit criteria:

- Causal validation is only attempted after attribution scoring is stable.

## Update Rule

When a task changes status, update:

1. this tasklist,
2. `docs/design/project_tracker.md`,
3. any relevant build-map section,
4. run manifests/status files if an experiment has run.
