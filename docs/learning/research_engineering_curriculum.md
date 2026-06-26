# Research Engineering Curriculum

This is the learning layer for the repo. It explains the concepts we are building, in the order they will appear. Update it whenever a new object, pattern, or script family is introduced.

## Learning Goal

Build the Pythia Assistant Axis attribution project while learning reusable research-engineering patterns:

- config-driven experiments,
- typed objects and schemas,
- durable artifacts,
- resumable execution,
- activation extraction,
- vector and PCA workflows,
- gradient attribution,
- small causal validation,
- report and decision logging.

## Core Pattern: Config to Artifact to Audit

Every serious stage should look like:

```text
config
-> builder / runner / analyzer
-> artifact files
-> manifest + status + progress
-> read-back validation
-> report or decision
```

This prevents fragile one-off notebooks and makes the project easier to resume.

## Component Types

### Builder

A builder constructs a durable artifact.

Examples in this repo:

- `RolloutCorpusBuilder`
- `AssistantAxisBuilder`
- `RoleGeometryBuilder`
- `ReportBuilder`

What to learn:

- deterministic records,
- validation before write,
- stable ids,
- manifest writing,
- config hashing.

### Runner

A runner executes expensive or external work.

Examples in this repo:

- `ActivationCacheRunner`
- `CheckpointSweepRunner`
- `GradientAttributionRunner`
- `SteeringRunner`

What to learn:

- batching,
- progress checkpoints,
- resume/skip behavior,
- structured logging,
- memory constraints,
- failure recovery.

### Analyzer

An analyzer consumes artifacts and produces metrics.

Examples in this repo:

- `TrajectoryAnalyzer`
- `RoleLoadingAnalyzer`
- `AttributionSummaryAnalyzer`

What to learn:

- tensor shapes,
- grouping by metadata,
- cosine similarity,
- PCA,
- interpretation boundaries,
- plotting without hiding raw values.

### Gate

A gate makes a proceed/pivot decision explicit.

Examples in this repo:

- final-AA sanity gate,
- checkpoint-transition gate,
- attribution-debug gate,
- causal-validation gate.

What to learn:

- explicit thresholds,
- negative results,
- short decision records,
- why "not ready yet" is a valid result.

## Concept Ladder

### Level 1: Repo Control Surface

Learn:

- `README.md` as the front door,
- `docs/design/project_tracker.md` as current state,
- `docs/design/tasklist.md` as the task queue,
- `docs/design/repo_build_map.md` as the system map.

Build:

- project scaffold,
- initial tasklist,
- object glossary.

### Level 2: Configs and Schemas

Learn:

- why configs are human-authored,
- why schemas are machine-checkable,
- how stable ids prevent confusion later.

Build:

- model config,
- dataset config,
- rollout config,
- record schemas.

### Level 3: Rollout Corpus

Learn:

- fixed semantic stimuli,
- role/default split,
- neutral controls,
- prompt/response token spans,
- generated text vs checkpoint-generated text,
- the difference between prompt-only rollout records and generated model responses,
- how a builder turns configs into JSONL plus a manifest.
- how an importer/validator defines a data contract before a model-backed generator exists.
- how a runner differs from a builder because it owns status, progress, logs, and resume behavior.

Build:

- rollout JSONL,
- rollout manifest,
- rollout inspector.
- generated response schemas,
- fixed-response generator harness,
- fixed-response importer,
- tiny response fixture.

Diagram focus:

- dataflow from configs to builder to artifacts,
- helper-function call graph,
- distinction between builder, importer, inspector/analyzer, and future runner.

Read:

- `docs/learning/rollout_corpus_walkthrough.md`
- `docs/design/fixed_response_generator_design.md`
- `docs/design/fixed_response_import_design.md`
- `docs/learning/failure_learning_log.md`

### Level 4: Activations and Tensor Shapes

Learn expected shapes:

```text
tokens: [batch, seq]
residual activations: [batch, seq, d_model]
pooled activations: [batch, d_model]
role/default means: [d_model]
AA vector: [d_model]
role matrix: [n_roles, d_model]
```

Build:

- model-runtime preflight,
- activation cache runner,
- activation index,
- response-token pooling,
- activation run inspector.

Preflight:

Before running Llama generation or Pythia activation caching, run:

```bash
.venv/bin/python scripts/system/check_model_runtime.py
```

This must pass before model downloads, generation, or activation runs. A module-discovery check is not enough; the preflight must actually import `torch` and run a tiny tensor operation.

Read-back validation:

After any activation run, inspect the run directory before interpreting vectors:

```bash
.venv/bin/python scripts/activations/inspect_activation_run.py --run-dir <activation-run-dir>
```

The inspector checks the state files, progress counts, activation index, tensor-file existence, response-token spans, and recorded activation shapes.

### Level 5: Axis and Geometry

Learn:

- default mean,
- role mean,
- contrast vector,
- unit normalization,
- PCA sign orientation,
- cosine-to-final,
- adjacent-checkpoint cosine.

Build:

- final AA,
- Assistant Axis vector schema,
- Assistant Axis builder,
- role PC1,
- role geometry builder,
- final-checkpoint geometry report,
- trajectory plots.

First AA formula:

```text
default_mean = mean(selected default prompt activations)
contrast_mean = mean(selected contrast role activations)
assistant_axis = normalize(default_mean - contrast_mean)
```

Read:

- `docs/design/assistant_axis_builder_design.md`
- `docs/design/role_geometry_builder_design.md`
- `docs/design/geometry_report_design.md`

### Level 6: Pythia Training Stream

Learn:

- raw document vs packed training sequence,
- `batch_idx`,
- `uid`,
- 2049-token sequence,
- checkpoint-window mapping,
- why `step30000 -> step40000` maps to `train-031000.parquet` through `train-040000.parquet`.

Build:

- training-window planner,
- sequence sampler,
- decode inspector.

### Level 7: Gradient Attribution

Learn:

- LM loss on packed sequences,
- backprop to residual-stream activations,
- gradient direction vs gradient-descent update direction,
- local AA score vs final AA score.

Build:

- attribution scorer,
- top/bottom sequence tables,
- source/mapping TODOs.

### Level 8: Causal Validation

Learn:

- why gradient scores are not influence functions,
- matched subsets,
- keyword baselines,
- small continued-pretraining experiments,
- post-intervention geometry checks.

Build:

- top/bottom/random/keyword subsets,
- tiny continued-pretraining validation,
- comparison report.

## Rules For Future Work

- Before adding a script, add or update the component entry in `docs/design/repo_build_map.md`.
- Before running an experiment, define the run artifact layout and manifest fields.
- After running anything expensive, update `docs/design/project_tracker.md` and `docs/design/tasklist.md`.
- If a result is ambiguous, write a decision record instead of pretending it is resolved.
- Before implementing activation caching or response generation, check `docs/learning/failure_learning_log.md` for known padding, empty-response, span, and resume failures.
