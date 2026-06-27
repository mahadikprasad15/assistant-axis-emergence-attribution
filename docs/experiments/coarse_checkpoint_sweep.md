# Coarse Checkpoint Sweep

## Question

When does the final Assistant-Axis-like geometry appear and stabilize across
Pythia-410M-deduped training?

## Setup

- Model: `EleutherAI/pythia-410m-deduped`
- Layer: `12`
- Pooling: `response_token_mean`
- Axis variant: `aa_main`
- Checkpoints: `step0`, `step1000`, `step5000`, `step10000`, `step20000`,
  `step40000`, `step80000`, `step143000`

## Artifacts

- HF dataset: `Prasadmahadik/assistant-axis-emergence-attribution`
- HF prefix: `pythia410m-mvp-v0`
- Sweep run: `artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/checkpoint-sweep-layer12/coarse8-full-v0`
- Trajectory run: `artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-layer12/coarse8-full-v0`
- Plot run: `artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-plots-layer12/coarse8-full-v0`

Artifact audit (2026-06-27): HF preserves the sweep summary, trajectory, and plots. Except for the separately uploaded final checkpoint, the original uploader omitted per-checkpoint activation, AA vector, PC1 vector, and geometry sibling runs.

## Main Metrics

| checkpoint | AA vs final | AA vs previous | PC1 vs final | PC1 vs previous | AA-PC1 | PC1 EVR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step0 | -0.0023 |  | -0.0042 |  | 0.8670 | 0.2254 |
| step1000 | 0.4622 | 0.0146 | 0.5356 | 0.0217 | 0.8715 | 0.4795 |
| step5000 | 0.7848 | 0.5694 | 0.8165 | 0.6730 | 0.9088 | 0.4326 |
| step10000 | 0.8396 | 0.8927 | 0.8739 | 0.9298 | 0.9107 | 0.4257 |
| step20000 | 0.8782 | 0.9222 | 0.9073 | 0.9493 | 0.9086 | 0.4210 |
| step40000 | 0.9328 | 0.9244 | 0.9491 | 0.9559 | 0.9282 | 0.4195 |
| step80000 | 0.9756 | 0.9501 | 0.9811 | 0.9655 | 0.9202 | 0.4078 |
| step143000 | 1.0000 | 0.9756 | 1.0000 | 0.9811 | 0.9226 | 0.4060 |

## Findings

The largest coarse unresolved transition is `step0 -> step1000`. However,
`step1000 -> step5000` is also substantial:

- AA-to-final gain from `step0 -> step1000`: about `+0.464`
- AA-to-final gain from `step1000 -> step5000`: about `+0.323`
- AA-to-final gain from `step5000 -> step10000`: about `+0.055`

After `step5000`, the direction is much more stable and later training mostly
refines the geometry.

## Decision

Densify both early training windows:

- `step0 -> step1000`
- `step1000 -> step5000`

## Interpretation Boundary

The sweep supports a tracked formation/stabilization story, not a claim that
assistantness appears from nothing. Prompt-induced role/default structure exists
at initialization, but training organizes it toward the final axis.
