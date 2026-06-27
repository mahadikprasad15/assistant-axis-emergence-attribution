# Early Dense Sweep: step0 to step1000

## Question

Is the large coarse `step0 -> step1000` transition an immediate first-update
effect, or does it happen later inside the first 1000 steps?

## Setup

- Model: `EleutherAI/pythia-410m-deduped`
- Layer: `12`
- Pooling: `response_token_mean`
- Axis variant: `aa_main`
- Checkpoints: `step0`, `step1`, `step2`, `step4`, `step8`, `step16`,
  `step32`, `step64`, `step128`, `step256`, `step512`, `step1000`
- Local endpoint for trajectory analysis: `step1000`

## Artifacts

- HF dataset: `Prasadmahadik/assistant-axis-emergence-attribution`
- HF prefix: `pythia410m-mvp-v0`
- Sweep run: `artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/checkpoint-sweep-layer12/early-dense-0-1000-full-v0`
- Trajectory run: `artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-layer12/early-dense-0-1000-full-v0`
- Plot run: `artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-plots-layer12/early-dense-0-1000-full-v0`

Artifact audit (2026-06-27): HF preserves the sweep summary, trajectory, and plots, but the original uploader omitted the per-checkpoint activation, AA vector, PC1 vector, and geometry sibling runs. Those tensors are not recoverable from the summary alone.

## Findings

The large `step0 -> step1000` transition is not primarily an immediate
`step0 -> step1` effect. The earliest transitions are comparatively small.

The strongest dense early candidate windows are:

- `step256 -> step512`
- `step128 -> step256`
- `step32 -> step64`
- `step16 -> step32`
- `step64 -> step128`

The current interpretation is that the role/default geometry begins weak or
unstable near initialization, then reorganizes mostly between roughly
`step16 -> step512`.

## Decision

Use these as first training-data attribution targets:

- `step128 -> step256`
- `step256 -> step512`

Use these as contextual secondary windows:

- `step16 -> step32`
- `step32 -> step64`
- `step64 -> step128`

Keep `step0 -> step1` or `step1 -> step2` as early controls.

## Limitation

This sweep explains only the `0 -> 1000` transition. The coarse sweep also
showed substantial continued stabilization from `step1000 -> step5000`, so that
window requires its own dense sweep.
