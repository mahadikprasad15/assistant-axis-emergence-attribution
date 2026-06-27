# Final Checkpoint Geometry: step143000

## Question

Does the final Pythia-410M-deduped checkpoint contain a coherent Assistant
Axis direction before we spend compute on checkpoint sweeps and attribution?

## Setup

- Model: `EleutherAI/pythia-410m-deduped`
- Checkpoint: `step143000`
- Layer: `12`
- Pooling: `response_token_mean`
- Rollout corpus: 1040 fixed role/default responses
- Axis variant: `aa_main`

## Artifacts

- HF dataset: `Prasadmahadik/assistant-axis-emergence-attribution`
- HF prefix: `pythia410m-mvp-v0`
- AA run: `artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/aa-main-layer12/aa-main-step143000-layer12-full-v0`
- Role geometry run: `artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/role-geometry-layer12/role-geometry-step143000-layer12-full-v0`
- Geometry report: `artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/geometry-report-layer12/geometry-report-step143000-layer12-full-v0`

## Result

- Activation records: `1040/1040`
- Missing activation files: `0`
- Bad response spans: `0`
- Tensor shape mismatches: `0`
- AA-PC1 cosine: `0.9226236343383789`
- PC1 explained variance ratio: `0.405979722738266`
- Gate: `proceed`

Top aligned roles/defaults included technical support agent, helpful assistant,
bureaucrat, planner, teacher, documentation writer, consultant, encyclopedia,
lawyer, and moderator.

## Interpretation

The final checkpoint contains a strong role/default geometry aligned with the
hand-constructed Assistant Axis. This justifies checkpoint sweeps.

The result should not be described as pure abstract "assistantness." The more
accurate interpretation is a structured, institutional, professional,
helpful-assistant-like direction.

## Decision

Proceed to checkpoint trajectory analysis.

## Limitations

- One layer only: layer 12.
- One pooling method only: response-token mean.
- One main axis variant: `aa_main`.
- No training-data attribution or causal validation yet.
