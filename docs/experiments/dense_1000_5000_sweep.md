# Dense Sweep: step1000 to step5000

## Question

The coarse sweep showed substantial continued Assistant Axis alignment between
`step1000 -> step5000`. This sweep asks whether that transition is spread
evenly or concentrated in a smaller post-1000 interval.

## Setup

- Model: `EleutherAI/pythia-410m-deduped`
- Layer: `12`
- Pooling: `response_token_mean`
- Axis variant: `aa_main`
- Checkpoints: `step1000`, `step2000`, `step3000`, `step4000`, `step5000`
- Local endpoint for trajectory analysis: `step5000`
- Run id used in Colab: `dense-1000-5000-full-v1`

## Artifacts

- HF dataset: `Prasadmahadik/assistant-axis-emergence-attribution`
- HF prefix: `pythia410m-mvp-v0`
- Sweep run: `artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/checkpoint-sweep-layer12/dense-1000-5000-full-v1`
- Trajectory run: `artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-layer12/dense-1000-5000-full-v1`
- Plot run: `artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-plots-layer12/dense-1000-5000-full-v1`

## Checkpoint Metrics

| checkpoint | AA vs step5000 | AA vs previous | PC1 vs step5000 | PC1 vs previous | AA-PC1 | PC1 EVR | AA loading corr vs step5000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step1000 | 0.5694 |  | 0.6730 |  | 0.8715 | 0.4794 | 0.9842 |
| step2000 | 0.7634 | 0.6231 | 0.8666 | 0.7484 | 0.8882 | 0.4216 | 0.9942 |
| step3000 | 0.8871 | 0.8178 | 0.9390 | 0.9130 | 0.8986 | 0.4206 | 0.9985 |
| step4000 | 0.9335 | 0.9113 | 0.9625 | 0.9476 | 0.9093 | 0.4264 | 0.9995 |
| step5000 | 1.0000 | 0.9335 | 1.0000 | 0.9625 | 0.9088 | 0.4326 | 1.0000 |

## Transition Scores

| window | transition score | AA adjacent cosine | PC1 adjacent cosine | AA loading adjacent corr | note |
| --- | ---: | ---: | ---: | ---: | --- |
| `step1000 -> step2000` | 0.7177 | 0.6231 | 0.7484 | 0.9852 | Strongest post-1000 stabilization window. |
| `step2000 -> step3000` | 0.2846 | 0.8178 | 0.9130 | 0.9961 | Continued but smaller refinement. |
| `step3000 -> step4000` | 0.1591 | 0.9113 | 0.9476 | 0.9985 | Mostly stable refinement. |
| `step4000 -> step5000` | 0.1113 | 0.9335 | 0.9625 | 0.9995 | Mostly stable refinement. |

## Findings

The large coarse `step1000 -> step5000` transition is concentrated most strongly
in `step1000 -> step2000`. Later windows still refine the direction, but with
substantially smaller transition scores.

## Decision

Use `step1000 -> step2000` as the primary post-1000 attribution window. Treat
`step2000 -> step3000` as a lower-priority secondary refinement window. Do not
use the full `step1000 -> step5000` window as the first attribution unit unless
we intentionally want a broad aggregate sample.
