# Gradient Attribution Extensions

## Why This Exists

The first attribution scorer should answer the observational question:

```text
Which packed training sequences exert first-order pressure toward the Assistant Axis?
```

The Shifting-the-Gradient-inspired follow-ups answer two stronger questions:

```text
Is that pressure low-dimensional?
Can changing the AA-aligned gradient component causally change AA formation?
```

These are important, but they should not block the first gradient scorer.

## Core Object

For a sampled packed training sequence, checkpoint `t`, and layer/site `h`:

```text
g_i = mean_tokens(∂L_i / ∂h_layer)
u_i = -g_i
```

`g_i` is the activation gradient. `u_i` is the gradient-descent update pressure
in activation space.

The first attribution score is:

```text
AA amplification score = cosine(u_i, v_AA)
                       = -cosine(g_i, v_AA)
```

Interpretation:

```text
positive high: sequence locally pushes toward the Assistant Axis
near zero: little AA-aligned pressure
negative: sequence locally pushes away from the Assistant Axis
```

This is principled because both the Assistant Axis and the activation gradient
live in the same residual-stream vector space.

## Extension 1: Gradient-Pressure PCA

After the scorer saves per-sequence update-pressure vectors, build:

```text
U = [u_1, u_2, ..., u_n]
```

Then compute PCA/SVD over rows or centered rows and report:

```text
PC1 explained variance
top-k explained variance
cos(PC1, v_AA_local)
cos(PC1, v_AA_final)
distribution of sequence scores along PC1
window-to-window differences
```

This tests whether AA-forming training pressure is concentrated in a
low-dimensional direction or spread across many directions.

Analyzer:

```text
scripts/analysis/analyze_gradient_pressure_pca.py
```

Canonical outputs:

```text
results/gradient_pressure_pca_summary.json
results/gradient_pressure_pca.csv
results/gradient_pressure_components.jsonl
results/pcs/*.pt
results/singular_values/*.pt
```

## Extension 2: Gradient Component Intervention Design

For a gradient vector:

```text
g_parallel = projection(g, v_AA)
g_perp = g - g_parallel
```

Candidate Shifting-style interventions:

```text
neutralize AA component:
  g' = g_perp

amplify AA component:
  g' = g_perp + alpha * g_parallel

attenuate AA component:
  g' = g_perp - alpha * g_parallel
```

This is a later causal-validation stage. It should be attempted only after:

1. The observational scorer is stable.
2. Top/bottom/random sequence groups are interpretable.
3. The gradient-pressure PCA report shows whether a low-dimensional AA-aligned
   component exists.

Planned runner:

```text
scripts/analysis/run_gradient_component_intervention.py
```

Possible outputs:

```text
results/intervention_summary.json
results/intervention_scores.jsonl
results/post_intervention_geometry.json
```

## Extension 3: Continued-Training Causal Validation

After scoring and ranking sequences, build small continued-training subsets:

```text
top AA-amplifying sequences
bottom AA-suppressing sequences
random matched controls
keyword/source controls if available
```

Then train tiny variants and recompute:

```text
AA vector
role PC1
AA-PC1 cosine
role loadings
behavior/steering probes if available
```

This is where stronger causal claims can start. Until then, the honest claim is:

```text
This packed sequence exerts first-order activation-space pressure toward/away
from the Assistant Axis at checkpoint t.
```

not:

```text
This document caused the Assistant Axis.
```

## Placement In Build Order

```text
Phase 6A: observational gradient scoring
Phase 6B: PCA/structure analysis over saved gradient-pressure vectors
Phase 6C: top/bottom attribution summaries and subset construction
Phase 7: steering and causal validation
```

The next implementation should still be Phase 6A: the gradient scorer.
