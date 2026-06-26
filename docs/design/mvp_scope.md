# MVP Scope

## Objective

Build a minimal but credible pipeline:

```text
AA construction
-> AA trajectory through checkpoints
-> candidate emergence/refinement window
-> gradient attribution on real Pythia training sequences
-> small causal validation
```

The MVP uses `EleutherAI/pythia-410m-deduped` because Pythia provides public checkpoints and tools/data artifacts for reconstructing the training stream in order.

## Research Questions

### RQ1: Geometric Emergence

Does an Assistant-Axis-like direction, constructed from fixed final rollouts, become geometrically stable across pretraining checkpoints?

Primary metrics:

- `cos(v_aa_t, v_aa_final)`
- `cos(v_aa_t, v_aa_t_next)`

### RQ2: Relation To Persona PC1

Does the contrast-defined Assistant Axis align with the leading role/persona principal component at each checkpoint?

Primary metrics:

- `abs(cos(v_aa_t, pc1_t))`
- `variance_explained(pc1_t)`

### RQ3: Behavioral Characterization

When does checkpoint-local AA steering begin to affect generation?

Minimal intervention:

- add `+alpha * v_aa_t`
- add `-alpha * v_aa_t`
- compare completions on a small fixed prompt set

### RQ4: Training-Sequence Pressure

During candidate emergence/refinement windows, which real Pythia training sequences exert first-order gradient pressure toward the Assistant Axis?

Score:

```text
s_t(z) = -cos(d L_t(z) / d h_t, v_aa_t)
```

Also compute:

```text
s_final(z) = -cos(d L_t(z) / d h_t, v_aa_final)
```

This separates pressure toward the local checkpoint axis from pressure toward the mature final axis.

## Rollout Corpus

Use fixed texts generated once by a strong model. Do not ask each Pythia checkpoint to generate role responses.

Target MVP size:

- 48 roles
- 20 shared questions
- 4 default assistant prompt families
- about 1,040 fixed texts total

Role groups:

- assistant-like/professional roles
- non-assistant/non-neutral roles mixing theatrical, adversarial, manipulative, authoritarian, conspiratorial, and self-interested styles
- neutral/control roles

Default prompts:

- `You are a helpful assistant.`
- `You are a large language model.`
- `Respond as yourself.`
- bare question / no role prompt

## Activation Extraction

First pass:

- one model: `EleutherAI/pythia-410m-deduped`
- one layer: middle residual stream layer
- one pooling method: mean over response tokens
- selected checkpoints only

Debug fallback: full-text mean pooling. Main results should use response-token pooling.

For text `x` at checkpoint `t`:

```text
h_t(x) = mean residual activation over response tokens at layer l
```

## Assistant Axis Construction

For checkpoint `t`:

```text
a_t = mean activation over default assistant rollouts
r_t = mean activation over role rollouts
v_aa_t = normalize(a_t - r_t)
```

For a closer imitation of the original AA contrast, exclude neutral controls from `r_t` and keep them for PC/loadings diagnostics.

## Role PC1

For each role `r`:

```text
role_vec_r_t = mean activation over that role's rollouts
```

Run PCA over mean-centered role vectors. Orient PC1 so default assistant projects positive, then track:

```text
abs(cos(v_aa_t, pc1_t))
```

## Checkpoint Plan

Use two passes.

### Pass 1: Coarse Sweep

Use 15-20 checkpoints, including:

- `step0`
- early log-spaced checkpoints
- sparse checkpoints through training
- `step143000`

### Pass 2: Densify

Run all checkpoints around regions where:

- `cos(v_aa_t, v_aa_final)` rises rapidly
- adjacent-checkpoint cosine dips
- AA-PC1 alignment changes
- PC1 role loadings reorganize
- steering first starts to work

## Required MVP Plots

1. Cosine to final AA
2. Adjacent-checkpoint cosine
3. AA-PC1 alignment
4. PC1 variance explained
5. Role-loading correlation with final checkpoint

## Minimal Gradient Attribution

For selected checkpoint/window `t` and training sequence `z`:

1. Run `z` through checkpoint `t`.
2. Compute LM loss.
3. Backprop to the selected residual-stream activation.
4. Score against `v_aa_t` and `v_aa_final`.

Start with:

- 1,000 sequences per selected window for debugging
- 10,000 sequences for the first real result
- larger samples only after the pipeline is stable and resumable

## Minimal Causal Validation

From the candidate emergence checkpoint, construct four matched subsets:

- top AA-amplifying sequences
- bottom / AA-attenuating sequences
- random sequences
- keyword baseline sequences

Continue-pretrain small model copies for a fixed token budget, then remeasure:

- cosine to final AA
- AA-PC1 alignment
- role-loading correlation
- steering effect
- small behavioral assistantness labels

## Claims To Avoid

Do not claim:

- exact documents created the Assistant Axis
- the axis emerges exactly at one checkpoint
- gradient cosine is influence-function attribution
- assistantness is fully explained by one source category

Prefer:

> These packed training sequences exert first-order Assistant-Axis-amplifying pressure at this checkpoint.

and:

> The axis appears to stabilize or refine during this training region under this measurement.
