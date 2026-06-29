# Assistant Axis Emergence and Training-Sequence Attribution

## Discussion report (evidence state: 2026-06-28)

### Executive summary

We studied when an Assistant-Axis-like activation direction forms during the training of `EleutherAI/pythia-410m-deduped`, then built a first-order method for asking which checkpoint-ordered training sequences exert pressure along that direction.

The checkpoint sweeps localize the largest observed reorganization to `step128 -> step512`, followed by consolidation through roughly `step2000`. The strongest individual interval is `step256 -> step512`: adjacent Assistant Axis (AA) cosine is only `0.3619`, adjacent role-PC1 cosine is `-0.6841`, and the transition score is `3.5843`. The final-checkpoint geometry is coherent: AA versus role-geometry PC1 cosine is about `0.9226`.

Attribution is currently a method-validated smoke experiment, not a scaled result. We scored the same 50 packed training sequences at `step256` and `step512`. Each packed row contains 2049 token ids and produces 2048 next-token losses. The score is the cosine between an AA target and the negative mean hidden-state loss gradient at layer 12. A centered PCA of the 50 per-sequence pressure vectors gave PC1 explained-variance ratio `0.155` and weak AA alignment. This does not yet support a claim that the dominant training pressure is one-dimensional or AA-directed. It motivated diagnostics for target choice, token-level cancellation, and numerical precision before scaling.

### 1. Research question and claim boundary

The experiment asks:

1. When does a stable Assistant-Axis-like direction emerge in Pythia training?
2. In the checkpoint windows where that direction changes most, which training sequences exert first-order pressure toward or away from a specified AA direction?

The defensible attribution claim is:

> A packed Pythia training sequence exerts first-order AA-amplifying or AA-opposing activation-space pressure at a specified checkpoint, layer, and axis target.

It is not yet defensible to say that a raw document created the Assistant Axis. Pythia trains on packed sequences that may combine document fragments; raw-document recovery is a separate provenance problem. Gradient alignment is also a local diagnostic, not proof of the realized optimizer update or long-horizon causal influence.

### 2. Model, semantic probe, and objects

- Model: `EleutherAI/pythia-410m-deduped` (410M parameters).
- Readout: residual-stream layer 12.
- Semantic corpus: 1040 frozen prompt-response records: 48 roles x 20 shared questions, plus 4 default prompt families x 20 questions.
- Role grouping: 16 assistant-like, 16 non-assistant/non-neutral contrast, and 16 neutral/control roles.
- Responses: one fixed 1040-record Llama response corpus, reused at every Pythia checkpoint so changing generations cannot masquerade as changing geometry.
- Rollout pooling: mean over response tokens only, to avoid directly pooling role instruction/prompt tokens.
- AA construction: mean(default activations) minus mean(contrast activations) for the `aa_main` variant.
- Independent geometry check: construct role/default mean vectors, center them, compute PC1 by SVD, orient PC1 toward AA, and compare AA-PC1 cosine and role loadings.

### 3. Phase I pipeline: axis construction through checkpoint sweeps

For each checkpoint:

1. Run the frozen 1040 rollout texts through Pythia.
2. cache layer-12 response-token-mean activations;
3. build the AA vector from default-versus-contrast means;
4. build role vectors and role-geometry PC1;
5. check AA-PC1 alignment and PC1 explained variance;
6. compare AA, PC1, and role loadings across checkpoints.

The persisted pipeline uses manifests, status/progress files, JSON/JSONL summaries, and tensor artifacts under `artifacts/runs/...`. A later audit found that the original remote upload preserved sweep summaries/plots but omitted most per-checkpoint sibling tensors; this was subsequently fixed for future uploads. Summary-level conclusions remain available, but omitted historical tensors cannot be reconstructed from cosine tables alone.

### 4. Phase I results

#### Coarse sweep

Checkpoints: `0, 1000, 5000, 10000, 20000, 40000, 80000, 143000`.

| checkpoint | AA vs final | PC1 vs final | AA-PC1 | PC1 EVR |
| --- | ---: | ---: | ---: | ---: |
| step0 | -0.0023 | -0.0042 | 0.8670 | 0.2254 |
| step1000 | 0.4622 | 0.5356 | 0.8715 | 0.4795 |
| step5000 | 0.7848 | 0.8165 | 0.9088 | 0.4326 |
| step10000 | 0.8396 | 0.8739 | 0.9107 | 0.4257 |
| step40000 | 0.9328 | 0.9491 | 0.9282 | 0.4195 |
| step143000 | 1.0000 | 1.0000 | 0.9226 | 0.4060 |

The largest coarse gaps were `step0 -> step1000` and `step1000 -> step5000`. After `step5000`, the direction mostly refines rather than reorganizes.

#### Dense localization

The early sweep used `0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000`. The strongest formation band was not the first update; it was concentrated later, especially `128 -> 512`.

| rank | window | transition score | adjacent AA cosine | adjacent PC1 cosine | adjacent loading corr. |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 256 -> 512 | 3.5843 | 0.3619 | -0.6841 | 0.5297 |
| 2 | 128 -> 256 | 2.6801 | 0.4498 | -0.6505 | 0.9765 |
| 3 | 512 -> 1000 | 0.9271 | 0.5901 | 0.7275 | 0.9683 |
| 4 | 1000 -> 2000 | 0.7177 | 0.6231 | 0.7484 | 0.9852 |
| 5 | 2000 -> 3000 | 0.2846 | 0.8178 | 0.9130 | 0.9961 |

Negative adjacent PC1 cosine in the two early windows indicates an unstable/reoriented principal direction, not automatically a semantic sign reversal: PC signs are arbitrary, and orientation conventions plus changing role geometry must be inspected together. The unusually low AA and PC1 continuity at `256 -> 512`, combined with low loading continuity, makes it the strongest candidate for attribution.

### 5. From checkpoint windows to actual training sequences

The practical stream is `pietrolesci/pile-deduped-pythia-preshuffled`, a Parquet repackaging of the checkpoint-ordered Pythia stream. Each row has `uid`, `batch_idx`, and 2049 token ids.

Pythia's documented training batch is 2,097,152 tokens. At 2048 prediction tokens per packed row, this corresponds to 1024 packed sequences per optimizer step. Consequently:

| window | optimizer steps | approximate packed rows seen | planned sample | planned fraction |
| --- | ---: | ---: | ---: | ---: |
| 128 -> 256 | 128 | 131,072 | 1,000 | 0.763% |
| 256 -> 512 | 256 | 262,144 | 1,000 | 0.381% |
| 512 -> 1000 | 488 | 499,712 | 1,000 | 0.200% |
| 1000 -> 2000 | 1,000 | 1,024,000 | 1,000 | 0.0977% |
| 2000 -> 3000 | 1,000 | 1,024,000 | 1,000 | 0.0977% |

The completed attribution smoke used 50 sequences from `256 -> 512`, only about `0.0191%` of that window's 262,144 packed rows. It contains 102,400 scored next-token positions in total (`50 x 2048`), but only 50 independent packed-sequence units for sequence-level statistics. Token count must therefore not be mistaken for sample size.

The stream object is already packed. One row can contain multiple documents or partial documents separated by EOD tokens. Sequence-level topical labels can therefore be mixtures, and a salient short span can be diluted by the other tokens in the row.

### 6. Phase II attribution logic

For a 2049-token row:

```text
input_ids = token_ids[:-1]   # 2048 tokens
targets   = token_ids[1:]    # 2048 next-token targets
```

At a chosen model checkpoint, we compute next-token cross-entropy, retain the gradient of the layer-12 hidden states, and define per-sequence update pressure:

```text
u_i = -mean_valid_tokens(dL_i / dh_layer)
score(i, axis) = cosine(u_i, axis)
```

Positive means the local negative-loss-gradient direction aligns with the named axis; negative means opposition. The same sampled sequences were scored at `step256` and `step512`, enabling checkpoint comparison without sample-composition confounding.

Target choice is part of the estimand, not a cosmetic option:

- `native_step256`: alignment with the axis represented by the model at the start checkpoint;
- `endpoint_step512`: alignment with the axis that exists after the window;
- `final_step143000`: alignment with the eventual mature axis.

A paired checkpoint delta should normally hold the target vector fixed. Comparing each checkpoint to its own native axis changes both the model and the ruler, so it answers a different question.

### 7. Attribution results so far

- Paired 50-sequence scoring at `step256` and `step512` completed successfully, validating model loading, hidden-state gradient extraction, durable result writing, resume behavior, and saved pressure vectors.
- Centered PCA over the 50 sequence-pressure vectors produced PC1 EVR `0.155`.
- PC1 was only weakly aligned with AA.
- Therefore the smoke data do not show a single dominant AA-aligned gradient-pressure mode.
- The run was too small to rank content types, compare tails reliably, or infer window-wide prevalence.
- The first scorer pooled gradients before taking cosine. This can hide cancellation: strongly positive and negative token pressures in one packed sequence may average to a weak vector. The current scorer adds per-token cosine distributions and can save all token-axis scores.
- The initial results also did not yet establish robustness to automatic mixed precision. A comparator now supports identical-sample `auto` versus `float32` checks; that VAST diagnostic remains pending according to the tracker.

No stronger numerical statement about score means, tails, or checkpoint deltas is preserved in the current repo summaries. Those values should be read from the original run artifacts before discussion, rather than reconstructed or guessed.

### 8. Main potential issues

1. **Axis validity versus prompt artifact.** High AA-PC1 agreement is encouraging, but both derive from the same frozen role/default corpus. The axis may encode style, verbosity, role instruction residue, or response distribution differences. Response-only pooling reduces direct prompt contamination but does not eliminate induced stylistic confounds.
2. **One layer only.** Layer 12 may not be where training pressure relevant to assistant behavior is clearest. A layer sweep could change both localization and attribution rankings.
3. **Axis drift.** Native, endpoint, and final axes differ substantially in the formation window. Results depend on which direction is treated as the target.
4. **Mean-gradient cancellation.** `cos(mean token gradient, axis)` can be near zero even when individual token positions have large opposing effects. Report both cosine-after-pooling and the distribution/mean of per-token cosines.
5. **Loss weighting.** Mean token loss gives every valid token equal weight. Repetitive/easy tokens dominate by count, while rare high-loss tokens may dominate gradient magnitude. Consider loss-, gradient-norm-, and span-aware summaries as diagnostics, not silent replacements.
6. **Packed mixtures.** A 2049-token row may combine unrelated documents. Sequence ranking is faithful to the training unit but weak for semantic interpretation.
7. **Sampling fraction and dependence.** Fifty rows are far too few for tail discovery. Rows within nearby batches may also be correlated, so nominal sequence count can overstate effective sample size.
8. **Window width asymmetry.** A fixed 1000-row sample represents 0.763% of `128 -> 256` but only 0.0977% of `1000 -> 2000`. Cross-window comparisons need equal-fraction sampling, uncertainty weighting, or an explicit reason for equal absolute sample size.
9. **Repeated exposure / epoch structure.** Attribution is to a position in the preshuffled stream. Similar or duplicated content elsewhere can distribute influence across multiple rows.
10. **First-order approximation.** Hidden-state gradient alignment is not TracIn, influence functions, or exact optimizer replay. It ignores parameter mapping, optimizer state, weight decay, momentum, and interactions among training examples.
11. **Checkpoint interval versus instantaneous scoring.** Scoring a row at `step256` estimates local pressure at that model state, although rows later in the `256 -> 512` window were actually encountered at progressively changed parameters. Endpoint scoring brackets the issue but does not reproduce the whole training trajectory.
12. **Numerics.** Small cosines can be sensitive to dtype, gradient scaling, normalization, and near-zero norms. Float32 agreement and norm diagnostics are prerequisites for scaling.
13. **Historical artifact completeness.** Some sweep summaries survived without per-checkpoint tensors. Future analysis should require tensor checksums and upload-completeness gates before deleting compute instances.

### 9. Recommended focus for the next attribution iteration

The narrowest useful next step is not all selected windows. It is a staged study of `step256 -> step512`:

1. Run an identical 50-100 row diagnostic in `float32` and automatic precision; require strong per-row agreement.
2. Score fixed `native_step256`, `endpoint_step512`, and `final_step143000` targets at both model checkpoints.
3. Save pooled pressure vectors and complete token-axis cosine arrays.
4. Quantify cancellation per sequence: pooled cosine, mean token cosine, positive-token fraction, opposing-tail mass, and gradient norm.
5. Decode and inspect only stratified extremes: stable-positive under all targets/checkpoints, sign-changing, high-cancellation, and stable-negative rows.
6. Scale to at least the planned 1000 uniformly sampled rows only after these diagnostics pass.
7. Add a matched control window (preferably `step0 -> step1` for formation specificity and a manageable data footprint; the very broad late control needs shard-efficient sampling).
8. Bootstrap confidence intervals over sequences and, where possible, batch blocks. Treat tokens as repeated observations within a sequence, not independent samples.
9. If semantic interpretation matters, split packed rows at EOD boundaries and report span-level diagnostics while retaining the packed-row score as the primary training-unit result.
10. Only after stable observational rankings should we test causality through tiny optimizer replay, gradient-component ablation, or continued-pretraining interventions.

### 10. Questions for another AI/researcher

1. What estimand is most appropriate: pressure toward the native axis, the endpoint axis, the final axis, or the observed axis displacement `normalize(v_endpoint - v_start)`?
2. Should the primary score remain activation-space pressure, or be mapped through the layer Jacobian to parameter-gradient alignment with the checkpoint-to-checkpoint weight update?
3. How should token-level evidence be aggregated when a packed row contains multiple documents and opposing spans?
4. Is uniform row sampling adequate, or should sampling be stratified by batch, loss, source mixture, EOD count, gradient norm, or approximate semantic clusters?
5. What control best distinguishes assistant-specific formation from generic early-training representation reorganization?
6. How should uncertainty account for correlations among rows in the same training batch or neighboring batches?
7. Would exact replay over a small subwindow give a more meaningful validation target than comparing instantaneous gradients at the two endpoints?
8. Is PC1 EVR `0.155` at `n=50` informative at all in 1024-dimensional hidden space without a permutation/noise baseline and out-of-sample stability test?

### Bottom line

The checkpoint evidence is strong enough to focus attribution on `step256 -> step512`, with `step128 -> step256` and `step512 -> step1000` as neighboring context. The attribution machinery works, but the substantive data-attribution result is not yet established. The immediate scientific problem is to fix the axis target, expose token cancellation, verify precision, and increase the sequence sample while preserving the packed-sequence claim boundary.
