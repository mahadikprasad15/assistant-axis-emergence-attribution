# Concept Attribution: Step256 to Step512 Plan

Status: **designed; implementation not started**

## Research Question

Which packed sequences seen between `step256` and `step512` are most associated
with formation or rotation toward the endpoint and final Assistant Axis?

## Frozen Decisions

| Decision | Value |
| --- | --- |
| Model used for scoring | `EleutherAI/pythia-410m-deduped@step256` |
| Training window | `batch_idx >= 256` and `< 512` |
| Master random sample | 5,000 |
| Vector Filter sample | 5,000 |
| Activation-gradient sample | 2,000 nested random |
| FOPCI sample | 250 nested random + 250 adaptive |
| Layer | 12 |
| Gradient dtype | float32 |
| Primary activation-gradient score | `dot(-mean(dL/dh), target)` |
| Primary FOPCI score | `-dot(grad_theta L_i, grad_theta S_target)` |
| Curvature approximation | identity |

## Primary Targets

- construction-split `step512` AA;
- construction-split final `step143000` AA;
- construction-split `step256 -> step512` innovation direction.

The native `step256` axis and existing all-question axes are diagnostics only.

## Validity Checks

- [ ] Construction and evaluation question IDs are disjoint.
- [ ] Both question halves cover all eight categories.
- [ ] Master/subset manifests contain stable IDs and hashes.
- [ ] Gradient runs use float32.
- [ ] Activation-gradient dot agrees between batch size 1 and a larger batch.
- [ ] FOPCI query gradient is nonzero and finite.
- [ ] FOPCI 50-record smoke completes before the 500-record run.
- [ ] Random and adaptive FOPCI results are reported separately.
- [ ] All artifacts are uploaded before destroying the compute instance.

## Comparison Outputs

- Pearson and Spearman correlations on shared records;
- top/bottom-k overlap;
- endpoint/final/innovation target agreement;
- random-half versus adaptive-half FOPCI summaries;
- decoded top, bottom, disagreement, and near-zero examples;
- source-file and batch-index distributions;
- explicit recommendation for causal continued-training subsets.

See `docs/design/concept_attribution_ladder_design.md` for formulas,
components, artifacts, and claim boundaries.
