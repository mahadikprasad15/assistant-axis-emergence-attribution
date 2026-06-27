# Chosen Attribution Windows

## Purpose

Record the checkpoint windows selected for first-pass training-data attribution
so the choice does not live only in chat history or plots.

## Primary Windows

| window | reason |
| --- | --- |
| `step128 -> step256` | Strong early-dense transition inside the first 1000 steps. |
| `step256 -> step512` | Strongest early-dense transition and likely central early reorganization window. |
| `step512 -> step1000` | Continued consolidation after the strongest early-dense reorganization. |
| `step1000 -> step2000` | Strongest narrowed post-1000 transition from the dense 1000-to-5000 sweep. |

## Secondary Context Windows

| window | reason |
| --- | --- |
| `step16 -> step32` | Early dense candidate transition. |
| `step32 -> step64` | Early dense candidate transition. |
| `step64 -> step128` | Early dense candidate transition. |
| `step2000 -> step3000` | Smaller continued refinement after the main post-1000 transition. |

## Consolidated Formation Band

The combined early-dense and 1000-to-5000 dense sweeps identify the useful
formation band as separate adjacent windows from `step128 -> step3000`, not as
one pooled window.

| rank | window | transition score | AA adjacent cosine | PC1 adjacent cosine | AA loading adjacent corr | note |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `step256 -> step512` | 3.5843 | 0.3619 | -0.6841 | 0.5297 | Strongest observed formation/reorganization window. |
| 2 | `step128 -> step256` | 2.6801 | 0.4498 | -0.6505 | 0.9765 | Major early reorganization; PC1 is unstable/flipping relative to adjacent checkpoint. |
| 3 | `step512 -> step1000` | 0.9271 | 0.5901 | 0.7275 | 0.9683 | Continued consolidation after the strongest early window. |
| 4 | `step1000 -> step2000` | 0.7177 | 0.6231 | 0.7484 | 0.9852 | Strongest post-1000 stabilization window. |
| 5 | `step2000 -> step3000` | 0.2846 | 0.8178 | 0.9130 | 0.9961 | Lower-priority continued refinement. |

Current summary claim:

> Most AA/role-geometry formation occurs between `step128` and `step512`, with
> continued stabilization through `step2000`.

## Controls

| window | reason |
| --- | --- |
| `step0 -> step1` | Very early update control. |
| `step1 -> step2` | Very early update control. |
| `step80000 -> step143000` | Late stable-training control. |

## First Debug Attribution Run

Use `1,000` packed training sequences per selected window before scaling to
larger samples.

First run:

1. `step128 -> step256`
2. `step256 -> step512`
3. `step512 -> step1000`
4. `step1000 -> step2000`
5. one early control: `step0 -> step1`
6. one late control: `step80000 -> step143000`

## Claim Boundary

The attribution unit is initially a packed Pythia training sequence, not an
original raw document. Decoded text can support interpretation, but source/raw
document mapping is a separate later layer.
