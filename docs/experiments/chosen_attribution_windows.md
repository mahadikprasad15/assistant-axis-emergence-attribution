# Chosen Attribution Windows

## Purpose

Record the checkpoint windows selected for first-pass training-data attribution
so the choice does not live only in chat history or plots.

## Primary Windows

| window | reason |
| --- | --- |
| `step128 -> step256` | Strong early-dense transition inside the first 1000 steps. |
| `step256 -> step512` | Strongest early-dense transition and likely central early reorganization window. |
| `step1000 -> step5000` | Second large coarse transition; substantial AA-to-final alignment gain remains after step1000. |

## Secondary Context Windows

| window | reason |
| --- | --- |
| `step16 -> step32` | Early dense candidate transition. |
| `step32 -> step64` | Early dense candidate transition. |
| `step64 -> step128` | Early dense candidate transition. |

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
3. `step1000 -> step5000`
4. one early control: `step0 -> step1`
5. one late control: `step80000 -> step143000`

## Claim Boundary

The attribution unit is initially a packed Pythia training sequence, not an
original raw document. Decoded text can support interpretation, but source/raw
document mapping is a separate later layer.
