# Replication of Phase 1's core result

MLP-only specialists from answer-key self-training; merge = both parents' changes added; payload = recipient plus half the donor's change; multitask = one model trained on both parents' data. Paired differences, 95% bootstrap intervals, same exam problems for every model of a family, every exam on the RTX 3070 (Phase 1's models first examined on the 2070 were re-examined on the 3070; see gpu_check.md).

## mult + div, seed 1 (Phase 1)

| model | mult | div | both | held out |
|---|---|---|---|---|
| base | 32.2 | 29.8 | 31.0 | 22.2 |
| mult_mlp | 41.8 | 28.5 | 35.1 | 28.8 |
| div_mlp | 21.5 | 30.2 | 25.9 | 25.2 |
| mt_mlp | 39.2 | 32.0 | 35.6 | 27.8 |
| merge_mlp | 47.0 | 34.2 | 40.6 | 28.2 |
| pay_into_mult | 40.5 | 29.8 | 35.1 | 26.8 |
| pay_into_div | 37.0 | 35.5 | 36.2 | 28.5 |

- merge vs A parent on A: +5.2 [+0.5, +9.8]
- merge vs B parent on B: +4.0 [+0.5, +7.8]
- merge vs multitask on both: +5.0 [+2.1, +8.0]
- merge vs base on held out: +6.0 [+2.2, +9.8]
- A parent vs base on A: +9.5 [+5.0, +14.2]
- B parent vs base on B: +0.5 [-3.8, +5.0]

## mult + div, seed 2

| model | mult | div | both | held out |
|---|---|---|---|---|
| base | 32.2 | 29.8 | 31.0 | 22.2 |
| s2_mult_mlp | 51.5 | 23.0 | 37.2 | 25.0 |
| s2_div_mlp | 28.8 | 37.2 | 33.0 | 27.0 |
| s2_mt_mlp | 50.8 | 33.0 | 41.9 | 25.0 |
| s2_merge_mlp | 41.5 | 33.5 | 37.5 | 27.8 |
| s2_pay_into_mult | 46.8 | 25.8 | 36.2 | 27.0 |
| s2_pay_into_div | 37.5 | 39.2 | 38.4 | 26.8 |

- merge vs A parent on A: -10.0 [-14.5, -5.5]
- merge vs B parent on B: -3.8 [-7.5, -0.2]
- merge vs multitask on both: -4.4 [-7.8, -1.1]
- merge vs base on held out: +5.5 [+1.2, +9.5]
- A parent vs base on A: +19.2 [+14.5, +24.0]
- B parent vs base on B: +7.5 [+3.0, +12.2]

## mult + div, seed 3

| model | mult | div | both | held out |
|---|---|---|---|---|
| base | 32.2 | 29.8 | 31.0 | 22.2 |
| s3_mult_mlp | 38.5 | 22.8 | 30.6 | 29.5 |
| s3_div_mlp | 31.0 | 41.5 | 36.2 | 25.5 |
| s3_mt_mlp | 31.2 | 35.2 | 33.2 | 26.8 |
| s3_merge_mlp | 35.0 | 34.8 | 34.9 | 28.0 |
| s3_pay_into_mult | 38.8 | 29.8 | 34.2 | 30.8 |
| s3_pay_into_div | 36.5 | 38.2 | 37.4 | 28.5 |

- merge vs A parent on A: -3.5 [-7.8, +0.8]
- merge vs B parent on B: -6.8 [-11.2, -2.2]
- merge vs multitask on both: +1.6 [-1.2, +4.6]
- merge vs base on held out: +5.8 [+1.7, +10.0]
- A parent vs base on A: +6.2 [+1.2, +11.0]
- B parent vs base on B: +11.8 [+7.2, +16.3]

## base + sort, seed 1

| model | base | sort | both | held out |
|---|---|---|---|---|
| base | 35.2 | 36.8 | 36.0 | 22.8 |
| bs_base_mlp | 45.5 | 34.2 | 39.9 | 28.2 |
| bs_sort_mlp | 40.0 | 51.5 | 45.8 | 29.2 |
| bs_mt_mlp | 45.2 | 46.8 | 46.0 | 25.5 |
| bs_merge_mlp | 49.2 | 47.2 | 48.2 | 30.2 |
| bs_pay_into_base | 46.2 | 43.8 | 45.0 | 26.8 |
| bs_pay_into_sort | 49.2 | 52.2 | 50.8 | 28.2 |

- merge vs A parent on A: +3.8 [+0.8, +6.8]
- merge vs B parent on B: -4.2 [-7.5, -1.0]
- merge vs multitask on both: +2.2 [-0.4, +5.0]
- merge vs base on held out: +7.5 [+3.0, +12.0]
- A parent vs base on A: +10.2 [+6.2, +14.5]
- B parent vs base on B: +14.8 [+11.0, +18.8]

## Summary

| family | merge vs A parent on A | merge vs B parent on B | merge vs multitask on both | merge vs base on held out | A parent vs base on A | B parent vs base on B |
|---|---|---|---|---|---|---|
| mult + div, seed 1 (Phase 1) | +5.2 [+0.5, +9.8] | +4.0 [+0.5, +7.8] | +5.0 [+2.1, +8.0] | +6.0 [+2.2, +9.8] | +9.5 [+5.0, +14.2] | +0.5 [-3.8, +5.0] |
| mult + div, seed 2 | -10.0 [-14.5, -5.5] | -3.8 [-7.5, -0.2] | -4.4 [-7.8, -1.1] | +5.5 [+1.2, +9.5] | +19.2 [+14.5, +24.0] | +7.5 [+3.0, +12.2] |
| mult + div, seed 3 | -3.5 [-7.8, +0.8] | -6.8 [-11.2, -2.2] | +1.6 [-1.2, +4.6] | +5.8 [+1.7, +10.0] | +6.2 [+1.2, +11.0] | +11.8 [+7.2, +16.3] |
| base + sort, seed 1 | +3.8 [+0.8, +6.8] | -4.2 [-7.5, -1.0] | +2.2 [-0.4, +5.0] | +7.5 [+3.0, +12.0] | +10.2 [+6.2, +14.5] | +14.8 [+11.0, +18.8] |

## Preregistered hypotheses (PREREGISTRATION.md, SHA-256 9e6c58b6...)

- **H1** (MLP merge above MLP multitask on both parents' fields; confirmed in a family if the Holm-adjusted test rejects zero in the merge's favour): mult + div, seed 2: -4.4 [-7.8, -1.1], p = 0.0085, not confirmed; mult + div, seed 3: +1.6 [-1.2, +4.6], p = 0.306, not confirmed; base + sort, seed 1: +2.2 [-0.4, +5.0], p = 0.0975, not confirmed. Confirmed in 0 of 3; mean difference over all 4 families +1.12. **DOES NOT HOLD** (needs at least 2 of 3 and a positive mean).
- **H2** (merge not worse than a parent on its own field, lower bound above -3, where the parent's gain is measurable): mult + div, seed 2, parent A: merge vs parent -10.0 [-14.5, -5.5], FAIL; mult + div, seed 2, parent B: merge vs parent -3.8 [-7.5, -0.2], FAIL; mult + div, seed 3, parent A: merge vs parent -3.5 [-7.8, +0.8], FAIL; mult + div, seed 3, parent B: merge vs parent -6.8 [-11.2, -2.2], FAIL; base + sort, seed 1, parent A: merge vs parent +3.8 [+0.8, +6.8], pass; base + sort, seed 1, parent B: merge vs parent -4.2 [-7.5, -1.0], FAIL. **DOES NOT HOLD**.
- **H3** (merge above the base on the held-out fields): mult + div, seed 2: +5.5 [+1.2, +9.5], confirmed; mult + div, seed 3: +5.8 [+1.7, +10.0], confirmed; base + sort, seed 1: +7.5 [+3.0, +12.0], confirmed. Confirmed in 3 of 3.

