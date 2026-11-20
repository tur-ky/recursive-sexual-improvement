# Replication of Phase 1's core result

MLP-only specialists from answer-key self-training; merge = both parents' changes added; payload = recipient plus half the donor's change; multitask = one model trained on both parents' data. Paired differences, 95% bootstrap intervals, same exam problems for every model of a family.

## mult + div, seed 1 (Phase 1)

| model | mult | div | both | held out |
|---|---|---|---|---|
| base | 32.2 | 29.8 | 31.0 | 22.2 |
| mult_mlp | 41.2 | 27.8 | 34.5 | 28.2 |
| div_mlp | 21.5 | 30.2 | 25.9 | 25.2 |
| mt_mlp | 39.2 | 32.0 | 35.6 | 27.8 |
| merge_mlp | 47.0 | 34.2 | 40.6 | 29.0 |
| pay_into_mult | 40.5 | 29.8 | 35.1 | 26.8 |
| pay_into_div | 37.0 | 35.5 | 36.2 | 28.5 |

- merge vs A parent on A: +5.8 [+1.2, +10.2]
- merge vs B parent on B: +4.0 [+0.5, +7.8]
- merge vs multitask on both: +5.0 [+2.1, +8.0]
- merge vs base on held out: +6.8 [+2.8, +10.5]
- A parent vs base on A: +9.0 [+4.5, +13.8]
- B parent vs base on B: +0.5 [-3.8, +5.0]

## Summary

| family | merge vs A parent on A | merge vs B parent on B | merge vs multitask on both | merge vs base on held out | A parent vs base on A | B parent vs base on B |
|---|---|---|---|---|---|---|
| mult + div, seed 1 (Phase 1) | +5.8 [+1.2, +10.2] | +4.0 [+0.5, +7.8] | +5.0 [+2.1, +8.0] | +6.8 [+2.8, +10.5] | +9.0 [+4.5, +13.8] | +0.5 [-3.8, +5.0] |
