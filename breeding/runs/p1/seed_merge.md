# Merge test: same-seed against different-seed parents

Design, outcomes and predictions: `SEEDMERGE-PLAN.md` (committed before any of these models existed). Every exam on tranny's RTX 3070, greedy, merged fp16 weights, Phase 1's exam (400 mult, 400 div, 400 modpow, 400 held out: code + clock). Intervals: paired bootstrap over exam problems within each field (4000 resamples, seed 0); they cover exam sampling only, not which seeds the six parents came from.

## Parents

| model | mult | div | modpow | held out |
|---|---|---|---|---|
| base | 32.2 | 29.8 | 39.5 | 22.2 |
| mult_mlp | 41.8 | 28.5 | 50.8 | 28.8 |
| s2_mult_mlp | 51.5 | 23.0 | 47.8 | 25.0 |
| s3_mult_mlp | 38.5 | 22.8 | 50.2 | 29.5 |
| div_mlp | 21.5 | 30.2 | 29.2 | 25.2 |
| s2_div_mlp | 28.8 | 37.2 | 22.8 | 27.0 |
| s3_div_mlp | 31.0 | 41.5 | 28.2 | 25.5 |

## Children

Merge cost = mean over the two fields of (child minus that field's parent), in points; 0 means the child keeps both parents' skill. Sign conflicts: share of changed weights where the two parents' changes have opposite signs. From plain: how far the combined change is from the plain sum (relative norm).

| method | mult parent | div parent | seeds | mult | div | merge cost | held out vs base | modpow vs base | sign conflicts | from plain |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | mult_mlp | div_mlp | same | 47.0 | 35.0 | +5.0 [+2.1, +7.9] | +6.0 [+2.0, +9.8] | +10.8 [+6.2, +15.2] | 0.474 | 0.00 |
| plain | mult_mlp | s2_div_mlp | different | 38.2 | 34.2 | -3.2 [-6.1, -0.4] | +5.2 [+1.0, +9.2] | +9.8 [+5.8, +13.8] | 0.500 | 0.00 |
| plain | mult_mlp | s3_div_mlp | different | 40.2 | 42.5 | -0.2 [-3.2, +2.8] | +7.5 [+3.5, +11.5] | +10.8 [+6.5, +15.2] | 0.499 | 0.00 |
| plain | s2_mult_mlp | div_mlp | different | 40.0 | 31.0 | -5.4 [-8.2, -2.6] | +4.2 [+0.5, +8.0] | +6.8 [+2.5, +11.0] | 0.500 | 0.00 |
| plain | s2_mult_mlp | s2_div_mlp | same | 41.5 | 33.2 | -7.0 [-9.9, -4.1] | +6.0 [+2.2, +10.0] | +8.8 [+4.8, +12.8] | 0.483 | 0.00 |
| plain | s2_mult_mlp | s3_div_mlp | different | 43.2 | 38.8 | -5.5 [-8.5, -2.4] | +6.8 [+2.8, +11.0] | +12.2 [+7.8, +16.8] | 0.500 | 0.00 |
| plain | s3_mult_mlp | div_mlp | different | 35.8 | 30.2 | -1.4 [-4.1, +1.5] | +5.2 [+1.2, +9.2] | +10.2 [+6.0, +14.5] | 0.500 | 0.00 |
| plain | s3_mult_mlp | s2_div_mlp | different | 41.5 | 34.2 | +0.0 [-3.0, +3.0] | +7.0 [+3.0, +11.0] | +8.2 [+4.2, +12.2] | 0.500 | 0.00 |
| plain | s3_mult_mlp | s3_div_mlp | same | 35.8 | 34.8 | -4.8 [-7.9, -1.6] | +6.2 [+2.0, +10.2] | +8.0 [+3.8, +12.8] | 0.489 | 0.00 |
| elect | mult_mlp | div_mlp | same | 47.0 | 35.5 | +5.2 [+2.5, +8.1] | +5.0 [+1.0, +8.8] | +8.5 [+4.2, +12.8] | 0.474 | 0.27 |
| elect | mult_mlp | s2_div_mlp | different | 38.2 | 34.8 | -3.0 [-6.1, -0.1] | +5.2 [+1.0, +9.2] | +11.5 [+7.8, +15.8] | 0.500 | 0.29 |
| elect | mult_mlp | s3_div_mlp | different | 40.2 | 43.2 | +0.1 [-2.9, +3.1] | +7.8 [+3.8, +11.5] | +11.0 [+6.5, +15.5] | 0.499 | 0.29 |
| elect | s2_mult_mlp | div_mlp | different | 41.0 | 31.5 | -4.6 [-7.4, -1.8] | +4.5 [+0.8, +8.2] | +6.0 [+1.8, +10.0] | 0.500 | 0.29 |
| elect | s2_mult_mlp | s2_div_mlp | same | 41.2 | 33.5 | -7.0 [-10.0, -4.0] | +5.8 [+1.8, +9.8] | +6.8 [+2.8, +11.0] | 0.483 | 0.28 |
| elect | s2_mult_mlp | s3_div_mlp | different | 42.5 | 38.2 | -6.1 [-9.2, -2.9] | +4.2 [+0.2, +8.5] | +12.5 [+8.0, +17.0] | 0.500 | 0.29 |
| elect | s3_mult_mlp | div_mlp | different | 36.8 | 32.2 | +0.1 [-2.6, +3.0] | +5.8 [+1.8, +9.8] | +11.2 [+7.0, +15.8] | 0.500 | 0.29 |
| elect | s3_mult_mlp | s2_div_mlp | different | 42.8 | 34.0 | +0.5 [-2.5, +3.5] | +7.8 [+3.8, +12.0] | +8.2 [+4.0, +12.5] | 0.500 | 0.29 |
| elect | s3_mult_mlp | s3_div_mlp | same | 34.5 | 33.8 | -5.9 [-8.9, -2.8] | +5.5 [+1.5, +9.5] | +8.0 [+3.8, +12.5] | 0.489 | 0.28 |
| ties | mult_mlp | div_mlp | same | 36.8 | 30.2 | -2.5 [-5.2, +0.2] | +5.5 [+1.2, +9.5] | +8.2 [+4.2, +12.2] | 0.474 | 0.61 |
| ties | mult_mlp | s2_div_mlp | different | 37.0 | 30.5 | -5.8 [-8.8, -2.9] | +4.5 [+0.2, +8.8] | +6.8 [+2.5, +10.8] | 0.500 | 0.61 |
| ties | mult_mlp | s3_div_mlp | different | 38.8 | 38.5 | -3.0 [-6.1, +0.1] | +3.8 [+0.0, +7.5] | +7.5 [+2.8, +12.0] | 0.499 | 0.61 |
| ties | s2_mult_mlp | div_mlp | different | 36.5 | 29.0 | -8.1 [-10.9, -5.2] | +2.0 [-1.8, +5.8] | +1.2 [-3.0, +5.8] | 0.500 | 0.61 |
| ties | s2_mult_mlp | s2_div_mlp | same | 39.5 | 32.0 | -8.6 [-11.5, -5.8] | +1.8 [-2.5, +5.8] | +7.0 [+3.0, +11.0] | 0.483 | 0.61 |
| ties | s2_mult_mlp | s3_div_mlp | different | 37.0 | 32.2 | -11.9 [-15.0, -8.8] | +5.0 [+1.2, +8.8] | +7.2 [+2.8, +11.8] | 0.500 | 0.61 |
| ties | s3_mult_mlp | div_mlp | different | 32.8 | 29.5 | -3.2 [-6.0, -0.6] | +6.5 [+2.5, +10.5] | +10.5 [+6.2, +14.8] | 0.500 | 0.61 |
| ties | s3_mult_mlp | s2_div_mlp | different | 34.8 | 33.5 | -3.8 [-6.8, -0.8] | +9.2 [+5.5, +13.5] | +11.0 [+7.0, +15.0] | 0.500 | 0.61 |
| ties | s3_mult_mlp | s3_div_mlp | same | 38.2 | 31.0 | -5.4 [-8.4, -2.4] | +8.0 [+3.8, +12.0] | +6.8 [+2.2, +11.2] | 0.489 | 0.62 |

### Merge cost, plain (rows: mult parent's seed, columns: div parent's seed)

| | p1 | s2 | s3 |
|---|---|---|---|
| p1 | +5.0 | -3.2 | -0.2 |
| s2 | -5.4 | -7.0 | -5.5 |
| s3 | -1.4 | +0.0 | -4.8 |

### Merge cost, elect (rows: mult parent's seed, columns: div parent's seed)

| | p1 | s2 | s3 |
|---|---|---|---|
| p1 | +5.2 | -3.0 | +0.1 |
| s2 | -4.6 | -7.0 | -6.1 |
| s3 | +0.1 | +0.5 | -5.9 |

### Merge cost, ties (rows: mult parent's seed, columns: div parent's seed)

| | p1 | s2 | s3 |
|---|---|---|---|
| p1 | -2.5 | -5.8 | -3.0 |
| s2 | -8.1 | -8.6 | -11.9 |
| s3 | -3.2 | -3.8 | -5.4 |

## Contrasts

- seed contrast, plain: same-seed minus different-seed merge cost: +0.4 [-1.1, +1.8], p = 0.634
- seed contrast, elect: same-seed minus different-seed merge cost: -0.4 [-1.8, +1.0], p = 0.614
- seed contrast, ties: same-seed minus different-seed merge cost: +0.5 [-0.8, +1.8], p = 0.501
- elect minus plain, merge cost, all 9: +0.2 [-0.3, +0.7], p = 0.434
- elect minus plain, merge cost, same-seed: -0.3 [-1.2, +0.5], p = 0.555
- elect minus plain, merge cost, different-seed: +0.5 [-0.2, +1.1], p = 0.167
- ties minus plain, merge cost, all 9: -3.3 [-4.4, -2.2], p < 0.0005
- ties minus plain, merge cost, same-seed: -3.2 [-4.9, -1.6], p < 0.0005
- ties minus plain, merge cost, different-seed: -3.3 [-4.6, -2.2], p < 0.0005
- elect minus plain, held out, all 9: -0.3 [-0.9, +0.2], p = 0.295
- ties minus plain, held out, all 9: -0.9 [-2.1, +0.3], p = 0.158

## Predictions (SEEDMERGE-PLAN.md), judged on the point estimate as written there

1. Sign-elect within 2 points of plain on different-seed pairs: +0.5 [-0.2, +1.1]: **held** (interval also inside +-2).
2. TIES worse than plain by more than 3 points: -3.3 [-4.4, -2.2]: **held**.
3. Seed contrast under plain: no direction predicted; see the contrasts above.

## Build check: rebuilt plain same-seed children against the original merge children

The original merges were exported as bf16 LoRA factors and baked by merge_lora.py; the rebuilt ones use fp32 factors (ties_build.py). Same exam, same GPU.

| pair | original | rebuilt | same final answer |
|---|---|---|---|
| p1 | 40.6 | 41.0 | 95.3% |
| s2 | 37.5 | 37.4 | 94.9% |
| s3 | 34.9 | 35.2 | 94.8% |
