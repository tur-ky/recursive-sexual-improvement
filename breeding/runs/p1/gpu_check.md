# Does the GPU change Phase 1's results?

Phase 1 first ran its exams on whichever of tranny's two cards was free. Each model below was examined on both the RTX 3070 and the RTX 2070 (same merged fp16 weights, same greedy decoding, same 1,600 problems).

## Same model, two cards

| model | first run on | same final answer | 3070 parents' fields | 2070 parents' fields | 3070 held out | 2070 held out |
|---|---|---|---|---|---|---|
| base | 3070 | 93% | 31.0 | 31.8 | 22.2 | 22.0 |
| mult_mlp | 2070 | 95% | 35.1 | 34.5 | 28.7 | 28.2 |
| div_mlp | 3070 | 96% | 25.9 | 26.0 | 25.2 | 24.8 |
| mult_all | 3070 | 96% | 35.0 | 35.1 | 27.3 | 27.0 |
| div_all | 2070 | 96% | 21.1 | 21.6 | 25.5 | 25.0 |
| mt_mlp | 3070 | 95% | 35.6 | 36.0 | 27.8 | 27.0 |
| mt_all | 2070 | 96% | 35.8 | 36.2 | 27.8 | 27.5 |
| merge_all | 3070 | 96% | 35.8 | 35.2 | 28.7 | 28.0 |
| merge_mlp | 2070 | 96% | 40.6 | 40.6 | 28.2 | 29.0 |
| pay_into_mult | 3070 | 95% | 35.1 | 35.1 | 26.8 | 27.8 |
| pay_into_div | 3070 | 95% | 36.2 | 36.2 | 28.5 | 29.0 |
| merge_all_family | 2070 | 96% | 36.9 | 36.5 | 31.5 | 31.2 |
| merge_mlp_vote | 2070 | 96% | 36.5 | 36.8 | 28.5 | 28.2 |
| pay_into_mult_a25 | 2070 | 95% | 33.8 | 33.2 | 27.5 | 27.5 |
| pay_into_mult_a75 | 2070 | 95% | 39.2 | 39.0 | 27.5 | 27.5 |
| pay_into_mult_late | 2070 | 96% | 33.6 | 33.9 | 28.5 | 29.2 |
| pay_into_div_middle | 2070 | 95% | 32.0 | 32.1 | 25.2 | 24.2 |

Mean accuracy difference over all 1,600 problems, 3070 minus 2070, across these 17 models: -0.02 points (range -0.6 to +0.6).

## Key comparisons in each exam set

Paired difference and 95% bootstrap interval. "As first run" mixes cards; the other two columns put both models of a comparison on one card.

| comparison | as first run | all 3070 | all 2070 |
|---|---|---|---|
| merge_mlp vs mt_mlp, parents' fields | +5.0 [+2.1, +8.0] | +5.0 [+2.1, +8.0] | +4.6 [+1.7, +7.6] |
| merge_mlp vs mult_mlp, mult | +5.8 [+1.2, +10.2] | +5.2 [+0.5, +9.8] | +5.8 [+1.2, +10.2] |
| merge_mlp vs div_mlp, div | +4.0 [+0.5, +7.8] | +4.0 [+0.5, +7.8] | +3.2 [+0.0, +6.8] |
| merge_mlp vs base, held out | +6.8 [+2.8, +10.5] | +6.0 [+2.2, +9.8] | +7.0 [+3.0, +10.8] |
| merge_all vs mt_all, parents' fields | -0.5 [-3.5, +2.5] | +0.0 [-2.9, +3.0] | -1.0 [-4.0, +2.0] |
| merge_all vs mult_all, mult | -6.2 [-10.8, -1.5] | -6.2 [-10.8, -1.5] | -6.5 [-11.0, -1.8] |
| pay_into_div vs div_mlp, div | +5.2 [+2.2, +8.5] | +5.2 [+2.2, +8.5] | +4.0 [+1.0, +7.0] |
| mult_mlp vs base, mult | +9.0 [+4.5, +13.8] | +9.5 [+5.0, +14.2] | +8.2 [+4.0, +12.8] |
| div_mlp vs base, div | +0.5 [-3.8, +5.0] | +0.5 [-3.8, +5.0] | +0.5 [-3.8, +5.0] |
