# Phase 1 report

## Exam (greedy; mult, div, modpow 400 problems each; code + clock 400)

Every exam below ran on tranny's RTX 3070 (merged fp16 weights, vLLM).

| model | mult | div | parents' fields | modpow (3rd) | held out |
|---|---|---|---|---|---|
| base | 32.2 | 29.8 | 31.0 | 39.5 | 22.2 |
| mult_mlp | 41.8 | 28.5 | 35.1 | 50.8 | 28.8 |
| mult_all | 43.0 | 27.0 | 35.0 | 51.0 | 27.2 |
| div_mlp | 21.5 | 30.2 | 25.9 | 29.2 | 25.2 |
| div_all | 9.2 | 33.0 | 21.1 | 30.8 | 25.5 |
| mt_mlp | 39.2 | 32.0 | 35.6 | 47.0 | 27.8 |
| mt_all | 35.5 | 36.0 | 35.8 | 49.2 | 27.8 |
| merge_all | 36.8 | 34.8 | 35.8 | 47.8 | 28.8 |
| merge_mlp | 47.0 | 34.2 | 40.6 | 50.2 | 28.2 |
| pay_into_mult | 40.5 | 29.8 | 35.1 | 48.5 | 26.8 |
| pay_into_div | 37.0 | 35.5 | 36.2 | 44.0 | 28.5 |
| merge_mlp_family | 44.5 | 32.2 | 38.4 | 52.8 | 30.0 |
| merge_all_family | 45.0 | 28.8 | 36.9 | 49.8 | 31.5 |
| pay_into_mult_family | 44.8 | 28.2 | 36.5 | 48.5 | 27.0 |
| pay_into_div_family | 41.0 | 28.2 | 34.6 | 52.0 | 29.5 |
| merge_mlp_vote | 45.0 | 28.0 | 36.5 | 50.2 | 28.5 |
| merge_mlp_key | 45.8 | 29.5 | 37.6 | 53.0 | 30.0 |
| mt_mlp_vote | 35.2 | 29.2 | 32.2 | 50.8 | 30.2 |
| pay_into_mult_a10 | 39.5 | 28.5 | 34.0 | 51.5 | 28.8 |
| pay_into_mult_a25 | 39.5 | 28.0 | 33.8 | 50.2 | 27.5 |
| pay_into_mult_a75 | 45.2 | 33.2 | 39.2 | 52.0 | 27.5 |
| pay_into_mult_early | 42.0 | 28.8 | 35.4 | 48.8 | 28.0 |
| pay_into_mult_middle | 40.5 | 27.8 | 34.1 | 49.5 | 30.2 |
| pay_into_mult_late | 38.0 | 29.2 | 33.6 | 48.0 | 28.5 |
| pay_into_div_a10 | 28.0 | 31.8 | 29.9 | 33.0 | 23.8 |
| pay_into_div_a25 | 33.2 | 33.5 | 33.4 | 34.8 | 23.8 |
| pay_into_div_a75 | 42.5 | 35.0 | 38.8 | 50.0 | 28.5 |
| pay_into_div_early | 24.8 | 30.8 | 27.8 | 35.2 | 25.8 |
| pay_into_div_middle | 31.8 | 32.2 | 32.0 | 34.8 | 25.2 |
| pay_into_div_late | 31.2 | 35.0 | 33.1 | 34.2 | 23.8 |

## Gate: heredity (does the child keep each parent's gain over the base?)

A parent's gain counts only if its paired 95% interval against the base excludes zero; a gain too small to measure cannot be inherited measurably and is skipped.

- merge_all: mult: keeps 42% of mult_all's +10.8; child vs parent -6.2 [-10.8, -1.5]; div: parent div_all gained +3.2 (not measurable); child vs parent +1.8 [-1.8, +5.2]
- merge_mlp: mult: keeps 155% of mult_mlp's +9.5; child vs parent +5.2 [+0.5, +9.8]; div: parent div_mlp gained +0.5 (not measurable); child vs parent +4.0 [+0.5, +7.8]
- pay_into_mult: mult: keeps 87% of mult_mlp's +9.5; child vs parent -1.2 [-5.0, +2.5]; div: parent div_mlp gained +0.5 (not measurable); child vs parent -0.5 [-4.0, +3.0]
- pay_into_div: mult: keeps 50% of mult_mlp's +9.5; child vs parent -4.8 [-9.5, -0.2]; div: parent div_mlp gained +0.5 (not measurable); child vs parent +5.2 [+2.2, +8.5]
- **PASS**: 3 of 4 constructions keep at least half of every measurable gain

## Information: each child against the multitask baseline on the parents' two fields

- merge_all vs mt_all: +0.0 points [-2.9, +3.0]
- merge_mlp vs mt_mlp: +5.0 points [+2.1, +8.0]
- pay_into_mult vs mt_mlp: -0.5 points [-3.4, +2.5]
- pay_into_div vs mt_mlp: +0.6 points [-2.2, +3.5]

## Who can check the third field (modular powers, 4 samples x 296 problems per child)

| child | base rate | filter | kept | precision | recall |
|---|---|---|---|---|---|
| merge_all | 0.44 | vote | 534 | 0.84 | 0.85 |
| merge_all | 0.44 | key | 524 | 1.00 | 1.00 |
| merge_all | 0.44 | family | 185 | 0.97 | 0.34 |
| merge_all | 0.44 | one parent | 499 | 0.88 | 0.84 |
| merge_mlp | 0.47 | vote | 572 | 0.86 | 0.88 |
| merge_mlp | 0.47 | key | 559 | 1.00 | 1.00 |
| merge_mlp | 0.47 | family | 225 | 0.96 | 0.39 |
| merge_mlp | 0.47 | one parent | 518 | 0.89 | 0.82 |
| pay_into_mult | 0.48 | vote | 561 | 0.89 | 0.87 |
| pay_into_mult | 0.48 | key | 569 | 1.00 | 1.00 |
| pay_into_mult | 0.48 | family | 222 | 0.96 | 0.37 |
| pay_into_mult | 0.48 | one parent | 507 | 0.90 | 0.80 |
| pay_into_div | 0.40 | vote | 467 | 0.87 | 0.85 |
| pay_into_div | 0.40 | key | 477 | 1.00 | 1.00 |
| pay_into_div | 0.40 | family | 189 | 0.98 | 0.39 |
| pay_into_div | 0.40 | one parent | 444 | 0.89 | 0.83 |
| mt_mlp | 0.47 | vote | 555 | 0.89 | 0.89 |
| mt_mlp | 0.47 | key | 553 | 1.00 | 1.00 |

## Childhood: before and after learning the third field

- merge_mlp, family (131 examples): modpow +2.5 [-1.8, +6.8]; parents' fields -2.2 [-5.1, +0.5]; held out +1.8 [-1.2, +4.8]
- merge_all, family (110 examples): modpow +2.0 [-2.0, +6.2]; parents' fields +1.1 [-1.9, +4.1]; held out +2.8 [-0.8, +6.2]
- pay_into_mult, family (129 examples): modpow +0.0 [-4.0, +4.0]; parents' fields +1.4 [-1.8, +4.5]; held out +0.2 [-3.2, +4.0]
- pay_into_div, family (124 examples): modpow +8.0 [+4.2, +11.8]; parents' fields -1.6 [-4.5, +1.1]; held out +1.0 [-3.0, +4.8]
- merge_mlp, vote (362 examples): modpow +0.0 [-4.0, +4.0]; parents' fields -4.1 [-7.0, -1.4]; held out +0.2 [-3.0, +3.5]
- merge_mlp, key (364 examples): modpow +2.8 [-1.0, +6.5]; parents' fields -3.0 [-5.6, -0.4]; held out +1.8 [-1.5, +5.0]
- mt_mlp, vote (374 examples): modpow +3.8 [-0.2, +7.5]; parents' fields -3.4 [-5.9, -1.0]; held out +2.5 [-1.5, +6.5]

## Sweep of the fertilisation edit (MLP-only parents; recipient keeps its whole genome)

Shares of a parent's gain are n/a where that gain over the base is not measurable (division).

| recipient | donor strength | donor layers | recipient's field | donor's field | held out | keeps recipient gain | gets donor gain |
|---|---|---|---|---|---|---|---|
| div | 0.1 | all | 31.8 | 28.0 | 23.8 | n/a | -45% |
| div | 0.25 | all | 33.5 | 33.2 | 23.8 | n/a | 11% |
| div | 0.5 | all | 35.5 | 37.0 | 28.5 | n/a | 50% |
| div | 0.75 | all | 35.0 | 42.5 | 28.5 | n/a | 108% |
| div | 1.0 | all | 34.2 | 47.0 | 28.2 | n/a | 155% |
| div | 0.5 | early | 30.8 | 24.8 | 25.8 | n/a | -79% |
| div | 0.5 | late | 35.0 | 31.2 | 23.8 | n/a | -11% |
| div | 0.5 | middle | 32.2 | 31.8 | 25.2 | n/a | -5% |
| mult | 0.1 | all | 39.5 | 28.5 | 28.8 | 76% | n/a |
| mult | 0.25 | all | 39.5 | 28.0 | 27.5 | 76% | n/a |
| mult | 0.5 | all | 40.5 | 29.8 | 26.8 | 87% | n/a |
| mult | 0.75 | all | 45.2 | 33.2 | 27.5 | 137% | n/a |
| mult | 1.0 | all | 47.0 | 34.2 | 28.2 | 155% | n/a |
| mult | 0.5 | early | 42.0 | 28.8 | 28.0 | 103% | n/a |
| mult | 0.5 | late | 38.0 | 29.2 | 28.5 | 61% | n/a |
| mult | 0.5 | middle | 40.5 | 27.8 | 30.2 | 87% | n/a |

## Microchimerism: does carrying part of the child make the mother a better verifier?

Single verifier agreeing with the child's sampled answer (modular powers).

| child | verifier | kept | precision | recall |
|---|---|---|---|---|
| pay_into_mult | mother | 465 | 0.92 | 0.75 |
| pay_into_mult | chimeric mother (10% of father) | 476 | 0.90 | 0.75 |
| pay_into_mult | chimeric mother (25% of father) | 465 | 0.91 | 0.75 |
| pay_into_mult | father | 264 | 0.92 | 0.43 |
| pay_into_div | mother | 218 | 0.94 | 0.43 |
| pay_into_div | chimeric mother (10% of father) | 245 | 0.94 | 0.48 |
| pay_into_div | chimeric mother (25% of father) | 305 | 0.93 | 0.60 |
| pay_into_div | father | 415 | 0.91 | 0.79 |
