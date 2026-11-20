# Transfer map

One MLP-only specialist per trainable field (rows), each examined with the base model on all 20 fields (columns; 160 problems each, 40 per level, levels 1 to 4; every exam on the RTX 3070). code and clock are never trained. Exploratory: not preregistered. Each specialist was trained once, from one seed, and the intervals below cover only which exam problems were drawn, not training: the replication found specialists of the same field differing by up to 13 points between seeds (runs/p1/replication.md). Treat single cells as leads to retrain with more seeds, not as findings.

## 1. Change against the base model

Scored with one rule for every field: the "Answer:" line, or a final \boxed{} when there is none. Cells in points. * the paired 95% bootstrap interval excludes zero; ** the cell also survives a Benjamini-Hochberg false discovery rate of 5% over all 360 cells of its table (at 360 cells, about 18 single * marks are expected by chance alone). [ ] marks each specialist's own field.

| specialist | modpow | base | sort | mult | date | div | letters | knights | boolean | grid | path | story | sets | lineq | prob | game | seq | roman | clock | code |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *base accuracy* | *46* | *44* | *36* | *39* | *22* | *30* | *19* | *14* | *37* | *74* | *21* | *33* | *51* | *73* | *83* | *11* | *49* | *49* | *29* | *22* |
| modpow (356 ex., levels 1,2,3,4) | [+2] | -8* | -4 | -7* | +2 | +1 | -3 | -1 | +1 | -14** | +7 | +11* | +1 | +8* | +1 | +1 | -4 | +2 | +0 | +4 |
| base (359 ex., levels 1,2,3) | +5 | [+2] | -1 | -1 | +4 | -2 | -1 | -1 | -1 | -9* | +2 | +2 | +5 | +6* | -2 | +0 | +7 | -6 | +0 | +0 |
| sort (484 ex., levels 1,2,3) | +5 | +0 | [+9*] | -14** | +0 | -6 | -6 | +4 | +8 | -19** | +4 | +11** | +1 | +10** | -1 | +4 | +2 | +2 | +5 | +2 |
| mult (334 ex., levels 1,2,3,4) | +4 | +0 | -2 | [-3] | -1 | -2 | -2 | -2 | +5 | -6 | +5 | +9* | -1 | +4 | +1 | +1 | -3 | +8* | +1 | +4 |
| date (307 ex., levels 1,2,3) | +2 | +1 | -6 | -8* | [+2] | -4 | -2 | -3 | +1 | -16** | +4 | +6 | +1 | +7* | +1 | +4 | +3 | -2 | +12** | +4 |
| div (552 ex., levels 1,2) | +1 | -9* | -5 | -9* | +0 | [+3] | +4 | -2 | +2 | -16** | +7 | -1 | +1 | +4 | +4 | +0 | +0 | -3 | +2 | +2 |
| letters (300 ex., levels 1,2,3,4) | +2 | +1 | -1 | -2 | +5 | -7* | [+6] | +2 | +4 | -11** | +1 | +6 | +4 | +6* | +3 | +3 | +3 | -1 | +4 | +1 |
| knights (238 ex., levels 1,2,3,4) | +1 | -4 | +0 | -1 | +4 | +3 | +3 | [+4] | -1 | -2 | +11* | +6 | +9* | +7* | -3 | -1 | +6 | -4 | +8 | +6 |
| boolean (558 ex., levels 1,2,3) | -2 | -6* | -2 | -7* | -2 | -2 | +4 | +3 | [+12**] | -18** | -1 | +2 | +4 | +7* | -2 | -1 | +0 | -2 | +0 | +2 |
| grid (541 ex., levels 2,3,4) | -8 | +4 | -2 | -1 | +1 | -6 | -2 | -1 | +1 | [+6] | +2 | +9* | +8* | +5* | +3 | +6 | +9* | -5 | +4 | +7* |
| path (342 ex., levels 1,2,3,4) | +2 | -6 | +0 | +13* | +2 | +1 | +2 | -2 | -8 | -6* | [+16**] | +2 | +4 | +6* | +4 | -1 | +7* | -4 | +9* | +5 |
| story (530 ex., levels 1,2,3,4) | +2 | +1 | +1 | -7* | +1 | -4 | -1 | +2 | -8* | -8* | +5 | [+13**] | +1 | +7* | +4 | +1 | -4 | -7 | +3 | +2 |
| sets (478 ex., levels 2,3,4) | +1 | +1 | -5 | +1 | +2 | -8* | -6 | +2 | -8* | -7* | +4 | +5 | [+12**] | +8* | +4 | -4 | -1 | -1 | +8 | +3 |
| lineq (241 ex., levels 4) | -2 | -5 | -18** | -11** | -5 | -1 | -2 | -1 | -1 | -26** | +1 | +9* | -1 | [+9**] | -1 | +4 | -1 | -1 | +3 | +0 |
| prob (524 ex., levels 2,4) | -2 | +3 | +2 | -5 | +6* | -2 | -1 | +2 | -9* | -7* | +4 | +2 | +2 | +5* | [+4] | +4 | +2 | -9* | +3 | +1 |
| game (209 ex., levels 1,2,3) | -12** | -6 | -1 | -11* | +1 | -3 | -1 | -3 | -6 | -8* | +8* | +7 | +6 | +6* | -3 | [+12*] | +4 | -3 | +7 | +5 |
| seq (401 ex., levels 2,3,4) | +2 | -3 | -2 | -8* | +2 | -9* | -8 | +4 | +1 | -6 | +4 | +12** | -6 | +9** | +2 | +1 | [+11**] | +2 | +7 | +3 |
| roman (450 ex., levels 1,2,3,4) | -1 | -6 | -8* | +1 | +0 | -8* | -3 | +4 | -1 | -10** | +3 | +10* | +10* | +4 | +1 | -2 | +5 | [+11*] | +3 | +3 |

## 2. Change against the average specialist

Most columns move the same way whichever field was trained (clock, lineq and story gain, grid loses): an effect of self-training at all, not of the field. Each cell here is the specialist minus the mean of the other specialists on the same problems, so what remains is specific to the pair. p from a one-sample t-test on the per-problem differences. Cells in points. * the paired 95% bootstrap interval excludes zero; ** the cell also survives a Benjamini-Hochberg false discovery rate of 5% over all 360 cells of its table (at 360 cells, about 18 single * marks are expected by chance alone). [ ] marks each specialist's own field.

| specialist | modpow | base | sort | mult | date | div | letters | knights | boolean | grid | path | story | sets | lineq | prob | game | seq | roman | clock | code |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| modpow (356 ex., levels 1,2,3,4) | [+2] | -6* | -1 | -3 | +1 | +4 | -2 | -2 | +2 | -4 | +2 | +4 | -3 | +2 | +0 | -1 | -7* | +4 | -5 | +1 |
| base (359 ex., levels 1,2,3) | +5* | [+5*] | +2 | +3 | +3 | +1 | +0 | -2 | -1 | +1 | -3 | -4 | +2 | -0 | -3 | -2 | +5 | -5 | -5 | -3 |
| sort (484 ex., levels 1,2,3) | +5* | +2 | [+12**] | -10** | -1 | -3 | -6* | +3 | +8* | -9** | -1 | +4 | -3 | +4* | -2 | +2 | -1 | +4 | +1 | -1 |
| mult (334 ex., levels 1,2,3,4) | +5* | +2 | +1 | [+1] | -3 | +1 | -1 | -3 | +6* | +5* | +0 | +2 | -5 | -2 | +0 | -1 | -6* | +10** | -4 | +1 |
| date (307 ex., levels 1,2,3) | +2 | +3 | -3 | -4 | [+1] | -1 | -1 | -4 | +2 | -6* | -0 | -1 | -2 | +0 | +0 | +3 | +1 | -1 | +9* | +1 |
| div (552 ex., levels 1,2) | +1 | -7* | -3 | -5* | -1 | [+7*] | +5 | -3 | +2 | -6* | +2 | -8* | -2 | -2 | +3 | -2 | -3 | -2 | -3 | -1 |
| letters (300 ex., levels 1,2,3,4) | +2 | +3 | +1 | +2 | +4 | -4 | [+7*] | +2 | +4 | -1 | -4 | -1 | +1 | -0 | +2 | +1 | +1 | +1 | -0 | -3 |
| knights (238 ex., levels 1,2,3,4) | +1 | -2 | +3 | +3 | +3 | +7* | +4 | [+3] | -0 | +8** | +6 | -1 | +6* | +0 | -5* | -3 | +3 | -3 | +3 | +3 |
| boolean (558 ex., levels 1,2,3) | -3 | -4 | -0 | -3 | -3 | +1 | +5 | +3 | [+14**] | -8* | -6* | -5 | +0 | +0 | -4* | -3 | -3 | -1 | -5 | -1 |
| grid (541 ex., levels 2,3,4) | -9* | +7** | +1 | +4 | -0 | -3 | -2 | -2 | +2 | [+17**] | -2 | +2 | +5 | -2 | +2 | +4 | +7* | -4 | -1 | +4* |
| path (342 ex., levels 1,2,3,4) | +3 | -4 | +3 | +19** | +1 | +5 | +4 | -3 | -8* | +4 | [+11**] | -4 | +0 | -0 | +3* | -3 | +5* | -3 | +5 | +2 |
| story (530 ex., levels 1,2,3,4) | +2 | +3 | +3 | -3 | -1 | -1 | +0 | +2 | -7* | +2 | +0 | [+7*] | -3 | +0 | +3 | -1 | -7* | -6* | -1 | -1 |
| sets (478 ex., levels 2,3,4) | +1 | +4 | -3 | +6* | +1 | -5* | -5 | +1 | -8** | +4 | -1 | -2 | [+9**] | +1 | +3 | -7* | -3 | +0 | +4 | -0 |
| lineq (241 ex., levels 4) | -2 | -3 | -16** | -7* | -7* | +3 | -1 | -2 | -1 | -17** | -4 | +3 | -4 | [+3*] | -2 | +3 | -3 | +0 | -1 | -3* |
| prob (524 ex., levels 2,4) | -2 | +6* | +5* | -1 | +5* | +1 | -0 | +2 | -9** | +4 | -0 | -4 | -1 | -2 | [+3] | +2 | -1 | -9* | -1 | -2 |
| game (209 ex., levels 1,2,3) | -13** | -4 | +1 | -7* | -1 | -0 | +0 | -4 | -5* | +2 | +3 | +0 | +2 | -0 | -5* | [+11**] | +1 | -2 | +3 | +2 |
| seq (401 ex., levels 2,3,4) | +2 | -1 | +1 | -4 | +1 | -6* | -7* | +3 | +1 | +4 | -0 | +6* | -10** | +2 | +1 | -1 | [+9**] | +3 | +3 | -0 |
| roman (450 ex., levels 1,2,3,4) | -1 | -4 | -5* | +5* | -1 | -5* | -2 | +4 | -1 | +0 | -2 | +3 | +7* | -2 | -1 | -4 | +3 | [+13**] | -1 | +0 |

## 3. Cosine between specialists' weight changes

Correlation between a pair's cosine and its mean pair-specific transfer (section 2, both directions averaged), over 153 pairs: r = -0.13.

| | modpow | base | sort | mult | date | div | letters | knights | boolean | grid | path | story | sets | lineq | prob | game | seq | roman |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| modpow |  | 0.06 | 0.04 | 0.15 | 0.06 | 0.03 | 0.02 | 0.01 | 0.04 | 0.04 | 0.04 | 0.03 | 0.05 | 0.11 | 0.06 | 0.02 | 0.16 | 0.02 |
| base | 0.06 |  | 0.04 | 0.06 | 0.06 | 0.04 | 0.03 | 0.06 | 0.05 | 0.07 | 0.09 | 0.04 | 0.05 | 0.04 | 0.08 | 0.07 | 0.07 | 0.03 |
| sort | 0.04 | 0.04 |  | 0.07 | 0.07 | 0.07 | 0.08 | -0.00 | 0.06 | 0.08 | 0.01 | 0.04 | 0.07 | 0.04 | 0.04 | -0.01 | 0.11 | 0.09 |
| mult | 0.15 | 0.06 | 0.07 |  | 0.07 | 0.05 | 0.05 | 0.01 | 0.05 | 0.05 | 0.03 | 0.04 | 0.06 | 0.06 | 0.06 | 0.01 | 0.14 | 0.05 |
| date | 0.06 | 0.06 | 0.07 | 0.07 |  | 0.04 | 0.03 | 0.02 | 0.06 | 0.06 | 0.04 | 0.06 | 0.07 | 0.13 | 0.07 | 0.01 | 0.12 | 0.03 |
| div | 0.03 | 0.04 | 0.07 | 0.05 | 0.04 |  | 0.08 | 0.01 | 0.05 | 0.05 | 0.01 | 0.03 | 0.04 | 0.02 | 0.04 | 0.01 | 0.05 | 0.09 |
| letters | 0.02 | 0.03 | 0.08 | 0.05 | 0.03 | 0.08 |  | 0.00 | 0.05 | 0.07 | 0.01 | 0.04 | 0.03 | -0.01 | 0.03 | 0.01 | 0.07 | 0.10 |
| knights | 0.01 | 0.06 | -0.00 | 0.01 | 0.02 | 0.01 | 0.00 |  | 0.03 | 0.06 | 0.08 | 0.03 | 0.04 | 0.01 | 0.04 | 0.12 | -0.02 | -0.02 |
| boolean | 0.04 | 0.05 | 0.06 | 0.05 | 0.06 | 0.05 | 0.05 | 0.03 |  | 0.06 | 0.03 | 0.04 | 0.07 | 0.04 | 0.05 | 0.02 | 0.06 | 0.04 |
| grid | 0.04 | 0.07 | 0.08 | 0.05 | 0.06 | 0.05 | 0.07 | 0.06 | 0.06 |  | 0.05 | 0.05 | 0.07 | 0.04 | 0.07 | 0.05 | 0.07 | 0.05 |
| path | 0.04 | 0.09 | 0.01 | 0.03 | 0.04 | 0.01 | 0.01 | 0.08 | 0.03 | 0.05 |  | 0.03 | 0.04 | 0.02 | 0.06 | 0.11 | 0.01 | -0.01 |
| story | 0.03 | 0.04 | 0.04 | 0.04 | 0.06 | 0.03 | 0.04 | 0.03 | 0.04 | 0.05 | 0.03 |  | 0.05 | 0.02 | 0.04 | 0.03 | 0.05 | 0.02 |
| sets | 0.05 | 0.05 | 0.07 | 0.06 | 0.07 | 0.04 | 0.03 | 0.04 | 0.07 | 0.07 | 0.04 | 0.05 |  | 0.07 | 0.06 | 0.04 | 0.07 | 0.03 |
| lineq | 0.11 | 0.04 | 0.04 | 0.06 | 0.13 | 0.02 | -0.01 | 0.01 | 0.04 | 0.04 | 0.02 | 0.02 | 0.07 |  | 0.06 | -0.01 | 0.14 | -0.01 |
| prob | 0.06 | 0.08 | 0.04 | 0.06 | 0.07 | 0.04 | 0.03 | 0.04 | 0.05 | 0.07 | 0.06 | 0.04 | 0.06 | 0.06 |  | 0.05 | 0.08 | 0.02 |
| game | 0.02 | 0.07 | -0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.12 | 0.02 | 0.05 | 0.11 | 0.03 | 0.04 | -0.01 | 0.05 |  | -0.03 | -0.01 |
| seq | 0.16 | 0.07 | 0.11 | 0.14 | 0.12 | 0.05 | 0.07 | -0.02 | 0.06 | 0.07 | 0.01 | 0.05 | 0.07 | 0.14 | 0.08 | -0.03 |  | 0.05 |
| roman | 0.02 | 0.03 | 0.09 | 0.05 | 0.03 | 0.09 | 0.10 | -0.02 | 0.04 | 0.05 | -0.01 | 0.02 | 0.03 | -0.01 | 0.02 | -0.01 | 0.05 |  |

## 4. Answer format: the same exams under the old fields' own rule

The first nine fields' checker accepts only an "Answer:" line. Share of old-field answers (modpow, base, sort, mult, date, div, letters, clock, code) that end in \boxed{} with no "Answer:" line:

| model | boxed instead of Answer: |
|---|---|
| *base model* | 10% |
| modpow specialist | 0% |
| base specialist | 0% |
| sort specialist | 1% |
| mult specialist | 0% |
| date specialist | 2% |
| div specialist | 4% |
| letters specialist | 9% |
| knights specialist | 37% |
| boolean specialist | 4% |
| grid specialist | 19% |
| path specialist | 18% |
| story specialist | 8% |
| sets specialist | 11% |
| lineq specialist | 2% |
| prob specialist | 11% |
| game specialist | 17% |
| seq specialist | 0% |
| roman specialist | 8% |

Change against the base under the old fields' rule (first nine fields strict, new fields as in section 1). Cells in points. * the paired 95% bootstrap interval excludes zero; ** the cell also survives a Benjamini-Hochberg false discovery rate of 5% over all 360 cells of its table (at 360 cells, about 18 single * marks are expected by chance alone). [ ] marks each specialist's own field.

| specialist | modpow | base | sort | mult | date | div | letters | knights | boolean | grid | path | story | sets | lineq | prob | game | seq | roman | clock | code |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *base accuracy* | *39* | *35* | *36* | *38* | *22* | *28* | *19* | *14* | *37* | *74* | *21* | *33* | *51* | *73* | *83* | *11* | *49* | *49* | *17* | *22* |
| modpow (356 ex., levels 1,2,3,4) | [+9*] | +1 | -4 | -6* | +2 | +2 | -2 | -1 | +1 | -14** | +7 | +11** | +1 | +8** | +1 | +1 | -4 | +2 | +12** | +4 |
| base (359 ex., levels 1,2,3) | +9* | [+12**] | -1 | -1 | +4 | +0 | +0 | -1 | -1 | -9** | +2 | +2 | +5 | +6* | -2 | +0 | +7* | -6 | +12** | +0 |
| sort (484 ex., levels 1,2,3) | +9* | +9* | [+9*] | -13** | +0 | -4 | -6 | +4 | +8 | -19** | +4 | +11** | +1 | +10** | -1 | +4 | +2 | +2 | +17** | +2 |
| mult (334 ex., levels 1,2,3,4) | +11** | +9* | -2 | [-2] | -1 | -1 | -1 | -2 | +5 | -6 | +5 | +9* | -1 | +4 | +1 | +1 | -3 | +8* | +12** | +4 |
| date (307 ex., levels 1,2,3) | +5 | +9* | -6 | -9* | [+2] | -2 | -1 | -3 | +1 | -16** | +4 | +6 | +1 | +7* | +1 | +4 | +3 | -2 | +24** | +4 |
| div (552 ex., levels 1,2) | -1 | -4 | -5 | -12** | +0 | [+5] | +4 | -2 | +2 | -16** | +7 | -1 | +1 | +4 | +4 | +0 | +0 | -3 | +11* | +2 |
| letters (300 ex., levels 1,2,3,4) | -16** | +9* | -1 | -14** | +5* | -5 | [+6] | +2 | +4 | -11** | +1 | +6 | +4 | +6* | +3 | +3 | +3 | -1 | +8 | +1 |
| knights (238 ex., levels 1,2,3,4) | -39** | -21** | -1 | -35** | +3 | +0 | -13** | [+4] | -1 | -2 | +11* | +6 | +9* | +7** | -3 | -1 | +6 | -4 | +1 | +3 |
| boolean (558 ex., levels 1,2,3) | -9* | -1 | -2 | -8* | -2 | -1 | +4 | +3 | [+12**] | -18** | -1 | +2 | +4 | +7** | -2 | -1 | +0 | -2 | +12** | +2 |
| grid (541 ex., levels 2,3,4) | -16** | -2 | -2 | -20** | +1 | -4 | -2 | -1 | +1 | [+6] | +2 | +9* | +8* | +5 | +3 | +6 | +9* | -5 | -1 | +7* |
| path (342 ex., levels 1,2,3,4) | -13** | -10* | +0 | -20** | +2 | +3 | +2 | -2 | -8 | -6* | [+16**] | +2 | +4 | +6* | +4 | -1 | +7* | -4 | +0 | +5 |
| story (530 ex., levels 1,2,3,4) | +2 | -1 | +1 | -10** | +1 | -2 | +0 | +2 | -8* | -8* | +5 | [+13**] | +1 | +7* | +4 | +1 | -4 | -7 | +10* | +2 |
| sets (478 ex., levels 2,3,4) | -18** | +4 | -5 | -16** | +2 | -6 | -5 | +2 | -8* | -7* | +4 | +5 | [+12**] | +8* | +4 | -4 | -1 | -1 | +19** | +3 |
| lineq (241 ex., levels 4) | +0 | +4 | -18** | -12** | -5 | +1 | -1 | -1 | -1 | -26** | +1 | +9* | -1 | [+9**] | -1 | +4 | -1 | -1 | +15** | +0 |
| prob (524 ex., levels 2,4) | -19** | -3 | +2 | -9* | +6* | -2 | -1 | +2 | -9* | -7* | +4 | +2 | +2 | +5* | [+4] | +4 | +2 | -9* | +10* | +1 |
| game (209 ex., levels 1,2,3) | -24** | -11* | -1 | -27** | +1 | -3 | -1 | -3 | -6 | -8* | +8* | +7 | +6 | +6** | -3 | [+12**] | +4 | -3 | +9* | +5 |
| seq (401 ex., levels 2,3,4) | +9* | +6 | -2 | -8* | +2 | -7 | -7 | +4 | +1 | -6 | +4 | +12** | -6 | +9** | +2 | +1 | [+11**] | +2 | +19** | +3 |
| roman (450 ex., levels 1,2,3,4) | -16** | -8 | -8* | -3 | +0 | -6* | -2 | +4 | -1 | -10** | +3 | +10* | +10* | +4 | +1 | -2 | +5 | [+11**] | +15** | +3 |
