# Merge test: same-seed against different-seed parents, and handling sign conflicts

Written 20 November 2026, before any model in this test has been built or examined. Exploratory (not preregistered
publicly), but the predictions and the analysis below are fixed before the data exist. Code: `seedmerge.py`,
`ties_build.py`.

## Why

turky asked how a merge can work when parents come from different seeds, and whether the positive or negative
direction ("valence") of each weight needs handling per pair. `seedcos.py` (`runs/p1/seed_cosine.txt`) showed:

- two specialists of the same field from different training seeds have weight changes with cosine 0.003 and agree in
  sign on 50% of weights, a coin flip; the random starting matrix of the low-rank update decides where a skill is
  written;
- specialists of different fields from the same seed overlap (cosine 0.04 to 0.15; 57 to 73% sign agreement on the
  weights both changed most). Every merge so far paired same-seed parents.

## Design

Parents: the six MLP-only specialists of the replication, multiplication and long division from three seeds
(Phase 1, s2, s3). Phase 1's pair and each replication pair share the identical random starting matrix A (same
`--seed` to `rsi.py train`; Phase 1 used the default 0).

Children: every multiplication parent with every division parent, a 3 x 3 grid. The 3 diagonal cells are same-seed
pairs, the 6 off-diagonal cells different-seed pairs. Each multiplication parent appears once on the diagonal and
twice off it, and so does each division parent, so the diagonal-minus-off-diagonal contrast is free of how strong
each parent happens to be.

Three ways to combine the two weight changes tau_m and tau_d, each applied to every MLP weight matrix separately:

1. **plain**: tau_m + tau_d (what reproduction does now; the merge child of Phase 1 and the replication).
2. **sign-elect**: where tau_m and tau_d agree in sign, tau_m + tau_d, as plain. Where they disagree, keep only the
   one with the larger magnitude, whole, instead of letting the two partly cancel. This changes only the conflicting
   weights, so it isolates the valence question.
3. **TIES** (Yadav et al., 2023, its defaults): keep each change's 20% largest entries, elect a sign per weight from
   the sum, average the kept entries that agree with it (disjoint mean), lambda 1.

All 27 children are built the same way on tranny (`ties_build.py`, fp32 factors to full weights, rounded once to
fp16), including the plain ones, so the three methods differ in nothing else. All are examined on tranny's RTX 3070
with greedy decoding on `data/p1/exam.jsonl` (400 multiplication, 400 division, 400 modular powers, 200 code tracing,
200 clock arithmetic). Parents and base reuse their existing 3070 exams on the same file. The rebuilt plain same-seed
children are also checked against the old merge exams (agreement on final answers), as a check of the build.

## Outcomes

- **Merge cost** (primary): for a child of mult parent i and div parent j, the mean over both fields of (child minus
  that field's parent), in points: 0.5 x [(child - mult_i) on mult + (child - div_j) on div]. Zero means the child
  keeps both parents' skill.
- Held out: code + clock, and modular powers, child minus base.

## Analysis

- Seed contrast, per method: mean merge cost of the 3 same-seed cells minus mean of the 6 different-seed cells.
- Method contrasts: sign-elect minus plain, and TIES minus plain, over all 9 cells and separately for same- and
  different-seed cells.
- 95% intervals from a paired bootstrap over exam problems (4,000 resamples, seed 0), recomputing every cell on the
  same resample. These intervals cover exam sampling only. Given the parents, a merge involves no randomness, so the
  remaining uncertainty is which seeds the six parents happened to come from; with three seeds a result is a lead for
  more seeds, not a law.

## Predictions (written before any result)

1. Sign-elect differs from plain by less than 2 points of merge cost in different-seed pairs: their sign conflicts
   are coincidences between two unrelated low-rank maps, so resolving them weight by weight should neither help nor
   hurt much. In same-seed pairs, I do not predict a direction.
2. TIES is worse than plain by more than 3 points of merge cost on average: trimming 80% of the entries of a
   low-rank update breaks the structure that makes it work, and TIES was designed for full fine-tunes.
3. Seed contrast under plain: no direction predicted. Different-seed changes do not overlap, so they cannot cancel
   each other; same-seed changes share input directions, which can help (division uses multiplication) or hurt.
