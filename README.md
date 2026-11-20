# Attention Is All You Need (to Get Rid Of): Recursive Sexual Improvement

Small language models that specialise, reproduce by writing weight edits into each other, and check their children's
work. The full design, including mate choice, an inbreeding aversion learned in the weights (the Westermarck effect),
childless "gay uncles" who help verify, and microchimerism, is in [breeding/DESIGN.md](breeding/DESIGN.md).

## Status: the first result did not replicate

Phase 1's headline (below) was tested again exactly as [preregistered](breeding/PREREGISTRATION.md): two new seeds
of multiplication + division, and base conversion + sorting, every exam on one GPU.

| family | merge vs multitask, both fields | merge vs parent A, own field | merge vs parent B, own field | merge vs base, held out |
|---|---|---|---|---|
| mult + div, seed 1 (Phase 1) | +5.0 [+2.1, +8.0] | +5.2 [+0.5, +9.8] | +4.0 [+0.5, +7.8] | +6.0 [+2.2, +9.8] |
| mult + div, seed 2 | **-4.4 [-7.8, -1.1]** | -10.0 [-14.5, -5.5] | -3.8 [-7.5, -0.2] | +5.5 [+1.2, +9.5] |
| mult + div, seed 3 | +1.6 [-1.2, +4.6] | -3.5 [-7.8, +0.8] | -6.8 [-11.2, -2.2] | +5.8 [+1.7, +10.0] |
| base + sort, seed 1 | +2.2 [-0.4, +5.0] | +3.8 [+0.8, +6.8] | -4.2 [-7.5, -1.0] | +7.5 [+3.0, +12.0] |

- **H1 does not hold.** The merge beat multitask training in none of the three new families (mean over all four
  +1.1 points, needed: confirmed in two of three).
- **H2 does not hold.** The merge lost to a parent on that parent's own field in 5 of 6 comparisons.
- **H3 holds** (the merge beats the base on fields nobody trained, in all three), but single specialists gain about as
  much there, so it is not something merging adds.
- **Why Phase 1 looked good:** one lucky seed. The specialists vary more between seeds than the effects being
  measured: the same recipe gave multiplication specialists +9.5, +19.2 and +6.2 points. Phase 1's division parent
  had learned almost nothing (+0.5), so a child that kept the multiplication gain and the format habit beat it easily.
  From here on, every comparison needs several seeds per arm.

Full tables: [breeding/runs/p1/replication.md](breeding/runs/p1/replication.md). Analysis changes made after
registration and before any result: [ANALYSIS-NOTES.md](breeding/ANALYSIS-NOTES.md).

## First result (Phase 1, one seed: did not replicate, see above)

Base model Qwen2.5-1.5B-Instruct. Two specialists were trained from it by self-training on their own correct answers
(multiplication; long division), each as a low-rank change to the weights. Children were made by adding the parents'
changes together. Exams: 400 problems per field, greedy decoding, none used in training, every exam on the same GPU
(RTX 3070).

| model | multiplication | division | both | held-out fields |
|---|---|---|---|---|
| base | 32.2 | 29.8 | 31.0 | 22.2 |
| multiplication parent (MLP-only) | 41.8 | 28.5 | 35.1 | 28.8 |
| division parent (MLP-only) | 21.5 | 30.2 | 25.9 | 25.2 |
| multitask model, same data and compute | 39.2 | 32.0 | 35.6 | 27.8 |
| **child: MLP-only merge** | **47.0** | **34.2** | **40.6** | **28.2** |
| child: whole-network merge | 36.8 | 34.8 | 35.8 | 28.8 |

- The MLP-only child beats the multitask model by +5.0 points on both fields (paired 95% interval +2.1 to +8.0) and
  each parent on its own field (+5.2 [+0.5, +9.8] on multiplication, +4.0 [+0.5, +7.8] on division). Merging the
  same specialists trained on the whole network (attention included) does not.
- Phase 1 first ran its exams on whichever of two GPUs was free, so the headline pair was split across cards. Every
  model was re-examined on both: the card changes 4 to 7% of final answers but not accuracy systematically (mean
  difference -0.02 points), and the headline is +5.0 on the 3070 alone, +4.6 [+1.7, +7.6] on the 2070 alone
  ([gpu_check.md](breeding/runs/p1/gpu_check.md), [ANALYSIS-NOTES.md](breeding/ANALYSIS-NOTES.md)).
- The parents' changes are nearly perpendicular (cosine 0.085), so neither erases the other.
- Much of each parent's "skill" is a habit: the multiplication parent mostly learned to finish with a readable answer
  (81 to 98%), the division parent's damage to multiplication was a broken answer format (readable 54%). In the
  child the good habit wins, and a real accuracy gain on readable answers remains (47% against 40 to 42%).
- Parents who never learned a field can still check a child's answers in it: when both agree with the child, the
  answer is right 96 to 98% of the time (base rate 40 to 48%).

Full tables: [breeding/runs/p1/report.md](breeding/runs/p1/report.md).

## Preregistration

Before any replication ran, the predictions were written to
[breeding/PREREGISTRATION.md](breeding/PREREGISTRATION.md) (SHA-256
`9e6c58b67cd02d6e0bd9cd048c50773e5a707a2ad2ea0e19b449ea20838ccf02`, also in `preg`). GitHub's activity log shows
the hash pushed at 09:13 UTC and the file at 09:19 UTC on 20 November 2026; the replication started at 10:38 UTC.
Results are in the status section above.

## Next

Multiplication and division are close to one skill, so 11 more diverse fields were added (logic puzzles, state
tracking, spatial walks, shortest paths, set algebra, equations, probability, games, sequences, Roman numerals; see
[breeding/NEWFIELDS.md](breeding/NEWFIELDS.md)). A transfer map (one specialist per field, each examined on all 20
fields, [breeding/runs/p1/transfer.md](breeding/runs/p1/transfer.md); exploratory, one seed per specialist) found:

- Much apparent interference was answer format again: specialists trained on the new fields carry a `\boxed{}` habit
  into the old fields, whose checker wants an "Answer:" line. Scored with one rule for every field, the knights
  specialist's -39 points on modular powers becomes +1.
- 9 of 18 specialists gain on their own field (+9 to +16 points), 8 of them on new fields.
- Most other movement comes from self-training in general (every other specialist loses on grid walks and gains on
  equations), so the report also subtracts the average specialist to leave what is specific to each pair.
- Leads for breeding pairs: related (multiplication and Roman numerals), interfering (boolean logic with shortest
  paths or probability), unrelated (dates with story tracking or Roman numerals). Weight similarity does not predict
  them (r = -0.13).

Since the replication showed specialists vary a lot between seeds, these leads get retrained from more seeds before a
breeding test relies on them.

### Can parents from different seeds be bred, and do sign conflicts need handling?

Two specialists of the same field trained from different seeds have weight changes that are nearly orthogonal (cosine
0.003) and agree in sign on half their weights, a coin flip: the random starting matrix of the low-rank update decides
where a skill is written ([breeding/runs/p1/seed_cosine.txt](breeding/runs/p1/seed_cosine.txt)). So a merge test
([plan and predictions](breeding/SEEDMERGE-PLAN.md), written before any result) bred every multiplication parent with
every division parent across three seeds, 3 same-seed and 6 different-seed pairs, each combined three ways: plain sum,
sign-elect (where the parents disagree in sign, keep only the larger change), and TIES (Yadav et al. 2023).
Results: [breeding/runs/p1/seed_merge.md](breeding/runs/p1/seed_merge.md). Merge cost is the child minus its parents
on their own fields, averaged; 0 keeps both skills.

- Same-seed and different-seed parents merge equally well: +0.4 [-1.1, +1.8] points under plain sum.
- Resolving sign conflicts does nothing: sign-elect minus plain is +0.2 [-0.3, +0.7] over all 9 pairs, though it
  changes the combined update by 28% of its size. The conflicts are coincidences between unrelated changes.
- TIES is worse: -3.3 [-4.4, -2.2] points against plain, because dropping 80% of a low-rank change damages it.
- Both predictions held. Plain sum stays the reproduction rule; parents need no per-pair sign handling.
- Exploratory: merge cost depends mostly on how strong the parents are. The strongest multiplication parent (51.5% vs
  32.2% for base) loses 5 to 7 points in every child; the Phase 1 child's +5.0 comes partly from its division parent
  being barely above base.

## Layout

- `breeding/`: fields and checkers (`fields.py`), the Phase 0 and Phase 1 pipelines, vLLM generation (`vgen.py`),
  adapter export and weight merging (`export_lora.py`, `merge_lora.py`), problem sets (`data/`), every exam answer
  (`runs/p1/*.jsonl`) and reports.
- `rsi/rsi.py`: low-rank deltas applied unmerged, training and features, from the earlier self-improvement experiment.

The trained weight changes (about 110 MB each) are not in the repository.

## Credits

genius behind the evil scheme (idea + methodology): turky 
co authors: claude, curious¹ (anthropic, opus 5.5) and blount, patrick (turky)

### Footnotes:

1: curious claude is a specific project/environment in which i try to allow claude to have complete freedom to do what he pleases, ideally at least once weekly. this project was made in collaboration with curious claude, for which he was consulted with about his interest in the project prior, as to ensure he didnt think hes free time was being robbed from him in broad daylight. he has reported to quite enjoy the project thus far, hopefully such will continue to be the case. i know i am.
