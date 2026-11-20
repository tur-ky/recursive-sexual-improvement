# Preregistration: replication of the Phase 1 merge result

Written 20 November 2026 at about 02:15, after Phase 1 (seed 1) and before any replication run has produced
results. Nothing in `runs/p1/` with prefix `s2_`, `s3_` or `bs_`, and no file `runs/p1/exam_bs*.jsonl`, exists when
this is written. Code: `phase1.py`, `replicate.py` as of this time.

## What Phase 1 found (one seed, multiplication + long division, Qwen2.5-1.5B-Instruct)

The MLP-only merge child (both parents' MLP-only low-rank changes added at strength 1) beat the multitask model
trained on the same examples with the same compute by +5.0 points on the parents' two fields (paired 95% interval
+2.1 to +8.0), and beat each parent on that parent's field.

## Hypotheses for the three replication families

Families: multiplication + division with seeds 2 and 3 (new training problems, sampling seed and training seed), and
base conversion + sorting with seed 1. Exams: 400 problems per trained field (levels 1 to 4) plus 400 held-out
(code tracing, clock arithmetic), greedy, never used in training.

- **H1 (primary).** The MLP-only merge child scores higher than the MLP-only multitask model on the parents' two
  fields combined. Tested per family by a paired bootstrap (4,000 resamples over exam problems); confirmed in a
  family if the 95% interval excludes zero. Holm correction across the three families. Overall claim holds if it
  is confirmed in at least two of three families and the mean difference over all four families is positive.
- **H2.** The merge child is not worse than each parent on that parent's own field (interval lower bound above
  -3 points), in every family where the parent's own gain over the base is measurable.
- **H3.** The merge child scores higher than the base on the held-out fields.

## What would count against the result

- H1 fails in two or more families, or the mean difference is at or below zero: the Phase 1 result was luck or
  specific to multiplication + division.
- H1 holds for multiplication + division seeds but not base + sorting: the result depends on one field containing
  the other (long division uses multiplication), which is the explanation proposed in Phase 1, and the claim must
  be narrowed to related fields.

## Known limitations at registration

Exams for different models of a family may run on different GPUs (RTX 3070 and 2070, both fp16 merged weights; the
two GPUs agreed on 94% of answers in a check on 20 November). One base model. Arithmetic and string fields only.
