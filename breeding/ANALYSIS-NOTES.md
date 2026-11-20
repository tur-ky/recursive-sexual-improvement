# Analysis notes added after preregistration, before any replication result

Written 20 November 2026, after `PREREGISTRATION.md` (SHA-256 `9e6c58b6...`) was made public and before the
replication (`replicate.py`) or the GPU check (`gpucheck.py`) had produced any output. None of this changes a
hypothesis, a threshold or a family; it fixes a measurement problem found in an audit and spells out how the registered
tests are computed.

1. **One GPU for every exam.** An audit of the Phase 1 logs showed that compared models were sometimes examined on
   different cards: the headline pair, merge_mlp (RTX 2070) against mt_mlp (RTX 3070), among them. The two cards gave
   the same final answer on 93 to 94% of problems in a check on 20 November. So:
   - every replication exam runs on the 3070 (`replicate.EXAM_GPU`); sampling for training data still uses both cards;
   - every Phase 1 model first examined on the 2070 is examined again on the 3070, and the core Phase 1 models first
     examined on the 3070 are examined again on the 2070 (`gpucheck.py`, results in `runs/p1/gpu_check.md`);
   - Phase 1's numbers in every report, and Phase 1's contribution to the four-family mean in H1, come from the
     all-3070 set, whatever the GPU check shows. The original mixed-card numbers stay on disk and in `gpu_check.md`.
2. **How H1's Holm correction is computed.** Per family, a two-sided bootstrap p-value from the registered bootstrap
   (4,000 resamples, seed 0): twice the smaller share of resamples on one side of zero. Holm's step-down over the three
   replication families at 0.05. A family confirms H1 if Holm rejects and the difference favours the merge. The
   unadjusted 95% interval is reported beside it.
3. **H2 and H3** are computed exactly as written: H2 skips a parent whose own gain over the base has a 95% interval
   touching zero; H3 is reported per family.

Code for all of this: `replicate.verdicts`, `phase1.exam_path`, `gpucheck.py`.
