# Attention Is All You Need (to Get Rid Of): Recursive Sexual Improvement

Small language models that specialise, reproduce by writing weight edits into each other, and check their children's
work. The full design, including mate choice, an inbreeding aversion learned in the weights (the Westermarck effect),
childless "uncles" who help verify, and microchimerism, is in [breeding/DESIGN.md](breeding/DESIGN.md).

## First result (Phase 1, one seed: preliminary)

Base model Qwen2.5-1.5B-Instruct. Two specialists were trained from it by self-training on their own correct answers
(multiplication; long division), each as a low-rank change to the weights. Children were made by adding the parents'
changes together. Exams: 400 problems per field, greedy decoding, none used in training.

| model | multiplication | division | both | held-out fields |
|---|---|---|---|---|
| base | 32.2 | 29.8 | 31.0 | 22.2 |
| multiplication parent (MLP-only) | 41.2 | 27.8 | 34.5 | 28.2 |
| division parent (MLP-only) | 21.5 | 30.2 | 25.9 | 25.2 |
| multitask model, same data and compute | 39.2 | 32.0 | 35.6 | 27.8 |
| **child: MLP-only merge** | **47.0** | **34.2** | **40.6** | **29.0** |
| child: whole-network merge | 36.8 | 34.8 | 35.8 | 28.8 |

- The MLP-only child beats the multitask model by +5.0 points on both fields (paired 95% interval +2.1 to +8.0) and
  each parent on its own field. Merging the same specialists trained on the whole network (attention included) does not.
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
`9e6c58b67cd02d6e0bd9cd048c50773e5a707a2ad2ea0e19b449ea20838ccf02`, also in `preg`). Two more seeds and a second
pair of fields (base conversion + sorting) are next.

## Layout

- `breeding/`: fields and checkers (`fields.py`), the Phase 0 and Phase 1 pipelines, vLLM generation (`vgen.py`),
  adapter export and weight merging (`export_lora.py`, `merge_lora.py`), problem sets (`data/`), every exam answer
  (`runs/p1/*.jsonl`) and reports.
- `rsi/rsi.py`: low-rank deltas applied unmerged, training and features, from the earlier self-improvement experiment.

The trained weight changes (about 110 MB each) are not in the repository.

## Credits

Idea and direction: turky. Experiments designed, built and run with Claude (Anthropic).
