# Phase 0 results (19 November 2026)

Base model Qwen2.5-1.5B-Instruct, greedy, 40 problems per field and level. Code: `fields.py` (generators and exact checkers, self-tested by `test_fields.py`), `run.py`, `bdist.py`, `verify0.py`; runs in `phase0.sh` and `phase0b.sh`.

## Fields

| field | L1 | L2 | L3 | L4 | mean | role |
|---|---|---|---|---|---|---|
| modular power | 90 | 70 | 17 | 5 | 46 | trainable |
| base conversion | 85 | 47 | 10 | 0 | 36 | trainable |
| sorting | 75 | 42 | 15 | 3 | 34 | trainable |
| multiplication | 85 | 23 | 15 | 7 | 33 | trainable |
| calendar dates | 82 | 17 | 7 | 0 | 27 | trainable |
| long division | 75 | 28 | 0 | 0 | 26 | trainable |
| letter counting | 33 | 17 | 28 | 5 | 21 | trainable |
| code tracing | 78 | 12 | 5 | 0 | 24 | held out |
| clock arithmetic | 35 | 23 | 10 | 3 | 17 | held out |

All nine are inside the 15 to 80% band. Three were changed after the first measurement:
- **Weekday arithmetic** (90% at every level: the model just reduces the day count mod 7) became **calendar dates** ("what is the date 141 days after September 3, 2029"), which needs month lengths.
- **Countdown** (0% with 3% of answers readable: the model searches until it runs out of tokens) was replaced by **clock arithmetic** as the second held-out field.
- **Letter counting** was shortened to 1 to 5 words. Its failures are real: the model counts the e's in "referee" as one, because it sees tokens rather than letters.

## Verifiers in a new field (long division, 1,200 base-model samples at temperature 0.8, 4 per problem, levels 2 to 4)

The base rate is 8.7% correct. The peer is a division specialist: one round of answer-key self-training on separate problems, MLP-only, which lifted greedy division accuracy on these problems from 9% to 20% (level 2: 21 to 39%).

| filter | kept | precision | recall |
|---|---|---|---|
| random | 1200 | 0.087 | 1.00 |
| **peer agrees** (the specialist gets the same answer) | 51 | **0.804** | 0.39 |
| majority of the model's own 4 samples | 93 | 0.656 | 0.59 |
| in-field probe (fitted on other division problems) | 19 | 0.579 | 0.11 |
| multiplication probe applied to division, unchanged | 39 | 0.179 | 0.07 |
| answer key | 104 | 1.000 | 1.00 |

Stop rule (a non-oracle verifier at least 15 points above random): **passed**, by three of them.

## What changes in the design

1. **Parental verification works in a field nobody had verified before.** A parent that has learned the field and agrees with the child's answer is right 80% of the time here, against 9% by chance. This is the core of the proposal, and it holds.
2. **A probe does not transfer between fields.** The multiplication probe is near chance on division. So a relative who has *not* learned the child's field cannot judge it with a borrowed probe. In the design, family members without the field either learn it, fit a probe in that field from a small labelled set, or stay off that child's panel.
3. **Majority vote is a usable, cheap filter.** Unlike the verbal self-check (which made the model worse on 19 November), agreement among the model's own samples is 66% precise with high recall. It belongs in the design as the fallback when no parent knows the field, and as a control arm, since it needs no family at all.
4. **Generation speed was the real cost, now fixed.** Division answers are long: 1,800 samples took 15 minutes on all three GPUs with Hugging Face generation. vLLM 0.34 is now installed on tranny's WD Blue SSD (below), and it reproduces the Hugging Face accuracy while running far faster.

## vLLM on tranny (installed and checked 19 November)

- Everything lives in `/mnt/wdblue/claude-breeding/` on tranny (WD Blue SATA SSD, 343 GB free afterwards): `vllm-env/` (vLLM 0.34.0, torch 2.14 cu130, 14 GB), caches, temp files and `work/`. `work/env.sh` points every cache (HF, vLLM, pip, temp) at the SSD; the root disk stayed at 13 GB free.
- `vgen.py` is the worker; `export_lora.py` turns any genome (stack of deltas with layer, module or rank filters) into one exact LoRA adapter by concatenating factors.
- **Same answers as Hugging Face.** 1,600 base-model problems over all nine fields: 27% with Hugging Face, 28% with vLLM; every field within 4 points. The division specialist as an exported LoRA: 19% against 20% with Hugging Face, while the same problems without the adapter score 7%, so the adapter is really applied.
- **Speed.** One 3070 generated 1,600 answers in 127 s (4,250 tokens/s); Hugging Face needed 321 s for 1,440 on all three GPUs. The 2070 (fp16) does 2,600 tokens/s without an adapter and about 1,200 with one.
- vLLM does not run on Windows, so the 5070 stays the training card.

## Merged weights instead of an adapter (20 November)

An unmerged adapter halved the 2070's speed, so `merge_lora.py` bakes a genome into a full copy of the weights (W + B A, summed in fp32, rounded once; about 2.9 GB and a few seconds). Same 300 division problems, greedy:

| GPU | model | tokens/s | accuracy | same answer as the 2070 adapter run |
|---|---|---|---|---|
| 2070 | adapter, fp16 | 1,184 | 19.0% | 100% |
| 2070 | **merged, fp16** | **2,671** | 19.0% | 93% |
| 3070 | adapter, bf16 | 3,977 | 22.7% | 64% |
| 3070 | merged, bf16 | 4,929 | 21.7% | 58% |
| 3070 | **merged, fp16** | **4,908** | 18.7% | 94% |
| 2070 | base model (no genome) | 2,592 | 7.3% | 9% |

- Merging restores the 2070 to plain-model speed (2.3 times faster) and makes the 3070 about 23% faster.
- **Use fp16 on both cards.** Rounding the merged weights to bf16 throws away 22% of the delta (fp16: 3%), and bf16 runs drift apart from everything else (58 to 64% agreement), while fp16 merged runs on either card agree with the 2070 adapter run on 93 to 94% of answers, at the same speed. The accuracy spread (18.7 to 22.7%) is within noise for 300 problems (one standard error is about 2.3 points).
- Each merged model needs its own engine start (about 25 to 40 s), whereas one engine can swap adapters. Merge for jobs of a few hundred answers or more; keep tiny jobs as adapters.
