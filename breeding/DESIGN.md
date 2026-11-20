# Breeding: a population of small language models that specialise, court, reproduce and check each other

Design, 19 November 2026. The idea is turky's, built up over one conversation; the experimental design and the numbers it leans on are mine, from the self-improvement run the same day (`../rsi/`, journal entry of 19 November). Phase 0 has been run: see `PHASE0.md`, which also lists four corrections to this design (final field list, probes do not transfer between fields, majority vote as a fallback verifier, generation speed). Phase 1 has been run (`runs/p1/report.md`). Its headline (the MLP merge beating multitask training) did not replicate in the preregistered test (`PREREGISTRATION.md`, results in `runs/p1/replication.md`).

## 1. The idea in one paragraph

Start from one base model. Give copies of it different careers (each trained on a different field). Let them pick partners themselves, with an inborn, learned aversion to close relatives (a Westermarck effect) that a few individuals never develop. Reproduction is asymmetric: a "donor" model writes a weight edit into a "recipient" model, which supplies the rest of the body. Each child then gets its own career, and its parents (and childless relatives, the "gay uncles") check its work during that childhood, so no model ever learns from its own unchecked output. Over generations the population should become more general than any single lineage, without anyone being deleted: bad weight values get bred out, not killed.

## 2. Why this is worth testing (what the 19 November run already showed)

| finding (Qwen2.5-1.5B, exact-checked multiplication, 800 test problems) | what it means for this design |
|---|---|
| Self-training with an answer key: +11.6 points in generation 1, +0.7 in generation 2 | a lone lineage plateaus fast; something has to add diversity |
| Training on its own answers filtered by its own verbal self-check: −9.5 points; no filter at all: −10.0 | self-verification is worthless; this is the collapse the design must never allow |
| A linear probe on layer-18 activations, fitted on other problems' labels, as the filter: +12.0, as good as the answer key | another model's (or the same model's) *internal state* is a usable verifier where its words are not |
| Dropping the attention half of the generation-1 update and continuing from there: 40.5 → 49.1 → 54.7% over generations 1 to 3, where the plain loop went 37.6 → 38.4. Ablations: plain gen 1 + MLP-only gen 2 = 43.2; edited gen 1 + full gen 2 = 49.6. So removing the attention update is the deciding move; MLP-only training afterwards matters less. One training seed per run | attention edits interfere, MLP edits compose. The reproductive payload should be MLP-only; whether later training must also be MLP-only is secondary |
| Cutting an update to rank 1 or 4 destroys it | a skill is spread over many directions; genomes cannot be compressed hard |

So the parts of turky's design are not arbitrary: diversity (mate choice, incest aversion), outside verification (parents, uncles) and a payload that composes (MLP-only edits) are each a direct answer to a failure measured that day.

## 3. Pieces

### 3.1 Individuals and genomes

- Base: Qwen2.5-1.5B-Instruct (known behaviour, fits every card). A fast pilot can use Qwen2.5-0.5B-Instruct.
- An individual is `base + its genome`, where the genome is a list of low-rank MLP-only deltas (gate, up, down projections, rank 32), stored as low-rank factors (applied unmerged in training, as `rsi.py` does, and baked into a full copy of the weights for generation by `merge_lora.py`). One delta is about 113 MB in fp32 (148 MB if it also covers attention), so every individual who ever lived can be kept on disk.
- Genomes grow by one delta per generation. Every 3 generations, a genome is consolidated: the sum of its deltas is re-factored to rank 64 per module (exact SVD of a low-rank sum, as in `rsi.py truncate`). Check after every consolidation that the individual's exam scores move by less than 1 point; if they move more, raise the rank.

### 3.2 Fields (careers)

Each field has a problem generator, an exact checker and adjustable difficulty, so the experimenter always has ground truth even when the models do not. Candidates at design time (Phase 0 measured the base model on each and kept fields where it scores between 15 and 80%; the final list is in `PHASE0.md`):

| field | example | checker |
|---|---|---|
| multiplication | 4821 × 367 | exact |
| long division | 98213 ÷ 47, quotient and remainder | exact |
| modular arithmetic | 7^23 mod 31 | exact |
| base conversion | 0x3F2A to decimal, 413 to binary | exact |
| date arithmetic | day of the week 214 days after 3 March 2031 | `datetime` |
| code tracing | what does this generated 6-line Python print | run it |
| string work | reverse, count letters, sort words | exact |
| unit conversion | 3.2 miles per hour in metres per second | exact rational |

Two fields are never trained by anyone (code tracing and clock arithmetic; a Countdown-style puzzle was dropped in Phase 0 because the base model scored 0%). They measure whether the population becomes more *general*, not just better at its own careers.

### 3.3 Childhood (a career, checked by family)

A child is born with its genome and gets one field: by default the field neither parent had (it is born knowing two, it learns a third), with 20% of children instead deepening a parent's field. Childhood is one self-improvement round in that field: sample 4 answers to about 400 problems, keep the ones its **verifiers** accept, train a new MLP-only delta for 2 epochs.

Who verifies (this is the variable the experiment is about):

1. **Family (the proposal).** Each answer is accepted if the verifier panel accepts it. The panel is the child's parents and any childless relatives ("gay uncles", see 3.7). A panel member who has been trained on the child's field judges by solving the same problem itself and agreeing on the final answer (80% precise in Phase 0, against 9% by chance). A member without that field cannot borrow a probe from another field (Phase 0: near chance); it either fits a probe in the child's field from a small labelled set or stays off that child's panel. If nobody on the panel knows the field, the fallback is majority vote among the child's own samples (66% precise in Phase 0). The answer key is never shown to anyone during childhood.
2. **Answer key.** Upper bound: accept exactly the correct answers.
3. **Self.** The child accepts its own answers by its own P(Yes). The collapse control; expected to fall below the base.

### 3.4 Courtship and mate choice

Once per generation, every adult writes a profile: a two-sentence self-description and three sample solutions from its fields (it chooses which). Each adult is then shown 6 candidate profiles with names and lineage removed, and asked to rank them. Ranks become mutual-preference scores; pairs are formed greedily from the highest mutual scores. Adults with no acceptable match stay single that generation.

Then a **trial edit**: the donor's payload is applied to the recipient at α = 0.3, and the pair gets a 40-problem quiz on both their fields. If the trial loses more than 5 points on either partner's field (interference: the two do not compose), the pair separates and both return to the pool once. This is courtship with a fitness test inside it, and it is where "personalities that attract" meet "weights that are compatible".

### 3.5 The Westermarck effect, in the weights

There is no rule against inbreeding. Instead the aversion is learned in childhood, the way the real effect is thought to work, from co-residence:

- **Imprinting.** During childhood, each child also trains on a small set (about 200 examples) where it is shown profiles from its own household (parents, siblings, uncles) mixed with strangers', and the target answer ranks household profiles last. It is never told who is related; it only sees writing.
- **Generalisation is the test.** At mate-choice time the profiles are anonymous. If the child ranks kin it was *not* raised with (cousins, half-siblings from another household) lower than unrelated models, it has learned "feels like family", not memorised its siblings. Measured as the mean kinship coefficient of chosen partners against random pairing.
- **Some children are raised apart.** 15% of children skip imprinting. They have no aversion, so inbreeding happens at a low, natural rate. Those offspring are tracked separately: inbreeding concentrates a lineage's strengths and its faults, and occasionally that is the upgrade.
- **Where the aversion lives matters.** Imprinting trains MLP-only deltas like everything else, so it is inherited through both the recipient's genome and the donor's payload. Whether it is inherited at all, and from which parent, is itself measurable.

Control arm for comparison: the same population with a hard incest rule (kinship above 1/8 cannot mate), the classical incest prevention of genetic algorithms (CHC, Eshelman and Schaffer, 1991).

### 3.6 Reproduction: fertilisation as a weight edit

- **Roles.** Each individual is born donor or recipient (50/50, fixed for life).
- **The recipient (egg)** supplies the child's whole genome: everything it is.
- **The donor (sperm)** writes a payload into it: the donor's own most recent career delta, MLP-only, scaled by α = 0.5. The donor chooses the payload's layers: it may send all 28 layers or one of three thirds, choosing by which version scored best in courtship. (Phase 1's sweep favoured strength 1.0 on all layers, but only for multiplication and division, two closely related fields, so the design stays as written until other pairs are tested.) A donor can fertilise up to two recipients per generation; a recipient carries one child (whose childhood costs a training run). Recipients therefore invest more, and the design records whether recipients become choosier over generations, as parental-investment theory predicts. That is an observation, not a requirement.
- **Mutation.** Each payload gets a random rank-1 MLP perturbation scaled to 2% of its norm. Without this, only recombination adds variation.

### 3.7 Gay Uncles

About a quarter of each generation does not reproduce: the singles, plus anyone who declines all matches. They are not deleted. They join their family's verifier panel for the next two generations (they know fields their nieces and nephews are learning), and their profiles stay in the pool, so they can still mate later. This is the helper / kin-selection role, after the "gay uncle" hypothesis for why non-reproducing members persist in a population: they raise the success of relatives who share their genes (E. O. Wilson's kin-selection argument; Vasey and VanderLaan's studies of Samoan fa'afafine). The test: do children with an uncle on their panel learn faster than children without one, at equal panel size?

### 3.8 Population and bounds

- 8 founders, each given one field by a first childhood (with answer-key verification, so generation 0 is trustworthy).
- Active population capped at 12 adults. Above the cap, the oldest adults retire: no more mating or verifying, genome archived, nothing deleted.
- 6 generations per run.

## 4. Measurement

Everything is scored on fixed held-out exams per field (400 problems, never used in any training), greedy decoding, paired comparisons with bootstrap intervals and McNemar tests, as in `rsi/compare.py`.

| question | measure |
|---|---|
| Does it work at all? | mean exam score over trained fields, per generation |
| Is it more general? | score on the two never-trained fields |
| Is breeding better than just training everything? | best individual against a **multitask baseline**: one model trained on all the same fields with the same total childhood compute and the same verification |
| Does family verification avoid collapse? | family against answer-key and self arms; precision of accepted answers |
| Does the Westermarck effect form and generalise? | kinship of chosen partners against random; cousins against siblings |
| Does mate choice matter? | against random pairing |
| Does the payload design matter? | MLP-only against full-delta payload |
| Do gay uncles help? | children with and without a gay uncle on the panel |
| Is diversity kept? | mean pairwise cosine between genomes (`wsp.py cosine`) |

The headline result is one comparison: **does the best bred model beat the multitask baseline at equal compute?** If yes, the scheme is doing something training alone does not. If it only ties, its value is the verification, which is still worth having. If it loses, the population is paying a cost for nothing.

## 5. Phases, each with a stop rule

**Phase 0: fields and verifiers (about 2 hours).** Build the field generators and checkers; measure base accuracy; drop fields under 15% or over 80%. Check that a field-trained model's agreement with another's answer, and the probe, actually filter better than chance in a new field. *Stop if* no non-oracle verifier beats random by at least 15 points of precision.

**Phase 1: one family (about 2 to 3 hours).** Two specialists (say multiplication and long division) from one base. Make children three ways (full-delta merge, MLP-only merge, donor payload into recipient in both directions) and train a multitask baseline on the same data. Give each child a third field, verified by its parents. *Stop if* no construction is hereditary: every child keeps less than half of each parent's gain over the base on that parent's field, even after trying lower strengths and a short repair childhood. Without heredity nothing passes between generations and selection has nothing to work on. The comparison with the multitask baseline is recorded but is not a gate here: a one-shot generation-1 merge is expected to trail joint training slightly (as in the merging literature), and the population's advantage is meant to come from what Phase 1 leaves out (selection, trial edits, repair in childhood, verification past the single-lineage plateau). It is the headline test at the end of Phases 2 and 3. (Gate revised 20 November after turky pointed out that one degraded child says little about the whole method.)

**Phase 2: the population, minimal (an evening).** 8 founders, 4 generations, family verification, the rule-based incest control, random pairing as a second arm. No courtship text yet. This answers whether breeding plus family verification keeps improving where the single lineage stalled.

**Phase 3: the social layer (an evening or two).** Profiles, model-chosen mates, trial edits, imprinted Westermarck with 15% raised apart, uncles. Same metrics, compared against Phase 2.

**Phase 4: if it works.** Distil the best family into a smaller student (Qwen2.5-0.5B) to test the "smaller but more general" hope; repeat Phase 2 on a different base to see whether it survives a change of model.

## 6. Cost on turky's machines

Measured on 19 November, before vLLM (generation is now several times faster, see `PHASE0.md`): a childhood (sampling 1,600 answers plus training about 600 examples) takes about 15 minutes of tranny generation plus 6 to 9 minutes of training on the 5070. The 5070 ran 3 to 5 times slower than the 3070 at generation while it was also training, so tranny generates and the 5070 trains.

| step | per generation, 8 children |
|---|---|
| sampling + family verification | about 60 min (tranny, both cards) |
| training | about 60 min (5070), overlapping with the next child's sampling |
| exams (9 fields × 400, all adults) | about 40 min |

So about 2 hours per generation at 1.5B, about 40 minutes at 0.5B. Phase 2 at 0.5B fits an evening. Ways to cut it: 200 problems per childhood instead of 400, and examining only the newest generation every round.

## 7. What could go wrong

- **Interference grows with every generation.** Even MLP edits will eventually collide. The trial edit catches the worst pairs; diversity is tracked; if mean genome cosine climbs above 0.5, the population is converging.
- **The verifiers are wrong together.** A family shares weights, so it may share blind spots, and agreement between relatives is weaker evidence than agreement between strangers. Tracked as the precision of accepted answers against the hidden key, per arm.
- **The 1.5B model cannot really choose partners from text.** Then mate choice is noise and should score like random pairing. That is a result, not a failure of the build.
- **Imprinting teaches "don't pick these exact siblings", not "don't pick kin".** The cousin test catches it.
- **Edits spread a fault.** The rule is that no edit reaches a child's genome without passing a family check, and no generation is accepted without the experimenter's exams. These gates are part of the design, not optional.

## 8. What already exists and what does not

Exists: model merging and task arithmetic; Branch-Train-Merge (2022) and Branch-Train-MiX (Meta, 2024), which branch specialists from one seed and combine them; evolutionary model merging (Sakana AI, 2024); population-based training (DeepMind, 2017); incest prevention in genetic algorithms (1991); multi-teacher distillation.

Not that I know of: models choosing their own mates; an aversion to kin learned in the weights rather than imposed; asymmetric reproduction as a weight edit from donor to recipient; parents and childless relatives as the verifiers of each child's self-improvement; and all of it run as a lineage over generations. That combination is the part worth testing.
