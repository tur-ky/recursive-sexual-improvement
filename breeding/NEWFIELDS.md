# Eleven new fields (20 November 2026)

turky pointed out that multiplication and long division are close to one skill, so Phase 1's design lessons cannot be
generalised from them, and that a population needs real diversity to be worth breeding. These fields add kinds of
thinking the first nine did not cover. Code: `fields.py` (`NEW_FIELDS`: generators, exact checkers), self-test
`test_fields_new.py` (0 failures; every level has at least 300 distinct problems, enough for an exam of 100 plus a
training set of 150 without overlap; the first nine fields re-score over 12,000 stored Phase 1 and replication answers
exactly as they were scored when generated). Built in a separate module while the replication ran, then folded into
`fields.py` once it had finished.

| field | kind of thinking | example |
|---|---|---|
| knights | propositional logic | knights and knaves, 2 to 5 islanders, unique solution checked by brute force |
| boolean | nested logic | three AND/OR/NOT expressions given variable values |
| grid | spatial tracking | walk a grid with absolute moves, then turns relative to facing |
| path | graph search | shortest route on a weighted road map (from level 2 never the direct road) |
| story | state tracking | who holds which object after gives, swaps and hand-overs |
| sets | set algebra | (A ∪ B) − (C ∩ A) |
| lineq | algebra | one equation, then 2 and 3 unknowns |
| prob | probability | exact fractions: dice of any size, marbles without replacement |
| game | game search | tic-tac-toe win and block, a take-away game, three-heap Nim |
| seq | induction | next term: arithmetic up to interleaved and alternating rules |
| roman | a symbol system | Roman numerals both ways, then Roman addition |

## Base model accuracy (Qwen2.5-1.5B-Instruct, greedy, 40 problems per level, RTX 2070)

| field | L1 | L2 | L3 | L4 | mean | readable (L1-L4) |
|---|---|---|---|---|---|---|
| knights | 30 | 10 | 8 | 5 | 13 | 98 / 88 / 58 / 45 |
| boolean | 75 | 52 | 12 | 2 | 36 | 95 / 100 / 48 / 20 |
| grid | 98 | 90 | 85 | 15 | 72 | 100 / 100 / 98 / 100 |
| path | 62 | 12 | 10 | 5 | 22 | 98 / 68 / 72 / 52 |
| story | 72 | 45 | 30 | 8 | 39 | 100 / 92 / 72 / 32 |
| sets | 95 | 78 | 35 | 20 | 57 | 100 / 100 / 92 / 92 |
| lineq | 98 | 98 | 95 | 25 | 79 | 98 / 98 / 95 / 30 |
| prob | 100 | 50 | 92 | 85 | 82 | 100 / 68 / 92 / 92 |
| game | 8 | 22 | 5 | 2 | 9 | 72 / 98 / 22 / 40 |
| seq | 100 | 85 | 15 | 20 | 55 | 100 / 100 / 98 / 98 |
| roman | 72 | 60 | 45 | 10 | 47 | 98 / 85 / 98 / 32 |

Every field has at least one level in the 15 to 80% band that Phase 0 required. Raw answers:
`runs/newfields_base40*.jsonl`.

## What changed while building them

- **Caesar cipher dropped** (0 to 2% at every level). The model shifts each letter correctly but garbles the word when
  writing it back out, because it sees tokens, not letters. Writing letters apart, and asking for spaced answers,
  did not help. Like Countdown in Phase 0, a field the base cannot do at all gives self-training nothing to learn
  from. Replaced by **sets**.
- **Tic-tac-toe became "game".** Only about 140 positions have a unique forcing move and 36 a unique non-losing move,
  too few for an exam plus training. Levels 3 and 4 are now a take-away game and three-heap Nim, which have large
  problem pools and need a real game-theory idea. Boards also list each side's cells and the eight winning lines
  (the base model scored 12% without them).
- **Grid** levels re-cut: relative turns scored 0%, so level 3 is ten absolute moves and level 4 four relative moves.
- **`\boxed{}` accepted for the new fields only.** The base model answered algebra and probability in `\boxed{}`
  instead of an `Answer:` line 50 to 75% of the time, which would have measured answer format rather than the skill.
  Phase 1 already showed that a lot of a specialist's gain can be format habit, so accepting both formats in new
  fields makes their gains more about the skill. The first nine fields keep their exact rule so earlier results stay
  comparable. Whether to extend this to them is a decision for later; it would change their base numbers.

## Transfer map (exploratory, one seed per specialist)

`transfer.py` trained one MLP-only specialist per trainable field (18) and examined each, with the base model, on all
20 fields: `runs/p1/transfer.md`.

- **Answer format first.** Specialists trained on new fields keep their own `\boxed{}` answers, so they carry that
  habit into the old fields, whose checker needs an "Answer:" line (the knights specialist does it on 37% of old-field
  answers). Under the old rule this looked like huge interference (knights specialist -39 points on modpow, -35 on
  mult); scored with one rule for every field it is +1 and -1. The map is scored with one rule for all fields; the
  old-rule table is kept in the report to show the effect.
- **Own field:** 9 of 18 specialists gain on their own field beyond exam noise (+9 to +16 points: sort, boolean,
  path, story, sets, lineq, game, seq, roman). Only one of them (sort) is an old field; mult -3, modpow, base and
  date +2.
- **Most movement is self-training in general, not the field:** all 17 other specialists lose on grid (-2 to -26)
  and gain on lineq (+4 to +10), and 16 of 17 gain on story. The report's second table subtracts the average specialist to leave what is specific
  to each pair.
- **Pair-specific leads** (to retrain with more seeds before use): related, mult and roman (+10 and +5, both
  directions), path then mult (+19, one way only), date then clock (+9; both are time arithmetic, and clock is never trained); interfering, boolean and
  path (-6, -8), boolean and prob (-4, -9), lineq then grid and sort (-17, -16); unrelated, date with story or roman
  (about -1 both ways).
- **Weight similarity does not predict any of this:** all specialists' weight changes are near perpendicular (cosine
  0.16 at most), and cosine is not correlated with pair-specific transfer (r = -0.13 over 153 pairs).

The replication found specialists of one field differing by up to 13 points between seeds, and these intervals cover
only exam sampling, so every lead above needs retraining from 2 or more further seeds before a breeding test relies
on it.
