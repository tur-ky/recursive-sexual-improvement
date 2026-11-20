"""Self-test for the fields added on 20 November (fields.NEW_FIELDS): every new field accepts its gold answer in
several phrasings (and in \boxed{}), rejects wrong and missing answers, and has enough distinct problems per level.
The Phase 0 fields must score every stored Phase 1 answer exactly as they did when it was generated."""
import collections
import json
from pathlib import Path

import fields as F

bad = 0


def fail(*a):
    global bad
    bad += 1
    print(*a)


RUNS = Path(__file__).resolve().parent / "runs" / "p1"
files = sorted(RUNS.glob("exam_*.jsonl"))[:6] + sorted(RUNS.glob("trainsamp_*.jsonl")) + sorted(RUNS.glob("child_*.jsonl"))[:1]
n = 0
for path in files:
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        n += 1
        if F.check(r, r["text"]) != (r["ok"], r["readable"]) or F.answer_of(r, r["text"]) != r["ans"]:
            fail("OLD FIELD SCORES DIFFERENTLY", path.name, r["pid"])
print(f"re-scored {n} stored answers from {len(files)} files")

WRONG = {"knights": "Alice: knave, Bob: knave, Carol: knave, Dave: knave, Erin: knave, Frank: knave, Grace: knave, "
                    "Heidi: knave", "boolean": "maybe", "grid": "(99, 99)", "path": "999", "story": "Nobody",
         "sets": "999", "lineq": "x = 999, y = 999, z = 999", "prob": "999/1000", "game": "Z 99", "seq": "-99999",
         "roman": "MMMM"}
for f in F.NEW_FIELDS:
    for level in (1, 2, 3, 4):
        ps = F.make(f, level, 200, seed=0)
        distinct = len({p["q"] for p in F.make(f, level, 1000, seed=0)})
        if distinct < 300:  # an exam (100 per level) plus a training set (150) must fit without overlap
            fail(f"FEW DISTINCT {f} L{level}: {distinct}/1000")
        golds = collections.Counter(p["gold"] for p in ps)
        top = golds.most_common(1)[0][1] / len(ps)
        for p in ps[:60]:
            s = p["show"]
            goods = [s, f"**{s}**", f"{s}."]
            if f == "knights":
                goods.append(", ".join(x.replace(": ", " is a ") for x in s.split(", ")))
            if f == "boolean":
                goods.append(s.upper())
            if f == "lineq" and len(p["vars"]) > 1:
                goods.append(s.replace(" = ", "=").replace(", ", "; "))
            if f == "prob":
                n, d = (s.split("/") + ["1"])[:2]
                goods += [f"\\frac{{{n}}}{{{d}}}", f"{int(n) * 2}/{int(d) * 2}"]
            if f == "roman" and p["kind"] == "roman":
                goods.append(s.lower())
            for g in goods:
                ok, readable = F.check(p, f"some working...\n\nAnswer: {g}")
                if not ok:
                    fail("REJECTED GOOD", f, level, repr(g), "gold", p["gold"])
            if F.answer_of(p, f"Answer: {WRONG[f]}") != p["gold"] and F.check(p, f"Answer: {WRONG[f]}")[0]:
                fail("ACCEPTED WRONG", f, level, WRONG[f], p["gold"])
            if not F.check(p, f"working\n\\[\n\\boxed{{{s}}}\n\\]")[0]:
                fail("REJECTED BOXED", f, level, s)
            if F.check(p, "no answer line here")[0]:
                fail("ACCEPTED MISSING", f, level)
            assert F.answer_of(p, f"Answer: {s}") == p["gold"]
        print(f"{f} L{level}: {distinct} distinct of 1000, most common answer {top:.0%}")
print("\nexample questions:")
for f in F.NEW_FIELDS:
    p = F.make(f, 3, 1, seed=1)[0]
    print(f"--- {f} (L3, answer {p['show']})\n{p['q']}")
print("FAILURES", bad)
