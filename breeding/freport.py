"""Accuracy by field and level.  python freport.py runs/a.jsonl [runs/b.jsonl ...]"""
import json, sys, collections

for path in sys.argv[1:]:
    t = collections.defaultdict(lambda: [0, 0, 0])
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            c = t[(r["field"], r["level"])]
            c[0] += r["ok"]; c[1] += r["readable"]; c[2] += 1
    print(f"== {path}")
    print(f"{'field':10s} " + " ".join(f"{'L' + str(i):>6s}" for i in (1, 2, 3, 4)) + "    mean  readable")
    for f in sorted({k[0] for k in t}):
        cells = [t[(f, lv)] for lv in (1, 2, 3, 4)]
        acc = [c[0] / c[2] if c[2] else float("nan") for c in cells]
        n = sum(c[2] for c in cells)
        print(f"{f:10s} " + " ".join(f"{x:6.2f}" for x in acc)
              + f"  {sum(c[0] for c in cells) / n:6.2f}  {sum(c[1] for c in cells) / n:6.2f}")
