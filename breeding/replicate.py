"""Replicate Phase 1's core result with new seeds and a second pair of fields.

Phase 1 (seed 1, multiplication + long division) found the MLP-only merge child beating both parents on their own
fields and beating a multitask model trained on the same data. Here the same recipe (answer-key specialists, MLP-only,
children by merge and by payload, multitask baseline) runs again with fresh training problems, sampling and training
seeds, and once on base conversion + sorting, two fields where neither is a sub-skill of the other.
Reuses phase1.py's machinery; outputs go to the same folders with a tag prefix.

python replicate.py           run every stage whose outputs are missing, then report
python replicate.py report    report only
"""
import sys
import threading

import numpy as np

import phase1 as P

RUNS = [  # tag, (field A, training levels), (field B, training levels), seed, exam
    ("s2", ("mult", [1, 2, 3]), ("div", [1, 2]), 2, "exam"),
    ("s3", ("mult", [1, 2, 3]), ("div", [1, 2]), 3, "exam"),
    ("bs", ("base", [1, 2, 3]), ("sort", [1, 2, 3]), 1, "exam_bs"),
]


def roles(tag, A, B):
    """Model names for one family. tag None is Phase 1 itself (unprefixed names)."""
    pre = f"{tag}_" if tag else ""
    return {"A": f"{pre}{A}_mlp", "B": f"{pre}{B}_mlp", "mt": f"{pre}mt_mlp", "merge": f"{pre}merge_mlp",
            "payA": f"{pre}pay_into_{A}", "payB": f"{pre}pay_into_{B}"}


for _tag, (_A, _), (_B, _), _seed, _ex in RUNS:
    _r = roles(_tag, _A, _B)
    _d = lambda n, a=1.0, t=_tag: P.d(f"{t}_{n}", a)  # noqa: E731
    P.G.update({_r["A"]: [_d(f"{_A}_mlp")], _r["B"]: [_d(f"{_B}_mlp")], _r["mt"]: [_d("mt_mlp")],
                _r["merge"]: [_d(f"{_A}_mlp"), _d(f"{_B}_mlp")],
                _r["payA"]: [_d(f"{_A}_mlp"), _d(f"{_B}_mlp", 0.5)],
                _r["payB"]: [_d(f"{_B}_mlp"), _d(f"{_A}_mlp", 0.5)]})


def exam_out(exam, m):
    return f"{exam}_{m}"


def problems(tag, A, la, B, lb, seed, exam):
    if not (P.DATA / f"{exam}.jsonl").exists():
        P.write_jsonl(P.DATA / f"{exam}.jsonl", P.make([A, B], [1, 2, 3, 4], 100, 9101)
                      + P.make(["code", "clock"], [1, 2, 3, 4], 50, 9002))
    seen = {p["q"] for p in P.read_jsonl(P.DATA / f"{exam}.jsonl")}
    for i, (f, lv) in enumerate(((A, la), (B, lb))):
        path = P.DATA / f"{tag}_train_{f}.jsonl"
        if path.exists():
            continue
        keep, qs = [], set(seen)
        for p in P.make([f], lv, 450 // len(lv), 500 + 10 * seed + i):
            if p["q"] not in qs:
                qs.add(p["q"])
                keep.append(p)
        P.write_jsonl(path, keep)


def run():
    for tag, (A, la), (B, lb), seed, exam in RUNS:
        problems(tag, A, la, B, lb, seed, exam)
    P.remote_round([(f"{tag}_base_{f}", "base", [P.job(f"{tag}_train_{f}", f"{tag}_trainsamp_{f}", 4, 0.8, seed)])
                    for tag, (A, _), (B, _), seed, _ in RUNS for f in (A, B)]
                   + [("base_bs", "base", [P.job("exam_bs", exam_out("exam_bs", "base"))])])
    errors, t = [], None

    def bg(plan):
        def go():
            try:
                P.remote_round(plan)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        th = threading.Thread(target=go)
        th.start()
        return th

    for tag, (A, _), (B, _), seed, exam in RUNS:
        for f in (A, B):
            if not (P.DATA / f"{tag}_sft_{f}.jsonl").exists():
                P.sft(P.read_jsonl(P.OUT / f"{tag}_trainsamp_{f}.jsonl"), lambda r: r["ok"], P.DATA / f"{tag}_sft_{f}.jsonl")
        if not (P.DATA / f"{tag}_sft_mt.jsonl").exists():
            P.write_jsonl(P.DATA / f"{tag}_sft_mt.jsonl",
                          P.read_jsonl(P.DATA / f"{tag}_sft_{A}.jsonl") + P.read_jsonl(P.DATA / f"{tag}_sft_{B}.jsonl"))
        for f in (A, B, "mt"):
            P.train(f"{tag}_{f}_mlp", "base", P.DATA / f"{tag}_sft_{f}.jsonl", P.MLP, seed)
        if t:
            t.join()
        if errors:
            raise errors[0]
        t = bg([(m, m, [P.job(exam, exam_out(exam, m))]) for m in roles(tag, A, B).values()])
    t.join()
    if errors:
        raise errors[0]


def report():
    out = ["# Replication of Phase 1's core result", "",
           "MLP-only specialists from answer-key self-training; merge = both parents' changes added; payload = recipient "
           "plus half the donor's change; multitask = one model trained on both parents' data. Paired differences, "
           "95% bootstrap intervals, same exam problems for every model of a family.", ""]
    summary = []
    for tag, (A, _), (B, _), seed, exam in [(None, ("mult", None), ("div", None), 1, "exam")] + RUNS:
        r = roles(tag, A, B)
        ex = {k: P.OUT / f"{exam_out(exam, m)}.jsonl" for k, m in {**r, "base": "base"}.items()}
        if not all(p.exists() for p in ex.values()):
            continue
        rows = {k: P.read_jsonl(p) for k, p in ex.items()}

        def acc(k, fs):
            rs = [x for x in rows[k] if x["field"] in fs]
            return 100 * sum(x["ok"] for x in rs) / len(rs)

        def paired(a, b, fs):
            ra = {x["pid"]: int(x["ok"]) for x in rows[a] if x["field"] in fs}
            rb = {x["pid"]: int(x["ok"]) for x in rows[b] if x["field"] in fs}
            dl = np.array([rb[i] - ra[i] for i in sorted(ra)])
            g = np.random.default_rng(0)
            lo, hi = np.percentile([dl[g.integers(0, len(dl), len(dl))].mean() for _ in range(4000)], [2.5, 97.5])
            return f"{100 * dl.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"

        name = f"{A} + {B}, seed {seed}" + (" (Phase 1)" if tag is None else "")
        both, held = {A, B}, P.F.HELD_OUT
        out += [f"## {name}", "", f"| model | {A} | {B} | both | held out |", "|---|---|---|---|---|"]
        for k in ("base", "A", "B", "mt", "merge", "payA", "payB"):
            out.append(f"| {r.get(k, 'base')} | {acc(k, {A}):.1f} | {acc(k, {B}):.1f} | {acc(k, both):.1f} | "
                       f"{acc(k, held):.1f} |")
        cmp = {"merge vs A parent on A": paired("A", "merge", {A}), "merge vs B parent on B": paired("B", "merge", {B}),
               "merge vs multitask on both": paired("mt", "merge", both),
               "merge vs base on held out": paired("base", "merge", held),
               "A parent vs base on A": paired("base", "A", {A}), "B parent vs base on B": paired("base", "B", {B})}
        out += [""] + [f"- {k}: {v}" for k, v in cmp.items()] + [""]
        summary.append((name, cmp))
    if summary:
        keys = list(summary[0][1])
        out += ["## Summary", "", "| family | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        out += [f"| {n} | " + " | ".join(c[k] for k in keys) + " |" for n, c in summary]
    text = "\n".join(out) + "\n"
    P.write_lines(P.OUT / "replication.md", text)
    print(text)


if __name__ == "__main__":
    if sys.argv[1:] != ["report"]:
        run()
    report()
