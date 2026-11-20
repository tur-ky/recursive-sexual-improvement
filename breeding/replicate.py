"""Replicate Phase 1's core result with new seeds and a second pair of fields.

Phase 1 (seed 1, multiplication + long division) found the MLP-only merge child beating both parents on their own
fields and beating a multitask model trained on the same data. Here the same recipe (answer-key specialists, MLP-only,
children by merge and by payload, multitask baseline) runs again with fresh training problems, sampling and training
seeds, and once on base conversion + sorting, two fields where neither is a sub-skill of the other.
Reuses phase1.py's machinery; outputs go to the same folders with a tag prefix. Every exam runs on the 3070 (see
ANALYSIS-NOTES.md: decided after registration and before any replication result); sampling uses both cards.

python replicate.py           run every stage whose outputs are missing, then report
python replicate.py report    report only
"""
import sys
import threading

import numpy as np

import phase1 as P

EXAM_GPU = 0  # tranny's 3070
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
                    for tag, (A, _), (B, _), seed, _ in RUNS for f in (A, B)])
    errors, t = [], None

    def bg(plan):
        def go():
            try:
                P.remote_round(plan, {lab: EXAM_GPU for lab, _, _ in plan})
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
        plan = [(m, m, [P.job(exam, exam_out(exam, m))]) for m in roles(tag, A, B).values()]
        if exam != "exam":  # a new exam needs its own base run (Phase 1's exam_base ran on the 3070)
            plan.append((f"base_{exam}", "base", [P.job(exam, exam_out(exam, "base"))]))
        t = bg(plan)
    t.join()
    if errors:
        raise errors[0]


def paired(rows_a, rows_b, fs):
    """Paired bootstrap as registered: 4,000 resamples over exam problems. Returns the mean difference b - a, its 95%
    interval (points) and a two-sided bootstrap p-value (twice the smaller share of resamples on one side of zero)."""
    ra = {x["pid"]: int(x["ok"]) for x in rows_a if x["field"] in fs}
    rb = {x["pid"]: int(x["ok"]) for x in rows_b if x["field"] in fs}
    dl = np.array([rb[i] - ra[i] for i in sorted(ra)])
    g = np.random.default_rng(0)
    bs = np.array([dl[g.integers(0, len(dl), len(dl))].mean() for _ in range(4000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return 100 * dl.mean(), 100 * lo, 100 * hi, min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))


def fmt(c):
    return f"{c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]"


def verdicts(fams):
    """The three hypotheses exactly as PREREGISTRATION.md states them. fams: [(name, tag, comparisons)]."""
    rep = [(n, c) for n, t, c in fams if t is not None]
    H = "merge vs multitask on both"
    out = ["## Preregistered hypotheses (PREREGISTRATION.md, SHA-256 9e6c58b6...)", ""]
    # H1: per family, the bootstrap p-value of merge - multitask; Holm over the three replication families
    order = sorted(rep, key=lambda x: x[1][H][3])
    holm, failed = {}, False
    for i, (n, c) in enumerate(order):
        failed = failed or c[H][3] > 0.05 / (len(order) - i)
        holm[n] = not failed and c[H][0] > 0
    mean4 = float(np.mean([c[H][0] for _, _, c in fams]))
    k = sum(holm.values())
    out.append(f"- **H1** (MLP merge above MLP multitask on both parents' fields; confirmed in a family if the "
               f"Holm-adjusted test rejects zero in the merge's favour): "
               + "; ".join(f"{n}: {fmt(c[H])}, p {f'= {c[H][3]:.3g}' if c[H][3] else '< 0.0005'}, "
                           f"{'confirmed' if holm[n] else 'not confirmed'}"
                           for n, c in rep)
               + f". Confirmed in {k} of {len(rep)}; mean difference over all {len(fams)} families {mean4:+.2f}. "
               f"**{'HOLDS' if k >= 2 and mean4 > 0 else 'DOES NOT HOLD'}** (needs at least 2 of 3 and a positive mean).")
    related = [n for n, t, _ in fams if t in ("s2", "s3")]
    if all(holm.get(n) for n in related) and not any(holm.get(n) for n, t, _ in fams if t == "bs"):
        out.append("- Counter-evidence clause: H1 holds for multiplication + division but not base conversion + "
                   "sorting, so the claim narrows to related fields.")
    # H2: merge not worse than each parent on that parent's field (lower bound above -3) where the parent's gain is real
    parts, ok = [], True
    for n, c in rep:
        for X in ("A", "B"):
            g = c[f"{X} parent vs base on {X}"]
            m = c[f"merge vs {X} parent on {X}"]
            if g[1] <= 0:
                parts.append(f"{n}, parent {X}: gain {fmt(g)} not measurable, skipped")
                continue
            ok &= m[1] > -3
            parts.append(f"{n}, parent {X}: merge vs parent {fmt(m)}, {'pass' if m[1] > -3 else 'FAIL'}")
    out.append("- **H2** (merge not worse than a parent on its own field, lower bound above -3, where the parent's "
               f"gain is measurable): {'; '.join(parts)}. **{'HOLDS' if ok else 'DOES NOT HOLD'}**.")
    h3 = {n: c["merge vs base on held out"] for n, c in rep}
    out.append("- **H3** (merge above the base on the held-out fields): "
               + "; ".join(f"{n}: {fmt(v)}, {'confirmed' if v[1] > 0 else 'not confirmed'}" for n, v in h3.items())
               + f". Confirmed in {sum(v[1] > 0 for v in h3.values())} of {len(h3)}.")
    return out + [""]


def report():
    out = ["# Replication of Phase 1's core result", "",
           "MLP-only specialists from answer-key self-training; merge = both parents' changes added; payload = recipient "
           "plus half the donor's change; multitask = one model trained on both parents' data. Paired differences, "
           "95% bootstrap intervals, same exam problems for every model of a family, every exam on the RTX 3070 "
           "(Phase 1's models first examined on the 2070 were re-examined on the 3070; see gpu_check.md).", ""]
    fams = []
    for tag, (A, _), (B, _), seed, exam in [(None, ("mult", None), ("div", None), 1, "exam")] + RUNS:
        r = roles(tag, A, B)
        ex = {k: P.exam_path(m, exam) for k, m in {**r, "base": "base"}.items()}
        if not all(p.exists() for p in ex.values()):
            continue
        rows = {k: P.read_jsonl(p) for k, p in ex.items()}

        def acc(k, fs):
            rs = [x for x in rows[k] if x["field"] in fs]
            return 100 * sum(x["ok"] for x in rs) / len(rs)

        name = f"{A} + {B}, seed {seed}" + (" (Phase 1)" if tag is None else "")
        both, held = {A, B}, P.F.HELD_OUT
        out += [f"## {name}", "", f"| model | {A} | {B} | both | held out |", "|---|---|---|---|---|"]
        for k in ("base", "A", "B", "mt", "merge", "payA", "payB"):
            out.append(f"| {r.get(k, 'base')} | {acc(k, {A}):.1f} | {acc(k, {B}):.1f} | {acc(k, both):.1f} | "
                       f"{acc(k, held):.1f} |")
        cmp = {"merge vs A parent on A": paired(rows["A"], rows["merge"], {A}),
               "merge vs B parent on B": paired(rows["B"], rows["merge"], {B}),
               "merge vs multitask on both": paired(rows["mt"], rows["merge"], both),
               "merge vs base on held out": paired(rows["base"], rows["merge"], held),
               "A parent vs base on A": paired(rows["base"], rows["A"], {A}),
               "B parent vs base on B": paired(rows["base"], rows["B"], {B})}
        out += [""] + [f"- {k}: {fmt(v)}" for k, v in cmp.items()] + [""]
        fams.append((name, tag, cmp))
    if fams:
        keys = list(fams[0][2])
        out += ["## Summary", "", "| family | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        out += [f"| {n} | " + " | ".join(fmt(c[k]) for k in keys) + " |" for n, _, c in fams] + [""]
    if len(fams) == len(RUNS) + 1:
        out += verdicts(fams)
    text = "\n".join(out) + "\n"
    P.write_lines(P.OUT / "replication.md", text)
    print(text)


if __name__ == "__main__":
    if sys.argv[1:] != ["report"]:
        run()
    report()
