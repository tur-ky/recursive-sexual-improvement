"""Transfer map: which fields help, ignore or hurt each other?

One MLP-only specialist per trainable field (answer-key self-training, the Phase 1 recipe: 4 samples per problem at
temperature 0.8, keep up to 2 correct per problem, rank-32 MLP delta, 2 epochs), then every specialist and the base
sit one exam covering all 20 fields (40 problems per level, levels 1 to 4). The table of changes against the base shows
which pairs of fields are related (gains spread), unrelated (nothing moves) or interfering (losses), so breeding tests
can pick pairs of each kind on purpose instead of only multiplication + division. Also the cosine between every pair
of specialists' weight changes, to compare relatedness in behaviour with relatedness in the weights.

Every exam on the RTX 3070 (see ANALYSIS-NOTES.md). Training levels: those where the base model scored 5 to 90% in
Phase 0 (old fields) or NEWFIELDS.md (new fields), so each specialist has something to learn from.

python transfer.py          run what is missing, then write runs/p1/transfer.md
python transfer.py report   report only
"""
import sys
import threading

import numpy as np

import fields as F
import phase1 as P

BASE_ACC = {  # base model, greedy, 40 problems per level: PHASE0.md and NEWFIELDS.md
    "modpow": (90, 70, 17, 5), "base": (85, 47, 10, 0), "sort": (75, 42, 15, 3), "mult": (85, 23, 15, 7),
    "date": (82, 17, 7, 0), "div": (75, 28, 0, 0), "letters": (33, 17, 28, 5),
    "knights": (30, 10, 8, 5), "boolean": (75, 52, 12, 2), "grid": (98, 90, 85, 15), "path": (62, 12, 10, 5),
    "story": (72, 45, 30, 8), "sets": (95, 78, 35, 20), "lineq": (98, 98, 95, 25), "prob": (100, 50, 92, 85),
    "game": (8, 22, 5, 2), "seq": (100, 85, 15, 20), "roman": (72, 60, 45, 10)}
TRAIN = list(BASE_ACC)
LEVELS = {f: [lv + 1 for lv, a in enumerate(acc) if 5 <= a <= 90] for f, acc in BASE_ACC.items()}
EXAM_FIELDS = TRAIN + sorted(F.HELD_OUT)
EXAM_GPU = 0
N_TRAIN = 480
for f in TRAIN:
    P.G[f"tm_{f}"] = [P.d(f"tm_{f}_mlp")]


def problems():
    if not (P.DATA / "exam_tm.jsonl").exists():
        P.write_jsonl(P.DATA / "exam_tm.jsonl", P.make(EXAM_FIELDS, [1, 2, 3, 4], 40, 9201))
    seen = {p["q"] for p in P.read_jsonl(P.DATA / "exam_tm.jsonl")}
    for i, f in enumerate(TRAIN):
        path = P.DATA / f"tm_train_{f}.jsonl"
        if path.exists():
            continue
        keep, qs = [], set(seen)
        for p in P.make([f], LEVELS[f], N_TRAIN // len(LEVELS[f]), 600 + i):
            if p["q"] not in qs:
                qs.add(p["q"])
                keep.append(p)
        P.write_jsonl(path, keep)


def exam_round(models):
    plan = [(f"tm_exam_{m}", m, [P.job("exam_tm", f"exam_tm_{m}")]) for m in models]
    P.remote_round(plan, {lab: EXAM_GPU for lab, _, _ in plan})


def run():
    problems()
    P.remote_round([(f"tm_base_{f}", "base", [P.job(f"tm_train_{f}", f"tm_trainsamp_{f}", 4, 0.8, 4)]) for f in TRAIN])
    errors, t = [], None

    def bg(models):
        def go():
            try:
                exam_round(models)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        th = threading.Thread(target=go)
        th.start()
        return th

    t = bg(["base"])
    batch = []
    for f in TRAIN:
        sft = P.DATA / f"tm_sft_{f}.jsonl"
        if not sft.exists():
            P.sft(P.read_jsonl(P.OUT / f"tm_trainsamp_{f}.jsonl"), lambda r: r["ok"], sft)
        P.train(f"tm_{f}_mlp", "base", sft, P.MLP)
        batch.append(f"tm_{f}")
        if len(batch) >= 3 and not t.is_alive():  # examine what is trained while the next ones train
            t.join()
            if errors:
                raise errors[0]
            t, batch = bg(batch), []
    t.join()
    if errors:
        raise errors[0]
    if batch:
        exam_round(batch)


def cosines():
    """Cosine between every pair of specialists' full weight changes, from the low-rank factors (no full matrices):
    <B1 A1, B2 A2> = trace((B1^T B2)(A2 A1^T))."""
    import torch
    from safetensors.torch import load_file
    fac = {}
    for f in TRAIN:
        path = P.HERE / "deltas" / "p1" / f"tm_{f}_mlp" / "delta.safetensors"
        if not path.exists():
            return None
        t = load_file(str(path))
        fac[f] = {k[:-2]: (t[k[:-2] + ".A"].double(), t[k[:-2] + ".B"].double()) for k in t if k.endswith(".A")}

    def dot(a, b):
        return sum(float(torch.trace((B1.T @ B2) @ (A2 @ A1.T))) for m, (A1, B1) in fac[a].items()
                   for A2, B2 in [fac[b][m]])

    norm = {f: dot(f, f) ** 0.5 for f in TRAIN}
    return {(a, b): dot(a, b) / (norm[a] * norm[b]) for a in TRAIN for b in TRAIN}


def one_rule(r):
    """Correct under one answer rule for all 20 fields: the "Answer:" line, else a final \\boxed{}. The new fields'
    checker already works this way; the old fields' checker needs the "Answer:" line. Specialists trained on new fields
    pick up the \\boxed{} habit from their own kept answers, so under the old fields' rule their losses there are
    mostly answer format (see the last table)."""
    a = F.last_answer(r["text"])
    return F.normalise(r["field"], r, F.boxed(r["text"]) if a is None else a) == r["gold"]


def report():
    from scipy.stats import binomtest, false_discovery_control, ttest_1samp
    ex = {m: {r["pid"]: r for r in P.read_jsonl(P.OUT / f"exam_tm_{m}.jsonl")}
          for m in ["base"] + [f"tm_{f}" for f in TRAIN] if (P.OUT / f"exam_tm_{m}.jsonl").exists()}
    if "base" not in ex:
        return
    specs = [f for f in TRAIN if f"tm_{f}" in ex]
    ids = {f: [i for i, r in ex["base"].items() if r["field"] == f] for f in EXAM_FIELDS}
    rules = {"one": lambda r: int(one_rule(r)), "strict": lambda r: int(r["ok"])}
    ok = {(rule, m, f): np.array([sc(ex[m][i]) for i in ids[f]])
          for rule, sc in rules.items() for m in ex for f in EXAM_FIELDS}
    rng = np.random.default_rng(0)

    def ci_excludes_zero(d):
        lo, hi = np.percentile(d[rng.integers(0, len(d), (4000, len(d)))].mean(1), [2.5, 97.5])
        return lo > 0 or hi < 0

    def grid(rule, centred):
        """{(spec, field): (points, bootstrap CI excludes 0, p)}. Raw: specialist minus base, p from an exact McNemar
        test. Centred: specialist minus the mean of the other specialists, per problem, p from a one-sample t-test."""
        g = {}
        for f in specs:
            for h in EXAM_FIELDS:
                a = ok[rule, f"tm_{f}", h]
                if centred:
                    d = a - np.mean([ok[rule, f"tm_{o}", h] for o in specs if o != f], axis=0)
                    p = float(ttest_1samp(d, 0).pvalue) if d.std() > 0 else 1.0
                else:
                    d = a - ok[rule, "base", h]
                    n01, n10 = int((d > 0).sum()), int((d < 0).sum())
                    p = binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
                g[f, h] = (100 * d.mean(), ci_excludes_zero(d), p)
        q = dict(zip(g, false_discovery_control([v[2] for v in g.values()])))
        return {k: (v[0], v[1], q[k] < 0.05) for k, v in g.items()}

    def table(g, base_row):
        rows = ["| specialist | " + " | ".join(EXAM_FIELDS) + " |", "|---" * (len(EXAM_FIELDS) + 1) + "|"]
        if base_row:
            rows.append("| *base accuracy* | " + " | ".join(f"*{100 * ok[base_row, 'base', h].mean():.0f}*"
                                                           for h in EXAM_FIELDS) + " |")
        for f in specs:
            cells = []
            for h in EXAM_FIELDS:
                v, sig, fdr = g[f, h]
                cell = f"{v:+.0f}{'**' if fdr else '*' if sig else ''}"
                cells.append(f"[{cell}]" if h == f else cell)
            rows.append(f"| {f} ({len(P.read_jsonl(P.DATA / f'tm_sft_{f}.jsonl'))} ex., levels "
                        f"{','.join(map(str, LEVELS[f]))}) | " + " | ".join(cells) + " |")
        return rows

    n = len(specs) * len(EXAM_FIELDS)
    key = (f"Cells in points. * the paired 95% bootstrap interval excludes zero; ** the cell also survives a "
           f"Benjamini-Hochberg false discovery rate of 5% over all {n} cells of its table (at {n} cells, about "
           f"{round(0.05 * n)} single * marks are expected by chance alone). [ ] marks each specialist's own field.")
    one, cen, strict = grid("one", False), grid("one", True), grid("strict", False)
    old = [h for h in EXAM_FIELDS if h not in F.NORM]
    out = ["# Transfer map", "",
           "One MLP-only specialist per trainable field (rows), each examined with the base model on all 20 fields "
           "(columns; 160 problems each, 40 per level, levels 1 to 4; every exam on the RTX 3070). code and clock are "
           "never trained. Exploratory: not preregistered. Each specialist was trained once, from one seed, and the "
           "intervals below cover only which exam problems were drawn, not training: the replication found specialists "
           "of the same field differing by up to 13 points between seeds (runs/p1/replication.md). Treat single "
           "cells as leads to retrain with more seeds, not as findings.", "",
           "## 1. Change against the base model", "",
           "Scored with one rule for every field: the \"Answer:\" line, or a final \\boxed{} when there is none. " + key,
           ""] + table(one, "one") + [
           "", "## 2. Change against the average specialist", "",
           "Most columns move the same way whichever field was trained (clock, lineq and story gain, grid loses): an "
           "effect of self-training at all, not of the field. Each cell here is the specialist minus the mean of the "
           "other specialists on the same problems, so what remains is specific to the pair. p from a one-sample "
           "t-test on the per-problem differences. " + key, ""] + table(cen, None)
    cos = cosines()
    if cos:
        pairs = [(a, b) for i, a in enumerate(specs) for b in specs[i + 1:]]
        sym = [(cen[a, b][0] + cen[b, a][0]) / 2 for a, b in pairs]
        r = float(np.corrcoef([cos[p] for p in pairs], sym)[0, 1])
        out += ["", "## 3. Cosine between specialists' weight changes", "",
                f"Correlation between a pair's cosine and its mean pair-specific transfer (section 2, both directions "
                f"averaged), over {len(pairs)} pairs: r = {r:+.2f}.", "",
                "| | " + " | ".join(TRAIN) + " |", "|---" * (len(TRAIN) + 1) + "|"]
        out += [f"| {a} | " + " | ".join("" if a == b else f"{cos[(a, b)]:.2f}" for b in TRAIN) + " |" for a in TRAIN]
    out += ["", "## 4. Answer format: the same exams under the old fields' own rule", "",
            "The first nine fields' checker accepts only an \"Answer:\" line. Share of old-field answers (" +
            ", ".join(old) + ") that end in \\boxed{} with no \"Answer:\" line:", "",
            "| model | boxed instead of Answer: |", "|---|---|"]
    for m in ["base"] + [f"tm_{f}" for f in specs]:
        rs = [ex[m][i] for h in old for i in ids[h]]
        share = np.mean([F.last_answer(r["text"]) is None and F.boxed(r["text"]) is not None for r in rs])
        out.append(f"| {'*base model*' if m == 'base' else m.removeprefix('tm_') + ' specialist'} | {100 * share:.0f}% |")
    out += ["", "Change against the base under the old fields' rule (first nine fields strict, new fields as in "
            "section 1). " + key, ""] + table(strict, "strict")
    text = "\n".join(out) + "\n"
    P.write_lines(P.OUT / "transfer.md", text)
    print(text)


if __name__ == "__main__":
    if sys.argv[1:] != ["report"]:
        run()
    report()
