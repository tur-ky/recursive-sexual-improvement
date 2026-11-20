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


def report():
    ex = {m: {r["pid"]: r for r in P.read_jsonl(P.OUT / f"exam_tm_{m}.jsonl")}
          for m in ["base"] + [f"tm_{f}" for f in TRAIN] if (P.OUT / f"exam_tm_{m}.jsonl").exists()}
    if "base" not in ex:
        return
    rng = np.random.default_rng(0)

    def diff(m, f):
        ids = [i for i, r in ex["base"].items() if r["field"] == f]
        d = np.array([int(ex[m][i]["ok"]) - int(ex["base"][i]["ok"]) for i in ids])
        lo, hi = np.percentile(d[rng.integers(0, len(d), (4000, len(d)))].mean(1), [2.5, 97.5])
        return 100 * d.mean(), lo > 0 or hi < 0

    base_acc = {f: 100 * np.mean([r["ok"] for r in ex["base"].values() if r["field"] == f]) for f in EXAM_FIELDS}
    out = ["# Transfer map", "",
           "Each row is a specialist trained on one field; each column is an exam field (160 problems, levels 1 to 4). "
           "Cells: change in accuracy against the base model, in points; * where the paired 95% bootstrap interval "
           "excludes zero. The diagonal is each specialist's own gain. code and clock are never trained.", "",
           "| specialist | " + " | ".join(EXAM_FIELDS) + " |", "|---" * (len(EXAM_FIELDS) + 1) + "|",
           "| *base accuracy* | " + " | ".join(f"*{base_acc[f]:.0f}*" for f in EXAM_FIELDS) + " |"]
    for f in TRAIN:
        m = f"tm_{f}"
        if m not in ex:
            continue
        cells = []
        for g in EXAM_FIELDS:
            v, sig = diff(m, g)
            cell = f"{v:+.0f}{'*' if sig else ''}"
            cells.append(f"**{cell}**" if g == f else cell)
        out.append(f"| {f} ({len(P.read_jsonl(P.DATA / f'tm_sft_{f}.jsonl'))} ex., levels "
                   f"{','.join(map(str, LEVELS[f]))}) | " + " | ".join(cells) + " |")
    cos = cosines()
    if cos:
        out += ["", "## Cosine between specialists' weight changes", "",
                "| | " + " | ".join(TRAIN) + " |", "|---" * (len(TRAIN) + 1) + "|"]
        out += [f"| {a} | " + " | ".join("" if a == b else f"{cos[(a, b)]:.2f}" for b in TRAIN) + " |" for a in TRAIN]
    text = "\n".join(out) + "\n"
    P.write_lines(P.OUT / "transfer.md", text)
    print(text)


if __name__ == "__main__":
    if sys.argv[1:] != ["report"]:
        run()
    report()
