"""Merge test of SEEDMERGE-PLAN.md: same-seed against different-seed parents, and three ways of combining their changes.

Every multiplication specialist (Phase 1, s2, s3) with every division specialist, 3 x 3 children, each combined three
ways (plain sum, sign-elect, TIES; see ties_build.py), all built the same way on tranny and examined on its RTX 3070
with Phase 1's exam. Parents and base reuse their existing 3070 exams of that file.

python seedmerge.py           build and examine every child whose exam is missing, then report
python seedmerge.py report    report only
"""
import json, subprocess, sys

import numpy as np

import phase1 as P

SEEDS = ["p1", "s2", "s3"]  # Phase 1 (training seed 0) and the replication's seeds 2 and 3
MULT = {"p1": "mult_mlp", "s2": "s2_mult_mlp", "s3": "s3_mult_mlp"}
DIV = {"p1": "div_mlp", "s2": "s2_div_mlp", "s3": "s3_div_mlp"}
OLD_MERGE = {"p1": "merge_mlp", "s2": "s2_merge_mlp", "s3": "s3_merge_mlp"}
METHODS = ["plain", "elect", "ties"]
EXAM_GPU = 0  # tranny's 3070
B = 4000


def name(meth, i, j):
    return f"sm_{meth}_{i}_{j}"


CHILDREN = [(meth, i, j) for meth in METHODS for i in SEEDS for j in SEEDS]
for _m, _i, _j in CHILDREN:
    P.G[name(_m, _i, _j)] = [P.d(MULT[_i]), P.d(DIV[_j])]
    P.BUILD[name(_m, _i, _j)] = (f"$PY ties_build.py deltas/p1/{MULT[_i]} deltas/p1/{DIV[_j]} "
                                 f"merged/p1/{name(_m, _i, _j)} --method {_m}")


def run():
    parents = list(MULT.values()) + list(DIV.values())
    have = subprocess.run(["ssh", P.REMOTE, f"ls {P.RW}/deltas/p1 2>/dev/null"], capture_output=True, text=True).stdout.split()
    P.sh(["ssh", P.REMOTE, f"mkdir -p {P.RW}/deltas/p1"])
    for p in parents:
        if p not in have:
            P.sh(["scp", "-q", "-r", str(P.HERE / "deltas" / "p1" / p), f"{P.REMOTE}:{P.RW}/deltas/p1/"])
    plan = [(name(*c), name(*c), [P.job("exam", f"exam_{name(*c)}")]) for c in CHILDREN]
    P.remote_round(plan, {lab: EXAM_GPU for lab, _, _ in plan})
    for c in CHILDREN:
        lg = P.OUT / "logs" / f"build_{name(*c)}.log"
        if not lg.exists():
            P.sh(["scp", "-q", f"{P.REMOTE}:{P.RW}/runs/p1/logs/build_{name(*c)}.log", str(lg)])


def build_stats(n):
    lg = P.OUT / "logs" / f"build_{n}.log"
    if not lg.exists():
        return {}
    lines = [l for l in lg.read_text().splitlines() if l.startswith("{")]
    return json.loads(lines[-1]) if lines else {}


def report():
    models = ["base"] + list(MULT.values()) + list(DIV.values()) + list(OLD_MERGE.values()) + [name(*c) for c in CHILDREN]
    missing = [m for m in models if not P.exam_path(m).exists()]
    if missing:
        print("missing exams:", ", ".join(missing))
        return
    rows = {m: {r["pid"]: r for r in P.read_jsonl(P.exam_path(m))} for m in models}
    pids = sorted(rows["base"])
    field = np.array([rows["base"][p]["field"] for p in pids])
    ok = np.array([[int(rows[m][p]["ok"]) for p in pids] for m in models], dtype=float)
    idx = {m: k for k, m in enumerate(models)}

    # accuracy per field, column 0 on the real exam, columns 1..B on problem resamples drawn within each field
    rng = np.random.default_rng(0)
    acc = {}
    for f in sorted(set(field)):
        cols = np.where(field == f)[0]
        w = np.ones((B + 1, len(cols)))
        for b in range(1, B + 1):
            w[b] = np.bincount(rng.integers(0, len(cols), len(cols)), minlength=len(cols))
        acc[f] = 100 * ok[:, cols] @ w.T / len(cols)  # models x (B + 1)
    n = {f: int((field == f).sum()) for f in acc}
    acc["held out"] = sum(acc[f] * n[f] for f in P.F.HELD_OUT) / sum(n[f] for f in P.F.HELD_OUT)
    A = lambda m, f: acc[f][idx[m]]  # noqa: E731

    def cost(c):
        meth, i, j = c
        return 0.5 * ((A(name(*c), "mult") - A(MULT[i], "mult")) + (A(name(*c), "div") - A(DIV[j], "div")))

    def ci(x):
        lo, hi = np.percentile(x[1:], [2.5, 97.5])
        p = min(1.0, 2 * min((x[1:] <= 0).mean(), (x[1:] >= 0).mean()))
        return x[0], lo, hi, p

    def fmt(x):
        e, lo, hi, p = ci(x)
        return f"{e:+.1f} [{lo:+.1f}, {hi:+.1f}]"

    def fmtp(x):
        e, lo, hi, p = ci(x)
        return f"{e:+.1f} [{lo:+.1f}, {hi:+.1f}], p {f'= {p:.3g}' if p else '< 0.0005'}"

    same = lambda c: c[1] == c[2]  # noqa: E731
    out = ["# Merge test: same-seed against different-seed parents", "",
           "Design, outcomes and predictions: `SEEDMERGE-PLAN.md` (committed before any of these models existed). "
           "Every exam on tranny's RTX 3070, greedy, merged fp16 weights, Phase 1's exam (400 mult, 400 div, "
           "400 modpow, 400 held out: code + clock). Intervals: paired bootstrap over exam problems within each field "
           f"({B} resamples, seed 0); they cover exam sampling only, not which seeds the six parents came from.", "",
           "## Parents", "", "| model | mult | div | modpow | held out |", "|---|---|---|---|---|"]
    for m in ["base"] + [MULT[s] for s in SEEDS] + [DIV[s] for s in SEEDS]:
        out.append(f"| {m} | " + " | ".join(f"{A(m, f)[0]:.1f}" for f in ("mult", "div", "modpow", "held out")) + " |")

    out += ["", "## Children", "",
            "Merge cost = mean over the two fields of (child minus that field's parent), in points; 0 means the child "
            "keeps both parents' skill. Sign conflicts: share of changed weights where the two parents' changes have "
            "opposite signs. From plain: how far the combined change is from the plain sum (relative norm).", "",
            "| method | mult parent | div parent | seeds | mult | div | merge cost | held out vs base | modpow vs base | "
            "sign conflicts | from plain |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in CHILDREN:
        st = build_stats(name(*c))
        out.append(f"| {c[0]} | {MULT[c[1]]} | {DIV[c[2]]} | {'same' if same(c) else 'different'} | "
                   f"{A(name(*c), 'mult')[0]:.1f} | {A(name(*c), 'div')[0]:.1f} | {fmt(cost(c))} | "
                   f"{fmt(A(name(*c), 'held out') - A('base', 'held out'))} | "
                   f"{fmt(A(name(*c), 'modpow') - A('base', 'modpow'))} | "
                   f"{st.get('sign_conflict_share', float('nan')):.3f} | {st.get('distance_from_plain', float('nan')):.2f} |")

    for meth in METHODS:
        out += ["", f"### Merge cost, {meth} (rows: mult parent's seed, columns: div parent's seed)", "",
                "| | " + " | ".join(SEEDS) + " |", "|---" * (len(SEEDS) + 1) + "|"]
        for i in SEEDS:
            out.append(f"| {i} | " + " | ".join(f"{cost((meth, i, j))[0]:+.1f}" for j in SEEDS) + " |")

    mean = lambda cs: sum(cost(c) for c in cs) / len(cs)  # noqa: E731
    by = lambda meth, f=lambda c: True: [c for c in CHILDREN if c[0] == meth and f(c)]  # noqa: E731
    diff = lambda c: not same(c)  # noqa: E731
    con = {}
    for meth in METHODS:
        con[f"seed contrast, {meth}: same-seed minus different-seed merge cost"] = mean(by(meth, same)) - mean(by(meth, diff))
    for meth in ("elect", "ties"):
        for label, f in (("all 9", lambda c: True), ("same-seed", same), ("different-seed", diff)):
            con[f"{meth} minus plain, merge cost, {label}"] = mean(by(meth, f)) - mean(by("plain", f))
    for meth in ("elect", "ties"):
        held = sum(A(name(*c), "held out") - A(name("plain", c[1], c[2]), "held out") for c in by(meth)) / 9
        con[f"{meth} minus plain, held out, all 9"] = held
    out += ["", "## Contrasts", ""] + [f"- {k}: {fmtp(v)}" for k, v in con.items()]

    e_off = ci(con["elect minus plain, merge cost, different-seed"])
    t_all = ci(con["ties minus plain, merge cost, all 9"])
    out += ["", "## Predictions (SEEDMERGE-PLAN.md), judged on the point estimate as written there", "",
            f"1. Sign-elect within 2 points of plain on different-seed pairs: {e_off[0]:+.1f} "
            f"[{e_off[1]:+.1f}, {e_off[2]:+.1f}]: **{'held' if abs(e_off[0]) < 2 else 'failed'}**"
            f"{' (interval also inside +-2)' if -2 < e_off[1] and e_off[2] < 2 else ''}.",
            f"2. TIES worse than plain by more than 3 points: {t_all[0]:+.1f} [{t_all[1]:+.1f}, {t_all[2]:+.1f}]: "
            f"**{'held' if t_all[0] < -3 else 'failed'}**.",
            "3. Seed contrast under plain: no direction predicted; see the contrasts above."]

    out += ["", "## Build check: rebuilt plain same-seed children against the original merge children", "",
            "The original merges were exported as bf16 LoRA factors and baked by merge_lora.py; the rebuilt ones use "
            "fp32 factors (ties_build.py). Same exam, same GPU.", "",
            "| pair | original | rebuilt | same final answer |", "|---|---|---|---|"]
    for s in SEEDS:
        o, r = OLD_MERGE[s], name("plain", s, s)
        agree = np.mean([rows[o][p]["ans"] == rows[r][p]["ans"] for p in pids])
        both = lambda m: 100 * np.mean([rows[m][p]["ok"] for p in pids if rows[m][p]["field"] in ("mult", "div")])  # noqa: E731
        out.append(f"| {s} | {both(o):.1f} | {both(r):.1f} | {agree:.1%} |")
    text = "\n".join(out) + "\n"
    P.write_lines(P.OUT / "seed_merge.md", text)
    print(text)


if __name__ == "__main__":
    if sys.argv[1:] != ["report"]:
        run()
    report()
