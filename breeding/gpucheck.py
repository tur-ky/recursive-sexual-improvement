"""Did the GPU change Phase 1's results?

Phase 1 spread its exams over tranny's RTX 3070 and 2070, so some compared models (the headline pair among them:
merge_mlp on the 2070, mt_mlp on the 3070) were examined on different cards. Here every Phase 1 model first examined on
the 2070 is examined again on the 3070, and the core models first examined on the 3070 are examined again on the 2070.
That gives a full one-GPU exam set on each card, and for every re-examined model a direct measure of how much the card
alone changes its answers. phase1.py's report then uses the 3070 set (phase1.exam_path).

python gpucheck.py           run what is missing, then write runs/p1/gpu_check.md and the Phase 1 report
python gpucheck.py report    report only
"""
import json, re, subprocess, sys

import numpy as np

import phase1 as P

CORE = ["base", "mult_mlp", "div_mlp", "mult_all", "div_all", "mt_mlp", "mt_all", "merge_all", "merge_mlp",
        "pay_into_mult", "pay_into_div"]
ALL = CORE + [f"{c}_{a}" for c, a in P.CHILDHOODS] + list(P.SWEEP)
CARD = {0: "3070", 1: "2070"}


def original_gpus():
    """GPU of each first exam run, from its vLLM log: Ampere (3070) uses FlashAttention, Turing (2070) cannot."""
    path = P.OUT / "gpu_original.json"
    if path.exists():
        return json.loads(path.read_text())
    txt = subprocess.run(["ssh", P.REMOTE, f"cd {P.RW}/runs/p1/logs && grep -m1 -o 'Using [A-Z_]* attention backend' "
                                           + " ".join(f"{m}.log" for m in ALL)],
                         capture_output=True, text=True, check=True).stdout
    found = {m: {"FLASH_ATTN": "3070", "TRITON_ATTN": "2070"}[b] for m, b in re.findall(r"^(\S+)\.log:Using (\w+) ", txt, re.M)}
    missing = set(ALL) - set(found)
    if missing:
        raise SystemExit(f"no backend line for {sorted(missing)}")
    P.write_lines(path, json.dumps(found, indent=1) + "\n")
    return found


def path(m, card, orig):
    """Exam answers of m on one card, or None if it was never examined there."""
    p = P.OUT / (f"exam_{m}.jsonl" if orig[m] == card else f"exam{card}_{m}.jsonl")
    return p if p.exists() else None


def run(orig):
    plan, gpu_of = [], {}
    for m in ALL:
        for gpu, card in CARD.items():
            if orig[m] != card and (card == "3070" or m in CORE):
                lab = f"g{card}_{m}"
                plan.append((lab, m, [P.job("exam", f"exam{card}_{m}")]))
                gpu_of[lab] = gpu
    P.remote_round(plan, gpu_of)


def report(orig):
    import fields as F
    rows = lambda p: {r["pid"]: r for r in P.read_jsonl(p)}  # noqa: E731
    sets = {card: {m: path(m, card, orig) for m in ALL} for card in CARD.values()}
    mixed = {m: P.OUT / f"exam_{m}.jsonl" for m in ALL}
    both, held = {"mult", "div"}, F.HELD_OUT
    out = ["# Does the GPU change Phase 1's results?", "",
           "Phase 1 first ran its exams on whichever of tranny's two cards was free. Each model below was examined on "
           "both the RTX 3070 and the RTX 2070 (same merged fp16 weights, same greedy decoding, same 1,600 problems).", "",
           "## Same model, two cards", "",
           "| model | first run on | same final answer | 3070 parents' fields | 2070 parents' fields | 3070 held out | "
           "2070 held out |", "|---|---|---|---|---|---|---|"]
    diffs = []
    for m in ALL:
        a, b = sets["3070"][m], sets["2070"][m]
        if not (a and b):
            continue
        A, B = rows(a), rows(b)
        same = np.mean([A[i]["ans"] == B[i]["ans"] for i in A])
        acc = lambda R, fs: 100 * np.mean([R[i]["ok"] for i in R if R[i]["field"] in fs])  # noqa: E731
        diffs.append(np.mean([A[i]["ok"] for i in A]) - np.mean([B[i]["ok"] for i in B]))
        out.append(f"| {m} | {orig[m]} | {100 * same:.0f}% | {acc(A, both):.1f} | {acc(B, both):.1f} | {acc(A, held):.1f} | "
                   f"{acc(B, held):.1f} |")
    out += ["", f"Mean accuracy difference over all 1,600 problems, 3070 minus 2070, across these {len(diffs)} models: "
            f"{100 * np.mean(diffs):+.2f} points (range {100 * min(diffs):+.1f} to {100 * max(diffs):+.1f}).", ""]

    def paired(pa, pb, fs):
        A, B = rows(pa), rows(pb)
        d = np.array([int(B[i]["ok"]) - int(A[i]["ok"]) for i in sorted(A) if A[i]["field"] in fs])
        g = np.random.default_rng(0)
        lo, hi = np.percentile([d[g.integers(0, len(d), len(d))].mean() for _ in range(4000)], [2.5, 97.5])
        return f"{100 * d.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"

    comps = [("merge_mlp vs mt_mlp, parents' fields", "mt_mlp", "merge_mlp", both),
             ("merge_mlp vs mult_mlp, mult", "mult_mlp", "merge_mlp", {"mult"}),
             ("merge_mlp vs div_mlp, div", "div_mlp", "merge_mlp", {"div"}),
             ("merge_mlp vs base, held out", "base", "merge_mlp", held),
             ("merge_all vs mt_all, parents' fields", "mt_all", "merge_all", both),
             ("merge_all vs mult_all, mult", "mult_all", "merge_all", {"mult"}),
             ("pay_into_div vs div_mlp, div", "div_mlp", "pay_into_div", {"div"}),
             ("mult_mlp vs base, mult", "base", "mult_mlp", {"mult"}),
             ("div_mlp vs base, div", "base", "div_mlp", {"div"})]
    out += ["## Key comparisons in each exam set", "",
            "Paired difference and 95% bootstrap interval. \"As first run\" mixes cards; the other two columns put both "
            "models of a comparison on one card.", "",
            "| comparison | as first run | all 3070 | all 2070 |", "|---|---|---|---|"]
    for name, a, b, fs in comps:
        cells = [paired(mixed[a], mixed[b], fs)]
        for card in ("3070", "2070"):
            pa, pb = sets[card][a], sets[card][b]
            cells.append(paired(pa, pb, fs) if pa and pb else "-")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(out) + "\n"
    P.write_lines(P.OUT / "gpu_check.md", text)
    print(text)


if __name__ == "__main__":
    o = original_gpus()
    if sys.argv[1:] != ["report"]:
        run(o)
    report(o)
    P.report()
