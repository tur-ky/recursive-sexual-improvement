"""Phase 1 of DESIGN.md: one family.

Two specialists (multiplication, long division) from one base, each trained twice on the same answer-key data:
MLP-only and all modules. Children: the full-delta merge, the MLP-only merge, and the donor payload (the other
parent's MLP delta at alpha 0.5) written into each parent. Baseline: one model trained on both fields' data (same
examples, same compute), MLP-only and all modules. Every model sits the same exam (greedy, vLLM on tranny, merged
fp16 weights; see PHASE0.md).

Then each child learns a third field (modular powers) in childhood: it samples 4 answers per problem and keeps the
ones both parents agree with, although neither parent was trained on that field. Controls on the MLP-merge child:
majority vote of its own samples, and the answer key. The multitask baseline has no parents and uses majority vote.

python phase1.py          run every stage whose outputs are missing (outputs are the checkpoints), then report
python phase1.py report   report only
"""
import collections, json, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fields as F  # noqa: E402

PY = str(HERE.parent / "rsi" / ".venv" / "Scripts" / "python.exe")
REMOTE, RW = "tranny", "/mnt/wdblue/claude-breeding/work"
DATA, OUT, SPECS, LORAS = HERE / "data" / "p1", HERE / "runs" / "p1", HERE / "specs" / "p1", HERE / "loras" / "p1"
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
MLP, ALL = "gate_proj,up_proj,down_proj", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
GPUS = [(0, 1.85), (1, 1.0)]  # tranny's 3070 and 2070, relative speed with merged fp16 weights (20 November)
LOW_PRIORITY = 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS: runs beside games on a home PC


def d(name, alpha=1.0):
    return {"path": f"deltas/p1/{name}", "alpha": alpha}


G = {  # genome of every model: the list of deltas added to the base
    "base": [],
    "mult_mlp": [d("mult_mlp")], "mult_all": [d("mult_all")],
    "div_mlp": [d("div_mlp")], "div_all": [d("div_all")],
    "mt_mlp": [d("mt_mlp")], "mt_all": [d("mt_all")],
    "merge_all": [d("mult_all"), d("div_all")],
    "merge_mlp": [d("mult_mlp"), d("div_mlp")],
    "pay_into_mult": [d("mult_mlp"), d("div_mlp", 0.5)],
    "pay_into_div": [d("div_mlp"), d("mult_mlp", 0.5)],
}
CHILDREN = ["merge_all", "merge_mlp", "pay_into_mult", "pay_into_div"]
PARENTS = {"merge_all": ("mult_all", "div_all"), "merge_mlp": ("mult_mlp", "div_mlp"),
           "pay_into_mult": ("mult_mlp", "div_mlp"), "pay_into_div": ("mult_mlp", "div_mlp")}
BASELINE_FOR = {"merge_all": "mt_all", "merge_mlp": "mt_mlp", "pay_into_mult": "mt_mlp", "pay_into_div": "mt_mlp"}
CHILDHOODS = [("merge_mlp", "family"), ("merge_all", "family"), ("pay_into_mult", "family"),
              ("pay_into_div", "family"), ("merge_mlp", "vote"), ("merge_mlp", "key"), ("mt_mlp", "vote")]
for _c, _arm in CHILDHOODS:
    G[f"{_c}_{_arm}"] = G[_c] + [d(f"grow_{_c}_{_arm}")]

# Sweep of the fertilisation edit itself (python phase1.py sweep): donor strength, and sending one third of the
# layers. Strength 0.5 on all layers is pay_into_*, strength 1.0 on all layers is merge_mlp.
THIRDS = {"early": list(range(0, 9)), "middle": list(range(9, 18)), "late": list(range(18, 28))}
SWEEP = {}
for _rec, _don in (("mult", "div"), ("div", "mult")):
    for _a in (0.1, 0.25, 0.75):
        SWEEP[f"pay_into_{_rec}_a{int(_a * 100)}"] = (_rec, _a, "all", [d(f"{_rec}_mlp"), d(f"{_don}_mlp", _a)])
    for _t, _layers in THIRDS.items():
        SWEEP[f"pay_into_{_rec}_{_t}"] = (_rec, 0.5, _t, [d(f"{_rec}_mlp"), {**d(f"{_don}_mlp", 0.5), "layers": _layers}])
G.update({k: v[3] for k, v in SWEEP.items()})
# Fetal microchimerism (turky's idea): after a payload birth the mother keeps a share of what the child has and she
# does not, i.e. part of the father's payload. Mother + 0.1 or 0.25 of the father's delta are the sweep models
# pay_into_<mother>_a10 / _a25; they also answer the child's third-field problems so they can be scored as verifiers
# against the plain mother (the control: same specialist, no chimera) and the father.
CHIMERA = {c: (f"{c.split('_')[2]}_mlp", [f"{c}_a10", f"{c}_a25"]) for c in ("pay_into_mult", "pay_into_div")}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_lines(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write(text)


def write_jsonl(path, rows):
    write_lines(path, "".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def sh(cmd):
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- problems
def make(fields, levels, n, seed):
    return [p for f in fields for lv in levels for p in F.make(f, lv, n, seed)]


def problems():
    if (DATA / "exam.jsonl").exists():
        return
    exam = make(["mult", "div", "modpow"], [1, 2, 3, 4], 100, 9001) + make(["code", "clock"], [1, 2, 3, 4], 50, 9002)
    seen = {p["q"] for p in exam}
    write_jsonl(DATA / "exam.jsonl", exam)
    for name, rows in {"train_mult": make(["mult"], [1, 2, 3], 150, 301),
                       "train_div": make(["div"], [1, 2], 225, 302),
                       "child": make(["modpow"], [2, 3], 150, 303)}.items():
        keep, qs = [], set(seen)
        for p in rows:  # nothing in training appears in the exam, and no problem twice
            if p["q"] not in qs:
                qs.add(p["q"])
                keep.append(p)
        write_jsonl(DATA / f"{name}.jsonl", keep)
    log("problems written")


# ---------------------------------------------------------------- tranny (vLLM) rounds
def job(problems, out, k=1, temp=0.0, seed=0):
    return {"problems": problems, "out": out, "k": k, "temp": temp, "seed": seed}


def exam_path(m, exam="exam"):
    """Exam answers of model m. Phase 1's exams were spread over both tranny GPUs; every model first examined on the
    2070 was examined again on the 3070 (gpucheck.py), and that run is used, so all compared models share one GPU."""
    p = OUT / f"{exam}3070_{m}.jsonl"
    return p if p.exists() else OUT / f"{exam}_{m}.jsonl"


def remote_round(plan, gpu_of=None):
    """plan: list of (label, model, [job]). Runs every missing job on tranny, one engine per label, labels spread
    over both GPUs (or on the GPU index gpu_of[label] gives), then copies the outputs back to runs/p1/."""
    gpu_of = gpu_of or {}
    plan = [(lab, m, [j for j in js if not (OUT / f"{j['out']}.jsonl").exists()]) for lab, m, js in plan]
    plan = [x for x in plan if x[2]]
    if not plan:
        return
    import export_lora
    sh(["ssh", REMOTE, f"mkdir -p {RW}/data/p1 {RW}/runs/p1/jobs {RW}/runs/p1/logs {RW}/loras/p1 {RW}/merged/p1"])
    sh(["scp", "-q", str(HERE / "vgen.py"), str(HERE / "merge_lora.py"), str(HERE / "fields.py"), f"{REMOTE}:{RW}/"])
    sh(["scp", "-q"] + [str(p) for p in DATA.glob("*.jsonl")] + [f"{REMOTE}:{RW}/data/p1/"])
    for lab, m, js in plan:
        if G[m] and not (LORAS / m).exists():
            spec = SPECS / f"{m}.json"
            write_lines(spec, json.dumps({"deltas": G[m]}))
            export_lora.export(spec, LORAS / m)
            sh(["scp", "-q", "-r", str(LORAS / m), f"{REMOTE}:{RW}/loras/p1/"])
        write_lines(OUT / "jobs" / f"{lab}.json", json.dumps(
            [{**j, "problems": f"data/p1/{j['problems']}.jsonl", "out": f"runs/p1/{j['out']}.jsonl"} for j in js]))
        sh(["scp", "-q", str(OUT / "jobs" / f"{lab}.json"), f"{REMOTE}:{RW}/runs/p1/jobs/"])
    # longest first onto whichever GPU finishes earliest
    size = {lab: sum(len(read_jsonl(DATA / f"{j['problems']}.jsonl")) * j["k"] for j in js) for lab, _, js in plan}
    load, queues = [0.0] * len(GPUS), [[] for _ in GPUS]
    for lab, m, js in sorted(plan, key=lambda x: -size[x[0]]):
        g = min((i for i in range(len(GPUS)) if gpu_of.get(lab, GPUS[i][0]) == GPUS[i][0]),
                key=lambda i: (load[i] + size[lab]) / GPUS[i][1])
        load[g] += size[lab]
        queues[g].append((lab, m))
    procs = []
    for (gpu, _), q in zip(GPUS, queues):
        if not q:
            continue
        lines = ["set -e", f"cd {RW}", "source env.sh"]
        for lab, m in q:
            model = BASE
            if G[m]:
                model = f"merged/p1/{m}"
                lines.append(f"[ -f {model}/model.safetensors ] || nice -n 5 $PY merge_lora.py loras/p1/{m} {model} "
                             f"--dtype float16 > runs/p1/logs/merge_{m}.log 2>&1")
            lines.append(f"CUDA_VISIBLE_DEVICES={gpu} nice -n 5 $PY vgen.py --model {model} --dtype float16 "
                         f"--jobs runs/p1/jobs/{lab}.json > runs/p1/logs/{lab}.log 2>&1")
        script = OUT / "jobs" / f"gpu{gpu}.sh"
        write_lines(script, "\n".join(lines) + "\n")
        sh(["scp", "-q", str(script), f"{REMOTE}:{RW}/runs/p1/jobs/"])
        procs.append((gpu, [lab for lab, _ in q], subprocess.Popen(["ssh", REMOTE, f"bash {RW}/runs/p1/jobs/gpu{gpu}.sh"])))
        log(f"tranny gpu{gpu}:", ", ".join(lab for lab, _ in q))
    bad = [(g, labs) for g, labs, p in procs if p.wait() != 0]
    if bad:
        raise RuntimeError(f"tranny jobs failed on {bad}; logs in {RW}/runs/p1/logs")
    for _, _, js in plan:
        for j in js:
            sh(["scp", "-q", f"{REMOTE}:{RW}/runs/p1/{j['out']}.jsonl", str(OUT / f"{j['out']}.jsonl")])
    log("round done:", ", ".join(lab for lab, _, _ in plan))


# ---------------------------------------------------------------- training (5070)
def train(name, genome, data, modules, seed=0):
    out = HERE / "deltas" / "p1" / name
    if (out / "delta.safetensors").exists():
        return
    spec = SPECS / f"train_{name}.json"
    write_lines(spec, json.dumps({"deltas": G[genome]}))
    lg = OUT / "logs" / f"train_{name}.log"
    lg.parent.mkdir(parents=True, exist_ok=True)
    log(f"training {name} on {genome} ({len(read_jsonl(data))} examples, {'MLP' if modules == MLP else 'all'})")
    with open(lg, "w") as fh:
        r = subprocess.run([PY, "run.py", "train", "--spec", str(spec.relative_to(HERE)),
                            "--data", str(data.relative_to(HERE)), "--out", f"deltas/p1/{name}", "--bs", "8",
                            "--accum", "2", "--epochs", "2", "--modules", modules, "--seed", str(seed)],
                           cwd=HERE, stdout=fh, stderr=subprocess.STDOUT, creationflags=LOW_PRIORITY)
    if r.returncode:
        raise RuntimeError(f"training {name} failed, see {lg}")


def sft(rows, accept, path):
    """Accepted answers, at most 2 distinct per problem."""
    by = collections.defaultdict(list)
    for r in rows:
        if accept(r) and r["text"] not in [x["text"] for x in by[r["pid"]]] and len(by[r["pid"]]) < 2:
            by[r["pid"]].append(r)
    write_jsonl(path, [r for v in by.values() for r in v])


# ---------------------------------------------------------------- verifiers for the third field
def filters(samples, parents):
    """parents: list of {pid: greedy answer}. Each filter maps a sampled row to accept / reject."""
    votes = collections.defaultdict(collections.Counter)
    for r in samples:
        if r["ans"] is not None:
            votes[r["pid"]][json.dumps(r["ans"])] += 1

    def majority(pid):
        mc = votes[pid].most_common(2)
        if not mc or mc[0][1] < 2 or (len(mc) > 1 and mc[1][1] == mc[0][1]):
            return None
        return mc[0][0]

    has = lambda r: r["ans"] is not None  # noqa: E731
    f = {"vote": lambda r: has(r) and json.dumps(r["ans"]) == majority(r["pid"]), "key": lambda r: bool(r["ok"])}
    if parents:
        f["family"] = lambda r: has(r) and all(P.get(r["pid"]) == r["ans"] for P in parents)
        f["one parent"] = lambda r: has(r) and any(P.get(r["pid"]) == r["ans"] for P in parents)
    return f


def child_filters(c):
    parents = [{r["pid"]: r["ans"] for r in read_jsonl(OUT / f"childp_{p}.jsonl")} for p in PARENTS.get(c, ())]
    samples = read_jsonl(OUT / f"child_{c}.jsonl")
    return samples, filters(samples, parents)


def childhood_data(c, arm):
    path = DATA / f"grow_{c}_{arm}.jsonl"
    if not path.exists():
        samples, f = child_filters(c)
        sft(samples, f[arm], path)
    return path


# ---------------------------------------------------------------- the run
def run():
    problems()
    remote_round([("base_tm", "base", [job("train_mult", "trainsamp_mult", 4, 0.8, 4)]),
                  ("base_td", "base", [job("train_div", "trainsamp_div", 4, 0.8, 4)])])
    for f in ("mult", "div"):
        if not (DATA / f"sft_{f}.jsonl").exists():
            sft(read_jsonl(OUT / f"trainsamp_{f}.jsonl"), lambda r: r["ok"], DATA / f"sft_{f}.jsonl")
    if not (DATA / "sft_mt.jsonl").exists():
        write_jsonl(DATA / "sft_mt.jsonl", read_jsonl(DATA / "sft_mult.jsonl") + read_jsonl(DATA / "sft_div.jsonl"))

    exam = lambda m: job("exam", f"exam_{m}")  # noqa: E731
    child = lambda m: job("child", f"child_{m}", 4, 0.8, 7)  # noqa: E731
    parent = lambda m: job("child", f"childp_{m}")  # noqa: E731
    errors = []

    def bg(plan):
        def go():
            try:
                remote_round(plan)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        t = threading.Thread(target=go)
        t.start()
        return t

    # base exam on tranny while the specialists train here
    t = bg([("base", "base", [exam("base")])])
    for f in ("mult", "div"):
        for kind, mods in (("mlp", MLP), ("all", ALL)):
            train(f"{f}_{kind}", "base", DATA / f"sft_{f}.jsonl", mods)
    t.join()
    if errors:
        raise errors[0]
    t = bg([(m, m, [exam(m)] + ([parent(m)] if m in ("mult_mlp", "div_mlp", "mult_all", "div_all") else []))
            for m in ("mult_mlp", "div_mlp", "mult_all", "div_all")]
           + [(c, c, [exam(c), child(c)]) for c in CHILDREN])
    train("mt_mlp", "base", DATA / "sft_mt.jsonl", MLP)
    train("mt_all", "base", DATA / "sft_mt.jsonl", ALL)
    t.join()
    if errors:
        raise errors[0]
    t = bg([("mt_mlp", "mt_mlp", [exam("mt_mlp"), child("mt_mlp")]), ("mt_all", "mt_all", [exam("mt_all")])])
    for c, arm in CHILDHOODS:
        if c in CHILDREN:
            train(f"grow_{c}_{arm}", c, childhood_data(c, arm), MLP)
    t.join()
    if errors:
        raise errors[0]
    train("grow_mt_mlp_vote", "mt_mlp", childhood_data("mt_mlp", "vote"), MLP)
    remote_round([(f"{c}_{arm}", f"{c}_{arm}", [exam(f"{c}_{arm}")]) for c, arm in CHILDHOODS])


# ---------------------------------------------------------------- report
def report():
    import numpy as np
    ex = {m: read_jsonl(exam_path(m)) for m in G if exam_path(m).exists()}
    groups = {"mult": {"mult"}, "div": {"div"}, "parents' fields": {"mult", "div"}, "modpow (3rd)": {"modpow"},
              "held out": F.HELD_OUT}

    def acc(m, fs):
        rows = [r for r in ex[m] if r["field"] in fs]
        return 100 * sum(r["ok"] for r in rows) / len(rows)

    orig = OUT / "gpu_original.json"  # GPU of each first exam run, read from the vLLM logs by gpucheck.py
    orig = json.loads(orig.read_text()) if orig.exists() else {}
    gpus = sorted({"3070" if exam_path(m).name.startswith("exam3070_") else orig.get(m, "?") for m in ex})
    out = ["# Phase 1 report", "", "## Exam (greedy; mult, div, modpow 400 problems each; code + clock 400)", "",
           f"Every exam below ran on tranny's RTX {' and '.join(gpus)} (merged fp16 weights, vLLM).", "",
           "| model | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|"]
    for m in ex:
        out.append(f"| {m} | " + " | ".join(f"{acc(m, fs):.1f}" for fs in groups.values()) + " |")

    def paired(a, b, fs):
        A = {r["pid"]: r["ok"] for r in ex[a] if r["field"] in fs}
        B = {r["pid"]: r["ok"] for r in ex[b] if r["field"] in fs}
        ids = sorted(set(A) & set(B))
        dlt = np.array([int(B[i]) - int(A[i]) for i in ids])
        rng = np.random.default_rng(0)
        lo, hi = np.percentile([dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(4000)], [2.5, 97.5])
        return 100 * dlt.mean(), 100 * lo, 100 * hi

    out += ["", "## Gate: heredity (does the child keep each parent's gain over the base?)", "",
            "A parent's gain counts only if its paired 95% interval against the base excludes zero; a gain too small "
            "to measure cannot be inherited measurably and is skipped.", ""]
    hereditary = []
    for c in CHILDREN:
        pm, pd = PARENTS[c]
        if not all(m in ex for m in (c, pm, pd, "base")):
            continue
        ok, parts = True, []
        for f, p in (("mult", pm), ("div", pd)):
            g, glo, _ = paired("base", p, {f})
            cd, clo, chi = paired(p, c, {f})
            if glo <= 0:
                parts.append(f"{f}: parent {p} gained {g:+.1f} (not measurable); child vs parent {cd:+.1f} "
                             f"[{clo:+.1f}, {chi:+.1f}]")
                continue
            keep = (acc(c, {f}) - acc("base", {f})) / g
            ok &= keep >= 0.5
            parts.append(f"{f}: keeps {keep:.0%} of {p}'s {g:+.1f}; child vs parent {cd:+.1f} [{clo:+.1f}, {chi:+.1f}]")
        hereditary.append(ok)
        out.append(f"- {c}: " + "; ".join(parts))
    if len(hereditary) == len(CHILDREN):
        out.append(f"- **{'PASS' if any(hereditary) else 'STOP (try lower strengths, a repair childhood)'}**: "
                   + (f"{sum(hereditary)} of {len(CHILDREN)} constructions keep at least half of every measurable gain"
                      if any(hereditary) else "no construction keeps half of the parents' measurable gains"))

    out += ["", "## Information: each child against the multitask baseline on the parents' two fields", ""]
    for c in CHILDREN:
        b = BASELINE_FOR[c]
        if c in ex and b in ex:
            dm, lo, hi = paired(b, c, {"mult", "div"})
            out.append(f"- {c} vs {b}: {dm:+.1f} points [{lo:+.1f}, {hi:+.1f}]")

    rows = []
    for c in CHILDREN + ["mt_mlp"]:
        if not (OUT / f"child_{c}.jsonl").exists() or not all(
                (OUT / f"childp_{p}.jsonl").exists() for p in PARENTS.get(c, ())):
            continue
        samples, f = child_filters(c)
        base_rate = sum(r["ok"] for r in samples) / len(samples)
        for name, fn in f.items():
            kept = [r for r in samples if fn(r)]
            prec = sum(r["ok"] for r in kept) / max(1, len(kept))
            rec = sum(r["ok"] for r in kept) / max(1, sum(r["ok"] for r in samples))
            rows.append(f"| {c} | {base_rate:.2f} | {name} | {len(kept)} | {prec:.2f} | {rec:.2f} |")
    if rows:
        out += ["", f"## Who can check the third field (modular powers, 4 samples x {len(read_jsonl(DATA / 'child.jsonl'))} "
                "problems per child)", "",
                "| child | base rate | filter | kept | precision | recall |", "|---|---|---|---|---|---|"] + rows

    out += ["", "## Childhood: before and after learning the third field", ""]
    for c, arm in CHILDHOODS:
        g = f"{c}_{arm}"
        if g in ex and c in ex:
            n = len(read_jsonl(DATA / f"grow_{c}_{arm}.jsonl"))
            parts = []
            for gname, fs in (("modpow", {"modpow"}), ("parents' fields", {"mult", "div"}), ("held out", F.HELD_OUT)):
                dm, lo, hi = paired(c, g, fs)
                parts.append(f"{gname} {dm:+.1f} [{lo:+.1f}, {hi:+.1f}]")
            out.append(f"- {c}, {arm} ({n} examples): " + "; ".join(parts))
    variants = [("merge_mlp", ("mult", 1.0, "all")), ("merge_mlp", ("div", 1.0, "all"))]
    variants += [(c, (c.split("_")[2], 0.5, "all")) for c in ("pay_into_mult", "pay_into_div")]
    variants += [(m, v[:3]) for m, v in SWEEP.items()]
    variants = [(m, v) for m, v in variants if m in ex and all(x in ex for x in ("base", "mult_mlp", "div_mlp"))]
    def share(m, f, parent):  # share of the parent's gain over the base, only where that gain is measurable
        if paired("base", parent, {f})[1] <= 0:
            return "n/a"
        return f"{(acc(m, {f}) - acc('base', {f})) / (acc(parent, {f}) - acc('base', {f})):.0%}"

    if any(m in SWEEP for m, _ in variants):
        out += ["", "## Sweep of the fertilisation edit (MLP-only parents; recipient keeps its whole genome)", "",
                "Shares of a parent's gain are n/a where that gain over the base is not measurable (division).", "",
                "| recipient | donor strength | donor layers | recipient's field | donor's field | held out | "
                "keeps recipient gain | gets donor gain |", "|---|---|---|---|---|---|---|---|"]
        for m, (rec, a, layers) in sorted(variants, key=lambda x: (x[1][0], x[1][2] != "all", x[1][2], x[1][1])):
            don = "div" if rec == "mult" else "mult"
            out.append(f"| {rec} | {a} | {layers} | {acc(m, {rec}):.1f} | {acc(m, {don}):.1f} | "
                       f"{acc(m, F.HELD_OUT):.1f} | {share(m, rec, f'{rec}_mlp')} | {share(m, don, f'{don}_mlp')} |")
    rows = []
    for c, (mother, chims) in CHIMERA.items():
        father = PARENTS[c][1] if PARENTS[c][0] == mother else PARENTS[c][0]
        verifiers = [("mother", mother)] + [(f"chimeric mother ({m[-2:]}% of father)", m) for m in chims] + [
            ("father", father)]
        if not (OUT / f"child_{c}.jsonl").exists() or not all((OUT / f"childp_{v}.jsonl").exists()
                                                               for _, v in verifiers):
            continue
        samples = read_jsonl(OUT / f"child_{c}.jsonl")
        n_ok = sum(r["ok"] for r in samples)
        for label, v in verifiers:
            P = {r["pid"]: r["ans"] for r in read_jsonl(OUT / f"childp_{v}.jsonl")}
            kept = [r for r in samples if r["ans"] is not None and P.get(r["pid"]) == r["ans"]]
            prec = sum(r["ok"] for r in kept) / max(1, len(kept))
            rows.append(f"| {c} | {label} | {len(kept)} | {prec:.2f} | {sum(r['ok'] for r in kept) / max(1, n_ok):.2f} |")
    if rows:
        out += ["", "## Microchimerism: does carrying part of the child make the mother a better verifier?", "",
                "Single verifier agreeing with the child's sampled answer (modular powers).", "",
                "| child | verifier | kept | precision | recall |", "|---|---|---|---|---|"] + rows
    text = "\n".join(out) + "\n"
    write_lines(OUT / "report.md", text)
    print(text)


if __name__ == "__main__":
    if sys.argv[1:] == ["sweep"]:
        chim = {m for _, ms in CHIMERA.values() for m in ms}
        remote_round([(m, m, [job("exam", f"exam_{m}")] + ([job("child", f"childp_{m}")] if m in chim else []))
                      for m in SWEEP])
    elif sys.argv[1:] != ["report"]:
        run()
    report()
