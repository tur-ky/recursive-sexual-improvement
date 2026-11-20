"""Shard a run.py gen/feats job over the 5070 here and tranny's 3070 + 2070 (same scheme as rsi/dist.py).

usage: python bdist.py gen --spec specs/x.json --problems data/p.jsonl --out runs/x.jsonl [--k 4 --temp 0.8]
Paths are relative to breeding/. Set BDIST_REMOTE_ONLY=1 to leave the 5070 free (e.g. while it trains).
"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY_LOCAL = str(ROOT.parent / "rsi" / ".venv" / "Scripts" / "python.exe")
REMOTE = "tranny"
RROOT = "breeding"  # on tranny, next to ~/rsi (which holds rsi.py and the venv)
WORKERS = [("local", "cuda:0", "bfloat16", 128, 0.30),
           ("remote", "cuda:0", "bfloat16", 96, 0.42),
           ("remote", "cuda:1", "float16", 72, 0.28)]
if os.environ.get("BDIST_REMOTE_ONLY"):
    WORKERS = [w for w in WORKERS if w[0] == "remote"]


def sh(cmd):
    return subprocess.run(cmd, check=True)


def push(paths):
    sh(["ssh", REMOTE, f"mkdir -p {RROOT}/runs/_shards {RROOT}/specs {RROOT}/deltas {RROOT}/data"])
    # ~/rsi on tranny belongs to the RSI run and is read-only here: run.py imports the rsi.py already there
    for p in paths:
        rel = Path(p).relative_to(ROOT).as_posix()
        sh(["ssh", REMOTE, f"mkdir -p {RROOT}/{Path(rel).parent.as_posix()}"])
        sh(["scp", "-q", "-r", str(p), f"{REMOTE}:{RROOT}/{rel}"])


def assign(n):
    order, acc = [], [0.0] * len(WORKERS)
    for _ in range(n):
        i = min(range(len(WORKERS)), key=lambda w: acc[w] / WORKERS[w][4])
        order.append(i)
        acc[i] += 1
    return order


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    opt = dict(zip(args[0::2], args[1::2]))
    src_key = "--problems" if cmd == "gen" else "--rows"
    src, out, spec = ROOT / opt[src_key], ROOT / opt["--out"], ROOT / opt["--spec"]
    rows = [l for l in src.read_text().splitlines() if l.strip()]
    order = assign(len(rows))
    tmp = ROOT / "runs" / "_shards"
    tmp.mkdir(parents=True, exist_ok=True)
    stem = out.stem
    pushes = [ROOT / "run.py", ROOT / "fields.py", spec]
    pushes += [ROOT / d["path"] for d in json.loads(spec.read_text()).get("deltas", [])]
    for w, (where, *_rest) in enumerate(WORKERS):
        sp = tmp / f"{stem}.in{w}.jsonl"
        sp.write_text("\n".join(r for r, i in zip(rows, order) if i == w) + "\n")
        if where == "remote":
            pushes.append(sp)
    push(pushes)
    extra = [x for kv in opt.items() if kv[0] not in ("--spec", src_key, "--out") for x in kv]
    procs = []
    for w, (where, dev, dt, bs, _) in enumerate(WORKERS):
        sp = f"runs/_shards/{stem}.in{w}.jsonl"
        so = f"runs/_shards/{stem}.out{w}" + (".pt" if cmd == "feats" else ".jsonl")
        a = [cmd, "--spec", opt["--spec"], src_key, sp, "--out", so, "--device", dev, "--dtype", dt,
             "--bs", str(bs)] + extra
        log = open(tmp / f"{stem}.log{w}", "w")
        if where == "local":
            p = subprocess.Popen([PY_LOCAL, "run.py"] + a, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        else:
            rc = (f"cd {RROOT} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 ../rsi/env/bin/python "
                  "run.py " + " ".join(f"'{x}'" for x in a))
            p = subprocess.Popen(["ssh", REMOTE, rc], stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, where, so))
    t0 = time.time()
    fails = [(where, so) for p, where, so in procs if p.wait() != 0]
    if fails:
        sys.exit(f"workers failed: {fails} (logs in runs/_shards)")
    for p, where, so in procs:
        if where == "remote":
            sh(["scp", "-q", f"{REMOTE}:{RROOT}/{so}", str(ROOT / so)])
    if cmd == "feats":
        import torch
        ds = [torch.load(ROOT / so) for _, _, so in procs]
        ptr, h, ok = [0] * len(WORKERS), [], []
        for i in order:
            h.append(ds[i]["h"][ptr[i]]); ok.append(ds[i]["ok"][ptr[i]]); ptr[i] += 1
        torch.save({"h": torch.stack(h), "ok": torch.stack(ok)}, out)
    else:
        lines = []
        for _, _, so in procs:
            lines += [l for l in (ROOT / so).read_text().splitlines() if l.strip()]
        out.write_text("\n".join(lines) + "\n")
    print(f"{cmd} done in {time.time() - t0:.0f}s -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
