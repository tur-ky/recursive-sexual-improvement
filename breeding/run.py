"""Model jobs for the breeding experiment, reusing rsi/rsi.py (unmerged low-rank deltas, training, features).

  python run.py probs --fields mult,div --levels 1,2,3,4 --n 30 --seed 0 --out data/x.jsonl
  python run.py gen   --spec specs/x.json --problems data/x.jsonl --out runs/x.jsonl [--k 4 --temp 0.8]
  python run.py train --spec specs/x.json --data data/sft.jsonl --out deltas/x [--modules gate_proj,up_proj,down_proj]
  python run.py feats --spec specs/x.json --rows runs/x.jsonl --out runs/x_feats.pt
Spec delta paths are relative to this folder (breeding/).
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rsi"))
import rsi  # noqa: E402
import fields as F  # noqa: E402

rsi.prompt_of = lambda r: r["q"]  # every rsi job now asks the field's own question


def fix_spec(path):
    """Rewrite delta paths (relative to breeding/) so rsi.apply_spec, which resolves against rsi/, finds them."""
    spec = json.loads(Path(path).read_text())
    for d in spec.get("deltas", []):
        d["path"] = str((HERE / d["path"]).resolve())
    out = HERE / "runs" / "_specs" / Path(path).name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec))
    return str(out)


def cmd_probs(a):
    rows = []
    for f in a.fields.split(","):
        for lv in map(int, a.levels.split(",")):
            rows += F.make(f, lv, a.n, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(len(rows), "problems ->", a.out)


def cmd_gen(a):
    import torch
    spec = json.loads(Path(fix_spec(a.spec)).read_text())
    tok, model, _ = rsi.load(spec, a.device, getattr(torch, a.dtype))
    probs = [json.loads(l) for l in open(a.problems) if l.strip()]
    jobs = [(i, k) for i in range(len(probs)) for k in range(a.k)]
    texts = [rsi.chat(tok, probs[i]["q"]) for i, _ in jobs]
    order = sorted(range(len(jobs)), key=lambda j: (len(texts[j]), j))
    outs = [None] * len(jobs)
    torch.manual_seed(a.seed)
    t0, ntok = time.time(), 0
    with torch.no_grad():
        for s in range(0, len(order), a.bs):
            idx = order[s:s + a.bs]
            enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True).to(model.device)
            kw = dict(max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id, repetition_penalty=1.0)
            if a.temp > 0:
                kw.update(do_sample=True, temperature=a.temp, top_p=1.0, top_k=0)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            g = model.generate(**enc, **kw)
            L = enc.input_ids.shape[1]
            for r, j in enumerate(idx):
                seq = g[r, L:]
                ntok += int((seq != tok.pad_token_id).sum())
                outs[j] = tok.decode(seq, skip_special_tokens=True)
            print(f"[gen {a.device}] {min(s + a.bs, len(order))}/{len(order)}  {ntok / (time.time() - t0):.0f} tok/s",
                  flush=True)
    with open(a.out, "w") as fh:
        for j, (i, k) in enumerate(jobs):
            p = probs[i]
            ok, readable = F.check(p, outs[j])
            fh.write(json.dumps({**p, "k": k, "text": outs[j], "ok": ok, "readable": readable,
                                 "ans": F.answer_of(p, outs[j])}) + "\n")


def cmd_train(a):
    a.spec = fix_spec(a.spec)
    a.out = str((HERE / a.out).resolve())
    rsi.cmd_train(a)


def cmd_feats(a):
    a.spec = fix_spec(a.spec)
    rsi.cmd_feats(a)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("probs")
    s.add_argument("--fields", required=True)
    s.add_argument("--levels", default="1,2,3,4")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    for name in ("gen", "train", "feats"):
        s = sub.add_parser(name)
        s.add_argument("--spec", required=True)
        s.add_argument("--device", default="cuda:0")
        s.add_argument("--dtype", default="bfloat16")
        s.add_argument("--bs", type=int, default=64)
        s.add_argument("--out", required=True)
        s.add_argument("--seed", type=int, default=0)
        if name == "gen":
            s.add_argument("--problems", required=True)
            s.add_argument("--k", type=int, default=1)
            s.add_argument("--temp", type=float, default=0.0)
            s.add_argument("--max_new", type=int, default=640)
        if name == "feats":
            s.add_argument("--rows", required=True)
        if name == "train":
            s.add_argument("--data", required=True)
            s.add_argument("--r", type=int, default=32)
            s.add_argument("--lr", type=float, default=2e-4)
            s.add_argument("--epochs", type=int, default=2)
            s.add_argument("--accum", type=int, default=2)
            s.add_argument("--max_len", type=int, default=1024)
            s.add_argument("--modules", default="gate_proj,up_proj,down_proj")
    a = ap.parse_args()
    {"probs": cmd_probs, "gen": cmd_gen, "train": cmd_train, "feats": cmd_feats}[a.cmd](a)


if __name__ == "__main__":
    main()
