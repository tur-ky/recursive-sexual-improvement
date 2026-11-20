"""Self-improvement + weight-space editing harness for a small local LM.

Every model in this experiment is  base + sum_i alpha_i * delta_i, where each
delta is a low-rank factor pair (B @ A) per linear module, optionally restricted
to some layers or truncated to a lower rank. Deltas are applied unmerged (as
extra low-rank paths), so small edits are not rounded away by bf16 weights.

Subcommands
  gen    sample or greedy-decode answers for a problem file, score them exactly
  judge  ask the model whether each of its own answers is correct (P(Yes))
  feats  hidden states at the last answer token, for probing
  train  train a new low-rank delta on top of a spec (rejection-sampled SFT)
  probs  make problem files
"""
import argparse, json, math, os, random, re, sys, time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------- problems
def make_problem(rng, a, b):
    x = rng.randint(10 ** (a - 1), 10 ** a - 1)
    y = rng.randint(10 ** (b - 1), 10 ** b - 1)
    return {"a": a, "b": b, "x": x, "y": y, "ans": x * y}


def prompt_of(p):
    return (f"What is {p['x']} × {p['y']}? Work it out step by step, then write the "
            f"final answer on the last line as \"Answer: <number>\".")


ANS_RE = re.compile(r"Answer:\s*\**\s*\$?\s*(-?[\d,\s]*\d)")


def parse_answer(text):
    m = ANS_RE.findall(text)
    if not m:
        return None
    try:
        return int(re.sub(r"[,\s]", "", m[-1]))
    except ValueError:
        return None


# ---------------------------------------------------------------- deltas
class DLinear(nn.Module):
    """Linear layer plus frozen low-rank deltas, an optional bias and an optional trainable LoRA."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.A = None  # [R, in]
        self.B = None  # [out, R]
        self.bias = None
        self.tA = None
        self.tB = None
        self.tscale = 1.0

    def add_factors(self, A, B):
        dev, dt = self.base.weight.device, self.base.weight.dtype
        A, B = A.to(dev, dt), B.to(dev, dt)
        self.A = A if self.A is None else torch.cat([self.A, A], 0)
        self.B = B if self.B is None else torch.cat([self.B, B], 1)

    def add_bias(self, v):
        v = v.to(self.base.weight.device, self.base.weight.dtype)
        self.bias = v if self.bias is None else self.bias + v

    def forward(self, x):
        y = self.base(x)
        if self.A is not None:
            y = y + (x @ self.A.T) @ self.B.T
        if self.bias is not None:
            y = y + self.bias
        if self.tA is not None:
            y = y + ((x @ self.tA.T) @ self.tB.T) * self.tscale
        return y


def wrap_model(model):
    mods = {}
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in TARGETS and isinstance(child, nn.Linear):
                d = DLinear(child)
                setattr(mod, child_name, d)
                mods[f"{name}.{child_name}"] = d
    return mods


def layer_of(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def truncate(A, B, rank):
    """Best rank-k approximation of B @ A, returned as factors."""
    A, B = A.float(), B.float()
    Qb, Rb = torch.linalg.qr(B)          # B = Qb Rb
    Qa, Ra = torch.linalg.qr(A.T)        # A^T = Qa Ra
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
    k = min(rank, S.numel())
    newB = (Qb @ U[:, :k]) * S[:k]
    newA = Vh[:k] @ Qa.T
    return newA, newB


def apply_spec(mods, spec):
    """spec = {"deltas": [{"path", "alpha", "layers"?, "modules"?, "rank"?}]}"""
    for d in spec.get("deltas", []):
        sd = load_file(str(ROOT / d["path"] / "delta.safetensors"))
        alpha = float(d.get("alpha", 1.0))
        layers = set(d["layers"]) if d.get("layers") is not None else None
        kinds = set(d["modules"]) if d.get("modules") else None
        names = sorted({k.rsplit(".", 1)[0] for k in sd})
        for n in names:
            if n not in mods:
                continue
            if layers is not None and layer_of(n) not in layers:
                continue
            if kinds is not None and n.split(".")[-1] not in kinds:
                continue
            if f"{n}.A" in sd:
                A, B = sd[f"{n}.A"].float(), sd[f"{n}.B"].float()
                if d.get("rank"):
                    A, B = truncate(A.cuda(), B.cuda(), int(d["rank"]))
                mods[n].add_factors(A, B * alpha)
            if f"{n}.bias" in sd:
                mods[n].add_bias(sd[f"{n}.bias"].float() * alpha)


def load(spec, device, dtype):
    if torch.cuda.get_device_capability(device)[0] < 8:  # Turing: no flash-attention kernels
        torch.backends.cuda.enable_flash_sdp(False)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype).to(device)
    mods = wrap_model(model)
    apply_spec(mods, spec)
    model.eval()
    return tok, model, mods


def chat(tok, user, assistant_prefix=""):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True) + assistant_prefix


# ---------------------------------------------------------------- gen
@torch.no_grad()
def cmd_gen(a):
    spec = json.loads(Path(a.spec).read_text())
    tok, model, _ = load(spec, a.device, getattr(torch, a.dtype))
    probs = [json.loads(l) for l in open(a.problems)]
    jobs = [(i, k) for i in range(len(probs)) for k in range(a.k)]
    texts = [chat(tok, prompt_of(probs[i])) for i, _ in jobs]
    order = sorted(range(len(jobs)), key=lambda j: (len(texts[j]), j))
    outs = [None] * len(jobs)
    torch.manual_seed(a.seed)
    t0, ntok = time.time(), 0
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
    with open(a.out, "w") as f:
        for j, (i, k) in enumerate(jobs):
            p = probs[i]
            pred = parse_answer(outs[j])
            nums = re.findall(r"\d[\d,]*", outs[j])
            loose = int(nums[-1].replace(",", "")) if nums else None
            f.write(json.dumps({**p, "pid": p.get("pid", i), "k": k, "text": outs[j], "pred": pred,
                                "ok": pred == p["ans"], "ok_loose": loose == p["ans"] or pred == p["ans"]}) + "\n")


# ---------------------------------------------------------------- judge
JUDGE_Q = ("Here is a question and a proposed solution.\n\nQuestion: What is {x} × {y}?\n\n"
           "Proposed solution:\n{text}\n\nIs the final answer of the proposed solution correct? "
           "Reply with only Yes or No.")


@torch.no_grad()
def cmd_judge(a):
    spec = json.loads(Path(a.spec).read_text())
    tok, model, _ = load(spec, a.device, getattr(torch, a.dtype))
    rows = [json.loads(l) for l in open(a.rows)]
    yes = tok.encode("Yes", add_special_tokens=False)[0]
    no = tok.encode("No", add_special_tokens=False)[0]
    texts = [chat(tok, JUDGE_Q.format(x=r["x"], y=r["y"], text=r["text"][-3000:])) for r in rows]
    order = sorted(range(len(rows)), key=lambda j: len(texts[j]))
    for s in range(0, len(order), a.bs):
        idx = order[s:s + a.bs]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        lp = torch.log_softmax(logits[:, [yes, no]], -1)
        for r, j in enumerate(idx):
            rows[j]["p_yes"] = float(lp[r, 0].exp())
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------- feats
@torch.no_grad()
def cmd_feats(a):
    """Residual stream at the last token of prompt+answer, every layer. Saved as fp16 [N, L+1, D]."""
    spec = json.loads(Path(a.spec).read_text())
    tok, model, _ = load(spec, a.device, getattr(torch, a.dtype))
    rows = [json.loads(l) for l in open(a.rows)]
    texts = [chat(tok, prompt_of(r)) + r["text"] for r in rows]
    feats = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda j: len(texts[j]))
    for s in range(0, len(order), a.bs):
        idx = order[s:s + a.bs]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True).to(model.device)
        hs = model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states  # tuple L+1 x [b, T, D]
        h = torch.stack([x[:, -1, :] for x in hs], 1).half().cpu()  # left padded: last pos is real
        for r, j in enumerate(idx):
            feats[j] = h[r]
    torch.save({"h": torch.stack(feats), "ok": torch.tensor([r["ok"] for r in rows])}, a.out)


# ---------------------------------------------------------------- train
def cmd_train(a):
    spec = json.loads(Path(a.spec).read_text())
    tok, model, mods = load(spec, a.device, torch.bfloat16)
    tok.padding_side = "right"
    for p in model.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(a.seed)
    params = []
    kinds = set(a.modules.split(",")) if a.modules else None
    train_mods = {n: m for n, m in mods.items() if kinds is None or n.split(".")[-1] in kinds}
    for n, m in train_mods.items():
        fin, fout = m.base.in_features, m.base.out_features
        m.tA = nn.Parameter((torch.randn(a.r, fin, generator=g) / math.sqrt(fin)).to(a.device, torch.float32))
        m.tB = nn.Parameter(torch.zeros(fout, a.r, device=a.device, dtype=torch.float32))
        m.tscale = 2.0  # lora_alpha / r with alpha = 2r
        params += [m.tA, m.tB]

    rows = [json.loads(l) for l in open(a.data)]
    end = tok.convert_tokens_to_ids("<|im_end|>")
    ex = []
    for r in rows:
        p = tok(chat(tok, prompt_of(r)), add_special_tokens=False).input_ids
        c = tok(r["text"], add_special_tokens=False).input_ids + [end]
        ex.append((p + c, [-100] * len(p) + c))
    ex = [e for e in ex if len(e[0]) <= a.max_len]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    steps_per_epoch = math.ceil(len(ex) / (a.bs * a.accum))
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, total // 20)) * max(0.0, 1 - s / total) if total else 1.0)
    rng = random.Random(a.seed)
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        idx = list(range(len(ex)))
        rng.shuffle(idx)
        # bucket by length inside chunks to cut padding
        chunks = [sorted(idx[i:i + a.bs * 16], key=lambda j: len(ex[j][0])) for i in range(0, len(idx), a.bs * 16)]
        batches = [c[i:i + a.bs] for c in chunks for i in range(0, len(c), a.bs)]
        rng.shuffle(batches)
        run_loss, n = 0.0, 0
        for bi, b in enumerate(batches):
            T = max(len(ex[j][0]) for j in b)
            ids = torch.full((len(b), T), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(b), T), -100, dtype=torch.long)
            att = torch.zeros((len(b), T), dtype=torch.long)
            for r, j in enumerate(b):
                x, y = ex[j]
                ids[r, :len(x)] = torch.tensor(x)
                lab[r, :len(y)] = torch.tensor(y)
                att[r, :len(x)] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids.to(a.device), attention_mask=att.to(a.device), labels=lab.to(a.device))
            (out.loss / a.accum).backward()
            run_loss += float(out.loss)
            n += 1
            if (bi + 1) % a.accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"[train] ep {ep} step {step}/{total} loss {run_loss / n:.4f} "
                          f"{time.time() - t0:.0f}s", flush=True)
                    run_loss, n = 0.0, 0
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {}
    for n_, m in train_mods.items():
        sd[f"{n_}.A"] = m.tA.detach().float().cpu().contiguous()
        sd[f"{n_}.B"] = (m.tB.detach().float() * m.tscale).cpu().contiguous()
    save_file(sd, str(out_dir / "delta.safetensors"))
    (out_dir / "meta.json").write_text(json.dumps({"parent": spec, "data": a.data, "r": a.r, "lr": a.lr,
                                                   "epochs": a.epochs, "n": len(ex)}, indent=1))
    print("saved", out_dir)


# ---------------------------------------------------------------- probs
def cmd_probs(a):
    rng = random.Random(a.seed)
    levels = [tuple(map(int, l.split("x"))) for l in a.levels.split(",")]
    with open(a.out, "w") as f:
        pid = 0
        for _ in range(a.n):
            for (x, y) in levels:
                p = make_problem(rng, x, y)
                p["pid"] = pid
                pid += 1
                f.write(json.dumps(p) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "judge", "feats", "train"):
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
            s.add_argument("--max_new", type=int, default=768)
        if name in ("judge", "feats"):
            s.add_argument("--rows", required=True)
        if name == "train":
            s.add_argument("--data", required=True)
            s.add_argument("--r", type=int, default=32)
            s.add_argument("--lr", type=float, default=2e-4)
            s.add_argument("--epochs", type=int, default=2)
            s.add_argument("--accum", type=int, default=2)
            s.add_argument("--max_len", type=int, default=1024)
            s.add_argument("--modules", default="", help="comma list of module kinds to train, e.g. gate_proj,up_proj,down_proj")
    s = sub.add_parser("probs")
    s.add_argument("--levels", required=True)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"gen": cmd_gen, "judge": cmd_judge, "feats": cmd_feats, "train": cmd_train, "probs": cmd_probs}[a.cmd](a)


if __name__ == "__main__":
    main()
