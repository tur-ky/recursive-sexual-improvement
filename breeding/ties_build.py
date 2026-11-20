"""Build a merged child's full weights from two parents' low-rank changes, combined weight by weight.

python ties_build.py deltas/p1/A deltas/p1/B merged/p1/x --method plain|elect|ties [--density 0.2]
Runs on tranny with the vLLM env (base model from its HF cache). For every weight matrix both parents changed, with
tau = B @ A for each parent (fp32):
  plain   tau_a + tau_b                                   (what export_lora + merge_lora give, from fp32 factors)
  elect   where the signs agree, tau_a + tau_b; where they disagree, only the larger of the two, whole
  ties    TIES-merging (Yadav et al. 2023): keep each change's `density` largest entries by size, elect a sign per
          weight from their sum, average the kept entries that agree with it (disjoint mean), lambda 1
The child is base + the combined change, summed in fp32 and rounded once to --dtype. Prints, over all changed matrices:
the share of weights where the parents disagree in sign, and how far the combined change is from the plain sum.
"""
import argparse, json, math, shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def combine(ta, tb, method, density):
    if method == "plain":
        return ta + tb
    if method == "elect":
        s = torch.sign(ta + tb)
        return ta * (torch.sign(ta) == s) + tb * (torch.sign(tb) == s)
    kept = []
    for t in (ta, tb):
        k = max(1, int(math.ceil(density * t.numel())))
        thr = t.abs().flatten().kthvalue(t.numel() - k + 1).values
        kept.append(t * (t.abs() >= thr))
    s = torch.sign(kept[0] + kept[1])
    on = [(t != 0) & (torch.sign(t) == s) for t in kept]
    return (kept[0] * on[0] + kept[1] * on[1]) / (on[0].float() + on[1].float()).clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--method", required=True, choices=["plain", "elect", "ties"])
    ap.add_argument("--density", type=float, default=0.2)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, a.dtype)
    fa, fb = (load_file(str(Path(p) / "delta.safetensors")) for p in (a.a, a.b))
    mods = sorted({k[:-2] for k in fa if k.endswith(".A")})
    if mods != sorted({k[:-2] for k in fb if k.endswith(".A")}):
        raise SystemExit("the two parents changed different modules")
    src = Path(snapshot_download(BASE, local_files_only=True))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix != ".safetensors":
            shutil.copy(f, out / f.name)
    mcfg = json.loads((out / "config.json").read_text())
    mcfg["torch_dtype"] = a.dtype
    (out / "config.json").write_text(json.dumps(mcfg, indent=2))

    used, n_w, n_conf, plain2, diff2, d2, err2 = set(), 0, 0, 0.0, 0.0, 0.0, 0.0
    for wf in sorted(src.glob("*.safetensors")):
        sd = load_file(str(wf))
        for k in list(sd):
            n = k[: -len(".weight")] if k.endswith(".weight") else None
            if n in mods:
                ta = fa[f"{n}.B"].to(dev).float() @ fa[f"{n}.A"].to(dev).float()
                tb = fb[f"{n}.B"].to(dev).float() @ fb[f"{n}.A"].to(dev).float()
                m = combine(ta, tb, a.method, a.density)
                n_w += ta.numel()
                n_conf += int((torch.sign(ta) * torch.sign(tb) < 0).sum())
                plain2 += float((ta + tb).pow(2).sum())
                diff2 += float((m - ta - tb).pow(2).sum())
                d2 += float(m.pow(2).sum())
                exact = sd[k].to(dev).float() + m
                sd[k] = exact.to(dtype).cpu()
                err2 += float((sd[k].to(dev).float() - exact).pow(2).sum())
                used.add(n)
            else:
                sd[k] = sd[k].to(dtype)
        save_file(sd, str(out / wf.name), metadata={"format": "pt"})
    if used != set(mods):
        raise SystemExit(f"{len(set(mods) - used)} changed modules not found in the base")
    print(json.dumps({"out": str(out), "method": a.method, "density": a.density if a.method == "ties" else None,
                      "modules": len(used), "sign_conflict_share": n_conf / n_w,
                      "norm_plain": plain2 ** 0.5, "norm_combined": d2 ** 0.5,
                      "distance_from_plain": (diff2 / plain2) ** 0.5, "rounding_error": (err2 / d2) ** 0.5}))


if __name__ == "__main__":
    main()
