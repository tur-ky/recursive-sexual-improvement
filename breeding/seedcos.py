"""How alike are specialists of the same field trained from different seeds?
Cosine between full weight changes, and how often two changes push a weight the same direction (sign agreement).

python seedcos.py > runs/p1/seed_cosine.txt      (needs a CUDA GPU; about a minute)
"""
import itertools, sys
from pathlib import Path
import torch
from safetensors.torch import load_file

D = Path(__file__).resolve().parent / "deltas" / "p1"
NAMES = ["mult_mlp", "s2_mult_mlp", "s3_mult_mlp", "tm_mult_mlp",
         "div_mlp", "s2_div_mlp", "s3_div_mlp", "tm_div_mlp"]


def factors(n):
    t = load_file(str(D / n / "delta.safetensors"))
    return {k[:-2]: (t[k[:-2] + ".A"].float().cuda(), t[k[:-2] + ".B"].float().cuda()) for k in t if k.endswith(".A")}


F = {n: factors(n) for n in NAMES}
mods = sorted(F[NAMES[0]])
print("modules:", len(mods), file=sys.stderr)

# cache full deltas per module per model would be too big for all at once; loop by module
dots = {(a, b): 0.0 for a in NAMES for b in NAMES}
agree = {(a, b): [0, 0] for a, b in itertools.combinations(NAMES, 2)}       # all entries
agree_top = {(a, b): [0, 0] for a, b in itertools.combinations(NAMES, 2)}   # entries in both models' top 10% by size
for m in mods:
    full = {n: (F[n][m][1] @ F[n][m][0]) for n in NAMES}
    top = {}
    for n, d in full.items():
        flat = d.abs().flatten()
        k = int(flat.numel() * 0.9)
        thr = flat.kthvalue(k).values if flat.numel() < 16_000_000 else flat[torch.randperm(flat.numel(), device=flat.device)[:2_000_000]].kthvalue(int(2_000_000 * 0.9)).values
        top[n] = d.abs() >= thr
    for a in NAMES:
        for b in NAMES:
            dots[(a, b)] += float((full[a].double() * full[b].double()).sum())
    for a, b in agree:
        s = (torch.sign(full[a]) == torch.sign(full[b]))
        agree[(a, b)][0] += int(s.sum()); agree[(a, b)][1] += s.numel()
        both = top[a] & top[b]
        agree_top[(a, b)][0] += int((s & both).sum()); agree_top[(a, b)][1] += int(both.sum())
    del full, top
    torch.cuda.empty_cache()

norm = {n: dots[(n, n)] ** 0.5 for n in NAMES}
short = {n: n.replace("_mlp", "") for n in NAMES}
print("cosine")
print("            " + " ".join(f"{short[b]:>9}" for b in NAMES))
for a in NAMES:
    print(f"{short[a]:>11} " + " ".join(f"{dots[(a, b)] / (norm[a] * norm[b]):9.3f}" for b in NAMES))
print("\nsize (Frobenius norm of the whole change):", {short[n]: round(norm[n], 2) for n in NAMES})
print("\nsign agreement: all weights | weights both models changed most (top 10% each), with count of such weights")
for a, b in agree:
    x, y = agree[(a, b)]; p, q = agree_top[(a, b)]
    print(f"{short[a]:>9} vs {short[b]:<9} {x / y:6.3f} | {p / max(q, 1):6.3f} ({q})")
