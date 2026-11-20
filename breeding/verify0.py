"""Phase 0 verifier test: in a field, which non-oracle filters pick out correct answers better than chance?

python verify0.py runs/samples.jsonl [--peer runs/peer.jsonl] [--feats runs/feats.pt] [--transfer ../rsi/probes/g0_big.pt]

  random    keep everything (precision = base rate)
  majority  keep a sample if it agrees with the plurality of the model's own samples for that problem (>= 2 votes)
  peer      keep a sample if it agrees with a field-trained peer's greedy answer to the same problem
  probe     linear probe on the sampling model's own activations, fitted on *other problems'* labels (grouped CV)
  transfer  the multiplication probe from 19 November applied as-is to this field (no labels from this field at all)
Prints precision (fraction of kept answers that are correct), recall (fraction of correct answers kept), and the
Phase 0 criterion: precision at least 15 points above random.
"""
import json, sys, collections
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
opt = dict(zip(sys.argv[2::2], sys.argv[3::2]))
y = np.array([int(r["ok"]) for r in rows])
lv = np.array([f"{r['field']}{r['level']}" for r in rows])
filters = {"random": np.ones(len(rows), bool)}

votes = collections.defaultdict(collections.Counter)
for r in rows:
    if r["ans"] is not None:
        votes[r["pid"]][r["ans"]] += 1
filters["majority"] = np.array([r["ans"] is not None and votes[r["pid"]].most_common(1)[0][0] == r["ans"]
                                and votes[r["pid"]].most_common(1)[0][1] >= 2 for r in rows])

if "--peer" in opt:
    peer = {json.loads(l)["pid"]: json.loads(l)["ans"] for l in open(opt["--peer"]) if l.strip()}
    filters["peer"] = np.array([r["ans"] is not None and peer.get(r["pid"]) == r["ans"] for r in rows])

if "--feats" in opt:
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    F = torch.load(opt["--feats"])
    H = F["h"].float().numpy()
    assert (F["ok"].numpy().astype(int) == y).all(), "feature/row order mismatch"
    groups = np.array([r["pid"] for r in rows])
    best = (-1, None, None)
    for L in range(8, H.shape[1]):
        X = H[:, L, :]
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        p = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups):
            p[te] = LogisticRegression(C=0.05, max_iter=3000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        prec = y[p > 0.5].mean() if (p > 0.5).any() else 0
        if prec > best[0]:
            best = (prec, L, p)
    print(f"in-field probe: best layer {best[1]}")
    filters["probe"] = best[2] > 0.5
    if "--transfer" in opt:
        P = torch.load(opt["--transfer"])
        L = P["layer"]
        s = H[:, L, :] @ P["w"].numpy() + P["b"]  # saved weights and intercept are in raw activation space
        filters["transfer"] = s > np.median(s) if (s > 0).mean() in (0.0, 1.0) else s > 0
        print(f"transfer probe: layer {L}, fraction scored positive {(s > 0).mean():.2f}"
              + (" (all one side, used the median instead)" if (s > 0).mean() in (0.0, 1.0) else ""))

filters["oracle"] = y.astype(bool)
base = y.mean()
print(f"n={len(y)} samples, base rate (random precision) {base:.3f}")
print(f"{'filter':9s} {'kept':>5s} {'precision':>9s} {'recall':>7s}  vs random")
for name, m in filters.items():
    prec = y[m].mean() if m.any() else float("nan")
    rec = (m & (y == 1)).sum() / max(1, y.sum())
    flag = "PASS" if name not in ("random", "oracle") and prec - base >= 0.15 else ""
    print(f"{name:9s} {m.sum():5d} {prec:9.3f} {rec:7.3f}  {prec - base:+.3f} {flag}")
