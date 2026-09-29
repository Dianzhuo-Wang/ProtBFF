#!/usr/bin/env python
"""RDE-Linear across MVA + CD-HIT thresholds. RDE's entropy features are frozen
(split-independent), so unlike the trained RDE-Network they can be re-fit per fold at any
threshold. 12 features = {ligand, ligand-neighbour, receptor} x {unbound,bound} x {wt,mt}
conformational entropies; ridge per fold, alpha val-selected (same protocol as the encoder
baselines). Sanity target: MVA-60 mean Pearson ~ 0.148 (matches rde_linear_mva.json)."""
import os, sys, pickle, json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protbff_train import load_clusters, load_folds, cluster_aware_val

ENT = "tuning_v1/out/rde_sweep/entropy.pkl"


def features():
    d = pickle.load(open(ENT, "rb"))
    ids, y, X = [], [], []
    for k in sorted(d):
        e = d[k]
        def agg(lst, keys):
            if not lst:
                return [0.0] * len(keys)
            return [float(np.sum([m.get(kk, 0.0) for m in lst])) for kk in keys]
        lig = agg(e["mutations"], ["H_ub_wt", "H_ub_mt", "H_b_wt", "H_b_mt"])
        nbr = agg(e["lignbrs"],   ["H_ub_wt", "H_ub_mt", "H_b_wt", "H_b_mt"])
        rc = e["receptors"]
        rec = [float(np.sum([m.get("H_ub", 0.0) for m in rc])),
               float(np.sum([m.get("H_ub", 0.0) for m in rc])),
               float(np.sum([m.get("H_b_wt", 0.0) for m in rc])),
               float(np.sum([m.get("H_b_mt", 0.0) for m in rc]))] if rc else [0.0] * 4
        feat = lig + nbr + rec; dg = float(e["ddG"])
        if not np.isfinite(dg) or not np.all(np.isfinite(feat)):
            continue
        X.append(feat); y.append(dg); ids.append(e["pdbcode"])
    return np.array(X), np.array(y), np.array(ids)


def sweep_one(X, y, ids, folds_dir, clusters):
    cmap = load_clusters(clusters); folds = load_folds(folds_dir, ids)
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
    frs, fss = [], []
    for fi, (tr_all, te) in enumerate(folds):
        if len(te) == 0 or len(tr_all) == 0:
            continue
        tr, val = cluster_aware_val(tr_all, ids, cmap, 0.10, 42 + fi)
        best_a, best_v = 1000.0, -np.inf
        for a in (1, 10, 100, 300, 1000, 3000):
            m = Ridge(alpha=a).fit(Xs[tr], y[tr]); vp = m.predict(Xs[val])
            v = pearsonr(y[val], vp)[0] if np.std(vp) > 1e-9 else -np.inf
            if v > best_v: best_v, best_a = v, a
        idx = np.concatenate([tr, val])
        m = Ridge(alpha=best_a).fit(Xs[idx], y[idx]); p = m.predict(Xs[te])
        frs.append(pearsonr(y[te], p)[0]); fss.append(spearmanr(y[te], p)[0])
    return dict(mean_r=float(np.mean(frs)), mean_s=float(np.mean(fss)),
                sem_r=float(np.std(frs) / len(frs) ** 0.5), n_folds=len(frs))


def main():
    X, y, ids = features()
    print(f"N={len(y)} feat={X.shape[1]} unique_pdb={len(set(ids))}", flush=True)
    out = {}
    for split, root, thrs in [("mva", "data/cross_validation_folds_mva", [90, 80, 60, 50, 40, 30]),
                              ("cdhit", "data/cross_validation_folds_final", [100, 95, 80, 60, 40])]:
        for t in thrs:
            fd = f"{root}/{t}_percent"
            cl = f"{fd}/clusters.tsv" if os.path.exists(f"{fd}/clusters.tsv") else \
                 "data/cross_validation_folds_mva/60_percent/clusters.tsv"
            r = sweep_one(X, y, ids, fd, cl)
            out[f"{split}_{t}"] = r
            print(f"  {split} {t}%: meanP={r['mean_r']:.3f}±{r['sem_r']:.3f} (n={r['n_folds']})", flush=True)
    json.dump(out, open("tuning_v1/out/rde_sweep/rde_linear_sweep.json", "w"), indent=2)
    print("SANITY mva 60% should be ~0.148 -> got", round(out["mva_60"]["mean_r"], 3))


if __name__ == "__main__":
    main()
