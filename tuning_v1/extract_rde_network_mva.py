#!/usr/bin/env python
"""Parse the RDE-Network MVA sweep runs (logs_mva/.../checkpoints/results_30000.csv) into
tuning_v1/out/rde_network_mva_sweep.json = {thr: {mean_r, sem_r}} (mean-of-folds Pearson),
using the MVA fold assignment at each threshold. plot_figure2_final.py reads this to draw the
full RDE-Network decay curve in Panel A."""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import pearsonr

LOGS = "/n/netscratch/shakhnovich_lab/Lab/jwang/rde_linear_mva/logs_mva"
MVA = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/data/cross_validation_folds_mva"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "rde_network_mva_sweep.json")
THRS = [30, 40, 50, 60, 80, 90]
IT = 30000


def foldmap(thr):
    m = {}
    base = f"{MVA}/{thr}_percent"
    for k in range(1, 11):
        f = f"{base}/fold_{k}/test_complex_ids.txt"
        if not os.path.exists(f):
            return None
        for line in open(f):
            c = line.strip()
            if c:
                m[c.split('_')[-1]] = k          # id "<idx>_<PDB>" -> PDB -> fold
    return m


def results_csv(thr):
    # the sweep tags dirs mva{thr}; results land in <dir>/checkpoints/results_{IT}.csv
    hits = glob.glob(f"{LOGS}/*mva{thr}*/checkpoints/results_{IT}.csv") \
        or glob.glob(f"{LOGS}/*-{thr})*/checkpoints/results_{IT}.csv")
    return sorted(hits, key=os.path.getmtime)[-1] if hits else None


def main():
    out = {}
    for thr in THRS:
        f = results_csv(thr)
        fm = foldmap(thr)
        if f is None:
            print(f"  {thr}%: results_{IT}.csv not found yet"); continue
        if fm is None:
            print(f"  {thr}%: MVA fold dir missing"); continue
        d = pd.read_csv(f)
        d["pdb"] = d["complex"].astype(str).str.split("_").str[0]
        d["fold"] = d["pdb"].map(fm)
        rs = [pearsonr(g.ddG, g.ddG_pred)[0] for _, g in d.groupby("fold") if len(g) > 2]
        mean_r, sem_r = float(np.mean(rs)), float(np.std(rs) / len(rs) ** 0.5)
        out[str(thr)] = {"mean_r": mean_r, "sem_r": sem_r, "nfolds": len(rs)}
        print(f"  {thr}%: mean-of-folds Pearson = {mean_r:.3f} ± {sem_r:.3f}  (nfolds={len(rs)}, n={len(d)})")
    if out:
        json.dump(out, open(OUT, "w"), indent=2)
        print("wrote", OUT)
        if "60" in out:
            print(f"SANITY: MVA-60 = {out['60']['mean_r']:.3f} (expect ~0.393 vs Antoine)")


if __name__ == "__main__":
    main()
