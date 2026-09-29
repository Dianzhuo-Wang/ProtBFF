#!/usr/bin/env python
"""Collect ProMIM MVA-threshold sweep results -> tuning_v1/out/promim_mva_sweep.json.

Per-fold results live in
  promim_mva/logs_mva_sweep/thr{T}/<run>_t{T}f{F}/checkpoints/combined_all_folds_results.csv
(one CSV per (threshold, fold); columns complex,mutstr,num_muts,ddG,ddG_pred,fold).
Mean-of-folds Pearson per threshold, matching how every other Fig-2 competitor is scored.
60% is taken from the existing promim_mva.json (the reported 0.399) unless --recompute60.
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import pearsonr

LOGS = "/n/netscratch/shakhnovich_lab/Lab/jwang/promim_mva/logs_mva_sweep"
OUT = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/tuning_v1/out"


def fold_r(thr, fold):
    hits = glob.glob(f"{LOGS}/thr{thr}/*t{thr}f{fold}*/checkpoints/combined_all_folds_results.csv")
    if not hits:
        return None
    csv = max(hits, key=os.path.getmtime)  # newest, in case of re-runs
    # mutstr may contain commas (multi-mutants) -> parse the always-numeric last 3 cols
    # (..., ddG, ddG_pred, fold) from the right rather than by header name.
    y, p = [], []
    for ln in open(csv).read().splitlines()[1:]:
        f = ln.split(",")
        y.append(float(f[-3])); p.append(float(f[-2]))
    y, p = np.array(y), np.array(p)
    m = np.isfinite(y) & np.isfinite(p)
    return float(pearsonr(y[m], p[m])[0]) if m.sum() > 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default="30,40,50,60,80,90")
    ap.add_argument("--recompute60", action="store_true",
                    help="score 60%% from the sweep too, instead of reusing promim_mva.json")
    a = ap.parse_args()
    thrs = [int(x) for x in a.thresholds.split(",")]

    out = {}
    for thr in thrs:
        if thr == 60 and not a.recompute60:
            continue
        rs = [r for f in range(10) if (r := fold_r(thr, f)) is not None]
        if rs:
            out[str(thr)] = {"mean_r": float(np.mean(rs)),
                             "sem_r": float(np.std(rs) / len(rs) ** 0.5),
                             "nfolds": len(rs), "fold_r": rs}
        print(f"thr {thr}: {len(rs)}/10 folds, "
              f"mean_r={np.mean(rs):.4f}" if rs else f"thr {thr}: NO folds yet")

    if not a.recompute60:
        m = json.load(open(f"{OUT}/promim_mva.json"))
        fr = m.get("fold_r", [])
        if fr:
            out["60"] = {"mean_r": float(np.mean(fr)),
                         "sem_r": float(np.std(fr) / len(fr) ** 0.5),
                         "nfolds": len(fr), "fold_r": fr}

    json.dump(out, open(f"{OUT}/promim_mva_sweep.json", "w"), indent=2)
    print("\nwrote promim_mva_sweep.json:")
    for k in sorted(out, key=lambda x: -int(x)):
        print(f"  {k}%: {out[k]['mean_r']:.3f} +/- {out[k]['sem_r']:.3f}  (n={out[k]['nfolds']})")


if __name__ == "__main__":
    main()
