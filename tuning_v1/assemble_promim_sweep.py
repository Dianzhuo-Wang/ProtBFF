#!/usr/bin/env python
"""Assemble the full ProMIM MVA identity sweep from the per-fold CSVs the Modal
sweep dumps into tuning_v1/out/promim_sweep_folds/ (thr{T}_fold{F}.csv), plus
thr60 from the earlier promim_mva.json. Writes tuning_v1/out/promim_mva_sweep.json.

mutstr may contain commas (multi-mutants), so ddG/ddG_pred are read from the
always-numeric last-3 columns. Metrics match every other Fig-2 model:
mean-of-folds Pearson/Spearman and per-fold (ddG>0) AUROC, averaged.
"""
import glob, json, os, numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

OUT = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/tuning_v1/out"
D = f"{OUT}/promim_sweep_folds"


def fold_vals(csv):
    y, p = [], []
    for ln in open(csv).read().splitlines()[1:]:
        f = ln.split(","); y.append(float(f[-3])); p.append(float(f[-2]))
    y, p = np.array(y), np.array(p); m = np.isfinite(y) & np.isfinite(p)
    return y[m], p[m]


def main():
    out = {}
    thrs = sorted({int(os.path.basename(c).split("_")[0][3:])
                   for c in glob.glob(f"{D}/thr*_fold*.csv")})
    for thr in thrs:
        csvs = sorted(glob.glob(f"{D}/thr{thr}_fold*.csv"))
        frs, fss, fas = [], [], []
        for c in csvs:
            y, p = fold_vals(c)
            if len(y) < 3:
                continue
            frs.append(pearsonr(y, p)[0]); fss.append(spearmanr(y, p)[0])
            lab = (y > 0).astype(int)
            fas.append(roc_auc_score(lab, p) if len(np.unique(lab)) > 1 else np.nan)
        out[str(thr)] = dict(mean_r=float(np.nanmean(frs)), mean_s=float(np.nanmean(fss)),
                             mean_auroc=float(np.nanmean(fas)),
                             sem_r=float(np.nanstd(frs) / len(frs) ** 0.5),
                             nfolds=len(frs), fold_r=[float(x) for x in frs])

    # thr60 from the standalone run
    p60 = f"{OUT}/promim_mva.json"
    if "60" not in out and os.path.exists(p60):
        m = json.load(open(p60)); fr = m.get("fold_r", [])
        if fr:
            out["60"] = dict(mean_r=float(np.mean(fr)), mean_s=float(m.get("mean_s", np.nan)),
                             mean_auroc=float(m.get("mean_auroc", np.nan)),
                             sem_r=float(np.std(fr) / len(fr) ** 0.5), nfolds=len(fr),
                             fold_r=[float(x) for x in fr])

    json.dump(out, open(f"{OUT}/promim_mva_sweep.json", "w"), indent=2)
    print("wrote promim_mva_sweep.json")
    for k in sorted(out, key=lambda x: int(x)):
        print(f"  {k}%: meanP {out[k]['mean_r']:.4f} +/- {out[k]['sem_r']:.4f}  "
              f"AUROC {out[k]['mean_auroc']:.4f}  (n={out[k]['nfolds']})")


if __name__ == "__main__":
    main()
