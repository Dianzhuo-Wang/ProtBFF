#!/usr/bin/env python
"""Rebuild a ProtBFF score cache with a DIFFERENT residue-pooling operator, to test
the score-weighted-average pooling proposed instead of the current score-scaled max-pool.

Replicates merge_scores.py exactly (same 5 scores, same block order
[interface, burial, 1-lddt, sasa, dihedral], same lddt inversion, same id extraction,
same skip rules) EXCEPT the pooling function:

  max  (current): max_i (score_i * diff_i)
  mean (proposed): sum_i (score_i * diff_i) / sum_i score_i     <-- score-weighted average
  sum            : sum_i (score_i * diff_i)

Note: for mean/sum (linear), if the per-residue reverse diff is the negation of the
forward diff, then X_r = -X_f exactly, so the input-level symmetric part (X_f+X_r) is 0.
max is nonlinear so X_f+X_r = range != 0. This script lets us measure the consequence.

  python build_wpool_cache.py --merged_dir <dir> --pool mean --out <cache.npz>
"""
import argparse, glob, os
import numpy as np


def pool(score_vec, diff_mat, how):
    w = diff_mat * score_vec[:, None]              # (L, D)
    if how == "max":
        return np.max(w, axis=0)
    if how == "sum":
        return np.sum(w, axis=0)
    if how == "mean":                               # score-weighted average
        return np.sum(w, axis=0) / (np.sum(score_vec) + 1e-8)
    raise ValueError(how)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", required=True)
    ap.add_argument("--pool", default="mean", choices=["max", "mean", "sum"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.merged_dir, "*.npz")))
    req = ["ddG", "lddt", "sasa", "dihedral", "burial", "interface"]
    Xf, Xr, y, ild, ids = [], [], [], [], []
    skipped = 0
    for fn in files:
        d = np.load(fn, allow_pickle=True)
        if any(k not in d for k in req) or "Xf" not in d or "Xr" not in d:
            skipped += 1; continue
        f1d = lambda k: np.asarray(d[k], float).reshape(-1)
        interface, burial, sasa, dihedral, lddt = (f1d("interface"), f1d("burial"),
                                                   f1d("sasa"), f1d("dihedral"), f1d("lddt"))
        # lddt NaN handling then inversion (as in merge_scores.py)
        nm = np.isnan(lddt)
        if nm.any():
            m = np.nanmean(lddt); lddt[nm] = 0.5 if np.isnan(m) else m
        lddt_inv = 1.0 - lddt
        diff_fwd = np.asarray(d["Xf"], np.float32); diff_rev = np.asarray(d["Xr"], np.float32)
        L = diff_fwd.shape[0]
        if any(v.shape[0] != L for v in (interface, burial, sasa, dihedral, lddt_inv)):
            skipped += 1; continue
        blocks_f, blocks_r = [], []
        for s in (interface, burial, lddt_inv, sasa, dihedral):   # merge_scores order
            blocks_f.append(pool(s, diff_fwd, a.pool)); blocks_r.append(pool(s, diff_rev, a.pool))
        Xf.append(np.concatenate(blocks_f)); Xr.append(np.concatenate(blocks_r))
        y.append(float(np.asarray(d["ddG"]).reshape(-1)[0]))
        iv = d["ilddt"] if "ilddt" in d else np.nan
        ild.append(float(np.asarray(iv).reshape(-1)[0]) if not np.isscalar(iv) else float(iv))
        pid = str(d["protein_id"]) if "protein_id" in d else os.path.basename(fn)
        ids.append(pid.split("_", 1)[1] if "_" in pid else pid)
    Xf, Xr = np.vstack(Xf).astype(np.float32), np.vstack(Xr).astype(np.float32)
    y = np.array(y, np.float32); ild = np.array(ild, np.float32); ids = np.array(ids)
    np.savez(a.out, Xf=Xf, Xr=Xr, y=y, ilddt=ild, ids=ids, mutations=None)
    S = (Xf + Xr) / 2.0
    print(f"pool={a.pool}  wrote {a.out}  Xf{Xf.shape}  skipped={skipped}")
    print(f"  input-level symmetric mag: ||S||/||Xf|| = {np.linalg.norm(S)/np.linalg.norm(Xf):.4f} "
          f"(near 0 => X_r=-X_f, no symmetric signal at input)")


if __name__ == "__main__":
    main()
