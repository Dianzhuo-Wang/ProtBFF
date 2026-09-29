"""Genuine out-of-fold ESM-C dual predictions on Modal, to compute a REAL ESM-C AUROC
(S5 Table currently shows 0.668, identical to ProSST and never computed from raw ESM-C preds).

Mirrors modal_protbff_arch.py but runs oof_preds.py, which dumps per-entry seed-ensembled
test predictions (idx, fold, y, pred). Sanity check on return: mean-of-folds Pearson must
reproduce the ESM-C dual value from arch_esmc.json (~0.469); then AUROC is computed locally
with the same (ddG>0) per-fold roc_auc definition used for every other model.

  modal run experiments/modal_oof_esmc.py
"""
import modal

BASE = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF"
CACHE = f"{BASE}/model_benchmarking/score_caches/skempi_esmc_score_cache.npz"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "scikit-learn")
    .add_local_dir(f"{BASE}/tuning_v1", "/root/tuning_v1", copy=True)
    .add_local_dir(f"{BASE}/data/cross_validation_folds_mva/60_percent", "/root/folds", copy=True)
    .add_local_file(CACHE, "/root/cache.npz", copy=True)
)

app = modal.App("protbff-oof-esmc", image=image)
vol = modal.Volume.from_name("protbff-arch-out", create_if_missing=True)


@app.function(gpu="A100", volumes={"/root/out": vol}, timeout=6 * 3600)
def run(embed_dim: int = 1152, readout: str = "dual", seeds: int = 5, out: str = "oof_dual_esmc.csv"):
    import subprocess, sys, os
    cmd = [sys.executable, "/root/tuning_v1/oof_preds.py",
           "--cache", "/root/cache.npz", "--folds_dir", "/root/folds",
           "--clusters", "/root/folds/clusters.tsv", "--readout", readout,
           "--embed_dim", str(embed_dim), "--seeds", str(seeds),
           "--out", f"/root/out/{out}"]
    print("RUN:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    vol.commit()
    txt = open(f"/root/out/{out}").read() if os.path.exists(f"/root/out/{out}") else None
    return rc, txt


@app.local_entrypoint()
def main():
    rc, txt = run.remote(embed_dim=1152, readout="dual", seeds=5, out="oof_dual_esmc.csv")
    print("returncode", rc)
    if txt:
        open(f"{BASE}/tuning_v1/out/oof_dual_esmc.csv", "w").write(txt)
        print(f"saved oof_dual_esmc.csv locally ({len(txt)} bytes)")
