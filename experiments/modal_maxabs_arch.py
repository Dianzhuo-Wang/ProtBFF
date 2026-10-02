"""Benchmark signed max-abs pooling + ProtBFF on ESM-C, MVA-60.

Signed max-abs (per dim, the element of largest |value|, sign kept) makes X_r = -X_f exactly,
so the symmetric channel is zero and the antisymmetric readout is exact/principled (no D/S story).
Expectation: dual == antisym (S=0), performance near the plain-max antisym baseline (0.444).

Compare to plain-max baselines: ESM-C antisym 0.444 / dual 0.469.

  modal run experiments/modal_maxabs_arch.py
"""
import modal

BASE = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF"
CACHE = f"{BASE}/model_benchmarking/score_caches/skempi_esmc_maxabs_cache.npz"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "scikit-learn")
    .add_local_dir(f"{BASE}/tuning_v1", "/root/tuning_v1", copy=True)
    .add_local_dir(f"{BASE}/data/cross_validation_folds_mva/60_percent", "/root/folds", copy=True)
    .add_local_file(CACHE, "/root/cache.npz", copy=True)
)
app = modal.App("protbff-maxabs", image=image)
vol = modal.Volume.from_name("protbff-arch-out", create_if_missing=True)


@app.function(gpu="A100", volumes={"/root/out": vol}, timeout=6 * 3600)
def run(embed_dim: int = 1152, variants: str = "antisym,dual", seeds: int = 5,
        out: str = "arch_esmc_maxabs.json"):
    import subprocess, sys, os
    cmd = [sys.executable, "/root/tuning_v1/protbff_arch.py",
           "--cache", "/root/cache.npz", "--folds_dir", "/root/folds",
           "--clusters", "/root/folds/clusters.tsv", "--variants", variants,
           "--embed_dim", str(embed_dim), "--seeds", str(seeds), "--out", f"/root/out/{out}"]
    print("RUN:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    vol.commit()
    return rc, (open(f"/root/out/{out}").read() if os.path.exists(f"/root/out/{out}") else None)


@app.local_entrypoint()
def main():
    rc, txt = run.remote(embed_dim=1152, variants="antisym,dual", seeds=5, out="arch_esmc_maxabs.json")
    print("returncode", rc)
    if txt:
        open(f"{BASE}/tuning_v1/out/arch_esmc_maxabs.json", "w").write(txt)
        print("saved arch_esmc_maxabs.json")
