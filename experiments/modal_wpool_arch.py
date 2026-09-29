"""Benchmark the score-weighted-MEAN pooling variant on MVA-60, with the appropriate
single-branch head (plus dual for contrast).

Rationale: weighted-mean pooling makes X_r = -X_f exactly, so the paired antisym/dual head
is degenerate (its symmetric branch sees ~nothing at the input). The honest head for a
weighted-mean pool is a single-branch regression on the pooled forward representation.
This runs `single` (the proper head) and `dual` (to show the paired structure adds nothing)
on both encoders, and protbff_arch also emits a ridge[D|S] reference (expected: S dead).

Compare to the max-pool baselines: ProSST antisym 0.422 / dual 0.451; ESM-C 0.444 / 0.469.

  modal run experiments/modal_wpool_arch.py
"""
import modal

BASE = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF"
CACHE_PROSST = f"{BASE}/model_benchmarking/score_caches/skempi_prosst_wmean_cache.npz"
CACHE_ESMC   = f"{BASE}/model_benchmarking/score_caches/skempi_esmc_wmean_cache.npz"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "scikit-learn")
    .add_local_dir(f"{BASE}/tuning_v1", "/root/tuning_v1", copy=True)
    .add_local_dir(f"{BASE}/data/cross_validation_folds_mva/60_percent", "/root/folds", copy=True)
    .add_local_file(CACHE_PROSST, "/root/prosst_wmean.npz", copy=True)
    .add_local_file(CACHE_ESMC, "/root/esmc_wmean.npz", copy=True)
)
app = modal.App("protbff-wpool", image=image)
vol = modal.Volume.from_name("protbff-arch-out", create_if_missing=True)


@app.function(gpu="A100", volumes={"/root/out": vol}, timeout=6 * 3600)
def run(cache: str, embed_dim: int, out: str, variants: str = "single,dual", seeds: int = 5):
    import subprocess, sys, os
    cmd = [sys.executable, "/root/tuning_v1/protbff_arch.py",
           "--cache", cache, "--folds_dir", "/root/folds",
           "--clusters", "/root/folds/clusters.tsv", "--variants", variants,
           "--embed_dim", str(embed_dim), "--seeds", str(seeds), "--out", f"/root/out/{out}"]
    print("RUN:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    vol.commit()
    return rc, (open(f"/root/out/{out}").read() if os.path.exists(f"/root/out/{out}") else None)


@app.local_entrypoint()
def main():
    jobs = [("/root/prosst_wmean.npz", 768, "arch_prosst_wmean.json"),
            ("/root/esmc_wmean.npz", 1152, "arch_esmc_wmean.json")]
    handles = [(out, run.spawn(cache=c, embed_dim=ed, out=out)) for c, ed, out in jobs]
    for out, h in handles:
        rc, txt = h.get()
        print("returncode", rc, out, flush=True)
        if txt:
            open(f"{BASE}/tuning_v1/out/{out}", "w").write(txt)
            print("saved", out, flush=True)
