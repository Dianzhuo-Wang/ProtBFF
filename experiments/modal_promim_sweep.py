"""Sweep ProMIM across MVA identity thresholds on Modal, one container per (threshold, fold).
Reuses the working promim-mva pipeline; 60% is already done (promim_mva.json), so this runs
30/40/50/80/90. The fold dir is selected per run via the MVA_FOLD_DIR env var (train script reads it).

  modal run experiments/modal_promim_sweep.py                 # full sweep (50 containers)
  modal run experiments/modal_promim_sweep.py::main --smoke   # 1 (thr,fold), 40 iters
"""
import modal

CODE = "/n/netscratch/shakhnovich_lab/Lab/jwang/promim_mva"
FOLDS_ALL = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/data/cross_validation_folds_mva"
THRESHOLDS = [80, 90]                                  # 30/40/50 done (promim_sweep_folds/), 60 in promim_mva.json

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", "numpy", "pandas", "scipy", "scikit-learn", "biopython",
        "tqdm", "easydict", "pyyaml", "torchmetrics", "matplotlib", "lmdb",
        "joblib", "tensorboard",
    )
    .add_local_dir(CODE, "/root/promim", copy=True)
    .add_local_dir(FOLDS_ALL, "/root/mva_folds_all", copy=True)
)
app = modal.App("promim-sweep", image=image)
vol = modal.Volume.from_name("promim-sweep-out", create_if_missing=True)


@app.function(gpu="H100", volumes={"/root/out": vol}, timeout=5 * 3600)
def train_fold(job):
    thr, fold, max_iters = job
    import os, re, subprocess, sys, glob
    os.chdir("/root/promim")
    os.environ["MVA_FOLD_DIR"] = f"/root/mva_folds_all/{thr}_percent"     # train script reads this
    base = open("configs/train/promim_ddg_skempi.yml").read()
    base = re.sub(r"max_iters:\s*\d+", f"max_iters: {max_iters}", base)
    base = re.sub(r"val_freq:\s*\d+", f"val_freq: {min(max_iters, 1000)}", base)
    cfg = f"configs/train/_modal_t{thr}_f{fold}.yml"
    open(cfg, "w").write(base)
    logdir = f"/root/out/thr{thr}/fold_{fold}"
    cmd = [sys.executable, "train_promim_skempi.py", "--config", cfg, "--num_cvfolds", "10",
           "--fold_only", str(fold), "--logdir", logdir, "--tag", f"t{thr}f{fold}",
           "--device", "cuda", "--num_workers", "4"]
    print(f"RUN thr={thr} fold={fold} (MVA_FOLD_DIR={os.environ['MVA_FOLD_DIR']})", flush=True)
    rc = subprocess.run(cmd).returncode
    vol.commit()
    hits = sorted(glob.glob(f"{logdir}/**/combined_all_folds_results.csv", recursive=True),
                  key=os.path.getmtime)
    csv = open(hits[-1]).read() if hits else None
    return thr, fold, rc, csv


@app.local_entrypoint()
def main(smoke: bool = False):
    import os, io, json, csv as _csv
    import numpy as np
    from scipy.stats import pearsonr

    if smoke:
        thr, fold, rc, txt = train_fold.remote((30, 0, 40))
        print(f"smoke thr{thr} fold{fold}: rc={rc} rows={0 if not txt else txt.count(chr(10)) - 1}")
        return

    base = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/tuning_v1/out"
    os.makedirs(f"{base}/promim_sweep_folds", exist_ok=True)
    jobs = [(thr, fold, 5000) for thr in THRESHOLDS for fold in range(10)]
    per_thr = {}
    for thr, fold, rc, txt in train_fold.map(jobs):
        n = 0 if not txt else txt.count(chr(10)) - 1
        print(f"thr{thr} fold{fold}: rc={rc} rows={n}", flush=True)
        if txt:
            open(f"{base}/promim_sweep_folds/thr{thr}_fold{fold}.csv", "w").write(txt)
            rows = list(_csv.DictReader(io.StringIO(txt)))
            y = np.array([float(r["ddG"]) for r in rows]); p = np.array([float(r["ddG_pred"]) for r in rows])
            if len(y) > 2:
                per_thr.setdefault(thr, []).append(pearsonr(y, p)[0])

    out = {str(thr): {"mean_r": float(np.mean(rs)), "sem_r": float(np.std(rs) / len(rs) ** 0.5),
                      "nfolds": len(rs)} for thr, rs in per_thr.items()}
    m = json.load(open(f"{base}/promim_mva.json")); fr = m.get("fold_r", [])   # reuse existing 60%
    if fr:
        out["60"] = {"mean_r": float(np.mean(fr)), "sem_r": float(np.std(fr) / len(fr) ** 0.5), "nfolds": len(fr)}
    json.dump(out, open(f"{base}/promim_mva_sweep.json", "w"), indent=2)
    print("\nwrote promim_mva_sweep.json:")
    for k in sorted(out, key=lambda x: -int(x)):
        print(f"  {k}%: {out[k]['mean_r']:.3f} +/- {out[k]['sem_r']:.3f}")
