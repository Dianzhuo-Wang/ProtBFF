# Pooling-operator ablation

How the per-residue biophysical scores are pooled into a single fixed-length vector matters.
ProtBFF uses a **score-scaled max** over residues: each residue's embedding is scaled by its
biophysical score, and the per-dimension maximum is kept. Max-pooling keeps both the
most-stabilizing and the most-destabilizing per-residue change in each dimension, which the
score-weighted mean averages away and signed max-abs keeps only one side of.

We compared three pooling operators with everything else held fixed (the antisymmetric
readout, the same MVA-60 folds, 5 seeds). Numbers are **mean-of-folds Pearson correlation**.

| Pooling operator | ProSST | ESM-C |
|---|---|---|
| **Max** (used in ProtBFF) | **0.422** | **0.444** |
| Signed max-abs | 0.357 | 0.404 |
| Score-weighted mean | 0.293 | 0.336 |

Max-pooling wins for both encoders. This is Table S10 in the supplement.

- **Max** — the main ProtBFF caches (`merge_scores.py` default). Numbers taken from the
  `antisym` entry of `prosst_max.json` / `esmc_max.json` (these files also hold the readout
  ablation; the max-vs-pooling comparison uses `antisym`).
- **Signed max-abs** — per dimension, the residue with the largest absolute value, sign kept.
  This makes the reverse input the exact negation of the forward one, so the symmetric channel
  is zero. See `prosst_maxabs.json` / `esmc_maxabs.json`.
- **Score-weighted mean** — `sum_i score_i * diff_i / sum_i score_i`. See
  `prosst_wmean.json` / `esmc_wmean.json`.

Each JSON holds, per readout, the per-fold Pearson/Spearman arrays (`fold_r`, `fold_s`),
their mean and SEM (`mean_r`, `sem_r`, `mean_s`, `sem_s`), and the pooled-across-folds values.

## Reproduce

1. Rebuild the score caches with the alternate pooling operator:

   ```bash
   python data_pipeline/build_wpool_cache.py --merged_dir <merged_output> --pool maxabs \
     --out skempi_esmc_maxabs_cache.npz
   python data_pipeline/build_wpool_cache.py --merged_dir <merged_output> --pool mean \
     --out skempi_esmc_wmean_cache.npz
   ```

   (`--pool max` reproduces the default ProtBFF cache.)

2. Train and evaluate on the MVA-60 folds:

   ```bash
   modal run experiments/modal_maxabs_arch.py      # ESM-C, signed max-abs
   modal run experiments/modal_prosst_maxabs.py    # ProSST, signed max-abs
   modal run experiments/modal_wpool_arch.py       # both encoders, score-weighted mean
   modal run experiments/modal_protbff_arch.py     # both encoders, default max (the main benchmark)
   ```

   or run `tuning_v1/protbff_arch.py` directly on a cache with
   `--variants antisym --folds_dir data/cross_validation_folds_mva/60_percent`.
