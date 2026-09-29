# ProtBFF: Biophysically Grounded Deep Learning for Protein–Protein ΔΔG Prediction

Code and homology-controlled data splits for the paper
**"Biophysically Grounded Deep Learning Improves Protein–Protein ΔΔG Prediction"**
(Feldman, Wang, Maechler, Shakhnovich).

ProtBFF is an encoder-agnostic module that injects five residue-level biophysical
descriptors (interface propensity, burial, dihedral change, SASA, lDDT) into
pretrained protein-language-model embeddings by per-residue multiplicative scaling,
then combines them with cross-embedding attention.

## Repository layout

```
data_pipeline/     biophysical-score computation, structure tokenization, embedding,
                   and cache assembly (merge_scores.py). prosst/ holds the ProSST
                   structure-quantizer code (weights not vendored, see below).
tuning_v1/         the attention model (protbff_arch.py), training/benchmark drivers,
                   metric extraction, and figure-generation scripts.
experiments/       Modal (serverless GPU) launchers for the benchmarks and sweeps.
data/
  cross_validation_folds_mva/    homology-controlled MVA (mutated-versus-all) splits,
                                 30–90% chain-identity thresholds. Each threshold has
                                 clusters.tsv, seq_edges.tsv, folds_XXpct.csv, and
                                 fold_{1..10}/{train,test}_complex_ids.txt.
  cross_validation_folds_final/  CD-HIT baseline splits (secondary reference).
  SKEMPI2_filtered_final.csv     the 335-complex / 6,631-mutation working set.
```

## The MVA-60 split

The primary evaluation uses **MVA-60**: complexes are linked by single-linkage
clustering whenever a mutated chain of one shares ≥60% sequence identity with any
chain of another (all-versus-all MMseqs2 chain comparison), and whole clusters are
assigned to folds so no cluster is split between train and test. This controls the
homology leakage that inflates standard SKEMPI2 benchmarks.

## Data not included here (large / third-party / regenerable)

- **FoldX-relaxed mutant structures** and **wildtype PDBs**: regenerate with FoldX
  (`data_pipeline`), or available on request.
- **Per-residue PLM embeddings and score caches** (`*.npz`, `*.pt`): regenerate with
  the pipeline; the encoders (ProSST, ESM2/ESM3/ESM-C, SaProt) are downloaded from
  their public releases.
- **ProSST structure quantizer** weights (`data_pipeline/prosst/structure/static/*.joblib`):
  obtain from the ProSST release (AI4Protein/ProSST).

## Reproducing the benchmark (outline)

1. Compute the five biophysical scores per complex (`data_pipeline/calculate_all_scores.py`).
2. Generate per-residue embeddings for wildtype and FoldX mutant structures.
3. Assemble a score cache with `data_pipeline/merge_scores.py`.
4. Run the model on the MVA-60 folds with `tuning_v1/protbff_arch.py`
   (`--variants antisym,dual`, `--folds_dir data/cross_validation_folds_mva/60_percent`).

Paths in some scripts are hardcoded to the development cluster; adjust for your setup.
