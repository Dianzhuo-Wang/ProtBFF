#!/usr/bin/env python
"""SI figure: the CD-HIT (leaky) identity sweep, moved out of main Fig 2. Shows that CD-HIT
clustering on concatenated chains does NOT deflate performance the way the honest MVA split does
-- ESM-C + ProtBFF, ESM-C bare, and both RDE competitors stay comparatively high across CD-HIT
thresholds. RDE-Network CD-HIT points come from the existing logs_skempi_grouped runs (30k iters,
mean-of-folds Pearson); RDE-Linear from rde_linear_sweep2.json (cdhit_*)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle_v2 import (apply_style, save, ENCODER_COLOR, ENCODER_LABEL, ENCODER_MARKER)
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
COMPETITOR = "#6b6a66"; RDELIN = "#b8873f"


def _load(fn):
    p = os.path.join(OUT, fn)
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    apply_style(base=18)
    order = ["100", "95", "80", "60", "40"]
    xlabels = ["original\n(100)", "95", "80", "60", "40"]
    fig, ax = plt.subplots(figsize=(9.5, 7.2))

    d = _load("thr_perf_cdhit_esmc.json"); dec = _load("decay_encoders.json")
    col, mk = ENCODER_COLOR["esmc"], ENCODER_MARKER["esmc"]
    xs, ys, es = [], [], []
    for i, t in enumerate(order):
        if t == "100" and "esmc" in dec:
            v = dec["esmc"]["nosplit"]; ys.append(v["protbff_r"]); es.append(v.get("protbff_r_sem", 0)); xs.append(i)
        elif t in d:
            ys.append(d[t]["protbff_r"]); es.append(d[t].get("protbff_r_sem", 0)); xs.append(i)
    ax.errorbar(xs, ys, yerr=es, fmt="-" + mk, color=col, capsize=4, markersize=11, lw=2.6,
                label="ESM-C + ProtBFF", zorder=6)

    bare = _load("bare_esmc_cdhit.json"); xs, ys, es = [], [], []
    for i, t in enumerate(order):
        if t in bare:
            ys.append(bare[t]["bare_r"]); es.append(bare[t].get("bare_r_sem", 0)); xs.append(i)
    if xs:
        ax.errorbar(xs, ys, yerr=es, fmt="--o", color=col, markerfacecolor="white",
                    markeredgecolor=col, markeredgewidth=2.0, lw=2.0, capsize=4, markersize=10,
                    label="ESM-C (bare)", zorder=4)

    rn = _load("rde_network_cdhit_sweep.json"); xs, ys, es = [], [], []
    for i, t in enumerate(order):
        if t in rn:
            xs.append(i); ys.append(rn[t]["mean_r"]); es.append(rn[t].get("sem_r", 0))
    if xs:
        ax.errorbar(xs, ys, yerr=es, fmt=":X", color=COMPETITOR, capsize=4, markersize=12, lw=2.2,
                    label="RDE-Network (competitor)", zorder=5)

    ax.set_xticks(range(len(order))); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Pearson correlation (mean-of-folds)")
    ax.set_xlabel("CD-HIT sequence-identity threshold (%)")
    ax.set_ylim(0.08, 0.68)
    ax.set_title("CD-HIT (leaky) identity sweep")
    ax.legend(loc="lower left", fontsize=13)
    save(fig, "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/figure_previews/si_cdhit_sweep")


if __name__ == "__main__":
    main()
