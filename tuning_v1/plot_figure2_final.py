#!/usr/bin/env python
"""Figure 2 (house style v2), 2-panel honest-split story:
  A = original(no-alignment) -> MVA identity sweep (ESM-C +/- ProtBFF; RDE-Network + RDE-Linear
      competitors decline the same way).
  C = performance vs dataset (leave-one-experimental-method-out), full width, single axis.
The CD-HIT (leaky) sweep now lives in the SI (plot_si_cdhit.py). Larger, legible fonts throughout;
panel C shows each method's % of the dataset clearly under its bar."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle_v2 import (apply_style, panel_label, save, ENCODER_COLOR, ENCODER_LABEL,
                         ENCODER_MARKER, MUTED, INK, INK2)
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
LEAKY_WASH = (0.85, 0.30, 0.26, 0.06)
ENCS = ["esmc"]                        # ESM-C only; trained non-PLM competitors overlaid
COMPETITOR = "#6b6a66"                 # RDE-Network (specialized non-PLM deep predictor)
DDAFF = "#8452c9"                      # DDAffinity (specialized non-PLM deep predictor)
RDE_LEAKY = 0.480                      # RDE-Network leaky benchmark (original split)
DDAFF_LEAKY, DDAFF_MVA60, DDAFF_SEM = 0.485, 0.340, 0.142 / (10 ** 0.5)  # reported -> honest MVA-60


def _load(fn):
    p = os.path.join(OUT, fn)
    return json.load(open(p)) if os.path.exists(p) else {}


def _sweep(ax, order, xlabels, title, highlight0=False):
    decay = _load("decay_encoders.json")
    for enc in ENCS:
        d = _load(f"thr_perf_mva_{enc}.json")
        col, mk = ENCODER_COLOR[enc], ENCODER_MARKER[enc]
        xs, ys, es = [], [], []
        for i, t in enumerate(order):
            if t == "100" and enc in decay:            # shared 'original / no-clustering' point
                v = decay[enc]["nosplit"]
                ys.append(v["protbff_r"]); es.append(v.get("protbff_r_sem", 0)); xs.append(i)
            elif t in d:
                ys.append(d[t]["protbff_r"]); es.append(d[t].get("protbff_r_sem", 0)); xs.append(i)
        ax.errorbar(xs, ys, yerr=es, fmt="-" + mk, color=col, capsize=4, markersize=11, lw=2.6,
                    label=f"{ENCODER_LABEL[enc]} + ProtBFF", zorder=6)
    # ESM-C bare (no ProtBFF) -> the ProtBFF gain across the sweep
    bare = _load("bare_esmc_mva.json")
    col = ENCODER_COLOR["esmc"]; xs, ys, es = [], [], []
    for i, t in enumerate(order):
        if t in bare:
            ys.append(bare[t]["bare_r"]); es.append(bare[t].get("bare_r_sem", 0)); xs.append(i)
    ax.errorbar(xs, ys, yerr=es, fmt="--o", color=col, markerfacecolor="white",
                markeredgecolor=col, markeredgewidth=2.0, lw=2.0, capsize=4, markersize=10,
                label="ESM-C (bare)", zorder=4)
    if highlight0:
        ax.axvspan(-0.4, 0.4, color=LEAKY_WASH, zorder=0)
        ax.text(0, ax.get_ylim()[1], "leaky", ha="center", va="bottom", fontsize=13,
                color="#b64342", style="italic")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Pearson correlation (mean-of-folds)")
    ax.set_ylim(0.08, 0.68)
    ax.set_title(title)


def _overlay_rde_network(ax, order):
    """RDE-Network (trained non-PLM specialized predictor). Uses the full MVA sweep if the
    retrain has produced it (rde_network_mva_sweep.json: {thr: {mean_r, sem_r}}); otherwise
    falls back to the two measured anchors (leaky benchmark -> honest MVA-60)."""
    d = _load("rde_network_mva_sweep.json")
    xs, ys, es = [], [], []
    for i, t in enumerate(order):
        if t == "100":                                 # leaky anchor shared with the wash
            xs.append(i); ys.append(RDE_LEAKY); es.append(0)
        elif t in d:
            xs.append(i); ys.append(d[t]["mean_r"]); es.append(d[t].get("sem_r", 0))
    if len(xs) > 2:                                     # full retrained curve
        ax.errorbar(xs, ys, yerr=es, fmt=":X", color=COMPETITOR, capsize=4, markersize=12,
                    lw=2.2, label="RDE-Network (competitor)", zorder=5)
    else:                                              # fallback: leaky -> MVA-60 only
        RDE_MVA60, SEM = 0.393, 0.087 / (10 ** 0.5)
        i60 = order.index("60") if "60" in order else None
        ax.errorbar([0, i60], [RDE_LEAKY, RDE_MVA60], yerr=[0, SEM], fmt=":X", color=COMPETITOR,
                    capsize=4, markersize=12, lw=2.2, label="RDE-Network (competitor)", zorder=5)


def _overlay_ddaffinity(ax, order):
    """DDAffinity (specialized non-PLM deep predictor), shown at its reported leaky benchmark and
    under the honest MVA-60 split. Its published 0.485 collapses to 0.340 under chain-level
    clustering -- the clearest single example that the reported numbers are inflated by homology."""
    i60 = order.index("60") if "60" in order else None
    ax.errorbar([0, i60], [DDAFF_LEAKY, DDAFF_MVA60], yerr=[0, DDAFF_SEM], fmt=":s", color=DDAFF,
                capsize=4, markersize=11, lw=2.2, label="DDAffinity (competitor)", zorder=5)


def panelA(ax):
    order = ["100", "90", "80", "60", "50", "40", "30"]
    xlabels = ["original\n(no align.)", "90", "80", "60", "50", "40", "30"]
    _sweep(ax, order, xlabels, "Original → MVA identity sweep", highlight0=True)
    _overlay_rde_network(ax, order)
    _overlay_ddaffinity(ax, order)
    ax.set_xlabel("MVA sequence-identity threshold (%)")
    ax.legend(loc="upper right", fontsize=13, ncol=1)
    panel_label(ax, "A", dx=-0.085)


def panelC(ax):
    """Performance vs dataset: leave-one-experimental-method-out. Single Pearson axis; error
    bars are the Fisher-z 95% CI from that method's test-set size n; the % of dataset is shown
    prominently under each method label."""
    d = _load("lomo_esmc.json")
    methods = sorted(d, key=lambda m: -d[m]["frac"])
    pears = [d[m]["pearson"] for m in methods]
    ns = [d[m]["n"] for m in methods]
    lo, hi = [], []
    for r, n in zip(pears, ns):                        # Fisher z 95% CI on Pearson r
        z, se = np.arctanh(r), 1.0 / np.sqrt(max(n - 3, 1))
        lo.append(r - np.tanh(z - 1.96 * se)); hi.append(np.tanh(z + 1.96 * se) - r)
    x = np.arange(len(methods))
    ax.bar(x, pears, yerr=[lo, hi], color=ENCODER_COLOR["esmc"], alpha=0.92, width=0.68, zorder=3,
           error_kw=dict(ecolor=INK2, lw=1.5, capsize=4))
    for xi, p, h in zip(x, pears, hi):
        ax.text(xi, p + h + 0.025, f"{p:.2f}", ha="center", va="bottom", fontsize=14,
                fontweight="bold", color=INK)
    ax.set_xticks(x)
    # two-line labels: bold method name + compact, clearly readable % of dataset
    ax.set_xticklabels([m for m in methods], fontsize=15, fontweight="bold", color=INK)
    for xi, m in zip(x, methods):
        ax.text(xi, -0.115, f"{d[m]['frac']*100:.0f}%", ha="center", va="top",
                fontsize=14.5, fontweight="bold", color=ENCODER_COLOR["esmc"],
                transform=ax.get_xaxis_transform())
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_xlabel("Held-out experimental method   (bold % = fraction of dataset)", labelpad=36)
    ax.set_ylabel("Pearson correlation")
    ax.set_ylim(0, 1.0)
    ax.set_title("Performance vs dataset: leave-one-experimental-method-out (ESM-C + ProtBFF)")
    panel_label(ax, "B", dx=-0.055)


def main():
    apply_style(base=18)
    fig = plt.figure(figsize=(13.5, 13.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.34)
    panelA(fig.add_subplot(gs[0, 0]))
    panelC(fig.add_subplot(gs[1, 0]))
    save(fig, "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/figure_previews/figure_2_final")


if __name__ == "__main__":
    main()
