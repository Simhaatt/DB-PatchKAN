"""Figure 10 - KAN response functions for named concepts.

Plots ONLY the saved fitted responses. Nothing is traced, smoothed or symmetrised.

Input: one .npz file with, for each concept key ("rate", "cycle"):
    {key}_x      (n_x,)           normalized concept grid, 0-1
    {key}_seeds  (n_seeds, n_x)   centered KAN response per seed, SOH percentage points
    {key}_lo     (n_x,)           lower edge of the 95 % seed range, as originally calculated
    {key}_hi     (n_x,)           upper edge of the 95 % seed range, as originally calculated
    {key}_mean   (n_x,)           optional; across-seed mean (computed from seeds if absent)

Usage:
    python fig10_kan_responses.py curves.npz out/Fig10
    python fig10_kan_responses.py curves.npz out/Fig10 --percentile-band
        (only if the saved band is unavailable: computes the 2.5-97.5 percentile range across
         seeds; the caption must then say so)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, GRID = "#1B1E22", "#DADDE1"
SEED, BAND, MEAN = "#B9BDC3", "#D6E4F2", "#3F6F9F"
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8, "text.color": INK,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.9, "font.weight": "bold", "axes.labelweight": "bold", "svg.fonttype": "path", "pdf.fonttype": 42,
})
PANELS = [("rate", "(a) D-BSI rate concept"), ("cycle", "(b) D-BSI cycle concept")]


def load(d, key, percentile_band):
    x = np.asarray(d[f"{key}_x"], float)
    s = np.asarray(d[f"{key}_seeds"], float)
    if s.ndim != 2 or s.shape[1] != x.size:
        raise ValueError(f"{key}_seeds must be (n_seeds, {x.size}); got {s.shape}")
    mean = np.asarray(d[f"{key}_mean"], float) if f"{key}_mean" in d else s.mean(axis=0)
    if f"{key}_lo" in d and f"{key}_hi" in d:
        lo, hi = np.asarray(d[f"{key}_lo"], float), np.asarray(d[f"{key}_hi"], float)
    elif percentile_band:
        lo, hi = np.percentile(s, [2.5, 97.5], axis=0)
    else:
        raise KeyError(f"{key}_lo/{key}_hi missing. Supply the saved 95 % seed range, or pass "
                       "--percentile-band and state the method in the caption.")
    return x, s, mean, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("stem")
    ap.add_argument("--percentile-band", action="store_true")
    a = ap.parse_args()
    d = np.load(a.npz)

    fig = plt.figure(figsize=(7.4, 3.0))
    rects = [(0.10, 0.27, 0.38, 0.60), (0.59, 0.27, 0.38, 0.60)]
    handles = None
    for (key, title), rect in zip(PANELS, rects):
        x, s, mean, lo, hi = load(d, key, a.percentile_band)
        ax = fig.add_axes(rect)
        for k, row in enumerate(s):
            ln, = ax.plot(x, row, color=SEED, lw=0.6, zorder=2)
            if k == 0:
                h_seed = ln
                h_seed.set_label("Individual fitted seeds")
        h_band = ax.fill_between(x, lo, hi, color=BAND, lw=0, zorder=1, label="95% seed range")
        h_mean, = ax.plot(x, mean, color=MEAN, lw=1.6, zorder=3, label="Across-seed mean")
        handles = [h_seed, h_band, h_mean]
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 5), [f"{v:.2f}" for v in np.linspace(0, 1, 5)])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        # y-limits follow the data only; no symmetrisation
        ymin = min(s.min(), lo.min(), mean.min())
        ymax = max(s.max(), hi.max(), mean.max())
        pad = 0.05 * (ymax - ymin or 1)
        ax.set_ylim(ymin - pad, ymax + pad)
        fig.text(rect[0] + rect[2] / 2, 0.035, title, fontsize=8.5, fontweight="bold",
                 ha="center", va="center")   # subcaption below panel
    fig.text(0.535, 0.135, "Normalized concept value", ha="center", fontsize=8, fontweight="bold")
    fig.text(0.015, 0.57, "Centered KAN response\n(SOH percentage points)", rotation=90,
             ha="center", va="center", fontsize=8, fontweight="bold", linespacing=1.2)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.535, 1.0), ncol=3,
               frameon=False, fontsize=8, handlelength=2.0, columnspacing=2.2)
    for ext in ("pdf", "svg"):
        fig.savefig(f"{a.stem}.{ext}", facecolor="white")
    fig.savefig(f"{a.stem}.png", dpi=300, facecolor="white")


if __name__ == "__main__":
    main()
