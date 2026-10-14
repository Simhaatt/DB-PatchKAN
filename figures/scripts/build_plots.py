"""Figures 7-9: data plots generated from the exact values in the figure specification.
Writes .pdf/.svg/.png plus a .drawio file that embeds the SVG for final layout work."""
import base64
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUTDIR, exist_ok=True)

INK, GRID = "#1B1E22", "#DADDE1"
BLUE, GRAY, GREEN, GOLD = "#6F93B8", "#C9CDD2", "#86AD88", "#CDAE62"
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8, "text.color": INK,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.9, "font.weight": "bold", "axes.labelweight": "bold",
    "axes.titleweight": "bold", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "svg.fonttype": "path", "pdf.fonttype": 42,
})


def style(ax, grid_axis):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def save(fig, key):
    stem = os.path.join(OUTDIR, key)
    for ext in ("pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", facecolor="white")
    fig.savefig(f"{stem}.png", dpi=300, facecolor="white")
    w, h = fig.get_size_inches()
    plt.close(fig)
    # draw.io wrapper: SVG embedded as an image cell at 1 unit = 1/100 in
    b64 = base64.b64encode(open(f"{stem}.svg", "rb").read()).decode()
    W, H = round(w * 100), round(h * 100)
    xml = f"""<mxfile host="app.diagrams.net" type="device">
  <diagram id="{key}" name="{key}">
    <mxGraphModel dx="900" dy="600" grid="1" gridSize="2" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0" background="#FFFFFF">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        <mxCell id="{key}-plot" value="" style="shape=image;html=1;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;image=data:image/svg+xml,{b64};" vertex="1" parent="1">
          <mxGeometry x="0" y="0" width="{W}" height="{H}" as="geometry"/>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""
    open(f"{stem}.drawio", "w").write(xml)


# ------------------------------------------------------------------ Figure 7
FIG7 = {
    "(a) RWTH": [("Peak C-rate", 46.9), ("High-current fraction", 51.9),
                 ("Electrothermal intensity", 46.3), ("Thermal-response proxy", 44.9),
                 ("Energy per 100 km", 61.9), ("Partial D-BSI", 88.5)],
    "(b) EVE": [("C-rate p95", 26.1), ("D-BSI", 67.6), ("Electrothermal response", 22.5),
                ("Energy per 100 km", 13.9), ("Temperature rise", 12.4),
                ("Maximum temperature", 1.2)],
}


def fig7():
    fig = plt.figure(figsize=(7.4, 2.6))
    boxes = [(0.205, 0.25, 0.265, 0.62), (0.695, 0.25, 0.265, 0.62)]
    fig.text(0.5, 0.955, "Driver-aware versus context-only", ha="center", va="center",
             fontsize=8.5, fontweight="bold")
    for (label, rows), rect in zip(FIG7.items(), boxes):
        ax = fig.add_axes(rect)
        names = [r[0] for r in rows]
        vals = np.array([r[1] for r in rows])
        y = np.arange(len(rows))
        ax.barh(y, vals, height=0.62, color=BLUE, zorder=2)
        for yi, v in zip(y, vals):
            ax.text(v + 1.5, yi, f"{v:.1f}%", va="center", ha="left", fontsize=7.5, zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="y", length=0, pad=4)
        style(ax, "x")
        ax.spines["left"].set_visible(False)
        ax.axvline(0, color=INK, lw=0.8, zorder=3)
        ax.set_xlabel("MAE reduction (%)", fontsize=8, fontweight="bold", labelpad=3)
        cx = rect[0] + rect[2] / 2
        fig.text(cx, 0.04, label, ha="center", va="center", fontsize=8.5, fontweight="bold")
    save(fig, "Fig7")


# ------------------------------------------------------------------ Figure 8
FIG8 = [("≤500 km", 0.2932, 0.3034), ("500–1000 km", 0.3825, 0.3818),
        ("1000–2000 km", 0.4710, 0.4638), (">2000 km", 0.6990, 0.6685)]


def fig8():
    fig = plt.figure(figsize=(5.5, 3.0))
    ax = fig.add_axes((0.12, 0.17, 0.85, 0.66))
    x = np.arange(len(FIG8))
    bw = 0.36
    p = np.array([r[1] for r in FIG8])
    m = np.array([r[2] for r in FIG8])
    b1 = ax.bar(x - bw / 2 - 0.01, p, bw, color=GRAY, label="Strict persistence", zorder=2)
    b2 = ax.bar(x + bw / 2 + 0.01, m, bw, color=BLUE, label="DB-PatchKAN", zorder=2)
    for bars, vals in ((b1, p), (b2, m)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.4f}", ha="center",
                    va="bottom", fontsize=7)
    ax.set_xticks(x, [r[0] for r in FIG8])
    ax.set_ylim(0, 0.8)
    ax.set_yticks(np.arange(0, 0.81, 0.1))
    ax.set_ylabel("MAE (Ah)", fontsize=8, fontweight="bold")
    ax.set_xlabel("Forecast distance", fontsize=8, fontweight="bold")
    ax.tick_params(axis="x", length=0)
    style(ax, "y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
              fontsize=8, handlelength=1.2, columnspacing=2.0)
    save(fig, "Fig8")


# ------------------------------------------------------------------ Figure 9
FIG9 = [("(a) Nissan", BLUE, [("Temperature bias", 0.770), ("Pack heterogeneity", 0.377),
                              ("SOC bias", 0.191), ("Capacity scale", 0.178)]),
        ("(b) RWTH", GREEN, [("Temperature bias", 0.685), ("Calendar activation energy", 0.612),
                             ("SOC bias", 0.214), ("Pack heterogeneity", 0.134)]),
        ("(c) EVE", GOLD, [("Pack heterogeneity", 0.627), ("Temperature bias", 0.557),
                           ("Capacity scale", 0.307), ("Cycle activation energy", 0.222)])]


def fig9():
    """Three panels side by side, subcaptions below each panel."""
    fig = plt.figure(figsize=(7.4, 2.4))
    lefts, width, bottom, height = (0.145, 0.475, 0.805), 0.165, 0.34, 0.62
    for k, ((label, color, rows), left) in enumerate(zip(FIG9, lefts)):
        ax = fig.add_axes((left, bottom, width, height))
        y = np.arange(len(rows))
        vals = [r[1] for r in rows]
        ax.barh(y, vals, height=0.62, color=color, zorder=2)
        for yi, v in zip(y, vals):
            ax.text(v + 0.02, yi, f"{v:.3f}", va="center", ha="left", fontsize=7.5, zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
        names = [r[0].replace("activation energy", "activation\nenergy") for r in rows]
        ax.set_yticks(y, names, fontsize=7.5, linespacing=1.0)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8], ["0", "0.2", "0.4", "0.6", "0.8"])
        ax.tick_params(axis="y", length=0, pad=3)
        style(ax, "x")
        ax.spines["left"].set_visible(False)
        ax.axvline(0, color=INK, lw=0.9, zorder=3)
        fig.text(left + width / 2, 0.05, label, ha="center", va="center", fontsize=8.5,
                 fontweight="bold")
    fig.text(0.5, 0.185, "Median absolute Spearman sensitivity, |ρ|", ha="center", va="center",
             fontsize=8, fontweight="bold")
    save(fig, "Fig9")


if __name__ == "__main__":
    fig7()
    fig8()
    fig9()
    print("plots ok")
