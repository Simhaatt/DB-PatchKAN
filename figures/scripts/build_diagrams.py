"""Figures 1-6 (architecture / workflow). Writes .drawio + .pdf/.svg/.png per figure."""
import os
import sys
from figlib import *

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUTDIR, exist_ok=True)
B, N = True, False


def L(*specs):
    """L(('text', size, bold), ...) helper passthrough."""
    return list(specs)


# ------------------------------------------------------------------ Figure 1
def fig1():
    f = Fig("Fig1", 740, 250, "Figure 1")
    M, w, g, y0, h = 18, 122, 23.5, 18, 214
    cards = [
        (("Driver and vehicle", "inputs"), DRV,
         ["Speed", "Acceleration", "Jerk", "Accelerator/braking", "Torque/route load", "Auxiliary load"]),
        (("Traction and", "regeneration"), DRV,
         ["Traction demand", "Regenerative demand", "Battery power demand"]),
        (("Immediate battery", "response"), BAT,
         ["Current $I(t)$", "Voltage $V(t)$", "Power $P(t)$", "C-rate", "Pack temperature", "SOC"]),
        (("Physical exposure", "quantities"), PHY,
         ["Current-squared dose", "Electrothermal\nexposure", "Temperature rise", "DoD/EFC rate",
          "C-rate slew", "Interaction exposure"]),
        (("Trip-level stress", "representation"), FUS, None),
    ]
    ids = []
    for i, (title, fill, items) in enumerate(cards):
        x = M + i * (w + g)
        cid = f.box(x, y0, w, h, fill=fill, container=True)
        ids.append(cid)
        f.text(x + 4, y0 + 8, w - 8, 30, [(title[0], 11, B), (title[1], 11, B)], parent=cid)
        f.line([(x + 10, y0 + 44), (x + w - 10, y0 + 44)], parent=cid)
        if items:
            s, lh, gap = 10.5, 10.5 * LH, 9
            hs = [len(t.split("\n")) * lh for t in items]
            tot = sum(hs) + gap * (len(items) - 1)
            yy = y0 + 48 + (h - 48 - 8 - tot) / 2
            for t, hh in zip(items, hs):
                f.text(x + 4, yy, w - 8, hh, [(p, s, N) for p in t.split("\n")], parent=cid)
                yy += hh + gap
        else:
            f.text(x + 4, y0 + 48, w - 8, 14, [("Six D-BSI components", 10, N)], parent=cid)
            gx, gy, gw, gh = x + 7, y0 + 63, w - 14, 50
            grid = f.box(gx, gy, gw, gh, fill="#FFFFFF", lw=1.0, r=3, parent=cid)
            comps = [("Rate", "Thermal"), ("SOC", "Cycle"), ("Transient", "Interaction")]
            for r_, (a, b_) in enumerate(comps):
                yy = gy + 4 + r_ * 14
                f.text(gx + 2, yy, gw / 2 - 2, 14, [(a, 10, N)], parent=cid)
                f.text(gx + gw / 2, yy, gw / 2 - 2, 14, [(b_, 10, N)], parent=cid)
            dbsi = f.box(x + 22, y0 + 127, w - 44, 24, fill=HL, stroke=HL_INK, lw=1.4, r=4,
                         lines=[("D-BSI", 11, B)], parent=cid)
            dur = f.box(x + 7, y0 + 165, w - 14, 36, fill="#FFFFFF", lw=1.0, r=4,
                        lines=[("Duration-weighted", 10.5, N), ("stress exposure", 10.5, N)], parent=cid)
            cx = x + w / 2
            f.arrow([(cx, gy + gh), (cx, y0 + 127)], src=grid, tgt=dbsi, lw=1.4)
            f.arrow([(cx, y0 + 151), (cx, y0 + 165)], src=dbsi, tgt=dur, lw=1.4)
    yc = y0 + h / 2
    for i in range(4):
        x1 = M + i * (w + g) + w
        f.arrow([(x1, yc), (x1 + g, yc)], src=ids[i], tgt=ids[i + 1])
    return f


# ------------------------------------------------------------------ Figure 2
def fig2():
    f = Fig("Fig2", 740, 336, "Figure 2")
    cols = [(18, 122), (162, 128), (312, 92), (426, 100), (548, 92), (662, 60)]
    rh = 116
    lanes = [("Driver stream", DRV, 12, 32, "$H_D$",
              ["Speed", "Acceleration", "Jerk", "Accelerator/brake", "Optional vehicle\ncontrols"]),
             ("Battery stream", BAT, 188, 208, "$H_B$",
              ["Current", "Voltage", "SOC", "Temperature"])]
    for name, fill, ty, y, hname, items in lanes:
        f.text(18, ty, 200, 16, [(name, 11.5, B)], align="left")
        ids = []
        # input
        x, w = cols[0]
        bid = f.box(x, y, w, rh, fill=fill, container=True)
        ids.append(bid)
        f.text(x + 4, y + 7, w - 8, 15, [("Input sequence", 11, B)], parent=bid)
        yy = y + 26
        for t in items:
            parts = t.split("\n")
            hh = len(parts) * 10 * LH
            f.text(x + 4, yy, w - 8, hh, [(p, 10, N) for p in parts], parent=bid)
            yy += hh + 1.5
        # overlapping patches
        x, w = cols[1]
        bid = f.box(x, y, w, rh, fill=fill, container=True)
        ids.append(bid)
        f.text(x + 4, y + 8, w - 8, 30, [("Overlapping", 11, B), ("patches", 11, B)], parent=bid)
        x0, ya, tw, th = x + 14, y + 56, 30, 20
        for k in range(4):
            f.box(x0 + 15 * k, ya + (10 if k % 2 else 0), tw, th, fill="none", lw=1.1, r=2,
                  parent=bid)
        f.text(x0 + 82, ya + 6, 20, 18, [("…", 12, N)], parent=bid)
        # embedding, transformer, encoded
        for (x, w), lines, bold in ((cols[2], ("Patch", "embedding"), B),
                                    (cols[3], ("Transformer", "encoder"), B),
                                    (cols[4], ("Encoded", "patches"), N)):
            ids.append(f.box(x, y, w, rh, fill=fill,
                             lines=[(lines[0], 11 if bold else 10.5, bold),
                                    (lines[1], 11 if bold else 10.5, bold)]))
        x, w = cols[5]
        ids.append(f.box(x, y + rh / 2 - 22, w, 44, fill=fill, lines=[(hname, 12, N)]))
        yc = y + rh / 2
        for i in range(5):
            xa = cols[i][0] + cols[i][1]
            f.arrow([(xa, yc), (cols[i + 1][0], yc)], src=ids[i], tgt=ids[i + 1])
    # shared settings between lanes
    ann = [["256-s input"], ["16-s patches, 8-s stride", "31 patches"], ["128-D embeddings"],
           ["3 encoder blocks"]]
    for (x, w), lines in zip(cols, ann):
        f.text(x - 10, 153, w + 20, 28, [(t, 10, N) for t in lines])
    return f


# ------------------------------------------------------------------ Figure 3
def fig3():
    f = Fig("Fig3", 740, 214, "Figure 3")
    hd = f.box(40, 32, 94, 44, fill=DRV, lines=[("Driver patches", 10.5, N), ("$H_D$", 11, N)])
    hb = f.box(40, 132, 94, 44, fill=BAT, lines=[("Battery patches", 10.5, N), ("$H_B$", 11, N)])
    at = f.box(162, 24, 128, 60, fill=ATT, lines=[("Driver queries", 11, B), ("battery", 11, B),
                                                  (r"$Q=H_D,\ K=V=H_B$", 10.5, N)])
    ab = f.box(162, 124, 128, 60, fill=ATT, lines=[("Battery queries", 11, B), ("driver", 11, B),
                                                   (r"$Q=H_B,\ K=V=H_D$", 10.5, N)])
    ud = f.box(310, 32, 94, 44, fill=DRV, lines=[("Updated driver", 10.5, N), ("patches", 10.5, N)])
    ub = f.box(310, 132, 94, 44, fill=BAT, lines=[("Updated battery", 10.5, N), ("patches", 10.5, N)])
    pool = f.box(424, 76, 92, 56, fill=CTL, lines=[("Patch attention", 11, B), ("pooling", 11, B)])
    agg = f.box(532, 76, 88, 56, fill=CTL, lines=[("Trip/segment", 11, B), ("aggregation", 11, B)])
    trip = f.box(636, 70, 86, 68, fill=FUS, stroke=HL_INK, lw=2.0,
                 lines=[("Trip", 10.5, N), ("representation", 10.5, N),
                        (r"$h_{\mathrm{trip}}$, 128-D", 11, N)])
    # inputs -> attention (crossing-free: one inner elbow, one outer loop)
    f.arrow([(134, 54), (162, 54)], src=hd, tgt=at)
    f.arrow([(134, 164), (162, 164)], src=hb, tgt=ab)
    f.arrow([(134, 144), (148, 144), (148, 72), (162, 72)], src=hb, tgt=at)
    f.arrow([(40, 54), (28, 54), (28, 200), (226, 200), (226, 184)], src=hd, tgt=ab)
    # attention -> updated
    f.arrow([(290, 54), (310, 54)], src=at, tgt=ud)
    f.arrow([(290, 154), (310, 154)], src=ab, tgt=ub)
    # merge into pooling
    f.arrow([(404, 54), (414, 54), (414, 92), (424, 92)], src=ud, tgt=pool)
    f.arrow([(404, 154), (414, 154), (414, 116), (424, 116)], src=ub, tgt=pool)
    f.arrow([(516, 104), (532, 104)], src=pool, tgt=agg)
    f.arrow([(620, 104), (636, 104)], src=agg, tgt=trip)
    return f


# ------------------------------------------------------------------ Figure 4
def fig4():
    f = Fig("Fig4", 740, 290, "Figure 4")
    yc = 108
    h1 = f.box(18, yc - 32, 112, 64, fill=FUS,
               lines=[("Trip representation", 10.5, N), (r"$h_{\mathrm{trip}}$ (128-D)", 11, N)])
    cp = f.box(150, yc - 32, 116, 64, fill=CTL,
               lines=[("Concept", 11, B), (r"predictor $g_\theta$", 11, B),
                      (r"$128\rightarrow128\rightarrow n_c$", 10.5, N)])
    x, y, w, h = 294, 18, 280, 180
    cv = f.box(x, y, w, h, fill=PHY, container=True)
    f.text(x + 6, y + 8, w - 12, 16, [(r"Named concept vector $\hat{\mathbf{c}}$", 11.5, B)], parent=cv)
    f.text(x + 6, y + 25, w - 12, 14, [("15 named concepts (Nissan configuration)", 10, N)], parent=cv)
    groups = [("Operating stress", ["D-BSI, cumulative stress,", "temperature, throughput"]),
              ("Driving behavior", ["acceleration/jerk, energy, regeneration,",
                                    "stop fraction, exploratory behavior proportions"]),
              ("Health and usage context", ["last available health anchor, age, mileage"])]
    gy = y + 50
    for k, (gname, items) in enumerate(groups):
        f.line([(x + 12, gy - 4), (x + w - 12, gy - 4)], parent=cv)
        f.text(x + 14, gy, w - 28, 14, [(gname, 10.5, B)], align="left", parent=cv)
        f.text(x + 14, gy + 14, w - 28, 12.5 * len(items), [(t, 10, N) for t in items],
               align="left", parent=cv)
        gy += 47
    hd = f.box(602, yc - 32, 120, 64, fill=OUT, lines=[("Concept-to-health", 11, B), ("head", 11, B)])
    tg = f.box(324, 234, 220, 40, fill=PHY,
               lines=[("Independently computed", 10.5, B), ("concept targets", 10.5, B)])
    f.arrow([(130, yc), (150, yc)], src=h1, tgt=cp)
    f.arrow([(266, yc), (294, yc)], src=cp, tgt=cv)
    f.arrow([(574, yc), (602, yc)], src=cv, tgt=hd)
    f.arrow([(434, 234), (434, 198)], src=tg, tgt=cv, dashed=True, lw=1.4)
    f.text(442, 208, 160, 16, [("Huber concept supervision", 10, N)], align="left")
    return f


# ------------------------------------------------------------------ Figure 5
def fig5():
    f = Fig("Fig5", 552, 236, "Figure 5")
    cin = f.box(16, 80, 104, 52, fill=PHY,
                lines=[("Named concept", 10.5, N), (r"vector $\hat{\mathbf{c}}$", 11, N)])
    heads = []
    for (y, fill, title, topo, desc, glyph) in (
            (16, KAN, "KAN head", r"$[n_c,\ 16,\ 8,\ n_{\mathrm{out}}]$",
             "Learnable univariate spline functions", True),
            (116, BAT, "Matched MLP control", r"$[n_c,\ 32,\ 16,\ n_{\mathrm{out}}]$",
             "Dense nonlinear mapping", False)):
        x, w, h = 160, 216, 80
        hid = f.box(x, y, w, h, fill=fill, container=True)
        heads.append(hid)
        top = y + 7 if glyph else y + 15
        f.text(x + 4, top, w - 8, 15, [(title, 11.5, B)], parent=hid)
        f.text(x + 4, top + 16, w - 8, 15, [(topo, 10.5, N)], parent=hid)
        f.text(x + 4, top + 32, w - 8, 14, [(desc, 10, N)], parent=hid)
        if glyph:
            shapes = [[(0, 12), (8, 11), (15, 7), (22, 3), (30, 2)],
                      [(0, 12), (8, 6), (15, 3), (22, 6), (30, 12)],
                      [(0, 8), (7, 3), (15, 10), (23, 4), (30, 7)]]
            gx0 = x + (w - (3 * 30 + 2 * 18)) / 2
            for k, s in enumerate(shapes):
                ox, oy = gx0 + k * 48, y + 58
                f.line([(ox + a, oy + b) for a, b in s], lw=1.2, color=INK, curved=True, parent=hid)
    outs = [f.box(400, 34, 136, 44, fill=OUT, lines=[(r"KAN residual $\hat{r}_{\mathrm{KAN}}$", 10.5, N)]),
            f.box(400, 134, 136, 44, fill=OUT, lines=[(r"MLP residual $\hat{r}_{\mathrm{MLP}}$", 10.5, N)])]
    f.arrow([(120, 106), (140, 106), (140, 56), (160, 56)], src=cin, tgt=heads[0])
    f.arrow([(120, 106), (140, 106), (140, 156), (160, 156)], src=cin, tgt=heads[1])
    f.arrow([(376, 56), (400, 56)], src=heads[0], tgt=outs[0])
    f.arrow([(376, 156), (400, 156)], src=heads[1], tgt=outs[1])
    f.text(16, 208, 520, 16,
           [("Shared encoder · concepts · targets · forecast anchor · data folds", 10.5, N)])
    return f


# ------------------------------------------------------------------ Figure 6
def fig6():
    """Two panels side by side; each: anchor (op) model term -> forecast, persistence below,
    subcaption under the panel."""
    f = Fig("Fig6", 740, 236, "Figure 6")
    panels = [
        (18, "(a) BMS-SOH loss forecast (Nissan)",
         [("Latest SOH", "anchor", "$H_k$"), ("Predicted", "non-negative loss", r"$\hat{L}_h \geq 0$"),
          ("Future SOH", r"$\hat{H}_{k+h} = H_k - \hat{L}_h$")], "−",
         r"Persistence: $\hat{H}_{k+h} = H_k$"),
        (382, "(b) Next capacity forecast (EVBattery)",
         [("Latest", "capacity", "$Q_k$"), ("Signed learned", "correction", r"$\widehat{\Delta Q}$"),
          ("Next capacity", r"$\hat{Q}_{k+1} = Q_k + \widehat{\Delta Q}$")], "+",
         r"Persistence: $\hat{Q}_{k+1} = Q_k$"),
    ]
    PW = 340
    for px, label, boxes, op, pers in panels:
        cx = px + PW / 2
        y, h = 14, 64
        yc = y + h / 2
        b1 = f.box(px, y, 140, h, fill=CTL,
                   lines=[(boxes[0][0], 11, B), (boxes[0][1], 11, B), (boxes[0][2], 11.5, N)])
        opc = f.box(cx - 13, yc - 13, 26, 26, fill="#FFFFFF", shape="ellipse", lw=1.4,
                    lines=[(op, 14, N)])
        b2 = f.box(px + PW - 140, y, 140, h, fill=CTL,
                   lines=[(boxes[1][0], 11, B), (boxes[1][1], 11, B), (boxes[1][2], 11.5, N)])
        b3 = f.box(cx - 125, 112, 250, 50, fill=OUT, lw=2.0,
                   lines=[(boxes[2][0], 11, B), (boxes[2][1], 11.5, N)])
        f.arrow([(px + 140, yc), (cx - 13, yc)], src=b1, tgt=opc, head=False, lw=1.5)
        f.arrow([(cx + 13, yc), (px + PW - 140, yc)], src=opc, tgt=b2, head=False, lw=1.5)
        f.arrow([(cx, yc + 13), (cx, 112)], src=opc, tgt=b3)
        f.box(cx - 100, 172, 200, 22, fill="#FFFFFF", lw=1.1, dashed=True, r=3,
              lines=[(pers, 10.5, N)])
        f.text(px, 208, PW, 18, [(label, 11.5, B)])
    return f


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6):
        f = fn()
        stem = os.path.join(OUTDIR, f.key)
        f.to_drawio(stem + ".drawio")
        w = f.render(stem)
        print(f.key, "ok", *(w or []), sep="\n" if w else " ")
