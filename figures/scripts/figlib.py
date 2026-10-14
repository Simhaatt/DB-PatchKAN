"""Tiny layout engine: one figure description -> editable .drawio + vector PDF/SVG/PNG.

Units: draw.io units (1 unit = 1/100 in = 0.72 pt). A 740-unit canvas is ~188 mm wide,
i.e. two-column journal width at 100 % zoom, so font sizes here are final print sizes.
"""
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse
from matplotlib.path import Path
from matplotlib.patches import PathPatch

PT = 0.72          # points per unit
INK = "#1B1E22"
ALL_BOLD = True    # user request: all text bold for print contrast
FONT_MPL = "Liberation Sans"   # metric-compatible with Arial
FONT_DIO = "Arial"
LH = 1.25          # line-height factor

plt.rcParams.update({
    "font.family": FONT_MPL,
    "mathtext.fontset": "custom",  # bold sans math to match bold Arial labels
    "mathtext.rm": "Liberation Sans:bold",
    "mathtext.it": "Liberation Sans:italic:bold",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.fallback": "stixsans",
    "svg.fonttype": "path",        # text converted to outlines in SVG
    "pdf.fonttype": 42,            # TrueType embedded in PDF
})

# role colours from the spec
DRV, BAT, PHY, FUS, OUT, CTL = "#F5E7D5", "#DDEAF5", "#E4F0E5", "#ECE8F5", "#E0F0EC", "#F1F2F4"
HL, HL_INK = "#C6D9EB", "#1C2F44"
ATT = "#F7F0D2"     # pale yellow (Fig 3 attention)
KAN = "#FAE3CC"     # pale orange (Fig 5 KAN head)
SEP = "#8E949C"     # light separator


def _math_to_dio(t):
    if ALL_BOLD:
        return re.sub(r"\$(.+?)\$", lambda m: r"\(\boldsymbol{" + m.group(1) + r"}\)", t)
    return re.sub(r"\$(.+?)\$", lambda m: r"\(" + m.group(1) + r"\)", t)


def _esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Fig:
    def __init__(self, key, W, H, name):
        self.key, self.W, self.H, self.name = key, W, H, name
        self.items, self.geo, self.n = [], {}, 0

    def _id(self, p):
        self.n += 1
        return f"{self.key.lower()}-{p}{self.n}"

    # ---------- primitives ----------
    def box(self, x, y, w, h, fill="#FFFFFF", stroke=INK, lw=1.4, dashed=False, r=5,
            shape="rect", lines=None, valign="middle", align="center", pad=6,
            container=False, parent=None, opacity=1.0):
        i = self._id("b")
        self.items.append(dict(kind="box", id=i, x=x, y=y, w=w, h=h, fill=fill, stroke=stroke,
                               lw=lw, dashed=dashed, r=r, shape=shape, lines=lines or [],
                               valign=valign, align=align, pad=pad, container=container,
                               parent=parent, opacity=opacity))
        self.geo[i] = (x, y, w, h)
        return i

    def text(self, x, y, w, h, lines, valign="middle", align="center", parent=None, color=INK,
             check_w=None):
        i = self._id("t")
        self.items.append(dict(kind="text", id=i, x=x, y=y, w=w, h=h, lines=lines, valign=valign,
                               align=align, parent=parent, color=color, pad=0,
                               check_w=check_w if check_w is not None else w))
        self.geo[i] = (x, y, w, h)
        return i

    def arrow(self, pts, src=None, tgt=None, dashed=False, lw=1.7, head=True, color=INK,
              curved=False, parent=None):
        i = self._id("e")
        self.items.append(dict(kind="edge", id=i, pts=pts, src=src, tgt=tgt, dashed=dashed, lw=lw,
                               head=head, color=color, curved=curved, parent=parent))
        return i

    def line(self, pts, lw=0.8, color=SEP, curved=False, parent=None):
        return self.arrow(pts, lw=lw, head=False, color=color, curved=curved, parent=parent)

    # ---------- draw.io ----------
    def _html(self, lines):
        out = []
        for t, size, bold in lines:
            s = _esc(_math_to_dio(t))
            if bold or ALL_BOLD:
                s = f"<b>{s}</b>"
            out.append(f'<span style="font-size:{size:g}px">{s}</span>')
        return "<br>".join(out)

    def _rel(self, parent, x, y):
        if parent is None:
            return x, y
        px, py, _, _ = self.geo[parent]
        return x - px, y - py

    def to_drawio(self, path):
        mx = ET.Element("mxfile", host="app.diagrams.net", type="device")
        dg = ET.SubElement(mx, "diagram", id=self.key, name=self.name)
        m = ET.SubElement(dg, "mxGraphModel", dx="900", dy="600", grid="1", gridSize="2",
                          guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1",
                          pageScale="1", pageWidth=str(self.W), pageHeight=str(self.H),
                          math="1", shadow="0", background="#FFFFFF")
        root = ET.SubElement(m, "root")
        ET.SubElement(root, "mxCell", id="0")
        ET.SubElement(root, "mxCell", id="1", parent="0")
        for it in self.items:
            par = it.get("parent") or "1"
            if it["kind"] in ("box", "text"):
                fs = max([l[1] for l in it["lines"]], default=10.5)
                if it["kind"] == "box":
                    if it["shape"] == "ellipse":
                        st = "ellipse;"
                    else:
                        st = f"rounded=1;absoluteArcSize=1;arcSize={2*it['r']:g};"
                    st += (f"whiteSpace=wrap;html=1;fillColor={it['fill']};strokeColor={it['stroke']};"
                           f"strokeWidth={it['lw']:g};")
                    if it["dashed"]:
                        st += "dashed=1;dashPattern=4 3;"
                    if it["container"]:
                        st += "container=1;collapsible=0;recursiveResize=0;"
                    if it["opacity"] < 1:
                        st += f"fillOpacity={int(it['opacity']*100)};"
                    color = INK if it["stroke"] in ("none",) else it["stroke"]
                else:
                    st = "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
                    color = it["color"]
                st += (f"fontFamily={FONT_DIO};fontColor={color};fontSize={fs:g};"
                       f"align={it['align']};verticalAlign={it['valign']};spacing={it['pad']};"
                       f"spacingTop=0;spacingBottom=0;")
                c = ET.SubElement(root, "mxCell", id=it["id"], value=self._html(it["lines"]),
                                  style=st, vertex="1", parent=par)
                x, y = self._rel(it.get("parent"), it["x"], it["y"])
                ET.SubElement(c, "mxGeometry", x=f"{x:g}", y=f"{y:g}", width=f"{it['w']:g}",
                              height=f"{it['h']:g}", **{"as": "geometry"})
            else:
                st = (f"edgeStyle=none;html=1;rounded=0;strokeColor={it['color']};"
                      f"strokeWidth={it['lw']:g};startArrow=none;")
                st += "endArrow=block;endFill=1;endSize=5;" if it["head"] else "endArrow=none;"
                if it["dashed"]:
                    st += "dashed=1;dashPattern=4 3;"
                if it["curved"]:
                    st += "curved=1;"
                attrs = dict(id=it["id"], style="", edge="1", parent=par)
                pts = it["pts"]
                for end, key, p in (("exit", "src", pts[0]), ("entry", "tgt", pts[-1])):
                    if it[key]:
                        bx, by, bw, bh = self.geo[it[key]]
                        st += (f"{end}X={(p[0]-bx)/bw:.4g};{end}Y={(p[1]-by)/bh:.4g};"
                               f"{end}Dx=0;{end}Dy=0;{end}Perimeter=0;")
                        attrs["source" if key == "src" else "target"] = it[key]
                attrs["style"] = st
                c = ET.SubElement(root, "mxCell", **attrs)
                g = ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})
                if not it["src"]:
                    x, y = self._rel(it.get("parent"), *pts[0])
                    ET.SubElement(g, "mxPoint", x=f"{x:g}", y=f"{y:g}", **{"as": "sourcePoint"})
                if not it["tgt"]:
                    x, y = self._rel(it.get("parent"), *pts[-1])
                    ET.SubElement(g, "mxPoint", x=f"{x:g}", y=f"{y:g}", **{"as": "targetPoint"})
                if len(pts) > 2:
                    arr = ET.SubElement(g, "Array", **{"as": "points"})
                    for p in pts[1:-1]:
                        x, y = self._rel(it.get("parent"), *p)
                        ET.SubElement(arr, "mxPoint", x=f"{x:g}", y=f"{y:g}")
        ET.indent(mx)
        ET.ElementTree(mx).write(path, encoding="utf-8", xml_declaration=False)

    # ---------- matplotlib render ----------
    def render(self, stem):
        fig = plt.figure(figsize=(self.W / 100, self.H / 100))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.W)
        ax.set_ylim(self.H, 0)
        ax.axis("off")
        texts = []
        z = 1
        for it in self.items:
            z += 1
            if it["kind"] == "box":
                if it["shape"] == "ellipse":
                    p = Ellipse((it["x"] + it["w"] / 2, it["y"] + it["h"] / 2), it["w"], it["h"])
                else:
                    p = FancyBboxPatch((it["x"], it["y"]), it["w"], it["h"],
                                       boxstyle=f"round,pad=0,rounding_size={it['r']}")
                p.set_facecolor(it["fill"] if it["fill"] != "none" else "none")
                p.set_alpha(None)
                p.set_edgecolor(it["stroke"] if it["stroke"] != "none" else "none")
                p.set_linewidth(it["lw"] * PT)
                if it["dashed"]:
                    p.set_linestyle((0, (4, 3)))
                p.set_zorder(z)
                ax.add_patch(p)
            if it["kind"] in ("box", "text") and it["lines"]:
                texts += self._draw_lines(ax, it, z + 0.5)
            if it["kind"] == "edge":
                self._draw_edge(ax, it, z)
        # overflow check
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        warn = []
        for t, avail, label in texts:
            bb = t.get_window_extent(rend)
            wu = bb.width / fig.dpi * 100
            if wu > avail + 0.5:
                warn.append(f"  overflow {wu:.1f} > {avail:.1f}: {label!r}")
        for ext in ("pdf", "svg"):
            fig.savefig(f"{stem}.{ext}", facecolor="white")
        fig.savefig(f"{stem}.png", dpi=300, facecolor="white")
        plt.close(fig)
        return warn

    def _draw_lines(self, ax, it, z):
        x, y, w, h = it["x"], it["y"], it["w"], it["h"]
        pad = it["pad"]
        heights = [s * LH for _, s, _ in it["lines"]]
        tot = sum(heights)
        if it["valign"] == "top":
            yy = y + pad
        elif it["valign"] == "bottom":
            yy = y + h - pad - tot
        else:
            yy = y + (h - tot) / 2
        if it["align"] == "left":
            xx, ha = x + pad, "left"
        elif it["align"] == "right":
            xx, ha = x + w - pad, "right"
        else:
            xx, ha = x + w / 2, "center"
        avail = (it.get("check_w") or w) - 2 * pad
        color = it.get("color", INK) if it["kind"] == "text" else (
            it["stroke"] if it["stroke"] not in ("none",) else INK)
        out = []
        for (t, s, b), hh in zip(it["lines"], heights):
            tt = ax.text(xx, yy + hh / 2, t, ha=ha, va="center_baseline", fontsize=s * PT,
                         fontweight="bold" if (b or ALL_BOLD) else "normal", color=color, zorder=z)
            out.append((tt, avail, t))
            yy += hh
        return out

    def _draw_edge(self, ax, it, z):
        pts = [tuple(p) for p in it["pts"]]
        lw = it["lw"] * PT
        ls = (0, (4, 3)) if it["dashed"] else "-"
        if it["curved"]:
            xs, ys = _catmull(pts)
            ax.plot(xs, ys, color=it["color"], lw=lw, ls=ls, zorder=z, solid_capstyle="round")
            return
        if it["head"]:
            L, Wd = 7.0, 6.0
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
            d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            ux, uy = (x1 - x0) / d, (y1 - y0) / d
            bx, by = x1 - ux * L, y1 - uy * L
            draw = pts[:-1] + [(bx + ux * 0.5, by + uy * 0.5)]
            tri = [(x1, y1), (bx - uy * Wd / 2, by + ux * Wd / 2), (bx + uy * Wd / 2, by - ux * Wd / 2)]
            ax.add_patch(Polygon(tri, closed=True, facecolor=it["color"], edgecolor=it["color"],
                                 lw=0.3, zorder=z + 0.1))
        else:
            draw = pts
        xs, ys = zip(*draw)
        ax.plot(xs, ys, color=it["color"], lw=lw, ls=ls, zorder=z, solid_joinstyle="miter",
                solid_capstyle="butt")


def _catmull(pts, n=16):
    P = [pts[0]] + pts + [pts[-1]]
    xs, ys = [], []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n + 1):
            t = k / n
            t2, t3 = t * t, t * t * t
            for j, arr in ((0, xs), (1, ys)):
                arr.append(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t +
                                  (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 +
                                  (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3))
    return xs, ys
