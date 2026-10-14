# DB-PatchKAN figures

## Folders

- `drawio/` holds the editable diagrams. Fig1–Fig6 are native draw.io shapes with LaTeX labels (Extras > Mathematical Typesetting is switched on in each file). Fig7–Fig9 hold the plot SVG as a single image so you can do final layout in draw.io.
- `vector/` has PDF and SVG versions of Fig1–Fig9. SVG text is converted to outlines and PDF fonts are embedded.
- `preview/` has 300-dpi PNGs for quick viewing only. Don't submit these.
- `supplied/` has your own Figure_8_high_visibility.pdf and Figure_10_high_visibility.pdf, added unchanged.
- `scripts/` has everything needed to rebuild: `build_diagrams.py` (Fig1–6), `build_plots.py` (Fig7–9), `fig10_kan_responses.py` (Fig10).

## Sizing

At 100 % zoom, 1 draw.io unit equals 1/100 in. The 740-unit canvases are 188 mm wide, which is two-column width. Font sizes are 10–11.5 units (7.2–8.3 pt) at that size. All text is bold, and equations use bold math (\boldsymbol in draw.io). Fig5, Fig8 and Fig9 are 140 mm (1.5-column). If you rescale a figure, scale the whole page, not individual labels.

## Notes on specific figures

- **Fig3.** Each attention module takes both streams. If both cross-stream connectors were drawn between the two columns they would have to cross. So the H_D to "Battery queries driver" connector runs around the outside, below the lower row, and the figure has no crossings.
- **Fig2.** The battery input lists only current, voltage, SOC and temperature. Add any other channel only if the model actually uses it.
- **Fig4.** "15 named concepts (Nissan configuration)" is copied from the spec. Check it against the final concept table.
- **Fig5.** The three spline glyphs are schematic. They are not fitted curves.
- **Fig7–9.** All values come from the tables in the figure specification. Bar ends, value labels and axis limits are generated in code.
- **Fig6, Fig7, Fig9, Fig10.** Panels sit side by side, with subcaptions under each panel.
- **Fig8.** This is the original-manuscript version. I did not draw the Stage 15 E2 alternative because it needs the saved bootstrap intervals.
- **Fig10.** Not drawn. The curves must come from the saved seed arrays. Run:

  `python scripts/fig10_kan_responses.py curves.npz Fig10`

  The script docstring describes the expected `.npz` keys. It refuses to run without the saved 95 % range unless you pass `--percentile-band`. If you use that flag, the caption must name the method.

## Rebuild

```
cd scripts
python build_diagrams.py ../vector_out
python build_plots.py ../vector_out
```
Requires matplotlib and Liberation Sans (metric-compatible with Arial).
