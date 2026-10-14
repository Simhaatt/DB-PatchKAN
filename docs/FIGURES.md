# Figure inventory and reproduction

The repository owner's `figures.zip` was supplied on 14 October 2026. Its files are included unchanged, with member hashes in `FIGURE_ARCHIVE_MANIFEST.json`. Archive README/scripts are retained as source documentation, not as new empirical evidence.

- `figures/drawio/Fig1.drawio` … `Fig9.drawio`: editable diagrams; numerical plots use embedded true vector geometries.
- `figures/vector/Fig1.pdf` … `Fig9.pdf` and matching SVGs: author-supplied vectors.
- `figures/preview/Fig1.png` … `Fig9.png`: supplied preview renders.
- `figures/supplied/Figure_8_high_visibility.pdf`: the alternate original-manuscript Figure 8 renderer.
- `figures/supplied/Figure_10_high_visibility.pdf`: actual five-seed KAN plot supplied in the archive.

The archive did **not** include a generated `Fig10.drawio`, `vector/Fig10.svg` or fitted-curve input NPZ. The actual high-visibility Figure 10 PNG/SVG/PDF, saved curve CSV and project rendering code are additionally included under `figures/high_visibility_8_10/` and `figures/data/`. These are from the same existing project outputs, not invented curves. The saved Figure 10 seed grid/mean/band can be exported to the input format expected by the supplied script:

```bash
python scripts/export_fig10_saved_npz.py --output outputs/fig10_saved_curves.npz
python figures/scripts/fig10_kan_responses.py outputs/fig10_saved_curves.npz outputs/Fig10
```

Use the supplied band, without `--percentile-band`, for exact data-method reproduction. The band is the original pointwise 2.5th–97.5th percentile range across five fitted seeds, not measurement uncertainty or an independent-vehicle confidence interval. Centering and normalized concept grids remain unchanged.

To reproduce the supplied schematic layouts and numerical bars:

```bash
python figures/scripts/build_diagrams.py outputs/figures_rebuilt
python figures/scripts/build_plots.py outputs/figures_rebuilt
```

The provided scripts require Matplotlib/NumPy and Liberation Sans for their intended appearance. Use their PDFs/SVGs directly if that font is unavailable. Installing a different font may alter layout; verify at journal width.

Figures 7–9 plot the original manuscript numerical tables. Figure 8 is not a selective Stage 15 KAN result; no Stage 15 intervals have been added to those original bars. Figure 7's historical response tasks must not be conflated with Stage 15 E4 truly future response tasks. Current paper caption/interpretation should follow the latest evidence review.

The uploaded figure files are not regenerated or renumbered by package assembly. The higher-visibility alternatives remain separate. Code is MIT-licensed; no separate CC-BY figure-reuse license has been asserted.
