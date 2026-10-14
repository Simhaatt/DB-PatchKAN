"""Redraw only Figures 8 and 10 using exact specified/saved numerical geometry.

Non-destructive output: figures/high_visibility_8_10. No model inference/training.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'high_visibility_8_10'
OUT.mkdir(exist_ok=True)
INK = '#30343A'
BLUE = '#2F658C'
SEED = '#9AA0A7'
BAND = '#D6E5F0'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 12, 'font.weight': 'normal', 'text.color': INK,
    'axes.labelsize': 13, 'axes.labelweight': 'bold', 'axes.labelcolor': INK,
    'axes.edgecolor': INK, 'axes.linewidth': 1.1,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'xtick.color': INK, 'ytick.color': INK, 'legend.fontsize': 12,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'axes.unicode_minus': False,
})


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=4, width=1.0, pad=6)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#DEE1E5', linewidth=0.65)


def save(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for obj in fig.findobj(Text):
        if obj.get_visible() and obj.get_text().strip():
            bbox = obj.get_window_extent(renderer)
            if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.x1+1 or bbox.y1 > fig.bbox.y1+1:
                clipped.append(obj.get_text())
    if clipped:
        raise ValueError(f'Clipped text in {stem}: {clipped}')
    fig.savefig(OUT/f'{stem}.svg')
    fig.savefig(OUT/f'{stem}.pdf', metadata={'Title': stem, 'Creator': 'DB-PatchKAN data-driven figure renderer'})
    fig.savefig(OUT/f'{stem}.png', dpi=600, pil_kwargs={'compress_level': 3})
    fig.savefig(OUT/f'{stem}_preview.png', dpi=160)
    plt.close(fig)
    return {'figure': stem, 'dpi': 600, 'clipped_text_count': len(clipped),
            'formats': ['png','svg','pdf'], 'minimum_tick_font_pt': 12}


def figure8():
    source = ROOT/'Figure_8_data.csv'
    rows = table(source)
    persistence = np.array([float(r['Strict_Persistence_MAE_Ah']) for r in rows])
    proposed = np.array([float(r['DB_PatchKAN_MAE_Ah']) for r in rows])
    assert np.array_equal(persistence, [.2932,.3825,.4710,.6990])
    assert np.array_equal(proposed, [.3034,.3818,.4638,.6685])
    labels = ['≤500 km', '500–1000 km', '1000–2000 km', '>2000 km']
    fig, ax = plt.subplots(figsize=(7.6,4.35))
    fig.subplots_adjust(left=.115,right=.985,bottom=.19,top=.82)
    x = np.arange(4)
    width = .34
    bars = [ax.bar(x-.19,persistence,width,color='#E0E2E5',edgecolor='#737981',linewidth=.85,
                   label='Strict persistence'),
            ax.bar(x+.19,proposed,width,color=BLUE,edgecolor=INK,linewidth=.85,label='DB-PatchKAN')]
    for group in bars:
        for bar in group:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.016,
                    f'{bar.get_height():.4f}',ha='center',va='bottom',fontsize=12)
    ax.set_xticks(x,labels)
    ax.set_xlim(-.65,3.65)
    ax.set_ylim(0,.80)
    ax.set_yticks(np.arange(0,.81,.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel('MAE (Ah)',labelpad=10)
    ax.set_xlabel('Forecast distance',labelpad=12)
    style(ax)
    fig.legend(handles=bars,labels=['Strict persistence','DB-PatchKAN'],
               loc='upper center',bbox_to_anchor=(.55,.975),ncol=2,frameon=False,
               columnspacing=2.2,handlelength=1.9)
    shutil.copy2(source,OUT/'Figure_8_data.csv')
    return save(fig,'Figure_8_high_visibility')


def figure10():
    source = ROOT/'Figure_10_data.csv'
    rows = table(source)
    assert rows and all(r['Actual_Fitted_Data']=='YES' for r in rows)
    assert all(r['Output_Name']=='current_soh' for r in rows)
    fig, axes = plt.subplots(1,2,figsize=(7.6,4.5))
    fig.subplots_adjust(left=.135,right=.955,bottom=.29,top=.81,wspace=.29)
    for panel,(ax,concept,title) in enumerate(zip(axes,['bsi_rate','bsi_cycle'],
                                                ['D-BSI rate concept','D-BSI cycle concept'])):
        subset = [r for r in rows if r['Concept_Name']==concept]
        fitted = [r for r in subset if r['Curve_Type']=='Fitted seed']
        aggregate = sorted([r for r in subset if r['Curve_Type']=='Across-seed aggregate'],
                           key=lambda r:float(r['Normalized_Concept_Value']))
        seeds = sorted(set(r['Seed'] for r in fitted),key=int)
        assert len(seeds)>=2 and len(aggregate)>1
        all_values = []
        for seed in seeds:
            curve = sorted([r for r in fitted if r['Seed']==seed],
                           key=lambda r:float(r['Normalized_Concept_Value']))
            xx = np.array([float(r['Normalized_Concept_Value']) for r in curve])
            yy = np.array([float(r['Centered_KAN_Response_pp']) for r in curve])
            assert np.isfinite(xx).all() and np.isfinite(yy).all()
            assert np.all(np.diff(xx)>0)
            all_values.extend(yy)
            ax.plot(xx,yy,color=SEED,linewidth=1.15,alpha=.88,zorder=2)
        xx = np.array([float(r['Normalized_Concept_Value']) for r in aggregate])
        yy = np.array([float(r['Centered_KAN_Response_pp']) for r in aggregate])
        low = np.array([float(r['Lower_pp']) for r in aggregate])
        high = np.array([float(r['Upper_pp']) for r in aggregate])
        assert np.all(low<=high) and np.isfinite(yy).all()
        ax.fill_between(xx,low,high,color=BAND,alpha=.85,linewidth=0,zorder=1)
        ax.plot(xx,yy,color=BLUE,linewidth=2.7,zorder=3)
        # Preserve each panel's real fitted range; do not force symmetry/monotonicity.
        ymin,ymax=min(all_values),max(all_values)
        padding=max((ymax-ymin)*.09,1e-6)
        ax.set_ylim(ymin-padding,ymax+padding)
        ax.set_xlim(0,1)
        ax.set_xticks([0,.25,.5,.75,1])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style(ax)
        ax.text(.5,-.245,f'({chr(97+panel)}) {title}',transform=ax.transAxes,
                ha='center',va='top',fontsize=12,fontweight='bold')
    fig.text(.016,.545,'Centered KAN response\n(SOH percentage points)',
             ha='left',va='center',rotation=90,fontsize=12.5,fontweight='bold')
    fig.text(.56,.045,'Normalized concept value',ha='center',va='bottom',
             fontsize=13,fontweight='bold')
    handles=[Line2D([0],[0],color=SEED,linewidth=1.15,label='Individual fitted seeds'),
             Patch(facecolor=BAND,edgecolor='none',label='95% seed range'),
             Line2D([0],[0],color=BLUE,linewidth=2.7,label='Across-seed mean')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.985),ncol=3,
               frameon=False,columnspacing=1.2,handlelength=1.8,fontsize=11.5)
    shutil.copy2(source,OUT/'Figure_10_data.csv')
    result=save(fig,'Figure_10_high_visibility')
    result['seed_counts']={c:len(set(r['Seed'] for r in rows if r['Concept_Name']==c
                                     and r['Curve_Type']=='Fitted seed')) for c in ['bsi_rate','bsi_cycle']}
    return result


def main():
    results=[figure8(),figure10()]
    provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [ROOT/'Figure_8_data.csv',ROOT/'Figure_10_data.csv']}
    (OUT/'render_manifest.json').write_text(json.dumps({'outputs':results,'source_sha256':provenance,
       'figure8_evidence':'Exact original manuscript row-weighted MAE values, not selective Stage15 forecasts.',
       'figure10_evidence':'Existing fitted/centered seed responses and saved pointwise .025/.975 quantiles.',
       'centering':'Each fitted seed response centered at normalized concept value 0.5 in original export.',
       'seed_band_is_not':'a measurement-error interval or independent-vehicle statistical confidence interval',
       'originals_overwritten':False},indent=2),encoding='utf-8')
    (OUT/'CAPTIONS.md').write_text('''# Figure captions and evidence notes

Figure 8. Next-checkpoint capacity-forecasting MAE by mileage horizon on the EVBattery cohort. Strict persistence performs better at the shortest distance, while the learned residual is more favorable over longer intervals. Bars reproduce the original manuscript row-weighted MAE values, not Stage 15 selective forecasts or vehicle-macro estimates. No significance markers or uncertainty bars are added.

Figure 10. Representative fitted KAN functions for rate- and cycle-related D-BSI concepts. Thin gray curves show individual fitted seeds; the blue line shows their mean; the pale band reproduces the saved pointwise 2.5th–97.5th seed quantiles. Responses are centered at normalized concept value 0.5. This seed band is not a measurement-error interval or a confidence interval for independent vehicles. Curves illustrate inspectable mappings and do not establish robust importance or causality. Panel y-limits retain the respective fitted ranges.

The CSV inputs are copied unchanged. Original figures remain untouched. SVG text remains editable; the PDF embeds TrueType fonts; PNG output uses 600 dpi. SVG may be imported into diagrams.net for layout work without redrawing the empirical curves.
''',encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
