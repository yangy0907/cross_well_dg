"""Editable vector overview aligned with the compact manuscript and released code."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / 'paper/gse_compact_20261104/figs'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(7.35, 7.2))
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis('off')
    ink, grey = '#243746', '#637786'
    blue, orange, teal = '#2E6289', '#A97122', '#26776D'
    texts = []

    def box(x, y, w, h, face='#F4F7FA', edge='#D2DDE4', lw=.85):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.18,rounding_size=1.0',
                                   facecolor=face, edgecolor=edge, linewidth=lw))

    def text(x, y, value, size=9, color=ink, weight='normal', ha='left', va='center'):
        artist = ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                         ha=ha, va=va, linespacing=1.5)
        texts.append(artist)

    def arrow(start, end, color=grey, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                                    linewidth=1.0, color=color,
                                    linestyle=(0, (3, 2)) if dashed else '-'))

    def label(y, letter, title):
        box(2, y - 1.3, 3.3, 3, ink, ink, 0)
        text(3.65, y + .15, letter, 8.3, 'white', 'bold', 'center')
        text(6.8, y + .15, title, 9.3, ink, 'bold')

    text(50, 97.7, 'Cross-well evaluation under explicit information budgets', 12.0,
         ink, 'bold', 'center')
    label(92.4, 'A', 'SPLIT WELLS BEFORE FITTING')
    box(2, 80.8, 46.5, 9.1, '#EEF4FA', '#B5CADB')
    text(4, 87.2, 'SOURCE WELLS', 9.1, blue, 'bold')
    text(4, 83.5, 'Training labels; separate validation wells', 8.7)
    box(51.5, 80.8, 46.5, 9.1, '#FBF5EA', '#E0CDAD')
    text(53.5, 87.2, 'TARGET WELLS', 9.1, orange, 'bold')
    text(53.5, 83.5, 'Curves and depth; labels for evaluation only', 8.4)
    text(50, 77.3, 'Global medians, standard deviations and imputation: fit on training rows only',
         8.5, grey, ha='center')

    label(72.2, 'B', 'CHOOSE ONE ADMISSIBLE FEATURE SET')
    cards = [
        (2, blue, '#EEF4FA', 'Inductive', 'Global-LGBM',
         'Global levels + masks\nNo target-well aggregates'),
        (34.7, orange, '#FBF5EA', 'Post-drilling', 'Hybrid-norel / Hybrid',
         'Global + full-well z + masks\nHybrid also uses relative depth'),
        (67.3, teal, '#EDF6F3', 'Causal prefix', 'Hybrid-while',
         'Global + prefix z + masks\nPast + current depths; no rel. depth'),
    ]
    for x, color, face, title, models, features in cards:
        box(x, 52.1, 30.7, 17.4, face, color, 1.05)
        text(x + 1.6, 66.3, title, 10, color, 'bold')
        text(x + 1.6, 62.6, models, 8.5, ink, 'bold')
        text(x + 1.6, 57.2, features, 8.1)
        arrow((x + 15.35, 51.9), (x + 15.35, 49.8))
    ax.plot([17.35, 82.65], [49.8, 49.8], color=grey, linewidth=1)
    arrow((50, 49.8), (50, 47.9))
    box(2, 39.1, 96, 8.6, '#F2F5F8', '#A7BAC7')
    text(50, 44.9, 'CPU LightGBM: source fitting and held-out-well early stopping',
         9.2, ink, 'bold', 'center')
    text(50, 41.2, 'Freeze source transforms and classifier before target prediction', 8.6,
         grey, ha='center')

    label(34.5, 'C', 'REPORT PERFORMANCE AND ITS INFORMATION SUPPORT')
    box(2, 24.2, 96, 7.6, '#F2F6FA', '#C5D3DE')
    text(50, 29.1, 'Macro-F1  |  Worst-fixed  |  W10', 10, ink, 'bold', 'center')
    text(50, 26, '18 closed splits; 2 open stresses; parent-grouped companion', 8.6,
         grey, ha='center')

    label(19.6, 'D', 'TEST INPUT AND INTERVAL DEPENDENCE')
    box(2, 5.0, 46.5, 12, '#F6F8FA', '#CBD7E0')
    text(4, 14.1, 'Curve comparison: refitted models', 8.8, blue, 'bold')
    text(4, 9.2, '4 versus 16 curves; identical labelled rows\nThree new parent-grouped partitions', 8.2)
    box(51.5, 5.0, 46.5, 12, '#F6F8FA', '#CBD7E0')
    text(53.5, 14.1, 'Coordinate stress: inference only', 8.8, teal, 'bold')
    text(53.5, 9.2, 'Top / bottom / both: 10% and 20%\nMatched retained depths; other features fixed', 8.1)
    text(50, 1.65, 'Fixed-taxonomy tails include class coverage. Centred voting is an optional offline diagnostic.',
         7.8, grey, ha='center')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extent = fig.get_window_extent(renderer)
    clipped = []
    for artist in texts:
        bounds = artist.get_window_extent(renderer)
        if bounds.x0 < extent.x0 or bounds.y0 < extent.y0 or bounds.x1 > extent.x1 or bounds.y1 > extent.y1:
            clipped.append(artist.get_text())
    assert not clipped, clipped
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'figM_method_overview.{suffix}', dpi=240, facecolor='white')
    (OUT.parent / 'audit/figure1_checks.json').write_text(json.dumps({
        'formats': ['pdf', 'svg', 'png'], 'clipped_text': clipped,
        'branches': ['global', 'hybrid_norel/hybrid', 'hybrid_while'],
        'new_content': ['matched 4/16-curve fitting', 'new parent-grouped partitions', 'inference-only interval-coordinate stress'],
        'target_labels': 'evaluation and coverage bookkeeping only; no target-label fitting or tuning',
        'original_figure_preserved': True}, indent=2), encoding='utf-8')
    plt.close(fig)
    draw_pareto()


def draw_pareto():
    import pandas as pd
    root = OUT.parents[2]
    source = root / 'results/followup/protocol1_multirep_well_id_closed_summary.csv'
    data = pd.read_csv(source)
    methods = [('lgbm_global', 'Global-LGBM', '#2E6289', 'o'),
               ('lgbm_hybrid_norel', 'Hybrid-norel', '#26776D', 's'),
               ('lgbm_hybrid', 'Hybrid', '#A97122', '^'),
               ('lgbm_hybrid_while', 'Hybrid-while', '#826293', 'D')]
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for method, label, color, marker in methods:
        row = data.loc[data.method == method].iloc[0]
        ax.errorbar(row.worst_fixed_mean, row.macro_legacy_mean,
                    xerr=row.worst_fixed_std, yerr=row.macro_legacy_std,
                    fmt=marker, color=color, ecolor=color, capsize=3, elinewidth=1.0,
                    markersize=6, label=label)
    ax.set(xlabel='Worst-fixed F1 (mean ± split SD)', ylabel='Pooled Macro-F1 (mean ± split SD)',
           title='Pooled performance and well tails on 18 closed splits')
    ax.grid(alpha=.2, linewidth=.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8.4, loc='lower right')
    fig.tight_layout()
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'fig_compact_pareto.{suffix}', dpi=240, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
