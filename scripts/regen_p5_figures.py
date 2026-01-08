"""Regenerate Phase 05 publication-grade figures from saved probing results (no retraining).
Saves PDF/SVG/PNG versions and performs basic smoke checks.
"""
from pathlib import Path
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import permutation_test
import numpy as np

OUTPUT_DIR = Path('results/probing')
RESULTS_FILE = OUTPUT_DIR / 'probing_results.json'

# Setup LaTeX if available, otherwise mathtext
import shutil
tex_exe = shutil.which('pdflatex') or shutil.which('xelatex') or shutil.which('lualatex')
if tex_exe:
    mpl.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Times'],
        'axes.labelsize': 11,
        'font.size': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
    })
    mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}\usepackage{bm}"
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    print(f"latex found: {tex_exe}")
else:
    mpl.rcParams.update({
        'text.usetex': False,
        'mathtext.fontset': 'cm',
        'font.family': 'serif',
    })
    print("latex not found: falling back to matplotlib mathtext")


def save_pub_fig(path_without_ext, fig=None, formats=('pdf','svg','png')):
    fig = fig if fig is not None else plt.gcf()
    base = Path(path_without_ext)
    saved = []
    for fmt in formats:
        p = base.with_suffix('.' + fmt)
        fig.savefig(p, dpi=300, bbox_inches='tight', format=fmt)
        saved.append(str(p.name))
    return saved


def main():
    with open(RESULTS_FILE) as f:
        res = json.load(f)

    layers = sorted(int(k) for k in res['layerwise_pd_probing'].keys())
    accuracies = [res['layerwise_pd_probing'][str(l)]['accuracy_mean'] for l in layers]
    accuracy_stds = [res['layerwise_pd_probing'][str(l)]['accuracy_std'] for l in layers]

    # compute permutation p-values
    p_vals = []
    selectivity = [res['selectivity_scores'][str(l)] for l in layers]
    for l in layers:
        pd_scores = res['layerwise_pd_probing'][str(l)]['accuracy_folds']
        ctrl_mean = res['control_probing'][str(l)]['accuracy_mean'] if str(l) in res['control_probing'] else res['control_probing'][l]['accuracy_mean']
        ctrl_scores = [ctrl_mean] * len(pd_scores)
        r = permutation_test((pd_scores, ctrl_scores), statistic=lambda x, y: np.mean(x) - np.mean(y),
                             permutation_type='independent', alternative='greater', n_resamples=2000, random_state=42)
        p_vals.append(r.pvalue)
    m = len(layers)
    p_vals_bonf = [min(p*m, 1.0) for p in p_vals]

    # 1) layerwise plot
    fig, ax = plt.subplots(figsize=(10,5))
    ax.errorbar(layers, accuracies, yerr=accuracy_stds, fmt='o-', capsize=5, color='tab:blue', linewidth=2)
    ax.axhline(0.5, color='red', linestyle='--', label='chance (0.5)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy (LOSO)')
    ax.set_title('Layer-wise PD probing accuracy')
    for i, pb in enumerate(p_vals_bonf):
        if pb < 0.05:
            ax.annotate('*', (layers[i], accuracies[i]+0.02), ha='center', fontsize=14)
    ax.set_ylim([0.4, 1.02])
    ax.grid(alpha=0.2)
    save_pub_fig(OUTPUT_DIR / 'fig_p5_01_layerwise_probing_regen', fig=fig)
    plt.close(fig)

    # 2) clinical feature heatmap
    features = list(res['clinical_feature_probing'].keys())
    heatmap = np.zeros((len(features), len(layers)))
    for i, feat in enumerate(features):
        for j, l in enumerate(layers):
            heatmap[i,j] = res['clinical_feature_probing'][feat][str(l)]['r2_mean']
    fig, ax = plt.subplots(figsize=(14,4))
    im = ax.imshow(heatmap, cmap='YlGnBu', aspect='auto', vmin=0, vmax=max(0.3, heatmap.max()))
    ax.set_xlabel('Layer')
    ax.set_ylabel('Clinical Feature')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    for i in range(len(features)):
        for j in range(len(layers)):
            ax.text(j, i, f'{heatmap[i,j]:.2f}', ha='center', va='center', fontsize=8)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$R^2$')
    save_pub_fig(OUTPUT_DIR / 'fig_p5_02_clinical_feature_heatmap_regen', fig=fig)
    plt.close(fig)

    # 3) selectivity bar plot
    fig, ax = plt.subplots(figsize=(10,3))
    ax.bar(layers, selectivity, color='tab:purple')
    for i, pb in enumerate(p_vals_bonf):
        if pb < 0.05:
            ax.annotate('*', (layers[i], selectivity[i]+0.02), ha='center')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Selectivity')
    ax.set_title('Selectivity (PD vs Random)')
    save_pub_fig(OUTPUT_DIR / 'fig_p5_03_selectivity_regen', fig=fig)
    plt.close(fig)

    # smoke checks
    files = [p.name for p in OUTPUT_DIR.iterdir() if p.suffix in ('.pdf','.svg','.png') and 'fig_p5_' in p.name]
    print('generated figures:', files)
    # ensure we have at least pdf for each
    expected = ['fig_p5_01_layerwise_probing_regen.pdf', 'fig_p5_02_clinical_feature_heatmap_regen.pdf', 'fig_p5_03_selectivity_regen.pdf']
    missing = [e for e in expected if not (OUTPUT_DIR / e).exists()]
    if missing:
        print('missing vectors:', missing)
    else:
        print('all vector figures present')

if __name__ == '__main__':
    main()
