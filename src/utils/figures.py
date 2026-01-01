"""
publication figure generator for isef-quality research figures.

generates comprehensive, multi-panel figures suitable for publication,
poster presentations, and competition submissions.

all figures use:
- times new roman font (serif) for publication standards
- latex rendering for mathematical notation
- 300+ dpi for print quality
- proper panel labeling (a, b, c...)
- consistent color schemes
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats


# set publication-quality defaults with times new roman and latex
def set_publication_style():
    """
    set matplotlib style for publication-quality figures.
    
    uses times new roman font and latex rendering for proper
    scientific publication formatting.
    """
    # try to use latex if available, fallback gracefully
    try:
        plt.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{times}\usepackage{amsmath}\usepackage{amssymb}'
        })
        _latex_available = True
    except:
        _latex_available = False
    
    plt.rcParams.update({
        # font settings - times new roman for publication
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
        'font.size': 11,
        
        # axes settings
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'axes.linewidth': 1.0,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'normal',
        'axes.spines.top': True,
        'axes.spines.right': True,
        
        # tick settings
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        
        # legend settings
        'legend.fontsize': 9,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        
        # figure settings
        'figure.dpi': 150,
        'figure.figsize': (7, 5),
        'figure.autolayout': False,
        
        # save settings - high quality for publication
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.format': 'pdf',
        
        # line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'lines.markeredgewidth': 1.0,
        
        # grid settings
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        
        # math text
        'mathtext.fontset': 'stix',
        
        # patch settings
        'patch.linewidth': 1.0,
        'patch.edgecolor': 'black',
    })


def set_latex_style():
    """
    set full latex style for maximum publication quality.
    
    requires latex installation on system.
    """
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'''
            \usepackage{times}
            \usepackage{amsmath}
            \usepackage{amssymb}
            \usepackage{bm}
        ''',
        'font.family': 'serif',
        'font.serif': ['Times'],
    })


def reset_style():
    """reset matplotlib to default style."""
    plt.rcdefaults()


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.08, fontsize: int = 14):
    """
    add panel label (a, b, c, etc.) to axes.
    
    args:
        ax: matplotlib axes
        label: label text (e.g., 'A', 'B', 'C')
        x: x position in axes coordinates
        y: y position in axes coordinates
        fontsize: font size
    """
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize, 
            fontweight='bold', va='top', ha='left')


def save_figure(fig, path: Union[str, Path], formats: List[str] = ['pdf', 'png', 'svg']):
    """
    save figure in multiple formats for publication.
    
    args:
        fig: matplotlib figure
        path: base path (without extension)
        formats: list of formats to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        fig.savefig(path.with_suffix(f'.{fmt}'), 
                   dpi=300 if fmt == 'png' else None,
                   bbox_inches='tight',
                   pad_inches=0.05)


# color palettes for different figure types
PALETTES = {
    'main': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B'],
    'categorical': ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3'],
    'diverging': ['#2166AC', '#67A9CF', '#F7F7F7', '#EF8A62', '#B2182B'],
    'sequential': ['#F7FBFF', '#C6DBEF', '#6BAED6', '#2171B5', '#08306B'],
    'hypothesis': ['#2E8B57', '#DC143C'],  # green for supported, red for not
}


class FigureGenerator:
    """
    generate publication-quality figures for the interpretability project.
    
    all figures follow isef/regeneron sts quality standards.
    """
    
    def __init__(self, output_dir: str = 'results/figures'):
        """
        args:
            output_dir: directory for saving figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        set_publication_style()
    
    def figure_1_overview(
        self,
        probing_results: Dict[int, Dict],
        patching_results: Dict[int, Dict],
        model_accuracy: float = 0.85,
        save: bool = True
    ) -> plt.Figure:
        """
        figure 1: project overview figure for abstract/poster.
        
        three-panel figure showing:
        a) layer-wise probing accuracy
        b) layer-wise patching recovery
        c) combined important layers identification
        
        args:
            probing_results: probing experiment results
            patching_results: patching experiment results
            model_accuracy: overall model accuracy
            save: whether to save figure
            
        returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
        
        colors = PALETTES['main']
        
        # panel a: probing accuracy
        ax1 = fig.add_subplot(gs[0])
        
        layers = sorted(probing_results.keys())
        accuracies = [probing_results[l].get('mean', probing_results[l].get('mean_score', 0)) for l in layers]
        stds = [probing_results[l].get('std', probing_results[l].get('std_score', 0)) for l in layers]
        
        ax1.errorbar(layers, accuracies, yerr=stds, marker='o', capsize=4,
                    color=colors[0], markerfacecolor='white', markeredgewidth=2)
        ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='chance')
        
        best_layer = layers[np.argmax(accuracies)]
        ax1.plot(best_layer, max(accuracies), 'r*', markersize=15, 
                label=f'best: layer {best_layer}', zorder=5)
        
        ax1.set_xlabel('transformer layer', fontweight='bold')
        ax1.set_ylabel('probing accuracy', fontweight='bold')
        ax1.set_title('(a) where is pd information encoded?', fontweight='bold')
        ax1.set_xticks(layers)
        ax1.set_ylim([0.4, 1.0])
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # add panel label
        ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold')
        
        # panel b: patching recovery
        ax2 = fig.add_subplot(gs[1])
        
        patching_layers = sorted(patching_results.keys())
        recoveries = [patching_results[l].get('mean_recovery', 0) for l in patching_layers]
        rec_stds = [patching_results[l].get('std_recovery', 0) for l in patching_layers]
        
        bar_colors = [colors[3] if r > 0.1 else colors[0] for r in recoveries]
        
        ax2.bar(patching_layers, recoveries, yerr=rec_stds, capsize=3,
               color=bar_colors, edgecolor='black', alpha=0.8)
        ax2.axhline(y=0.1, color='black', linestyle='--', linewidth=1.5, 
                   label='importance threshold')
        ax2.axhline(y=0, color='gray', linewidth=1)
        
        ax2.set_xlabel('transformer layer', fontweight='bold')
        ax2.set_ylabel('logit difference recovery', fontweight='bold')
        ax2.set_title('(b) which layers are causally important?', fontweight='bold')
        ax2.set_xticks(patching_layers)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold')
        
        # panel c: combined analysis
        ax3 = fig.add_subplot(gs[2])
        
        # compute correlation between probing and patching
        common = set(layers) & set(patching_layers)
        if len(common) >= 3:
            common_layers = sorted(common)
            probe_vals = [probing_results[l].get('mean', probing_results[l].get('mean_score', 0)) for l in common_layers]
            patch_vals = [patching_results[l].get('mean_recovery', 0) for l in common_layers]
            
            ax3.scatter(probe_vals, patch_vals, s=100, c=colors[0], edgecolors='black', zorder=5)
            
            for layer, pv, rv in zip(common_layers, probe_vals, patch_vals):
                ax3.annotate(f'L{layer}', (pv, rv), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9)
            
            # trend line
            r, p = stats.spearmanr(probe_vals, patch_vals)
            z = np.polyfit(probe_vals, patch_vals, 1)
            poly = np.poly1d(z)
            x_line = np.linspace(min(probe_vals), max(probe_vals), 100)
            ax3.plot(x_line, poly(x_line), '--', color=colors[1], alpha=0.7,
                    label=f'r = {r:.2f}, p = {p:.3f}')
            
            ax3.set_xlabel('probing accuracy', fontweight='bold')
            ax3.set_ylabel('patching recovery', fontweight='bold')
            ax3.set_title('(c) correlation: encoding vs causality', fontweight='bold')
            ax3.legend(loc='lower right')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'insufficient data', ha='center', va='center',
                    transform=ax3.transAxes)
        
        ax3.text(-0.15, 1.05, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold')
        
        # main title
        fig.suptitle(
            f'mechanistic interpretability of wav2vec2 for pd detection (model accuracy: {model_accuracy:.1%})',
            fontsize=14, fontweight='bold', y=1.02
        )
        
        if save:
            fig.savefig(self.output_dir / 'figure1_overview.png', dpi=300, bbox_inches='tight')
            fig.savefig(self.output_dir / 'figure1_overview.pdf', bbox_inches='tight')
        
        return fig
    
    def figure_2_clinical_encoding(
        self,
        clinical_results: Dict[str, Dict[int, Dict]],
        feature_categories: Optional[Dict[str, List[str]]] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        figure 2: clinical feature encoding heatmap.
        
        shows which layers encode which clinical features,
        organized by feature type (phonatory vs prosodic).
        
        args:
            clinical_results: nested dict feature -> layer -> metrics
            feature_categories: dict mapping category to feature names
            save: whether to save
            
        returns:
            matplotlib figure
        """
        if feature_categories is None:
            feature_categories = {
                'phonatory': ['jitter_local', 'jitter_rap', 'shimmer_local', 'shimmer_apq3'],
                'prosodic': ['f0_mean', 'f0_std', 'hnr']
            }
        
        # get all features and layers
        all_features = list(clinical_results.keys())
        if not all_features:
            raise ValueError("no clinical results available")
        
        sample_feat = all_features[0]
        layers = sorted(clinical_results[sample_feat].keys())
        n_layers = len(layers)
        
        # build matrix
        feature_order = []
        for cat, feats in feature_categories.items():
            feature_order.extend([f for f in feats if f in all_features])
        
        # add any remaining features
        feature_order.extend([f for f in all_features if f not in feature_order])
        
        matrix = np.zeros((len(feature_order), n_layers))
        
        for i, feat in enumerate(feature_order):
            for j, layer in enumerate(layers):
                if layer in clinical_results.get(feat, {}):
                    data = clinical_results[feat][layer]
                    if isinstance(data, dict):
                        matrix[i, j] = data.get('mean', data.get('r2_score', 0))
                    else:
                        matrix[i, j] = float(data)
        
        # create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # create heatmap
        im = ax.imshow(matrix, cmap='viridis', aspect='auto', vmin=0)
        
        # add value annotations
        for i in range(len(feature_order)):
            for j in range(n_layers):
                val = matrix[i, j]
                color = 'white' if val > 0.3 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       fontsize=9, color=color)
        
        # set ticks
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(layers)
        ax.set_yticks(range(len(feature_order)))
        ax.set_yticklabels(feature_order)
        
        ax.set_xlabel('transformer layer', fontweight='bold', fontsize=12)
        ax.set_ylabel('clinical feature', fontweight='bold', fontsize=12)
        ax.set_title('clinical feature encoding across layers (r² score)', 
                    fontweight='bold', fontsize=14)
        
        # add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('r² score', fontsize=11)
        
        # add category annotations
        y_pos = 0
        for cat, feats in feature_categories.items():
            cat_feats = [f for f in feats if f in feature_order]
            if cat_feats:
                cat_end = y_pos + len(cat_feats) - 1
                ax.axhline(y=cat_end + 0.5, color='white', linewidth=2)
                ax.text(-0.7, (y_pos + cat_end) / 2, cat.upper(), 
                       ha='right', va='center', fontweight='bold', fontsize=10, rotation=90)
                y_pos = cat_end + 1
        
        # highlight best layer per feature
        for i, feat in enumerate(feature_order):
            best_j = np.argmax(matrix[i, :])
            rect = plt.Rectangle((best_j - 0.5, i - 0.5), 1, 1, 
                                 fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'figure2_clinical_encoding.png', dpi=300, bbox_inches='tight')
            fig.savefig(self.output_dir / 'figure2_clinical_encoding.pdf', bbox_inches='tight')
        
        return fig
    
    def figure_3_hypothesis_summary(
        self,
        hypothesis_results: Dict[str, Dict],
        save: bool = True
    ) -> plt.Figure:
        """
        figure 3: hypothesis testing summary.
        
        visual summary of all three hypothesis tests.
        
        args:
            hypothesis_results: dict with h1, h2, h3 test results
            save: whether to save
            
        returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        hypotheses = ['hypothesis_1', 'hypothesis_2', 'hypothesis_3']
        titles = [
            'H1: Clinical Encoding',
            'H2: Causal Dependency', 
            'H3: Generalization'
        ]
        descriptions = [
            'Phonatory features in\nearly layers (2-4)\nProsodic features in\nmiddle layers (5-8)',
            'Model predictions\ncausally depend on\nclinical representations',
            'Clinical alignment\npredicts cross-dataset\ngeneralization'
        ]
        
        for ax, h_key, title, desc in zip(axes, hypotheses, titles, descriptions):
            result = hypothesis_results.get(h_key, {})
            supported = result.get('supported', False)
            
            # background color based on support
            color = '#90EE90' if supported else '#FFB6C1'  # light green or light red
            ax.set_facecolor(color)
            
            # verdict text
            verdict = 'SUPPORTED' if supported else 'NOT SUPPORTED'
            verdict_color = '#006400' if supported else '#8B0000'  # dark green or dark red
            
            ax.text(0.5, 0.85, title, ha='center', va='top', fontweight='bold', 
                   fontsize=14, transform=ax.transAxes)
            
            ax.text(0.5, 0.60, desc, ha='center', va='top', fontsize=10, 
                   transform=ax.transAxes, style='italic')
            
            ax.text(0.5, 0.25, verdict, ha='center', va='center', fontweight='bold',
                   fontsize=16, color=verdict_color, transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=verdict_color, linewidth=2))
            
            # add key metric if available
            if h_key == 'hypothesis_1':
                phonatory = result.get('phonatory_mean_layer')
                prosodic = result.get('prosodic_mean_layer')
                if phonatory and prosodic:
                    ax.text(0.5, 0.08, f'Phonatory: L{phonatory:.1f}, Prosodic: L{prosodic:.1f}',
                           ha='center', va='bottom', fontsize=9, transform=ax.transAxes)
            
            elif h_key == 'hypothesis_2':
                corr = result.get('probing_patching_correlation', {})
                if corr:
                    r = corr.get('spearman_r', 0)
                    p = corr.get('spearman_p', 1)
                    ax.text(0.5, 0.08, f'Correlation: r={r:.2f}, p={p:.3f}',
                           ha='center', va='bottom', fontsize=9, transform=ax.transAxes)
            
            elif h_key == 'hypothesis_3':
                corr = result.get('correlation')
                if corr:
                    if isinstance(corr, dict):
                        r = corr.get('spearman_r', 0)
                    else:
                        r = corr
                    ax.text(0.5, 0.08, f'Alignment-Gen. r={r:.2f}',
                           ha='center', va='bottom', fontsize=9, transform=ax.transAxes)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            for spine in ax.spines.values():
                spine.set_linewidth(2)
        
        fig.suptitle('hypothesis testing results', fontweight='bold', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'figure3_hypothesis_summary.png', dpi=300, bbox_inches='tight')
            fig.savefig(self.output_dir / 'figure3_hypothesis_summary.pdf', bbox_inches='tight')
        
        return fig
    
    def figure_4_cross_dataset(
        self,
        evaluation_matrix: np.ndarray,
        dataset_names: List[str],
        alignment_scores: Optional[Dict[str, float]] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        figure 4: cross-dataset generalization.
        
        shows nxn evaluation matrix and alignment-generalization relationship.
        
        args:
            evaluation_matrix: [n_datasets, n_datasets] accuracy matrix
            dataset_names: names of datasets
            alignment_scores: optional clinical alignment scores per dataset
            save: whether to save
            
        returns:
            matplotlib figure
        """
        n_datasets = len(dataset_names)
        
        if alignment_scores:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig, ax1 = plt.subplots(figsize=(8, 6))
        
        # panel a: evaluation matrix
        im = ax1.imshow(evaluation_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0)
        
        # add values
        for i in range(n_datasets):
            for j in range(n_datasets):
                val = evaluation_matrix[i, j]
                color = 'white' if val < 0.7 else 'black'
                ax1.text(j, i, f'{val:.2f}', ha='center', va='center', 
                        fontsize=11, fontweight='bold', color=color)
        
        ax1.set_xticks(range(n_datasets))
        ax1.set_yticks(range(n_datasets))
        ax1.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax1.set_yticklabels(dataset_names)
        
        ax1.set_xlabel('test dataset', fontweight='bold')
        ax1.set_ylabel('train dataset', fontweight='bold')
        ax1.set_title('(a) cross-dataset evaluation matrix', fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label('accuracy', fontsize=11)
        
        # highlight diagonal
        for i in range(n_datasets):
            rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, 
                                 fill=False, edgecolor='black', linewidth=3)
            ax1.add_patch(rect)
        
        # panel b: alignment vs generalization (if available)
        if alignment_scores:
            # compute generalization score (mean off-diagonal performance)
            gen_scores = []
            for i, name in enumerate(dataset_names):
                off_diag = [evaluation_matrix[i, j] for j in range(n_datasets) if i != j]
                gen_scores.append(np.mean(off_diag) if off_diag else 0)
            
            align_vals = [alignment_scores.get(name, 0) for name in dataset_names]
            
            ax2.scatter(align_vals, gen_scores, s=150, c=PALETTES['main'][0], 
                       edgecolors='black', zorder=5)
            
            for name, av, gv in zip(dataset_names, align_vals, gen_scores):
                ax2.annotate(name, (av, gv), xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            # trend line
            if len(align_vals) >= 3:
                r, p = stats.spearmanr(align_vals, gen_scores)
                z = np.polyfit(align_vals, gen_scores, 1)
                poly = np.poly1d(z)
                x_line = np.linspace(min(align_vals), max(align_vals), 100)
                ax2.plot(x_line, poly(x_line), '--', color=PALETTES['main'][1],
                        label=f'r = {r:.2f}, p = {p:.3f}')
                ax2.legend(loc='lower right')
            
            ax2.set_xlabel('clinical alignment score', fontweight='bold')
            ax2.set_ylabel('generalization score', fontweight='bold')
            ax2.set_title('(b) alignment predicts generalization', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'figure4_cross_dataset.png', dpi=300, bbox_inches='tight')
            fig.savefig(self.output_dir / 'figure4_cross_dataset.pdf', bbox_inches='tight')
        
        return fig
    
    def figure_5_attention_analysis(
        self,
        head_importance: Dict[Tuple[int, int], float],
        n_layers: int = 12,
        n_heads: int = 12,
        save: bool = True
    ) -> plt.Figure:
        """
        figure 5: attention head importance analysis.
        
        heatmap of attention head patching importance.
        
        args:
            head_importance: dict mapping (layer, head) to importance score
            n_layers: number of layers
            n_heads: number of heads per layer
            save: whether to save
            
        returns:
            matplotlib figure
        """
        # build matrix
        matrix = np.zeros((n_layers, n_heads))
        
        for (layer, head), score in head_importance.items():
            if 0 <= layer < n_layers and 0 <= head < n_heads:
                matrix[layer, head] = score
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # create diverging colormap centered at 0
        vmax = max(abs(matrix.min()), abs(matrix.max()))
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        
        ax.set_xlabel('attention head', fontweight='bold', fontsize=12)
        ax.set_ylabel('transformer layer', fontweight='bold', fontsize=12)
        ax.set_title('attention head importance (patching recovery)', fontweight='bold', fontsize=14)
        
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('recovery score', fontsize=11)
        
        # highlight top 10 important heads
        flat_scores = [(score, layer, head) for (layer, head), score in head_importance.items()]
        flat_scores.sort(reverse=True)
        
        for i, (score, layer, head) in enumerate(flat_scores[:10]):
            rect = plt.Rectangle((head - 0.5, layer - 0.5), 1, 1, 
                                 fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(head, layer, f'{i+1}', ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'figure5_attention_heads.png', dpi=300, bbox_inches='tight')
            fig.savefig(self.output_dir / 'figure5_attention_heads.pdf', bbox_inches='tight')
        
        return fig
    
    def generate_all_figures(
        self,
        probing_results: Dict,
        patching_results: Dict,
        clinical_results: Dict,
        hypothesis_results: Dict,
        cross_dataset_results: Optional[Dict] = None,
        head_importance: Optional[Dict] = None
    ) -> Dict[str, plt.Figure]:
        """
        generate all publication figures.
        
        args:
            probing_results: probing experiment results
            patching_results: patching experiment results
            clinical_results: clinical feature probing results
            hypothesis_results: hypothesis test results
            cross_dataset_results: optional cross-dataset results
            head_importance: optional head importance scores
            
        returns:
            dict of figure name to figure object
        """
        figures = {}
        
        try:
            figures['figure1'] = self.figure_1_overview(probing_results, patching_results)
        except Exception as e:
            warnings.warn(f"could not generate figure 1: {e}")
        
        try:
            figures['figure2'] = self.figure_2_clinical_encoding(clinical_results)
        except Exception as e:
            warnings.warn(f"could not generate figure 2: {e}")
        
        try:
            figures['figure3'] = self.figure_3_hypothesis_summary(hypothesis_results)
        except Exception as e:
            warnings.warn(f"could not generate figure 3: {e}")
        
        if cross_dataset_results:
            try:
                matrix = cross_dataset_results.get('evaluation_matrix')
                names = cross_dataset_results.get('dataset_names', [])
                if matrix is not None and names:
                    figures['figure4'] = self.figure_4_cross_dataset(
                        np.array(matrix), names
                    )
            except Exception as e:
                warnings.warn(f"could not generate figure 4: {e}")
        
        if head_importance:
            try:
                figures['figure5'] = self.figure_5_attention_analysis(head_importance)
            except Exception as e:
                warnings.warn(f"could not generate figure 5: {e}")
        
        return figures


def create_poster_figure(
    results_dict: Dict,
    output_path: str,
    figsize: Tuple[int, int] = (48, 36)
) -> plt.Figure:
    """
    create comprehensive poster figure.
    
    args:
        results_dict: all experiment results
        output_path: save path
        figsize: figure size in inches
        
    returns:
        matplotlib figure
    """
    set_publication_style()
    
    fig = plt.figure(figsize=figsize)
    
    # create complex grid layout
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # title and abstract section (top)
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.7, 
                 'Mechanistic Interpretability of Wav2Vec2\nfor Parkinson\'s Disease Detection',
                 ha='center', va='center', fontsize=36, fontweight='bold')
    ax_title.text(0.5, 0.3,
                 'Probing, Patching, and Predicting: Understanding What Speech AI Learns',
                 ha='center', va='center', fontsize=24, style='italic')
    
    # TODO: add remaining panels based on results_dict
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


# =============================================================================
# PRE-PROCESSING FIGURES
# =============================================================================

def figure_data_distribution(
    dataset_info: Dict[str, Dict],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 6)
) -> plt.Figure:
    """
    Create figure showing data distribution across datasets.
    
    Shows sample counts, class balance, and demographic information.
    
    Args:
        dataset_info: Dict with dataset names as keys, containing:
            - n_pd: number of PD samples
            - n_hc: number of healthy control samples
            - mean_age: mean age (optional)
            - std_age: std of age (optional)
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    datasets = list(dataset_info.keys())
    n_datasets = len(datasets)
    
    # Panel A: Sample counts by class
    ax = axes[0]
    x = np.arange(n_datasets)
    width = 0.35
    
    pd_counts = [dataset_info[d].get('n_pd', 0) for d in datasets]
    hc_counts = [dataset_info[d].get('n_hc', 0) for d in datasets]
    
    bars1 = ax.bar(x - width/2, pd_counts, width, label='PD', 
                   color=COLORS['main']['primary'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, hc_counts, width, label='HC',
                   color=COLORS['main']['secondary'], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Distribution by Class')
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', '\n') for d in datasets], fontsize=8)
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add count labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)
    
    add_panel_label(ax, 'A')
    
    # Panel B: Class balance pie charts
    ax = axes[1]
    
    total_pd = sum(pd_counts)
    total_hc = sum(hc_counts)
    
    sizes = [total_pd, total_hc]
    labels = [f'PD\n(n={total_pd})', f'HC\n(n={total_hc})']
    colors = [COLORS['main']['primary'], COLORS['main']['secondary']]
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       autopct='%1.1f%%', shadow=False, startangle=90,
                                       wedgeprops=dict(edgecolor='black', linewidth=0.5))
    ax.set_title('Overall Class Distribution')
    
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    add_panel_label(ax, 'B', x=-0.1)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def figure_audio_preprocessing(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 8)
) -> plt.Figure:
    """
    Create figure showing audio preprocessing pipeline.
    
    Displays waveform, spectrogram, and mel-spectrogram.
    
    Args:
        waveform: Audio waveform as numpy array
        sample_rate: Sample rate in Hz
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Panel A: Waveform
    ax = axes[0]
    time = np.arange(len(waveform)) / sample_rate
    ax.plot(time, waveform, color=COLORS['main']['primary'], linewidth=0.5, alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Raw Audio Waveform')
    ax.set_xlim([0, time[-1]])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    add_panel_label(ax, 'A')
    
    # Panel B: Spectrogram
    ax = axes[1]
    
    # Compute spectrogram
    n_fft = 1024
    hop_length = 256
    
    # Simple spectrogram using numpy
    num_frames = 1 + (len(waveform) - n_fft) // hop_length
    spectrogram = np.zeros((n_fft // 2 + 1, num_frames))
    
    window = np.hanning(n_fft)
    for i in range(num_frames):
        start = i * hop_length
        frame = waveform[start:start + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame))
        spectrogram[:, i] = spectrum
    
    # Convert to dB
    spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
    
    freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)
    times = np.arange(num_frames) * hop_length / sample_rate
    
    im = ax.pcolormesh(times, freqs, spectrogram_db, shading='auto', cmap='viridis')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram')
    ax.set_ylim([0, sample_rate / 2])
    
    cbar = plt.colorbar(im, ax=ax, label='Power (dB)')
    add_panel_label(ax, 'B')
    
    # Panel C: Mel-frequency bands (simplified)
    ax = axes[2]
    
    n_mels = 80
    mel_min = 0
    mel_max = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    
    # Plot mel filterbank frequencies
    ax.fill_between(np.arange(n_mels), hz_points[:-2], hz_points[2:], 
                    alpha=0.3, color=COLORS['main']['primary'])
    ax.plot(np.arange(n_mels), (hz_points[:-2] + hz_points[2:]) / 2, 
            color=COLORS['main']['primary'], linewidth=1.5)
    ax.set_xlabel('Mel Filter Index')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Mel Filterbank Coverage')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    add_panel_label(ax, 'C')
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def figure_feature_extraction_pipeline(
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 5)
) -> plt.Figure:
    """
    Create schematic figure of the feature extraction pipeline.
    
    Shows flow from raw audio through Wav2Vec2 to classifier.
    
    Args:
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Define boxes
    boxes = [
        {'x': 0.5, 'y': 2.5, 'w': 1.5, 'h': 1.2, 'label': 'Raw\nAudio', 'color': COLORS['categorical'][0]},
        {'x': 2.5, 'y': 2.5, 'w': 1.5, 'h': 1.2, 'label': 'Feature\nExtractor', 'color': COLORS['categorical'][1]},
        {'x': 4.5, 'y': 2.5, 'w': 1.5, 'h': 1.2, 'label': 'Transformer\nEncoder', 'color': COLORS['categorical'][2]},
        {'x': 6.5, 'y': 2.5, 'w': 1.5, 'h': 1.2, 'label': 'Layer\nOutputs', 'color': COLORS['categorical'][3]},
        {'x': 8.5, 'y': 2.5, 'w': 1.0, 'h': 1.2, 'label': 'Probe/\nClassifier', 'color': COLORS['categorical'][4]},
    ]
    
    # Draw boxes
    for box in boxes:
        rect = plt.Rectangle((box['x'], box['y']), box['w'], box['h'],
                             facecolor=box['color'], edgecolor='black', linewidth=1.5,
                             alpha=0.7)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, box['label'],
               ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=1.5)
    for i in range(len(boxes) - 1):
        start_x = boxes[i]['x'] + boxes[i]['w']
        end_x = boxes[i+1]['x']
        y = boxes[i]['y'] + boxes[i]['h'] / 2
        ax.annotate('', xy=(end_x, y), xytext=(start_x, y),
                   arrowprops=arrow_style)
    
    # Add title
    ax.text(5, 5.2, 'Wav2Vec2 Feature Extraction Pipeline', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add layer annotations
    ax.text(6.5 + 0.75, 1.8, 'Layers 0-23', ha='center', va='top', fontsize=8, style='italic')
    
    # Add dimension annotations
    annotations = [
        (1.25, 2.3, '$T \\times 1$'),
        (3.25, 2.3, '$T\' \\times 512$'),
        (5.25, 2.3, '$T\' \\times 768$'),
        (7.25, 2.3, '$24 \\times T\' \\times 768$'),
        (9.0, 2.3, '$P(\\text{PD})$'),
    ]
    for x, y, label in annotations:
        ax.text(x, y, label, ha='center', va='top', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


# =============================================================================
# TRAINING FIGURES
# =============================================================================

def figure_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 4)
) -> plt.Figure:
    """
    Create training curves figure showing loss and accuracy over epochs.
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        train_accs: Optional training accuracy per epoch
        val_accs: Optional validation accuracy per epoch
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    has_accuracy = train_accs is not None and val_accs is not None
    n_cols = 2 if has_accuracy else 1
    
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Panel A: Loss curves
    ax = axes[0]
    ax.plot(epochs, train_losses, 'o-', color=COLORS['main']['primary'], 
            label='Training', markersize=4, linewidth=1.5)
    ax.plot(epochs, val_losses, 's-', color=COLORS['main']['secondary'], 
            label='Validation', markersize=4, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_loss = min(val_losses)
    ax.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.annotate(f'Best: {best_loss:.4f}', xy=(best_epoch, best_loss),
               xytext=(best_epoch + 1, best_loss + 0.1),
               arrowprops=dict(arrowstyle='->', color='gray'),
               fontsize=8)
    
    add_panel_label(ax, 'A')
    
    # Panel B: Accuracy curves (if provided)
    if has_accuracy:
        ax = axes[1]
        ax.plot(epochs, train_accs, 'o-', color=COLORS['main']['primary'], 
                label='Training', markersize=4, linewidth=1.5)
        ax.plot(epochs, val_accs, 's-', color=COLORS['main']['secondary'], 
                label='Validation', markersize=4, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Validation Accuracy')
        ax.legend(frameon=True, fancybox=False, edgecolor='black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0, 1])
        
        # Mark best epoch
        best_acc_epoch = np.argmax(val_accs) + 1
        best_acc = max(val_accs)
        ax.axvline(x=best_acc_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.annotate(f'Best: {best_acc:.3f}', xy=(best_acc_epoch, best_acc),
                   xytext=(best_acc_epoch + 1, best_acc - 0.1),
                   arrowprops=dict(arrowstyle='->', color='gray'),
                   fontsize=8)
        
        add_panel_label(ax, 'B')
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def figure_learning_rate_schedule(
    lrs: List[float],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (5, 3.5)
) -> plt.Figure:
    """
    Create learning rate schedule visualization.
    
    Args:
        lrs: Learning rate at each step/epoch
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    steps = np.arange(len(lrs))
    ax.plot(steps, lrs, color=COLORS['main']['primary'], linewidth=1.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Use scientific notation if needed
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
    
    # Add warmup annotation if applicable
    if len(lrs) > 10:
        max_lr_idx = np.argmax(lrs)
        if max_lr_idx > 0 and max_lr_idx < len(lrs) // 2:
            ax.axvline(x=max_lr_idx, color='gray', linestyle='--', alpha=0.5)
            ax.annotate('Warmup', xy=(max_lr_idx/2, max(lrs)*0.5),
                       ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


# =============================================================================
# EVALUATION FIGURES
# =============================================================================

def figure_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ['HC', 'PD'],
    normalize: bool = True,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (4, 3.5)
) -> plt.Figure:
    """
    Create publication-quality confusion matrix figure.
    
    Args:
        cm: Confusion matrix as 2D numpy array
        class_names: List of class names
        normalize: Whether to show percentages
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        fmt = '.1f'
        vmax = 100
    else:
        cm_display = cm
        fmt = 'd'
        vmax = None
    
    im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues', vmin=0, vmax=vmax)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Percentage' if normalize else 'Count', rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add text annotations
    thresh = cm_display.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f'{cm_display[i, j]:.1f}%\n({cm[i, j]})'
            else:
                text = f'{cm[i, j]}'
            ax.text(j, i, text,
                   ha="center", va="center",
                   color="white" if cm_display[i, j] > thresh else "black",
                   fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def figure_roc_curves(
    roc_data: Dict[str, Dict],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (5, 4.5)
) -> plt.Figure:
    """
    Create ROC curves figure for multiple models/datasets.
    
    Args:
        roc_data: Dict with model/dataset names as keys, containing:
            - fpr: False positive rates
            - tpr: True positive rates  
            - auc: Area under curve
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    colors = COLORS['categorical']
    
    for i, (name, data) in enumerate(roc_data.items()):
        fpr = data['fpr']
        tpr = data['tpr']
        auc = data['auc']
        
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=1.5,
                label=f'{name} (AUC = {auc:.3f})')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def figure_precision_recall(
    pr_data: Dict[str, Dict],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (5, 4.5)
) -> plt.Figure:
    """
    Create Precision-Recall curves figure.
    
    Args:
        pr_data: Dict with model names as keys, containing:
            - precision: Precision values
            - recall: Recall values
            - ap: Average precision
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    colors = COLORS['categorical']
    
    for i, (name, data) in enumerate(pr_data.items()):
        precision = data['precision']
        recall = data['recall']
        ap = data['ap']
        
        ax.plot(recall, precision, color=colors[i % len(colors)], linewidth=1.5,
                label=f'{name} (AP = {ap:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc="lower left", frameon=True, fancybox=False, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def figure_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_names: List[str] = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 4)
) -> plt.Figure:
    """
    Create bar chart comparing metrics across models/datasets.
    
    Args:
        metrics: Dict with model names as keys, containing metric values
        metric_names: List of metric names to display
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    models = list(metrics.keys())
    n_models = len(models)
    n_metrics = len(metric_names)
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    colors = COLORS['categorical']
    
    for i, model in enumerate(models):
        values = [metrics[model].get(m.lower(), 0) for m in metric_names]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, 
                     color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, rotation=90)
    
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([0, 1.15])
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


# =============================================================================
# INTERPRETABILITY FIGURES
# =============================================================================

def figure_layerwise_probing_detailed(
    probing_results: Dict[str, np.ndarray],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 5)
) -> plt.Figure:
    """
    Create detailed layer-wise probing results figure.
    
    Args:
        probing_results: Dict containing:
            - accuracies: Array of accuracies per layer
            - f1_scores: Array of F1 scores per layer (optional)
            - std_accuracies: Standard deviations (optional)
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    accuracies = probing_results.get('accuracies', probing_results.get('accuracy', []))
    std_accs = probing_results.get('std_accuracies', None)
    f1_scores = probing_results.get('f1_scores', None)
    
    n_layers = len(accuracies)
    layers = np.arange(n_layers)
    
    has_f1 = f1_scores is not None
    
    fig, axes = plt.subplots(1, 2 if has_f1 else 1, figsize=figsize)
    if not has_f1:
        axes = [axes]
    
    # Panel A: Accuracy
    ax = axes[0]
    if std_accs is not None:
        ax.fill_between(layers, np.array(accuracies) - np.array(std_accs),
                        np.array(accuracies) + np.array(std_accs),
                        alpha=0.3, color=COLORS['main']['primary'])
    ax.plot(layers, accuracies, 'o-', color=COLORS['main']['primary'], 
            markersize=5, linewidth=1.5)
    
    # Mark best layer
    best_layer = np.argmax(accuracies)
    ax.axvline(x=best_layer, color='gray', linestyle='--', alpha=0.5)
    ax.scatter([best_layer], [accuracies[best_layer]], s=100, 
              facecolors='none', edgecolors=COLORS['main']['accent'], linewidth=2,
              zorder=5, label=f'Best: Layer {best_layer}')
    
    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('Probing Accuracy')
    ax.set_title('Layer-wise Probing Accuracy')
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(layers[::2])
    
    add_panel_label(ax, 'A')
    
    # Panel B: F1 Score (if available)
    if has_f1:
        ax = axes[1]
        ax.plot(layers, f1_scores, 's-', color=COLORS['main']['secondary'], 
                markersize=5, linewidth=1.5)
        
        best_f1_layer = np.argmax(f1_scores)
        ax.axvline(x=best_f1_layer, color='gray', linestyle='--', alpha=0.5)
        ax.scatter([best_f1_layer], [f1_scores[best_f1_layer]], s=100,
                  facecolors='none', edgecolors=COLORS['main']['accent'], linewidth=2,
                  zorder=5, label=f'Best: Layer {best_f1_layer}')
        
        ax.set_xlabel('Transformer Layer')
        ax.set_ylabel('F1 Score')
        ax.set_title('Layer-wise Probing F1 Score')
        ax.legend(frameon=True, fancybox=False, edgecolor='black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(layers[::2])
        
        add_panel_label(ax, 'B')
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def figure_activation_patching_detailed(
    patching_results: Dict,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 5)
) -> plt.Figure:
    """
    Create detailed activation patching results figure.
    
    Args:
        patching_results: Dict containing:
            - effects: 2D array of patching effects (layers x positions)
            - layer_effects: 1D array of layer-wise effects
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    effects = patching_results.get('effects', None)
    layer_effects = patching_results.get('layer_effects', patching_results.get('mean_effect', []))
    
    has_2d = effects is not None and len(np.array(effects).shape) == 2
    
    fig, axes = plt.subplots(1, 2 if has_2d else 1, figsize=figsize)
    if not has_2d:
        axes = [axes]
    
    # Panel A: Layer-wise effects
    ax = axes[0]
    n_layers = len(layer_effects)
    layers = np.arange(n_layers)
    
    # Color bars by effect magnitude
    colors = [COLORS['diverging']['positive'] if e > 0 else COLORS['diverging']['negative'] 
              for e in layer_effects]
    
    bars = ax.bar(layers, layer_effects, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('Patching Effect')
    ax.set_title('Layer-wise Patching Effects')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(layers[::2])
    
    add_panel_label(ax, 'A')
    
    # Panel B: 2D heatmap (if available)
    if has_2d:
        ax = axes[1]
        
        effects_arr = np.array(effects)
        
        # Use diverging colormap centered at 0
        vmax = np.abs(effects_arr).max()
        
        im = ax.imshow(effects_arr, aspect='auto', cmap='RdBu_r', 
                       vmin=-vmax, vmax=vmax)
        
        ax.set_xlabel('Time Position')
        ax.set_ylabel('Layer')
        ax.set_title('Patching Effect Heatmap')
        
        cbar = plt.colorbar(im, ax=ax, label='Effect Magnitude')
        
        add_panel_label(ax, 'B')
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def figure_attention_patterns(
    attention_weights: np.ndarray,
    layer_idx: int = 0,
    head_idx: int = 0,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 5)
) -> plt.Figure:
    """
    Create attention pattern visualization.
    
    Args:
        attention_weights: Attention weights array [heads, seq, seq] or [seq, seq]
        layer_idx: Layer index for title
        head_idx: Head index for title
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Handle different input shapes
    if len(attention_weights.shape) == 3:
        attn = attention_weights[head_idx]
    else:
        attn = attention_weights
    
    im = ax.imshow(attn, cmap='Blues', aspect='auto')
    
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(f'Attention Pattern (Layer {layer_idx}, Head {head_idx})')
    
    cbar = plt.colorbar(im, ax=ax, label='Attention Weight')
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def figure_clinical_correlations(
    correlations: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 6)
) -> plt.Figure:
    """
    Create clinical feature correlation heatmap.
    
    Args:
        correlations: Nested dict of layer -> feature -> correlation value
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    # Convert to matrix
    layers = sorted(correlations.keys(), key=lambda x: int(x.replace('layer_', '')))
    features = list(next(iter(correlations.values())).keys())
    
    matrix = np.zeros((len(layers), len(features)))
    for i, layer in enumerate(layers):
        for j, feature in enumerate(features):
            matrix[i, j] = correlations[layer].get(feature, 0)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Use diverging colormap
    vmax = np.abs(matrix).max()
    
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_xticklabels([f.replace('_', '\n') for f in features], fontsize=8)
    ax.set_yticklabels([l.replace('layer_', 'L') for l in layers], fontsize=8)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_xlabel('Clinical Feature')
    ax.set_ylabel('Layer')
    ax.set_title('Layer-Feature Correlation Matrix')
    
    cbar = plt.colorbar(im, ax=ax, label='Correlation ($r$)')
    
    # Add text annotations for significant correlations
    for i in range(len(layers)):
        for j in range(len(features)):
            if np.abs(matrix[i, j]) > 0.3:  # Only annotate significant correlations
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                              ha="center", va="center", fontsize=6,
                              color="white" if np.abs(matrix[i, j]) > 0.5 else "black")
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


# =============================================================================
# SUPPLEMENTARY FIGURES
# =============================================================================

def figure_statistical_tests(
    test_results: Dict[str, Dict],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 4)
) -> plt.Figure:
    """
    Create visualization of statistical test results.
    
    Args:
        test_results: Dict with test names as keys, containing:
            - statistic: Test statistic value
            - p_value: P-value
            - effect_size: Effect size (optional)
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    tests = list(test_results.keys())
    n_tests = len(tests)
    
    # Panel A: P-values
    ax = axes[0]
    p_values = [test_results[t]['p_value'] for t in tests]
    log_p = [-np.log10(p + 1e-300) for p in p_values]
    
    colors = [COLORS['main']['accent'] if p < 0.05 else COLORS['main']['muted'] for p in p_values]
    bars = ax.barh(np.arange(n_tests), log_p, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add significance threshold lines
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=1, label='$p = 0.05$')
    ax.axvline(x=-np.log10(0.01), color='orange', linestyle='--', linewidth=1, label='$p = 0.01$')
    
    ax.set_yticks(np.arange(n_tests))
    ax.set_yticklabels(tests, fontsize=8)
    ax.set_xlabel('$-\\log_{10}(p)$')
    ax.set_title('Statistical Significance')
    ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    add_panel_label(ax, 'A')
    
    # Panel B: Effect sizes
    ax = axes[1]
    effect_sizes = [test_results[t].get('effect_size', 0) for t in tests]
    
    colors = [COLORS['sequential']['high'] if abs(e) > 0.5 
              else COLORS['sequential']['medium'] if abs(e) > 0.3 
              else COLORS['sequential']['low'] for e in effect_sizes]
    
    bars = ax.barh(np.arange(n_tests), effect_sizes, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add effect size thresholds
    ax.axvline(x=0.2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0.8, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_yticks(np.arange(n_tests))
    ax.set_yticklabels(tests, fontsize=8)
    ax.set_xlabel("Effect Size (Cohen's $d$)")
    ax.set_title('Effect Sizes')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add annotations
    ax.text(0.2, n_tests, 'Small', ha='center', va='bottom', fontsize=7, color='gray')
    ax.text(0.5, n_tests, 'Medium', ha='center', va='bottom', fontsize=7, color='gray')
    ax.text(0.8, n_tests, 'Large', ha='center', va='bottom', fontsize=7, color='gray')
    
    add_panel_label(ax, 'B')
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def figure_model_architecture(
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 8)
) -> plt.Figure:
    """
    Create schematic of Wav2Vec2 model architecture.
    
    Args:
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Wav2Vec2 Architecture for PD Detection', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Feature extractor block
    rect = plt.Rectangle((1.5, 9), 7, 1.5, facecolor=COLORS['categorical'][0], 
                         edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(rect)
    ax.text(5, 9.75, 'CNN Feature Extractor', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(5, 9.25, '7 layers of temporal convolutions', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # Arrow
    ax.annotate('', xy=(5, 8.8), xytext=(5, 9),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Transformer blocks
    for i in range(4):
        y = 7.5 - i * 1.5
        color = COLORS['categorical'][1] if i < 3 else COLORS['categorical'][2]
        alpha = 0.7 if i < 3 else 0.9
        
        rect = plt.Rectangle((1.5, y), 7, 1.2, facecolor=color, 
                             edgecolor='black', linewidth=1.5, alpha=alpha)
        ax.add_patch(rect)
        
        if i < 3:
            ax.text(5, y + 0.6, f'Transformer Block {i}', ha='center', va='center', 
                    fontsize=10, fontweight='bold')
        else:
            ax.text(5, y + 0.75, '...', ha='center', va='center', fontsize=14)
            ax.text(5, y + 0.3, 'Transformer Block 23', ha='center', va='center', 
                    fontsize=10, fontweight='bold')
        
        # Arrow
        if i < 3:
            ax.annotate('', xy=(5, y - 0.2), xytext=(5, y),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Classification head
    rect = plt.Rectangle((2.5, 1), 5, 1.2, facecolor=COLORS['categorical'][4], 
                         edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(rect)
    ax.text(5, 1.6, 'Classification Head', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(5, 1.2, 'Mean pooling + Linear + Softmax', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # Arrow to output
    ax.annotate('', xy=(5, 0.5), xytext=(5, 1),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Output
    ax.text(5, 0.3, '$P(\\text{PD})$', ha='center', va='center', fontsize=12)
    
    # Side annotations
    ax.annotate('', xy=(0.8, 4.5), xytext=(0.8, 7.5),
               arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    ax.text(0.5, 6, 'Probing\nLayers', ha='center', va='center', 
            fontsize=8, rotation=90, color='gray')
    
    ax.annotate('', xy=(9.2, 4.5), xytext=(9.2, 7.5),
               arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    ax.text(9.5, 6, 'Patching\nTargets', ha='center', va='center', 
            fontsize=8, rotation=90, color='gray')
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def figure_experimental_design(
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 5)
) -> plt.Figure:
    """
    Create experimental design flowchart.
    
    Args:
        output_path: Optional path to save figure
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.7, 'Experimental Design', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # Phase boxes
    phases = [
        {'x': 0.3, 'y': 3.5, 'w': 1.8, 'h': 1.5, 'label': 'Phase 1:\nData\nCollection', 
         'color': COLORS['categorical'][0]},
        {'x': 2.5, 'y': 3.5, 'w': 1.8, 'h': 1.5, 'label': 'Phase 2:\nModel\nFine-tuning', 
         'color': COLORS['categorical'][1]},
        {'x': 4.7, 'y': 3.5, 'w': 1.8, 'h': 1.5, 'label': 'Phase 3:\nLayer\nProbing', 
         'color': COLORS['categorical'][2]},
        {'x': 6.9, 'y': 3.5, 'w': 1.8, 'h': 1.5, 'label': 'Phase 4:\nActivation\nPatching', 
         'color': COLORS['categorical'][3]},
        {'x': 3.6, 'y': 0.8, 'w': 2.8, 'h': 1.2, 'label': 'Phase 5:\nInterpretable Prediction Interface', 
         'color': COLORS['categorical'][4]},
    ]
    
    for phase in phases:
        rect = plt.Rectangle((phase['x'], phase['y']), phase['w'], phase['h'],
                             facecolor=phase['color'], edgecolor='black', 
                             linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(phase['x'] + phase['w']/2, phase['y'] + phase['h']/2, phase['label'],
               ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Horizontal arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=1.5)
    for i in range(3):
        start_x = phases[i]['x'] + phases[i]['w']
        end_x = phases[i+1]['x']
        y = phases[i]['y'] + phases[i]['h'] / 2
        ax.annotate('', xy=(end_x, y), xytext=(start_x, y), arrowprops=arrow_style)
    
    # Arrow to Phase 5
    ax.annotate('', xy=(5, 2), xytext=(5, 3.5), arrowprops=arrow_style)
    
    # Add dataset boxes at bottom
    datasets = ['Italian PVS', 'MDVR-KCL', 'Arkansas']
    for i, ds in enumerate(datasets):
        x = 1.5 + i * 2.5
        rect = plt.Rectangle((x, 0.2), 2, 0.5, facecolor=COLORS['main']['light'],
                             edgecolor='black', linewidth=0.5, alpha=0.5)
        ax.add_patch(rect)
        ax.text(x + 1, 0.45, ds, ha='center', va='center', fontsize=7)
    
    ax.text(5, -0.1, 'Datasets', ha='center', va='top', fontsize=8, style='italic')
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
