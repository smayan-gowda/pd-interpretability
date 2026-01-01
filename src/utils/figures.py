"""
publication figure generator for isef-quality research figures.

generates comprehensive, multi-panel figures suitable for publication,
poster presentations, and competition submissions.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats


# set publication-quality defaults
def set_publication_style():
    """set matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.linewidth': 1.5,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.framealpha': 0.9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
    })


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
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
