"""
visualization utilities for mechanistic interpretability.

comprehensive plotting functions for probing results, activation patterns,
and clinical feature analysis.

publication-quality styling with times new roman and latex support.
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy import stats


def set_visualization_style() -> None:
    """
    set publication-quality visualization style.
    
    uses times new roman fonts and professional formatting
    consistent with figures.py styling.
    """
    plt.rcParams.update({
        # font settings - times new roman for publication
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
        'font.size': 10,
        
        # axes settings
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # tick settings
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        
        # legend settings
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        
        # figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # math text
        'mathtext.fontset': 'stix',
    })


# apply style on import
set_visualization_style()


def plot_layerwise_probing(
    results: Dict[int, Dict[str, float]],
    title: str = "Layer-wise PD Classification Probing Accuracy",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7.5, 4),
    chance_level: float = 0.5
) -> plt.Figure:
    """
    plot probing accuracy across layers.

    args:
        results: dict mapping layer_idx to metrics dict
        title: plot title
        save_path: path to save figure
        figsize: figure size
        chance_level: chance accuracy level

    returns:
        matplotlib figure
    """
    set_visualization_style()
    
    layers = sorted(results.keys())
    accuracies = [results[l]['mean'] for l in layers]
    stds = [results[l]['std'] for l in layers]

    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        layers, accuracies, yerr=stds,
        marker='o', capsize=4, linewidth=1.5, markersize=6,
        color='#2E86AB', label='Probing Accuracy'
    )

    ax.axhline(
        y=chance_level, color='gray', linestyle='--',
        linewidth=1, label='Chance', alpha=0.7
    )

    best_layer = layers[np.argmax(accuracies)]
    best_acc = max(accuracies)
    ax.plot(best_layer, best_acc, 'r*', markersize=15, label=f'Best (Layer {best_layer})')

    ax.set_xlabel('Transformer Layer')

    ax.set_ylabel('Probing Accuracy')
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.set_ylim([max(0.3, min(accuracies) - 0.1), min(1.0, max(accuracies) + 0.1)])
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_clinical_feature_heatmap(
    results: Dict[str, Dict[int, Dict[str, float]]],
    feature_names: Optional[List[str]] = None,
    metric: str = 'mean',
    title: str = "Clinical Feature Encoding Across Layers",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7.5, 5),
    cmap: str = 'viridis',
    annot: bool = True
) -> plt.Figure:
    """
    create heatmap of clinical feature encoding across layers.

    args:
        results: nested dict feature_name -> layer_idx -> metrics
        feature_names: list of features to plot (none = all)
        metric: which metric to plot ('mean', 'std')
        title: plot title
        save_path: path to save
        figsize: figure size
        cmap: colormap
        annot: whether to annotate cells

    returns:
        matplotlib figure
    """
    set_visualization_style()
    
    if feature_names is None:
        feature_names = list(results.keys())

    feature_names = [f for f in feature_names if f in results]

    if not feature_names:
        raise ValueError("no valid features found in results")

    layers = sorted(list(results[feature_names[0]].keys()))

    matrix = np.zeros((len(feature_names), len(layers)))

    for i, feat in enumerate(feature_names):
        for j, layer in enumerate(layers):
            if layer in results[feat]:
                matrix[i, j] = results[feat][layer].get(metric, 0)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix,
        xticklabels=layers,
        yticklabels=feature_names,
        cmap=cmap,
        annot=annot,
        fmt='.2f',
        ax=ax,
        cbar_kws={'label': f'{metric} score'},
        vmin=0,
        vmax=1 if metric in ['mean', 'accuracy'] else None,
        annot_kws={'fontsize': 8}
    )

    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('Clinical Feature')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_patching_results(
    patching_results: Dict[int, Dict[str, float]],
    title: str = "Activation Patching: Logit Difference Recovery",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7.5, 4),
    threshold: float = 0.5
) -> plt.Figure:
    """
    plot activation patching results showing causal importance.

    args:
        patching_results: dict mapping layer to recovery metrics
        title: plot title
        save_path: save path
        figsize: figure size
        threshold: importance threshold

    returns:
        matplotlib figure
    """
    set_visualization_style()
    
    layers = sorted(patching_results.keys())
    mean_recovery = [patching_results[l]['mean_recovery'] for l in layers]
    std_recovery = [patching_results[l]['std_recovery'] for l in layers]

    colors = ['#E63946' if m > threshold else '#2E86AB' for m in mean_recovery]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(layers, mean_recovery, yerr=std_recovery, capsize=3, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1, 
               label=f'Importance Threshold ({threshold})')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('Logit Difference Recovery')
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')

    important_layers = [l for l, m in zip(layers, mean_recovery) if m > threshold]
    if important_layers:
        ax.text(
            0.02, 0.98,
            f'Important Layers: {important_layers}',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_clinical_comparison(
    features_df: pd.DataFrame,
    feature_names: List[str],
    label_col: str = 'label',
    title: str = "Clinical Features: PD vs Healthy Control",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7.5, 8)
) -> plt.Figure:
    """
    plot clinical feature distributions for pd vs hc.

    args:
        features_df: dataframe with features and labels
        feature_names: list of features to plot
        label_col: column name for labels
        title: plot title
        save_path: save path
        figsize: figure size

    returns:
        matplotlib figure
    """
    set_visualization_style()
    
    available_features = [f for f in feature_names if f in features_df.columns]

    if not available_features:
        raise ValueError("no valid features in dataframe")

    n_features = len(available_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feature in enumerate(available_features):
        ax = axes[i]

        hc_vals = features_df[features_df[label_col] == 0][feature].dropna()
        pd_vals = features_df[features_df[label_col] == 1][feature].dropna()

        if len(hc_vals) > 0 and len(pd_vals) > 0:
            ax.hist(hc_vals, alpha=0.6, label='Healthy', bins=20, color='#2E86AB', edgecolor='black', linewidth=0.5)
            ax.hist(pd_vals, alpha=0.6, label='Parkinson', bins=20, color='#A23B72', edgecolor='black', linewidth=0.5)

            t_stat, p_val = stats.ttest_ind(hc_vals, pd_vals)

            ax.axvline(hc_vals.mean(), color='#2E86AB', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.axvline(pd_vals.mean(), color='#A23B72', linestyle='--', linewidth=1.5, alpha=0.8)

            sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

            ax.set_title(f"{feature}\n$p$={p_val:.3f} {sig_marker}", fontsize=9)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, linestyle=':')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=11, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ['Healthy', 'Parkinson'],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (4, 3.5),
    normalize: bool = False
) -> plt.Figure:
    """
    plot confusion matrix.

    args:
        cm: confusion matrix
        class_names: class labels
        title: plot title
        save_path: save path
        figsize: figure size
        normalize: whether to normalize

    returns:
        matplotlib figure
    """
    set_visualization_style()
    
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        ax=ax,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        annot_kws={'fontsize': 10}
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

    if not normalize:
        total = cm.sum()
        accuracy = np.trace(cm) / total
        ax.text(
            0.02, 0.98,
            f'Accuracy: {accuracy:.2%}',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (5, 4.5)
) -> plt.Figure:
    """
    plot roc curve.

    args:
        y_true: true labels
        y_scores: prediction scores
        title: plot title
        save_path: save path
        figsize: figure size

    returns:
        matplotlib figure
    """
    set_visualization_style()
    
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color='#2E86AB', linewidth=1.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Chance')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    layer_idx: int,
    head_idx: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 5)
) -> plt.Figure:
    """
    plot attention weights heatmap.

    args:
        attention_weights: attention tensor [num_heads, seq_len, seq_len] or [seq_len, seq_len]
        layer_idx: layer index
        head_idx: head index (none if already single head)
        title: plot title
        save_path: save path
        figsize: figure size

    returns:
        matplotlib figure
    """
    set_visualization_style()
    
    if attention_weights.ndim == 3:
        if head_idx is None:
            attn = attention_weights.mean(axis=0)
            head_str = "Avg All Heads"
        else:
            attn = attention_weights[head_idx]
            head_str = f"Head {head_idx}"
    else:
        attn = attention_weights
        head_str = ""

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attn, cmap='viridis', aspect='auto')

    if title is None:
        title = f"Attention Weights - Layer {layer_idx} {head_str}"

    ax.set_title(title)
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_results_dashboard(
    results_dict: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7.5, 8)
) -> plt.Figure:
    """
    create comprehensive results dashboard.

    args:
        results_dict: dict with keys: 'probing', 'patching', 'test_metrics', 'confusion_matrix'
        save_path: save path
        figsize: figure size

    returns:
        matplotlib figure
    """
    set_visualization_style()
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    if 'probing' in results_dict:
        ax1 = fig.add_subplot(gs[0, 0])
        layers = sorted(results_dict['probing'].keys())
        accuracies = [results_dict['probing'][l]['mean'] for l in layers]
        ax1.plot(layers, accuracies, marker='o', linewidth=1.5, markersize=5, color='#2E86AB')
        ax1.set_title('Probing Accuracy Across Layers')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True, alpha=0.3, linestyle=':')

    if 'patching' in results_dict:
        ax2 = fig.add_subplot(gs[0, 1])
        layers = sorted(results_dict['patching'].keys())
        recovery = [results_dict['patching'][l]['mean_recovery'] for l in layers]
        ax2.bar(layers, recovery, alpha=0.8, color='#2E86AB', edgecolor='black', linewidth=0.5)
        ax2.set_title('Activation Patching Recovery')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Recovery')
        ax2.grid(True, alpha=0.3, axis='y', linestyle=':')

    if 'confusion_matrix' in results_dict:
        ax3 = fig.add_subplot(gs[1, 0])
        cm = np.array(results_dict['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
                   annot_kws={'fontsize': 10})
        ax3.set_title('Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')

    if 'test_metrics' in results_dict:
        ax4 = fig.add_subplot(gs[1, 1])
        metrics = results_dict['test_metrics']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        metric_values = [metrics.get(m.lower(), 0) for m in metric_names]
        bars = ax4.barh(metric_names, metric_values, color='#2E86AB', 
                       edgecolor='black', linewidth=0.5)
        ax4.set_xlim(0, 1)
        ax4.set_title('Test Metrics')
        for i, v in enumerate(metric_values):
            ax4.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=8)

    fig.suptitle('Mechanistic Interpretability Results', fontsize=12, y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
