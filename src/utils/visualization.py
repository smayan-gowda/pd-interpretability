"""
visualization utilities for mechanistic interpretability.

comprehensive plotting functions for probing results, activation patterns,
and clinical feature analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy import stats


def plot_layerwise_probing(
    results: Dict[int, Dict[str, float]],
    title: str = "layer-wise pd classification probing accuracy",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
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
    layers = sorted(results.keys())
    accuracies = [results[l]['mean'] for l in layers]
    stds = [results[l]['std'] for l in layers]

    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        layers, accuracies, yerr=stds,
        marker='o', capsize=5, linewidth=2, markersize=8,
        color='#2E86AB', label='probing accuracy'
    )

    ax.axhline(
        y=chance_level, color='gray', linestyle='--',
        linewidth=2, label='chance', alpha=0.7
    )

    best_layer = layers[np.argmax(accuracies)]
    best_acc = max(accuracies)
    ax.plot(best_layer, best_acc, 'r*', markersize=20, label=f'best (layer {best_layer})')

    ax.set_xlabel('transformer layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('probing accuracy', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(layers)
    ax.set_ylim([max(0.3, min(accuracies) - 0.1), min(1.0, max(accuracies) + 0.1)])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_clinical_feature_heatmap(
    results: Dict[str, Dict[int, Dict[str, float]]],
    feature_names: Optional[List[str]] = None,
    metric: str = 'mean',
    title: str = "clinical feature encoding across layers",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
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
        vmax=1 if metric in ['mean', 'accuracy'] else None
    )

    ax.set_xlabel('transformer layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('clinical feature', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_patching_results(
    patching_results: Dict[int, Dict[str, float]],
    title: str = "activation patching: logit difference recovery",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
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
    layers = sorted(patching_results.keys())
    mean_recovery = [patching_results[l]['mean_recovery'] for l in layers]
    std_recovery = [patching_results[l]['std_recovery'] for l in layers]

    colors = ['red' if m > threshold else 'steelblue' for m in mean_recovery]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(layers, mean_recovery, yerr=std_recovery, capsize=4, color=colors, alpha=0.7)

    ax.axhline(y=threshold, color='black', linestyle='--', linewidth=2, label=f'importance threshold ({threshold})')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel('transformer layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('logit difference recovery', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(layers)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')

    important_layers = [l for l, m in zip(layers, mean_recovery) if m > threshold]
    if important_layers:
        ax.text(
            0.02, 0.98,
            f'important layers: {important_layers}',
            transform=ax.transAxes,
            fontsize=11,
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
    title: str = "clinical features: pd vs healthy control",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
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
    available_features = [f for f in feature_names if f in features_df.columns]

    if not available_features:
        raise ValueError("no valid features in dataframe")

    n_features = len(available_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feature in enumerate(available_features):
        ax = axes[i]

        hc_vals = features_df[features_df[label_col] == 0][feature].dropna()
        pd_vals = features_df[features_df[label_col] == 1][feature].dropna()

        if len(hc_vals) > 0 and len(pd_vals) > 0:
            ax.hist(hc_vals, alpha=0.6, label='healthy', bins=20, color='#2E86AB', edgecolor='black')
            ax.hist(pd_vals, alpha=0.6, label='parkinson', bins=20, color='#A23B72', edgecolor='black')

            t_stat, p_val = stats.ttest_ind(hc_vals, pd_vals)

            ax.axvline(hc_vals.mean(), color='#2E86AB', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(pd_vals.mean(), color='#A23B72', linestyle='--', linewidth=2, alpha=0.8)

            sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

            ax.set_title(f"{feature}\np={p_val:.3f} {sig_marker}", fontsize=10, fontweight='bold')
            ax.set_xlabel('value', fontsize=9)
            ax.set_ylabel('count', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, linestyle=':')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ['healthy', 'parkinson'],
    title: str = "confusion matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
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
        cbar_kws={'label': 'proportion' if normalize else 'count'}
    )

    ax.set_xlabel('predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('actual', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')

    if not normalize:
        total = cm.sum()
        accuracy = np.trace(cm) / total
        ax.text(
            0.02, 0.98,
            f'accuracy: {accuracy:.2%}',
            transform=ax.transAxes,
            fontsize=12,
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
    title: str = "roc curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
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
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color='#2E86AB', linewidth=3, label=f'roc curve (auc = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='chance')

    ax.set_xlabel('false positive rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('true positive rate', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
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
    figsize: Tuple[int, int] = (10, 8)
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
    if attention_weights.ndim == 3:
        if head_idx is None:
            attn = attention_weights.mean(axis=0)
            head_str = "avg all heads"
        else:
            attn = attention_weights[head_idx]
            head_str = f"head {head_idx}"
    else:
        attn = attention_weights
        head_str = ""

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attn, cmap='viridis', aspect='auto')

    if title is None:
        title = f"attention weights - layer {layer_idx} {head_str}"

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('key position', fontsize=12)
    ax.set_ylabel('query position', fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('attention weight', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_results_dashboard(
    results_dict: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12)
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
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    if 'probing' in results_dict:
        ax1 = fig.add_subplot(gs[0, :2])
        layers = sorted(results_dict['probing'].keys())
        accuracies = [results_dict['probing'][l]['mean'] for l in layers]
        ax1.plot(layers, accuracies, marker='o', linewidth=2, markersize=8)
        ax1.set_title('probing accuracy across layers', fontweight='bold')
        ax1.set_xlabel('layer')
        ax1.set_ylabel('accuracy')
        ax1.grid(True, alpha=0.3)

    if 'patching' in results_dict:
        ax2 = fig.add_subplot(gs[1, :2])
        layers = sorted(results_dict['patching'].keys())
        recovery = [results_dict['patching'][l]['mean_recovery'] for l in layers]
        ax2.bar(layers, recovery, alpha=0.7)
        ax2.set_title('activation patching recovery', fontweight='bold')
        ax2.set_xlabel('layer')
        ax2.set_ylabel('recovery')
        ax2.grid(True, alpha=0.3, axis='y')

    if 'confusion_matrix' in results_dict:
        ax3 = fig.add_subplot(gs[0, 2])
        cm = np.array(results_dict['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title('confusion matrix', fontweight='bold')
        ax3.set_xlabel('predicted')
        ax3.set_ylabel('actual')

    if 'test_metrics' in results_dict:
        ax4 = fig.add_subplot(gs[1, 2])
        metrics = results_dict['test_metrics']
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        metric_values = [metrics.get(m, 0) for m in metric_names]
        ax4.barh(metric_names, metric_values)
        ax4.set_xlim(0, 1)
        ax4.set_title('test metrics', fontweight='bold')
        for i, v in enumerate(metric_values):
            ax4.text(v + 0.01, i, f'{v:.3f}', va='center')

    fig.suptitle('mechanistic interpretability results', fontsize=18, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
