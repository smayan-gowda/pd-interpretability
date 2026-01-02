"""
generate publication-quality figures for phase 2 clinical baseline.

creates all figures needed to document the clinical baseline results
with times new roman fonts, latex formatting, and 300 dpi output.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.metrics import roc_curve, auc

# add latex to path if not present
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ.get('PATH', '')


def set_publication_style():
    """configure matplotlib for publication-quality figures with latex."""
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.fancybox': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def load_results():
    """load clinical baseline results."""
    results_path = project_root / 'results' / 'clinical_baseline_results.json'
    with open(results_path) as f:
        results = json.load(f)
    
    subjects_path = project_root / 'results' / 'clinical_baseline_subjects.csv'
    subjects_df = pd.read_csv(subjects_path)
    
    features_path = project_root / 'data' / 'clinical_features' / 'italian_pvs_features.csv'
    features_df = pd.read_csv(features_path)
    
    return results, subjects_df, features_df


def fig1_feature_importance(results, save_path):
    """
    figure 1: clinical feature importance ranking.
    
    bar chart showing random forest feature importances
    with error bars from cross-validation.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    importance_list = results['feature_importance']
    features = [item['feature'] for item in importance_list]
    values = [item['importance'] for item in importance_list]
    
    sorted_idx = np.argsort(values)[::-1]
    features = [features[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    
    bars = ax.barh(range(len(features)), values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([f.replace('_', ' ') for f in features])
    ax.set_xlabel(r'Feature Importance')
    ax.set_title(r'Clinical Feature Importance for PD Classification')
    ax.invert_yaxis()
    
    ax.axvline(x=np.mean(values), color='red', linestyle='--', linewidth=1, alpha=0.7, label=r'mean')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig2_confusion_matrix(features_df, save_path):
    """
    figure 2: confusion matrix heatmap.
    
    shows classification performance with percentages.
    reconstructed from features_df labels.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.impute import SimpleImputer
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    feature_cols = [c for c in features_df.columns 
                    if c not in ['subject_id', 'label', 'path', 'diagnosis']]
    
    df_clean = features_df.dropna(subset=feature_cols)
    
    X = df_clean[feature_cols].values
    y = df_clean['label'].values
    groups = df_clean['subject_id'].values
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    y_true_all = []
    y_pred_all = []
    
    logo = LeaveOneGroupOut()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        rf.fit(X_scaled[train_idx], y[train_idx])
        y_pred = rf.predict(X_scaled[test_idx])
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(y_pred)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[r'Healthy', r"Parkinson's"],
        yticklabels=[r'Healthy', r"Parkinson's"],
        ax=ax, cbar=False,
        linewidths=0.5, linecolor='black'
    )
    
    for i in range(2):
        for j in range(2):
            pct = cm_normalized[i, j] * 100
            ax.text(j + 0.5, i + 0.7, r'(%.1f\%%)' % pct, 
                   ha='center', va='center', fontsize=8, color='gray')
    
    ax.set_xlabel(r'Predicted Label')
    ax.set_ylabel(r'True Label')
    ax.set_title(r'Confusion Matrix (Random Forest, LOSO CV)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig3_statistical_comparison(results, features_df, save_path):
    """
    figure 3: statistical comparison of clinical features between pd and hc.
    
    violin plots showing distribution differences with significance markers.
    """
    stat_list = results['statistical_comparison']
    stat_results = {item['feature']: item for item in stat_list}
    
    top_features = ['shimmer_apq5', 'shimmer_dda', 'shimmer_apq3', 'jitter_ppq5']
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        hc_data = features_df[features_df['label'] == 0][feature].dropna()
        pd_data = features_df[features_df['label'] == 1][feature].dropna()
        
        parts = ax.violinplot([hc_data, pd_data], positions=[0, 1], showmeans=True, showmedians=False)
        
        colors = ['#2ecc71', '#e74c3c']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
            if partname in parts:
                parts[partname].set_color('black')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r'Healthy', r"Parkinson's"])
        
        feat_stat = stat_results.get(feature, {})
        p_val = feat_stat.get('p_value', 1.0)
        cohens_d = feat_stat.get('cohens_d', 0)
        
        # determine significance level
        if p_val < 0.001:
            sig_text = r'$p < 0.001$'
        elif p_val < 0.01:
            sig_text = r'$p < 0.01$'
        elif p_val < 0.05:
            sig_text = r'$p < 0.05$'
        else:
            sig_text = r'n.s.'
        
        ax.set_ylabel(feature.replace('_', ' '))
        ax.set_title(r"Cohen's $d$ = %.2f, " % cohens_d + sig_text, fontsize=9)
    
    fig.suptitle(r'Clinical Feature Distributions: PD vs Healthy Controls', fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig4_per_subject_accuracy(subjects_df, save_path):
    """
    figure 4: per-subject accuracy distribution.
    
    histogram showing accuracy distribution across all 61 subjects
    with separate coloring for hc and pd.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    hc_acc = subjects_df[subjects_df['diagnosis'] == 'hc']['accuracy']
    pd_acc = subjects_df[subjects_df['diagnosis'] == 'pd']['accuracy']
    
    bins = np.linspace(0, 1, 11)
    
    ax.hist(hc_acc, bins=bins, alpha=0.7, label=r'Healthy ($n$=%d)' % len(hc_acc), 
            color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.hist(pd_acc, bins=bins, alpha=0.7, label=r"Parkinson's ($n$=%d)" % len(pd_acc), 
            color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=hc_acc.mean(), color='#27ae60', linestyle='--', linewidth=2, 
               label=r'HC $\mu$: %.2f' % hc_acc.mean())
    ax.axvline(x=pd_acc.mean(), color='#c0392b', linestyle='--', linewidth=2, 
               label=r'PD $\mu$: %.2f' % pd_acc.mean())
    
    ax.set_xlabel(r'Classification Accuracy')
    ax.set_ylabel(r'Number of Subjects')
    ax.set_title(r'Per-Subject Classification Accuracy Distribution (LOSO CV)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig5_model_comparison(results, save_path):
    """
    figure 5: model comparison bar chart.
    
    comparing svm and random forest with error bars.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    
    models = ['SVM (RBF)', 'Random Forest']
    accuracies = [
        results['svm']['accuracy_mean'],
        results['random_forest']['accuracy_mean']
    ]
    stds = [
        results['svm']['accuracy_std'],
        results['random_forest']['accuracy_std']
    ]
    
    colors = ['#3498db', '#9b59b6']
    bars = ax.bar(models, accuracies, yerr=stds, capsize=8, 
                  color=colors, edgecolor='black', linewidth=0.8)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Chance level')
    ax.axhspan(0.70, 0.85, alpha=0.2, color='green', label='Target range (70-85%)')
    
    for bar, acc, std in zip(bars, accuracies, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                r'%.1f\%%' % (acc*100), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel(r'Accuracy')
    ax.set_title(r'Clinical Baseline Model Comparison (LOSO CV)')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig6_fold_performance(results, save_path):
    """
    figure 6: per-fold (per-subject) performance across cv folds.
    
    line plot showing variance across loso folds.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    rf_scores = results['random_forest']['per_fold_scores']
    svm_scores = results['svm']['per_fold_scores']
    folds = range(1, len(rf_scores) + 1)
    
    ax.plot(folds, rf_scores, 'o-', color='#9b59b6', alpha=0.7, 
            label=r"RF ($\mu$: %.2f)" % np.mean(rf_scores), markersize=4)
    ax.plot(folds, svm_scores, 's-', color='#3498db', alpha=0.7, 
            label=r"SVM ($\mu$: %.2f)" % np.mean(svm_scores), markersize=4)
    
    ax.axhline(y=np.mean(rf_scores), color='#9b59b6', linestyle='--', alpha=0.5)
    ax.axhline(y=np.mean(svm_scores), color='#3498db', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance')
    
    ax.set_xlabel(r'Fold (Subject Left Out)')
    ax.set_ylabel(r'Accuracy')
    ax.set_title(r'Leave-One-Subject-Out Cross-Validation Performance')
    ax.set_xlim([0, len(folds) + 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right', fontsize=8)
    
    ax.fill_between(folds, 0.7, 0.85, alpha=0.1, color='green')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig7_feature_correlation(features_df, save_path):
    """
    figure 7: clinical feature correlation matrix.
    
    heatmap showing inter-feature correlations.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    feature_cols = [c for c in features_df.columns if c not in ['subject_id', 'label', 'path', 'diagnosis']]
    corr_matrix = features_df[feature_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt='.2f',
        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        ax=ax, square=True, linewidths=0.5,
        cbar_kws={'shrink': 0.7, 'label': 'Correlation'},
        annot_kws={'size': 6}
    )
    
    # format x labels with underscores replaced by spaces, rotated for readability
    ax.set_xticklabels([l.get_text().replace('_', ' ') for l in ax.get_xticklabels()], 
                       rotation=55, ha='right', fontsize=7)
    ax.set_yticklabels([l.get_text().replace('_', ' ') for l in ax.get_yticklabels()], 
                       rotation=0, fontsize=7)
    ax.set_title(r'Clinical Feature Correlation Matrix')
    
    # add extra bottom margin to prevent text cutoff
    plt.subplots_adjust(bottom=0.25, left=0.2)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def main():
    """generate all phase 2 figures."""
    set_publication_style()
    
    print("loading results...")
    results, subjects_df, features_df = load_results()
    
    figures_dir = project_root / 'results' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("\ngenerating publication-quality figures...")
    print("=" * 50)
    
    fig1_feature_importance(results, figures_dir / 'fig_p2_01_feature_importance.pdf')
    fig1_feature_importance(results, figures_dir / 'fig_p2_01_feature_importance.png')
    
    fig2_confusion_matrix(features_df, figures_dir / 'fig_p2_02_confusion_matrix.pdf')
    fig2_confusion_matrix(features_df, figures_dir / 'fig_p2_02_confusion_matrix.png')
    
    fig3_statistical_comparison(results, features_df, figures_dir / 'fig_p2_03_statistical_comparison.pdf')
    fig3_statistical_comparison(results, features_df, figures_dir / 'fig_p2_03_statistical_comparison.png')
    
    fig4_per_subject_accuracy(subjects_df, figures_dir / 'fig_p2_04_per_subject_accuracy.pdf')
    fig4_per_subject_accuracy(subjects_df, figures_dir / 'fig_p2_04_per_subject_accuracy.png')
    
    fig5_model_comparison(results, figures_dir / 'fig_p2_05_model_comparison.pdf')
    fig5_model_comparison(results, figures_dir / 'fig_p2_05_model_comparison.png')
    
    fig6_fold_performance(results, figures_dir / 'fig_p2_06_fold_performance.pdf')
    fig6_fold_performance(results, figures_dir / 'fig_p2_06_fold_performance.png')
    
    fig7_feature_correlation(features_df, figures_dir / 'fig_p2_07_feature_correlation.pdf')
    fig7_feature_correlation(features_df, figures_dir / 'fig_p2_07_feature_correlation.png')
    
    print("=" * 50)
    print(f"\nall figures saved to: {figures_dir}")
    print("\nfigure summary:")
    print("  fig_p2_01: feature importance ranking")
    print("  fig_p2_02: confusion matrix")
    print("  fig_p2_03: statistical comparison (pd vs hc)")
    print("  fig_p2_04: per-subject accuracy distribution")
    print("  fig_p2_05: model comparison (svm vs rf)")
    print("  fig_p2_06: loso cv fold performance")
    print("  fig_p2_07: feature correlation matrix")


if __name__ == '__main__':
    main()
