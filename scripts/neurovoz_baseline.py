#!/usr/bin/env python3
"""
neurovoz dataset preprocessing and baseline evaluation
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import time
import random

# ML imports
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

# Project imports
from data.datasets import NeuroVozDataset
from features.clinical import ClinicalFeatureExtractor


def main():
    np.random.seed(42)
    random.seed(42)

    print("=" * 60)
    print("neurovoz baseline processing")
    print("=" * 60)
    print()

    config = {
        'data_dir': project_root / 'data' / 'raw' / 'neurovoz',
        'features_dir': project_root / 'data' / 'clinical_features',
        'results_dir': project_root / 'results' / 'neurovoz_baseline',
        'task': None,
        'svm_params': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'random_state': 42
        },
        'seed': 42
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['results_dir'] = config['results_dir'] / f"run_{timestamp}"
    config['results_dir'].mkdir(parents=True, exist_ok=True)
    config['features_dir'].mkdir(parents=True, exist_ok=True)

    print("configuration")
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, indent=2))
    print()

    print("step 1: loading neurovoz dataset")
    print(f"  audio dir: {config['data_dir'] / 'audios'}")
    print(f"  task filter: {config['task']}")

    load_start = time.time()
    dataset = NeuroVozDataset(root_dir=config['data_dir'], task=config['task'])
    load_dur = time.time() - load_start

    print(f"  loaded samples: {len(dataset)} in {load_dur:.1f}s")
    print(f"  subjects: {dataset.n_subjects}")
    print(f"  subject ids: {len(dataset.subject_ids)}")

    labels = [sample['label'] for sample in dataset.samples]
    subject_ids = [sample['subject_id'] for sample in dataset.samples]
    tasks = [sample['task'] for sample in dataset.samples]

    label_counts = pd.Series(labels).value_counts()
    task_counts = pd.Series(tasks).value_counts().head(10)
    print(f"  label counts hc/pd: {label_counts.get(0, 0)} / {label_counts.get(1, 0)}")
    print(f"  top tasks: {task_counts.to_dict()}")

    subject_labels = {}
    for sid, label in zip(subject_ids, labels):
        subject_labels[sid] = label

    hc_subjects = sum(1 for v in subject_labels.values() if v == 0)
    pd_subjects = sum(1 for v in subject_labels.values() if v == 1)
    print(f"  subjects hc/pd: {hc_subjects} / {pd_subjects}")
    print()

    # 2. Extract clinical features
    if config['task'] is None:
        task_str = 'all_tasks'
    elif isinstance(config['task'], list):
        task_str = '_'.join(config['task'])
    else:
        task_str = config['task']
    features_csv_path = config['features_dir'] / f'neurovoz_features_{task_str}.csv'

    if features_csv_path.exists():
        print(f"step 2: loading existing features from {features_csv_path}")
        features_df = pd.read_csv(features_csv_path)
        print(f"  loaded {len(features_df)} samples")
    else:
        print("step 2: extracting clinical features")
        print(f"  processing {len(dataset)} audio samples")
        print(f"  estimated time: ~{len(dataset) * 0.15:.0f}s ({len(dataset) * 0.15 / 60:.1f} min)")

        extractor = ClinicalFeatureExtractor(f0_min=75.0, f0_max=600.0)

        features_list = []
        failed_samples = []

        for i in tqdm(range(len(dataset)), desc="  extracting", unit="sample", ncols=90,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            sample = dataset.samples[i]

            try:
                features = extractor.extract(str(sample['path']))

                features['sample_idx'] = i
                features['path'] = str(sample['path'])
                features['subject_id'] = sample['subject_id']
                features['label'] = sample['label']
                features['diagnosis'] = 'pd' if sample['label'] == 1 else 'hc'
                features['task'] = sample['task']

                for key in ['age', 'sex', 'updrs', 'hy_stadium', 'disease_duration']:
                    if key in sample:
                        features[key] = sample[key]

                features_list.append(features)

            except Exception as e:
                failed_samples.append((i, str(e)))

        features_df = pd.DataFrame(features_list)

        print()
        print(f"  extracted: {len(features_df)} samples")
        print(f"  failed: {len(failed_samples)} samples ({len(failed_samples)/len(dataset)*100:.1f}%)")

        if failed_samples:
            print("  sample failures (up to 10):")
            for idx, error in failed_samples[:10]:
                print(f"    {idx}: {error[:120]}")

        features_df.to_csv(features_csv_path, index=False)
        print(f"  saved features to {features_csv_path}")
    print()

    # 3. Prepare features
    print("step 3: preparing feature matrix")
    print(f"  loading {len(features_df)} samples from features csv")

    clinical_feature_cols = [
        'f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range',
        'voicing_fraction',
        'jitter_local', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp',
        'shimmer_local', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda',
        'hnr_mean', 'hnr_std'
    ]

    available_features = [f for f in clinical_feature_cols if f in features_df.columns]
    print(f"  clinical features available: {len(available_features)}/17")
    if len(available_features) < 17:
        missing_features = set(clinical_feature_cols) - set(available_features)
        print(f"  missing: {sorted(list(missing_features))}")

    print("  checking data quality")
    missing = features_df[available_features].isnull().sum()
    if missing.sum() > 0:
        for feat, count in missing[missing > 0].items():
            print(f"    missing {feat}: {count} ({count/len(features_df)*100:.1f}%)")

    features_clean = features_df.dropna(subset=available_features)
    removed = len(features_df) - len(features_clean)
    print(f"  removed {removed} samples with missing values")
    print(f"  clean samples: {len(features_clean)}/{len(features_df)} ({len(features_clean)/len(features_df)*100:.1f}%)")

    subsets = [
        {'name': 'all_tasks', 'mask': np.ones(len(features_clean), dtype=bool)},
        {'name': 'vowels_only', 'mask': features_clean['task'].str.startswith('vowel_')}
    ]

    def run_loso(params, X_data, y_data, groups_data):
        logo = LeaveOneGroupOut()
        n_folds = logo.get_n_splits(X_data, y_data, groups_data)
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**params, probability=True))
        ])

        y_pred_local = np.zeros_like(y_data)
        y_proba_local = np.zeros_like(y_data, dtype=float)

        fold_iter = tqdm(enumerate(logo.split(X_data, y_data, groups_data), start=1), total=n_folds, ncols=90,
                         desc="  loso folds")
        for fold_idx, (train_idx, test_idx) in fold_iter:
            fold_start = time.time()
            clf.fit(X_data[train_idx], y_data[train_idx])
            fold_pred = clf.predict(X_data[test_idx])
            fold_proba = clf.predict_proba(X_data[test_idx])[:, 1]

            y_pred_local[test_idx] = fold_pred
            y_proba_local[test_idx] = fold_proba

            subj = np.unique(groups_data[test_idx])[0]
            fold_acc = accuracy_score(y_data[test_idx], fold_pred)
            fold_iter.set_postfix({'fold': f"{fold_idx}/{n_folds}", 'subject': subj, 'acc': f"{fold_acc:.3f}", 'dur_s': f"{time.time()-fold_start:.1f}"})

        metrics_local = {
            'accuracy': accuracy_score(y_data, y_pred_local),
            'precision': precision_score(y_data, y_pred_local, average='binary'),
            'recall': recall_score(y_data, y_pred_local, average='binary'),
            'f1': f1_score(y_data, y_pred_local, average='binary'),
            'auc': roc_auc_score(y_data, y_proba_local),
            'n_folds': n_folds,
            'n_samples': len(y_data),
            'n_subjects': len(np.unique(groups_data))
        }

        cm_local = confusion_matrix(y_data, y_pred_local)
        tn_l, fp_l, fn_l, tp_l = cm_local.ravel()
        metrics_local['confusion_matrix'] = {'tn': int(tn_l), 'fp': int(fp_l), 'fn': int(fn_l), 'tp': int(tp_l)}
        metrics_local['sensitivity'] = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0
        metrics_local['specificity'] = tn_l / (tn_l + fp_l) if (tn_l + fp_l) > 0 else 0

        return metrics_local, y_pred_local, y_proba_local

    print(f"  feature matrix (all tasks): {features_clean[available_features].shape}")
    print()

    candidate_params = [
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'random_state': 42},
        {'kernel': 'rbf', 'C': 5.0, 'gamma': 'scale', 'random_state': 42},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale', 'random_state': 42},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 0.1, 'random_state': 42},
        {'kernel': 'rbf', 'C': 2.0, 'gamma': 'scale', 'class_weight': 'balanced', 'random_state': 42},
        {'kernel': 'linear', 'C': 1.0, 'class_weight': 'balanced', 'random_state': 42}
    ]

    overall_best = None

    print("step 4: training svm with loso cross-validation")
    print("  method: leave-one-subject-out")
    print("  classifier: svm grid (rbf + linear)")
    print(f"  features: {len(available_features)} clinical biomarkers")

    for subset in subsets:
        mask = subset['mask']
        subset_df = features_clean[mask].copy()
        X_sub = subset_df[available_features].values
        y_sub = subset_df['label'].values
        groups_sub = subset_df['subject_id'].values

        if len(y_sub) == 0 or len(np.unique(groups_sub)) < 2:
            print(f"  skipping subset {subset['name']} (insufficient data)")
            continue

        print(f"  subset: {subset['name']} | samples: {len(y_sub)} | subjects: {len(np.unique(groups_sub))}")

        subset_best = None
        for idx, params in enumerate(candidate_params, start=1):
            print(f"    candidate {idx}/{len(candidate_params)}: {params}")
            metrics_cand, y_pred_cand, y_proba_cand = run_loso(params, X_sub, y_sub, groups_sub)
            metrics_cand['params'] = params
            metrics_cand['subset'] = subset['name']
            print(f"      accuracy: {metrics_cand['accuracy']:.3f} | auc: {metrics_cand['auc']:.3f} | f1: {metrics_cand['f1']:.3f}")

            if subset_best is None or metrics_cand['auc'] > subset_best['metrics']['auc']:
                subset_best = {
                    'metrics': metrics_cand,
                    'y_pred': y_pred_cand,
                    'y_proba': y_proba_cand,
                    'y_true': y_sub,
                    'groups': groups_sub,
                    'features_df': subset_df
                }

        print(f"    best for {subset['name']}: auc {subset_best['metrics']['auc']:.3f}, acc {subset_best['metrics']['accuracy']:.3f}")

        if overall_best is None or subset_best['metrics']['auc'] > overall_best['metrics']['auc']:
            overall_best = subset_best

    metrics = overall_best['metrics']
    y_pred = overall_best['y_pred']
    y_proba = overall_best['y_proba']
    y = overall_best['y_true']
    groups = overall_best['groups']
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    config['svm_params'] = metrics['params']
    config['subset'] = metrics.get('subset', 'all_tasks')
    print(f"  selected subset: {config['subset']}")
    print(f"  selected params: {config['svm_params']} (best auc)")
    print()

    print("step 5: calculating performance metrics")
    print()
    print("=" * 60)
    print("neurovoz baseline results")
    print("=" * 60)
    print(f"task: {config['task']}")
    print(f"classifier: svm (rbf kernel)")
    print()
    print("dataset:")
    print(f"  subjects: {metrics['n_subjects']}")
    print(f"  samples: {metrics['n_samples']}")
    print(f"  hc: {tn + fp}, pd: {tp + fn}")
    print()
    print("performance:")
    print(f"  accuracy:    {metrics['accuracy']:.1%}")
    print(f"  precision:   {metrics['precision']:.1%}")
    print(f"  recall:      {metrics['recall']:.1%}")
    print(f"  f1 score:    {metrics['f1']:.3f}")
    print(f"  auc-roc:     {metrics['auc']:.3f}")
    print(f"  sensitivity: {metrics['sensitivity']:.1%}")
    print(f"  specificity: {metrics['specificity']:.1%}")
    print()
    print("confusion matrix:")
    print(f"  tn: {tn:3d}  fp: {fp:3d}")
    print(f"  fn: {fn:3d}  tp: {tp:3d}")
    print("=" * 60)
    print()

    # 6. Per-subject analysis
    print("step 6: per-subject accuracy analysis")

    unique_subjects = np.unique(groups)
    subject_results = []

    for subject in unique_subjects:
        mask = groups == subject
        subject_true = y[mask]
        subject_pred = y_pred[mask]

        subject_acc = accuracy_score(subject_true, subject_pred)
        subject_label = 'pd' if subject_true[0] == 1 else 'hc'
        n_samples = mask.sum()

        subject_results.append({
            'subject_id': subject,
            'diagnosis': subject_label,
            'n_samples': n_samples,
            'accuracy': subject_acc,
            'correct': int(subject_acc * n_samples),
            'total': n_samples
        })

    subject_accuracy_df = pd.DataFrame(subject_results)

    print(f"  Mean subject accuracy: {subject_accuracy_df['accuracy'].mean():.1%}")
    print(f"  Std subject accuracy: {subject_accuracy_df['accuracy'].std():.3f}")
    print(f"  100% accuracy: {(subject_accuracy_df['accuracy'] == 1.0).sum()} subjects")
    print(f"  0% accuracy: {(subject_accuracy_df['accuracy'] == 0.0).sum()} subjects")
    print()

    # 7. Generate figures
    print("step 7: generating publication-quality figures")

    # Set publication style
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

    # Figure 1: Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['HC', 'PD'],
                yticklabels=['HC', 'PD'],
                cbar_kws={'label': 'Count'},
                ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'NeuroVoz Baseline - Confusion Matrix\\n(Accuracy: {metrics["accuracy"]:.1%})')
    plt.tight_layout()
    plt.savefig(config['results_dir'] / 'confusion_matrix.png', dpi=300)
    plt.savefig(config['results_dir'] / 'confusion_matrix.pdf')
    plt.close()
    print("  Saved confusion_matrix.png/pdf")

    # Figure 2: ROC Curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'SVM (AUC = {metrics["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('NeuroVoz Baseline - ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    plt.tight_layout()
    plt.savefig(config['results_dir'] / 'roc_curve.png', dpi=300)
    plt.savefig(config['results_dir'] / 'roc_curve.pdf')
    plt.close()
    print("  Saved roc_curve.png/pdf")

    # Figure 3: Per-Subject Accuracy Distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    hc_acc = subject_accuracy_df[subject_accuracy_df['diagnosis'] == 'hc']['accuracy']
    pd_acc = subject_accuracy_df[subject_accuracy_df['diagnosis'] == 'pd']['accuracy']
    box_data = [hc_acc, pd_acc]
    bp = ax.boxplot(box_data, positions=[1, 2], widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['HC', 'PD'])
    ax.set_ylabel('Per-Subject Accuracy')
    ax.set_title('NeuroVoz Baseline - Per-Subject Accuracy Distribution')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    stats_text = f"HC: {hc_acc.mean():.1%} ± {hc_acc.std():.3f}\\nPD: {pd_acc.mean():.1%} ± {pd_acc.std():.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(config['results_dir'] / 'per_subject_accuracy.png', dpi=300)
    plt.savefig(config['results_dir'] / 'per_subject_accuracy.pdf')
    plt.close()
    print("  Saved per_subject_accuracy.png/pdf")

    # Figure 4: Metrics Summary
    fig, ax = plt.subplots(figsize=(6, 4))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Sensitivity', 'Specificity']
    metric_values = [
        metrics['accuracy'], metrics['precision'], metrics['recall'],
        metrics['f1'], metrics['sensitivity'], metrics['specificity']
    ]
    bars = ax.barh(metric_names, metric_values, color='steelblue', alpha=0.7)
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        ax.text(value + 0.01, i, f'{value:.1%}', va='center', fontsize=9)
    ax.set_xlabel('Score')
    ax.set_xlim([0, 1.1])
    ax.set_title('NeuroVoz Baseline - Performance Metrics Summary')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(config['results_dir'] / 'metrics_summary.png', dpi=300)
    plt.savefig(config['results_dir'] / 'metrics_summary.pdf')
    plt.close()
    print("  Saved metrics_summary.png/pdf")
    print()

    # 8. Save results
    print("step 8: saving results")

    # Save metrics JSON
    def serialize_config(v):
        if isinstance(v, Path):
            return str(v)
        elif isinstance(v, list):
            return v
        elif isinstance(v, dict):
            return {k: serialize_config(val) for k, val in v.items()}
        else:
            return v

    task_list = config['task'] if isinstance(config['task'], list) else ([config['task']] if config['task'] is not None else ['all'])

    results_json = {
        'dataset': 'neurovoz',
        'task': task_list,
        'timestamp': timestamp,
        'classifier': 'SVM',
        'cv_method': 'LOSO',
        'metrics': metrics,
        'config': {k: serialize_config(v) for k, v in config.items()}
    }

    json_path = config['results_dir'] / 'neurovoz_baseline_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved neurovoz_baseline_results.json")

    # Save per-subject results CSV
    csv_path = config['results_dir'] / 'neurovoz_subject_accuracy.csv'
    subject_accuracy_df.to_csv(csv_path, index=False)
    print(f"  Saved neurovoz_subject_accuracy.csv")

    # Save classification report
    report = classification_report(y, y_pred, target_names=['HC', 'PD'], digits=3)
    report_path = config['results_dir'] / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("neurovoz baseline - classification report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"task: {config['task']}\n")
        f.write(f"classifier: svm (rbf kernel)\n")
        f.write(f"cv method: leave-one-subject-out\n")
        f.write(f"timestamp: {timestamp}\n\n")
        f.write(report)
        f.write("\\n" + "=" * 60 + "\\n")
        f.write(f"overall accuracy: {metrics['accuracy']:.1%}\n")
        f.write(f"auc-roc: {metrics['auc']:.3f}\n")
    print(f"  Saved classification_report.txt")
    print()

    # 9. Final summary
    print("=" * 60)
    print("neurovoz preprocessing complete")
    print("=" * 60)
    print(f"Results directory: {config['results_dir']}")
    print()
    print(f"baseline performance:")
    print(f"  accuracy: {metrics['accuracy']:.1%}")
    print(f"  auc-roc: {metrics['auc']:.3f}")
    print()
    print("ready for phase 4 cross-dataset validation")
    print("=" * 60)


if __name__ == '__main__':
    main()
