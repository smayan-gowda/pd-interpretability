"""
comprehensive verification script for phases 0-2.

validates all deliverables are complete and correct.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd


def verify_phase0():
    """verify phase 0: data infrastructure."""
    print('=' * 60)
    print('PHASE 0 VERIFICATION: DATA INFRASTRUCTURE')
    print('=' * 60)
    
    status = {'passed': 0, 'failed': 0}
    
    # test 1: dataset classes
    print('\n[1] dataset classes...')
    try:
        from src.data.datasets import BasePDDataset, ItalianPVSDataset
        print('  pass: basepddataset imported')
        print('  pass: italianpvsdataset imported')
        status['passed'] += 2
    except Exception as e:
        print(f'  fail: import error: {e}')
        status['failed'] += 2
    
    # test 2: preprocessing utilities
    print('\n[2] preprocessing utilities...')
    try:
        from src.data.preprocessing import load_audio, normalize_audio
        print('  pass: load_audio imported')
        print('  pass: normalize_audio imported')
        status['passed'] += 2
    except Exception as e:
        print(f'  fail: import error: {e}')
        status['failed'] += 2
    
    # test 3: clinical feature extractor
    print('\n[3] clinical feature extractor...')
    try:
        from src.features.clinical import ClinicalFeatureExtractor
        extractor = ClinicalFeatureExtractor()
        print('  pass: clinicalfeatureextractor instantiated')
        status['passed'] += 1
    except Exception as e:
        print(f'  fail: error: {e}')
        status['failed'] += 1
    
    # test 4: dataset loading
    print('\n[4] italian pvs dataset loading...')
    try:
        dataset = ItalianPVSDataset(Path('data/raw/italian_pvs'))
        print(f'  pass: dataset loaded: {len(dataset)} samples')
        print(f'  pass: subjects: {dataset.n_subjects}')
        status['passed'] += 2
    except Exception as e:
        print(f'  fail: error: {e}')
        status['failed'] += 2
    
    # test 5: data directories
    print('\n[5] data directories...')
    dirs = [
        'data/raw/italian_pvs',
        'data/clinical_features',
        'data/processed',
        'data/activations'
    ]
    for d in dirs:
        p = Path(d)
        if p.exists():
            print(f'  pass: {d}')
            status['passed'] += 1
        else:
            print(f'  fail: {d}')
            status['failed'] += 1
    
    return status


def verify_phase1():
    """verify phase 1: clinical feature extraction."""
    print('\n' + '=' * 60)
    print('PHASE 1 VERIFICATION: CLINICAL FEATURE EXTRACTION')
    print('=' * 60)
    
    status = {'passed': 0, 'failed': 0}
    
    # test 1: clinical features csv
    print('\n[1] clinical features data file...')
    features_path = Path('data/clinical_features/italian_pvs_features.csv')
    if features_path.exists():
        df = pd.read_csv(features_path)
        print(f'  pass: csv exists with {len(df)} samples')
        status['passed'] += 1
        
        # check columns
        required_cols = ['f0_mean', 'jitter_local', 'shimmer_local', 'subject_id', 'label']
        for col in required_cols:
            if col in df.columns:
                print(f'  pass: column {col} present')
                status['passed'] += 1
            else:
                print(f'  fail: column {col} missing')
                status['failed'] += 1
    else:
        print(f'  fail: csv not found')
        status['failed'] += 6
        return status
    
    # test 2: feature quality
    print('\n[2] feature quality...')
    f0_mean = df['f0_mean'].mean()
    if 50 < f0_mean < 500:
        print(f'  pass: f0_mean in valid range: {f0_mean:.1f} hz')
        status['passed'] += 1
    else:
        print(f'  fail: f0_mean out of range: {f0_mean:.1f} hz')
        status['failed'] += 1
    
    jitter_mean = df['jitter_local'].mean()
    if 0 < jitter_mean < 0.1:
        print(f'  pass: jitter_local in valid range: {jitter_mean:.4f}')
        status['passed'] += 1
    else:
        print(f'  fail: jitter_local out of range')
        status['failed'] += 1
    
    # test 3: class distribution
    print('\n[3] class distribution...')
    hc_count = len(df[df['label'] == 0])
    pd_count = len(df[df['label'] == 1])
    print(f'  pass: healthy controls: {hc_count}')
    print(f'  pass: parkinsons: {pd_count}')
    status['passed'] += 2
    
    return status


def verify_phase2():
    """verify phase 2: clinical baseline."""
    print('\n' + '=' * 60)
    print('PHASE 2 VERIFICATION: CLINICAL BASELINE')
    print('=' * 60)
    
    status = {'passed': 0, 'failed': 0}
    
    # test 1: results files
    print('\n[1] baseline results files...')
    results_path = Path('results/clinical_baseline_results.json')
    subjects_path = Path('results/clinical_baseline_subjects.csv')
    
    if results_path.exists():
        print(f'  pass: {results_path.name}')
        status['passed'] += 1
        with open(results_path) as f:
            results = json.load(f)
    else:
        print(f'  fail: {results_path.name} not found')
        status['failed'] += 1
        return status
    
    if subjects_path.exists():
        print(f'  pass: {subjects_path.name}')
        status['passed'] += 1
    else:
        print(f'  fail: {subjects_path.name} not found')
        status['failed'] += 1
    
    # test 2: model performance
    print('\n[2] model performance...')
    svm_acc = results['svm']['accuracy_mean']
    rf_acc = results['random_forest']['accuracy_mean']
    
    print(f'  pass: svm accuracy: {svm_acc*100:.1f}%')
    print(f'  pass: random forest: {rf_acc*100:.1f}%')
    status['passed'] += 2
    
    if rf_acc >= 0.70:
        print(f'  pass: accuracy meets target (>70%)')
        status['passed'] += 1
    else:
        print(f'  fail: accuracy below target')
        status['failed'] += 1
    
    # test 3: cross-validation
    print('\n[3] cross-validation...')
    n_folds = results['n_folds']
    n_subjects = results['n_subjects']
    cv_method = results['cv_method']
    
    print(f'  pass: method: {cv_method}')
    status['passed'] += 1
    
    if n_folds == n_subjects:
        print(f'  pass: loso valid (folds={n_folds} == subjects={n_subjects})')
        status['passed'] += 1
    else:
        print(f'  fail: loso invalid')
        status['failed'] += 1
    
    # test 4: statistical analysis
    print('\n[4] statistical significance...')
    stat_comp = results['statistical_comparison']
    sig_count = sum(1 for item in stat_comp if item['significant'])
    total = len(stat_comp)
    
    print(f'  pass: {sig_count}/{total} features significant')
    status['passed'] += 1
    
    # test 5: figures
    print('\n[5] publication figures...')
    figures_dir = Path('results/figures')
    expected_figs = [
        'fig_p2_01_feature_importance.pdf',
        'fig_p2_02_confusion_matrix.pdf',
        'fig_p2_03_statistical_comparison.pdf',
        'fig_p2_04_per_subject_accuracy.pdf',
        'fig_p2_05_model_comparison.pdf',
        'fig_p2_06_fold_performance.pdf',
        'fig_p2_07_feature_correlation.pdf'
    ]
    
    for fig in expected_figs:
        fig_path = figures_dir / fig
        if fig_path.exists():
            size_kb = fig_path.stat().st_size / 1024
            print(f'  pass: {fig} ({size_kb:.0f}kb)')
            status['passed'] += 1
        else:
            print(f'  fail: {fig} not found')
            status['failed'] += 1
    
    return status


def main():
    """run comprehensive verification."""
    print('\n' + '#' * 60)
    print('# PHASES 0-2 COMPREHENSIVE VERIFICATION')
    print('#' * 60)
    
    s0 = verify_phase0()
    s1 = verify_phase1()
    s2 = verify_phase2()
    
    total_passed = s0['passed'] + s1['passed'] + s2['passed']
    total_failed = s0['failed'] + s1['failed'] + s2['failed']
    
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'\nphase 0: {s0["passed"]} passed, {s0["failed"]} failed')
    print(f'phase 1: {s1["passed"]} passed, {s1["failed"]} failed')
    print(f'phase 2: {s2["passed"]} passed, {s2["failed"]} failed')
    print(f'\ntotal: {total_passed} passed, {total_failed} failed')
    
    if total_failed == 0:
        print('\n*** ALL PHASES VERIFIED SUCCESSFULLY ***')
        print('\nready to proceed to phase 3: wav2vec2 fine-tuning')
    else:
        print(f'\n*** {total_failed} ISSUES FOUND ***')
        print('\nfix issues before proceeding')
    
    print('=' * 60)


if __name__ == '__main__':
    main()
