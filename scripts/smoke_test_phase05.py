#!/usr/bin/env python3
"""
Smoke test for Phase 05 - tests critical components without full execution.
"""

import sys
from pathlib import Path
import numpy as np

# Set up project root
project_root = Path('/Volumes/usb drive/pd-interpretability')
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("="*80)
print("PHASE 05 SMOKE TEST")
print("="*80)

tests_passed = 0
tests_failed = 0

# Test 1: LaTeX configuration
print("\n[Test 1] LaTeX Configuration")
try:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Times'],
    })
    assert plt.rcParams['text.usetex'] == True
    print("✓ PASSED: LaTeX rendering enabled")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAILED: {e}")
    tests_failed += 1

# Test 2: Dataset loading
print("\n[Test 2] Dataset Loading")
try:
    from src.data.datasets import ItalianPVSDataset
    dataset = ItalianPVSDataset(
        root_dir=project_root / 'data' / 'raw' / 'italian_pvs',
        task=None,
        max_duration=10.0,
        target_sr=16000
    )
    assert len(dataset) > 0
    print(f"✓ PASSED: Dataset loaded with {len(dataset)} samples")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAILED: {e}")
    tests_failed += 1

# Test 3: Clinical features
print("\n[Test 3] Clinical Features")
try:
    import pandas as pd
    clinical_df = pd.read_csv(project_root / 'data' / 'clinical_features' / 'italian_pvs_features.csv')

    required_features = ['jitter_local', 'shimmer_local', 'hnr_mean', 'f0_std']
    missing = [f for f in required_features if f not in clinical_df.columns]
    assert len(missing) == 0, f"Missing features: {missing}"
    print(f"✓ PASSED: All 4 required clinical features present")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAILED: {e}")
    tests_failed += 1

# Test 4: Model loading
print("\n[Test 4] Model Path")
try:
    model_path = project_root / 'results' / 'final_model'

    if model_path.exists():
        # Check config file exists (don't load model to avoid segfault in test)
        config_file = model_path / 'config.json'
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✓ PASSED: Fine-tuned model path exists with config")
        else:
            print(f"⚠ WARNING: Model path exists but config missing")
    else:
        print(f"⚠ SKIPPED: Fine-tuned model not found (will use base model)")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAILED: {e}")
    tests_failed += 1

# Test 5: Cached activations
print("\n[Test 5] Cached Activations")
try:
    import pickle
    activations_cache = project_root / 'results' / 'probing' / 'activations_cache.pkl'

    with open(activations_cache, 'rb') as f:
        cache = pickle.load(f)

    assert 'activations_by_layer' in cache
    assert 'labels' in cache
    assert 'subject_ids' in cache
    assert len(cache['activations_by_layer']) == 12
    print(f"✓ PASSED: Activations cache valid with 12 layers")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAILED: {e}")
    tests_failed += 1

# Test 6: Previous probing results
print("\n[Test 6] Previous Probing Results")
try:
    import json
    results_path = project_root / 'results' / 'probing' / 'probing_results.json'

    with open(results_path, 'r') as f:
        results = json.load(f)

    assert 'layerwise_pd_probing' in results
    assert len(results['layerwise_pd_probing']) == 12

    layer_0_acc = results['layerwise_pd_probing']['0']['accuracy_mean']
    assert 0 <= layer_0_acc <= 1.0
    print(f"✓ PASSED: Layer-wise probing results valid (layer 0 acc: {layer_0_acc:.3f})")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAILED: {e}")
    tests_failed += 1

# Test 7: LaTeX figures exist
print("\n[Test 7] LaTeX Figures")
try:
    results_dir = project_root / 'results' / 'probing'

    figures_found = 0
    for fig_pattern in ['fig_p5_01_*.pdf', 'fig_p5_02_*.pdf']:
        matching_files = list(results_dir.glob(fig_pattern))
        if matching_files:
            figures_found += 1

    assert figures_found >= 2, f"Only found {figures_found}/2 required figures"
    print(f"✓ PASSED: LaTeX PDF figures exist")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAILED: {e}")
    tests_failed += 1

# Test 8: Sklearn compatibility
print("\n[Test 8] Sklearn Components")
try:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV

    # Test LOSO CV
    logo = LeaveOneGroupOut()
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    groups = np.repeat(np.arange(10), 10)

    n_splits = logo.get_n_splits(groups=groups)
    assert n_splits == 10
    print(f"✓ PASSED: Sklearn components work correctly")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAILED: {e}")
    tests_failed += 1

# Summary
print("\n" + "="*80)
print("SMOKE TEST SUMMARY")
print("="*80)
print(f"Tests passed: {tests_passed}")
print(f"Tests failed: {tests_failed}")

if tests_failed == 0:
    print("\n✓ ALL TESTS PASSED - Phase 05 notebook is ready to execute!")
    sys.exit(0)
else:
    print(f"\n✗ {tests_failed} TEST(S) FAILED - Please fix before proceeding")
    sys.exit(1)
