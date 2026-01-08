#!/usr/bin/env python3
"""
Verification script for Phase 05 Probing Experiments.
Checks that all components are in place and notebook is ready to execute.
"""

import sys
from pathlib import Path
import json

# Set up project root
project_root = Path('/Volumes/usb drive/pd-interpretability')
sys.path.insert(0, str(project_root))

print("=" * 80)
print("PHASE 05 PROBING EXPERIMENTS VERIFICATION")
print("=" * 80)

errors = []
warnings = []
successes = []

# Check 1: Notebook exists and is valid JSON
notebook_path = project_root / 'notebooks' / 'cpu' / '05_probing_experiments.ipynb'
if notebook_path.exists():
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        successes.append(f"✓ Notebook found: {notebook_path}")
        successes.append(f"✓ Notebook is valid JSON with {len(notebook['cells'])} cells")
    except Exception as e:
        errors.append(f"✗ Notebook JSON invalid: {e}")
else:
    errors.append(f"✗ Notebook not found: {notebook_path}")

# Check 2: Results directory exists
results_dir = project_root / 'results' / 'probing'
if results_dir.exists():
    successes.append(f"✓ Results directory exists: {results_dir}")
else:
    warnings.append(f"⚠ Results directory missing: {results_dir}")

# Check 3: Activations cache
activations_cache = results_dir / 'activations_cache.pkl'
if activations_cache.exists():
    size_mb = activations_cache.stat().st_size / (1024 * 1024)
    successes.append(f"✓ Activations cache exists: {size_mb:.1f} MB")
else:
    warnings.append(f"⚠ Activations cache missing (will need to extract ~10 min)")

# Check 4: Layer-wise results cache
layerwise_cache = results_dir / 'layerwise_results_cache.pkl'
if layerwise_cache.exists():
    size_kb = layerwise_cache.stat().st_size / 1024
    successes.append(f"✓ Layer-wise results cache exists: {size_kb:.1f} KB")
else:
    warnings.append(f"⚠ Layer-wise results cache missing (will need to train ~1+ hour)")

# Check 5: PDF figures (LaTeX-rendered)
expected_figures = [
    'fig_p5_01_layerwise_probing.pdf',
    'fig_p5_02_clinical_feature_heatmap.pdf',
]

for fig_name in expected_figures:
    fig_path = results_dir / fig_name
    # Also check for _regen suffix
    fig_path_regen = results_dir / fig_name.replace('.pdf', '_regen.pdf')

    if fig_path.exists():
        size_kb = fig_path.stat().st_size / 1024
        successes.append(f"✓ Figure exists: {fig_name} ({size_kb:.1f} KB)")
    elif fig_path_regen.exists():
        size_kb = fig_path_regen.stat().st_size / 1024
        successes.append(f"✓ Figure exists (regen): {fig_name.replace('.pdf', '_regen.pdf')} ({size_kb:.1f} KB)")
    else:
        warnings.append(f"⚠ Figure missing: {fig_name}")

# Check 6: Results JSON
results_json = results_dir / 'probing_results.json'
if results_json.exists():
    try:
        with open(results_json, 'r') as f:
            results = json.load(f)
        successes.append(f"✓ Results JSON exists with keys: {list(results.keys())}")
    except Exception as e:
        warnings.append(f"⚠ Results JSON exists but invalid: {e}")
else:
    warnings.append(f"⚠ Results JSON missing")

# Check 7: Clinical features CSV
clinical_features_csv = project_root / 'data' / 'clinical_features' / 'italian_pvs_features.csv'
if clinical_features_csv.exists():
    successes.append(f"✓ Clinical features CSV exists")
else:
    errors.append(f"✗ Clinical features CSV missing: {clinical_features_csv}")

# Check 8: Fine-tuned model
model_path = project_root / 'results' / 'final_model'
if model_path.exists():
    successes.append(f"✓ Fine-tuned model exists: {model_path}")
else:
    warnings.append(f"⚠ Fine-tuned model missing (will use base model)")

# Check 9: LaTeX installation
import subprocess
try:
    result = subprocess.run(['which', 'pdflatex'], capture_output=True, text=True)
    if result.returncode == 0:
        latex_path = result.stdout.strip()
        successes.append(f"✓ LaTeX installed: {latex_path}")
    else:
        errors.append(f"✗ LaTeX not found (required for publication-quality figures)")
except Exception as e:
    errors.append(f"✗ Could not check LaTeX: {e}")

# Check 10: Required Python packages
required_packages = [
    'torch',
    'transformers',
    'sklearn',
    'matplotlib',
    'seaborn',
    'scipy',
    'tqdm',
]

for pkg in required_packages:
    try:
        __import__(pkg.replace('-', '_'))
        successes.append(f"✓ Package installed: {pkg}")
    except ImportError:
        errors.append(f"✗ Package missing: {pkg}")

# Print results
print("\n" + "=" * 80)
print("SUCCESSES:")
print("=" * 80)
for success in successes:
    print(success)

if warnings:
    print("\n" + "=" * 80)
    print("WARNINGS:")
    print("=" * 80)
    for warning in warnings:
        print(warning)

if errors:
    print("\n" + "=" * 80)
    print("ERRORS:")
    print("=" * 80)
    for error in errors:
        print(error)

# Final status
print("\n" + "=" * 80)
if errors:
    print("STATUS: FAILED - Please fix errors before proceeding")
    sys.exit(1)
elif warnings:
    print("STATUS: READY WITH WARNINGS - Notebook can run but some operations may be slow")
    sys.exit(0)
else:
    print("STATUS: READY - All checks passed!")
    sys.exit(0)
