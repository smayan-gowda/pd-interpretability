#!/usr/bin/env python3
"""
Execute Phase 05 probing experiments notebook programmatically.
This avoids Jupyter kernel memory issues.
"""

import sys
from pathlib import Path
import json

# Set up project root
project_root = Path('/Volumes/usb drive/pd-interpretability')
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print("="*80)

# Load notebook
notebook_path = project_root / 'notebooks' / 'cpu' / '05_probing_experiments.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Execute each code cell
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        cell_id = cell.get('id', 'unknown')
        source = ''.join(cell['source'])

        # Skip empty cells
        if not source.strip():
            continue

        print(f"\n{'='*80}")
        print(f"Executing Cell {i} (id: {cell_id})")
        print(f"{'='*80}")
        print(source[:200] + ('...' if len(source) > 200 else ''))
        print()

        try:
            exec(source, globals())
            print(f"✓ Cell {i} completed successfully")
        except Exception as e:
            print(f"✗ Cell {i} failed with error: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next cell to see all errors

print("\n" + "="*80)
print("Execution complete!")
print("="*80)
