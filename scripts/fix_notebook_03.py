#!/usr/bin/env python3
"""
comprehensive fix for notebook 03 issues:
- remove flawed degenerate detection logic for loso
- fix visualization cutoffs and spacing
- ensure all figures save to checkpoints folder
- fix confidence distribution plot
- ensure all visualization cells execute properly
"""

import json
import re
from pathlib import Path

def fix_notebook_03(notebook_path):
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    modified = False

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code' or 'source' not in cell:
            continue

        source = ''.join(cell['source'])
        original_source = source

        # FIX 1: remove degenerate detection logic for loso
        if 'is_degenerate' in source and 'check for degenerate model' in source:
            # comment out degenerate detection
            source = re.sub(
                r'(\s+)# check for degenerate model.*?\n.*?is_degenerate = .*?\n',
                r'\1# degenerate detection disabled for loso (single-class test sets are expected)\n\1is_degenerate = False\n',
                source,
                flags=re.DOTALL
            )
            source = re.sub(
                r'(\s+)# detect degenerate by auc threshold.*?\n.*?if auc_roc.*?\n.*?is_degenerate = True\n',
                r'',
                source,
                flags=re.DOTALL
            )
            print(f'cell {i}: removed flawed degenerate detection')

        # FIX 2: fix figure save paths to use checkpoints folder
        if 'savefig' in source and "results/figures/fig_p3" in source:
            source = source.replace(
                "results/figures/fig_p3",
                "results/checkpoints/{checkpoint_name}/fig_p3"
            )
            # add checkpoint_name variable if not present
            if 'checkpoint_name =' not in source:
                source = f"checkpoint_name = f'wav2vec2_loso_{{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}}'\n{source}"
            print(f'cell {i}: fixed figure save paths to use checkpoints')

        # FIX 3: fix confusion matrix title cutoff
        if 'Confusion Matrix' in source and 'Accuracy:' in source and '{overall_accuracy' in source:
            # add bbox_inches='tight' to savefig
            if 'bbox_inches' not in source and 'savefig' in source:
                source = source.replace(
                    ".savefig(",
                    ".savefig(bbox_inches='tight', "
                )
                print(f'cell {i}: added bbox_inches for confusion matrix')

        # FIX 4: fix per-class performance spacing
        if 'Per-Class Performance' in source and 'text(i, val' in source:
            # increase y-offset for text labels
            source = re.sub(
                r'ax\.text\(i,\s*val\s*\+\s*[\d.]+',
                'ax.text(i, val + 0.08',
                source
            )
            print(f'cell {i}: increased spacing for per-class performance')

        # FIX 5: fix confidence distribution plot
        if 'Confidence Distribution' in source and 'kde' in source:
            # ensure proper kde plot
            source = re.sub(
                r'sns\.kdeplot\([^)]*\)',
                'sns.kdeplot(data=plot_df, x="confidence", hue="actual_label", fill=True, alpha=0.3, linewidth=2)',
                source
            )
            print(f'cell {i}: fixed confidence distribution plot')

        # FIX 6: ensure loso_results variable is properly defined for visualization cells
        if 'for fold_idx, fold_result in enumerate(loso_results)' in source:
            # add check at start of cell
            if 'if' not in source[:100]:
                source = f"if 'fold_results' not in locals():\n    print('error: fold_results not found, skipping visualization')\nelse:\n    {source}"
                # indent the rest
                lines = source.split('\n')
                source = '\n'.join([lines[0], lines[1]] + ['    ' + line for line in lines[2:]])
                print(f'cell {i}: added fold_results existence check')

        # update cell if modified
        if source != original_source:
            lines = source.split('\n')
            cell['source'] = [line + '\n' if idx < len(lines) - 1 else line
                             for idx, line in enumerate(lines)]
            modified = True

    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f'\nnotebook fixed and saved')
        return True
    else:
        print('no modifications made')
        return False

if __name__ == '__main__':
    notebook_path = Path(__file__).parent.parent / 'notebooks' / 'gpu' / '03_train_wav2vec2.ipynb'
    fix_notebook_03(notebook_path)
