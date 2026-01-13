#!/usr/bin/env python3
"""
convert colab notebooks to chameleon cloud format
removes google.colab imports and updates paths
"""

import json
import sys
from pathlib import Path

def convert_notebook(notebook_path: Path, project_root: str = "/home/cc/projects/pd-interpretability"):
    """convert a single notebook from colab to chameleon format"""

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    modified = False
    cells_to_remove = []

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            original_source = source

            # remove google colab imports and drive mounting
            if 'from google.colab import drive' in source or 'drive.mount' in source:
                cells_to_remove.append(i)
                modified = True
                continue

            # replace colab paths with chameleon paths
            source = source.replace('/content/drive/MyDrive/pd-interpretability', project_root)
            source = source.replace('/content/drive/.shortcut-targets-by-id/', project_root + '/')

            # update any remaining /content references
            source = source.replace('/content/', '/home/cc/')

            if source != original_source:
                cell['source'] = source.split('\n')
                # ensure each line ends with \n except the last
                cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                                 for i, line in enumerate(cell['source'])]
                modified = True

    # remove cells marked for deletion (in reverse order to preserve indices)
    for i in reversed(cells_to_remove):
        nb['cells'].pop(i)
        print(f"  removed cell {i} (google colab specific)")

    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        return True
    return False

def main():
    if len(sys.argv) > 1:
        notebook_path = Path(sys.argv[1])
        if convert_notebook(notebook_path):
            print(f"converted {notebook_path}")
        else:
            print(f"no changes needed for {notebook_path}")
    else:
        # convert all gpu notebooks
        gpu_notebooks = Path('notebooks/gpu').glob('*.ipynb')
        for nb_path in sorted(gpu_notebooks):
            print(f"\nconverting {nb_path}...")
            if convert_notebook(nb_path):
                print(f"  -> converted")
            else:
                print(f"  -> no changes needed")

if __name__ == '__main__':
    main()
