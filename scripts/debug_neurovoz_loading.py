#!/usr/bin/env python3
"""
debug script to identify why only 248/2976 neurovoz samples are loading.
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

print("=" * 70)
print("NEUROVOZ DATASET LOADING DEBUG")
print("=" * 70)
print()

# direct file analysis without pandas import
audio_dir = project_root / 'data' / 'raw' / 'neurovoz' / 'audios'

print(f"Audio directory: {audio_dir}")
print(f"Directory exists: {audio_dir.exists()}")
print()

if not audio_dir.exists():
    print("ERROR: Audio directory not found!")
    sys.exit(1)

# count all wav files
all_files = list(audio_dir.glob("*.wav"))
print(f"Total WAV files: {len(all_files)}")

# filter out mac os metadata files
clean_files = [f for f in all_files if not f.name.startswith('.') and not f.name.startswith('._')]
print(f"After filtering ._* files: {len(clean_files)}")
print()

# analyze filename patterns
print("Filename pattern analysis:")
print("-" * 70)

parseable = []
unparseable = []
task_counts = Counter()
diagnosis_counts = Counter()
parsing_failures = defaultdict(list)

for audio_file in sorted(clean_files)[:20]:  # sample first 20 for debugging
    filename = audio_file.stem
    print(f"\nFile: {audio_file.name}")
    
    # attempt to parse
    parts = filename.split('_')
    print(f"  Split into {len(parts)} parts: {parts}")
    
    if len(parts) < 3:
        print(f"  FAIL: Not enough parts (need 3, got {len(parts)})")
        unparseable.append(filename)
        parsing_failures['too_few_parts'].append(filename)
        continue
    
    diagnosis = parts[0]
    task_raw = parts[1]
    id_raw = parts[2]
    
    print(f"  Diagnosis: {diagnosis}")
    print(f"  Task: {task_raw}")
    print(f"  ID: {id_raw}")
    
    # check diagnosis
    if diagnosis not in ['HC', 'PD']:
        print(f"  FAIL: Invalid diagnosis '{diagnosis}' (need HC or PD)")
        unparseable.append(filename)
        parsing_failures['invalid_diagnosis'].append(filename)
        continue
    
    # check ID is numeric
    try:
        subject_num = int(id_raw)
        print(f"  Subject number: {subject_num}")
    except ValueError:
        print(f"  FAIL: ID '{id_raw}' is not numeric")
        unparseable.append(filename)
        parsing_failures['non_numeric_id'].append(filename)
        continue
    
    # task parsing
    task_upper = task_raw.upper()
    
    # check if vowel
    vowel_map = {
        'A1': 'vowel_a1', 'A2': 'vowel_a2', 'A3': 'vowel_a3',
        'E1': 'vowel_e1', 'E2': 'vowel_e2', 'E3': 'vowel_e3',
        'I1': 'vowel_i1', 'I2': 'vowel_i2', 'I3': 'vowel_i3',
        'O1': 'vowel_o1', 'O2': 'vowel_o2', 'O3': 'vowel_o3',
        'U1': 'vowel_u1', 'U2': 'vowel_u2', 'U3': 'vowel_u3',
    }
    
    if task_upper in vowel_map:
        task_name = vowel_map[task_upper]
        print(f"  Task name: {task_name} (vowel)")
    elif task_upper == 'FREE':
        task_name = 'free_speech'
        print(f"  Task name: {task_name} (free speech)")
    else:
        task_name = task_raw.lower()
        print(f"  Task name: {task_name} (spanish word)")
    
    print(f"  SUCCESS: Parsed successfully")
    parseable.append(filename)
    task_counts[task_name] += 1
    diagnosis_counts[diagnosis] += 1

print("\n" + "=" * 70)
print("SUMMARY (first 20 files)")
print("=" * 70)
print(f"Parseable: {len(parseable)}")
print(f"Unparseable: {len(unparseable)}")
print()

if parsing_failures:
    print("Parsing failure reasons:")
    for reason, files in parsing_failures.items():
        print(f"  {reason}: {len(files)} files")
        for f in files[:3]:
            print(f"    - {f}")
    print()

print("Task distribution (first 20 files):")
for task, count in sorted(task_counts.items()):
    print(f"  {task}: {count}")
print()

print("Diagnosis distribution (first 20 files):")
for diag, count in sorted(diagnosis_counts.items()):
    print(f"  {diag}: {count}")
print()

# now analyze ALL files
print("=" * 70)
print("FULL DATASET ANALYSIS")
print("=" * 70)
print()

all_parseable = []
all_unparseable = []
all_task_counts = Counter()
all_diagnosis_counts = Counter()

for audio_file in clean_files:
    filename = audio_file.stem
    parts = filename.split('_')
    
    if len(parts) < 3:
        all_unparseable.append(filename)
        continue
    
    diagnosis = parts[0]
    task_raw = parts[1]
    id_raw = parts[2]
    
    if diagnosis not in ['HC', 'PD']:
        all_unparseable.append(filename)
        continue
    
    try:
        subject_num = int(id_raw)
    except ValueError:
        all_unparseable.append(filename)
        continue
    
    # parse task
    task_upper = task_raw.upper()
    if task_upper in vowel_map:
        task_name = vowel_map[task_upper]
    elif task_upper == 'FREE':
        task_name = 'free_speech'
    else:
        task_name = task_raw.lower()
    
    all_parseable.append(filename)
    all_task_counts[task_name] += 1
    all_diagnosis_counts[diagnosis] += 1

print(f"Total parseable: {len(all_parseable)} / {len(clean_files)}")
print(f"Total unparseable: {len(all_unparseable)}")
print()

if all_unparseable:
    print("Sample unparseable files:")
    for f in all_unparseable[:10]:
        print(f"  {f}")
    print()

print("Full task distribution:")
for task, count in sorted(all_task_counts.items()):
    print(f"  {task}: {count}")
print()

print("Full diagnosis distribution:")
for diag, count in sorted(all_diagnosis_counts.items()):
    print(f"  {diag}: {count}")
print()

print("=" * 70)
print("EXPECTED VS ACTUAL")
print("=" * 70)
print(f"Expected: ~2,976 files")
print(f"Found: {len(clean_files)} files")
print(f"Parseable: {len(all_parseable)} files ({len(all_parseable)/len(clean_files)*100:.1f}%)")
print()

if len(all_parseable) < 2000:
    print("WARNING: Only {len(all_parseable)} files parseable - investigation needed!")
    print("This explains why baseline script only loaded 248 samples.")
else:
    print("SUCCESS: Most files are parseable!")
    print("The bug must be elsewhere (metadata loading or feature extraction).")

print("=" * 70)
