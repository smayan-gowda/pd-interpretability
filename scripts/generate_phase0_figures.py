"""
generate publication-quality figures for phase 0 data infrastructure.

documents the dataset composition, audio quality, and data pipeline
with times new roman fonts and 300 dpi output.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torchaudio


def set_publication_style():
    """configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
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
        'mathtext.fontset': 'stix',
    })


def collect_dataset_statistics():
    """collect statistics from italian pvs dataset structure or clinical features."""
    # try to get stats from clinical features file first (more reliable)
    features_path = project_root / 'data' / 'clinical_features' / 'italian_pvs_features.csv'
    
    if features_path.exists():
        features_df = pd.read_csv(features_path)
        stats_data = []
        samples_per_subject = []
        
        for subj in features_df['subject_id'].unique():
            subj_df = features_df[features_df['subject_id'] == subj]
            diagnosis = 'pd' if subj_df['label'].iloc[0] == 1 else 'hc'
            n_samples = len(subj_df)
            
            # determine age group from subject naming patterns
            if diagnosis == 'pd':
                age_group = 'pd'
            else:
                # heuristic: check if subject id is all caps (elderly) or mixed case (young)
                subj_str = str(subj)
                if subj_str.isupper() or len(subj_str.split()) >= 2:
                    age_group = 'elderly'
                else:
                    age_group = 'young'
            
            stats_data.append({
                'subjects': subj,
                'diagnosis': diagnosis,
                'n_samples': n_samples,
                'age_group': age_group
            })
            samples_per_subject.append(n_samples)
        
        return pd.DataFrame(stats_data), samples_per_subject
    
    # fallback: scan raw data directory
    data_root = project_root / 'data' / 'raw' / 'italian_pvs'
    stats_data = []
    samples_per_subject = []
    
    # young healthy control (15)
    young_hc = data_root / '15 young healthy control'
    if young_hc.exists():
        for subject_dir in young_hc.iterdir():
            if subject_dir.is_dir():
                audio_files = list(subject_dir.glob('*.txt')) + list(subject_dir.glob('*.wav'))
                n_files = len(audio_files)
                stats_data.append({
                    'subjects': subject_dir.name,
                    'diagnosis': 'hc',
                    'n_samples': n_files,
                    'age_group': 'young'
                })
                samples_per_subject.append(n_files)
    
    # elderly healthy control (22)
    elderly_hc = data_root / '22 elderly healthy control'
    if elderly_hc.exists():
        for subject_dir in elderly_hc.iterdir():
            if subject_dir.is_dir():
                audio_files = list(subject_dir.glob('*.txt')) + list(subject_dir.glob('*.wav'))
                n_files = len(audio_files)
                stats_data.append({
                    'subjects': subject_dir.name,
                    'diagnosis': 'hc',
                    'n_samples': n_files,
                    'age_group': 'elderly'
                })
                samples_per_subject.append(n_files)
    
    # pd patients (28)
    pd_dir = data_root / '28 people with parkinson\'s disease'
    if pd_dir.exists():
        for subgroup in pd_dir.iterdir():
            if subgroup.is_dir():
                for subject_dir in subgroup.iterdir():
                    if subject_dir.is_dir():
                        audio_files = list(subject_dir.glob('*.txt')) + list(subject_dir.glob('*.wav'))
                        n_files = len(audio_files)
                        stats_data.append({
                            'subjects': subject_dir.name,
                            'diagnosis': 'pd',
                            'n_samples': n_files,
                            'age_group': 'pd'
                        })
                        samples_per_subject.append(n_files)
    
    return pd.DataFrame(stats_data), samples_per_subject


def fig1_dataset_composition(stats_df, save_path):
    """
    figure 1: dataset composition overview.
    
    bar chart showing subject and sample distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # subjects by diagnosis
    ax1 = axes[0]
    subject_counts = stats_df.groupby('diagnosis').size()
    colors = {'hc': '#2ecc71', 'pd': '#e74c3c'}
    bars = ax1.bar(['Healthy Controls', "Parkinson's Disease"], 
                   [subject_counts.get('hc', 0), subject_counts.get('pd', 0)],
                   color=[colors['hc'], colors['pd']], edgecolor='black', linewidth=0.5)
    
    # add value labels on bars
    for bar, count in zip(bars, [subject_counts.get('hc', 0), subject_counts.get('pd', 0)]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Number of Subjects')
    ax1.set_title('Subject Distribution by Diagnosis')
    ax1.set_ylim(0, max(subject_counts) * 1.15)
    
    # samples by diagnosis
    ax2 = axes[1]
    sample_counts = stats_df.groupby('diagnosis')['n_samples'].sum()
    bars2 = ax2.bar(['Healthy Controls', "Parkinson's Disease"], 
                    [sample_counts.get('hc', 0), sample_counts.get('pd', 0)],
                    color=[colors['hc'], colors['pd']], edgecolor='black', linewidth=0.5)
    
    for bar, count in zip(bars2, [sample_counts.get('hc', 0), sample_counts.get('pd', 0)]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Number of Audio Samples')
    ax2.set_title('Audio Sample Distribution by Diagnosis')
    ax2.set_ylim(0, max(sample_counts) * 1.15)
    
    fig.suptitle('Italian PVS Dataset Composition', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig2_samples_per_subject(stats_df, save_path):
    """
    figure 2: samples per subject distribution.
    
    histogram showing variability in samples per subject.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    hc_samples = stats_df[stats_df['diagnosis'] == 'hc']['n_samples']
    pd_samples = stats_df[stats_df['diagnosis'] == 'pd']['n_samples']
    
    bins = range(0, max(stats_df['n_samples']) + 5, 2)
    
    ax.hist(hc_samples, bins=bins, alpha=0.7, label='Healthy Controls', 
            color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.hist(pd_samples, bins=bins, alpha=0.7, label="Parkinson's Disease", 
            color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    ax.axvline(hc_samples.mean(), color='#27ae60', linestyle='--', linewidth=1.5, 
               label=f'HC mean: {hc_samples.mean():.1f}')
    ax.axvline(pd_samples.mean(), color='#c0392b', linestyle='--', linewidth=1.5, 
               label=f'PD mean: {pd_samples.mean():.1f}')
    
    ax.set_xlabel('Number of Audio Samples per Subject')
    ax.set_ylabel('Number of Subjects')
    ax.set_title('Distribution of Audio Samples per Subject')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig3_age_group_distribution(stats_df, save_path):
    """
    figure 3: subject distribution by age group.
    
    stacked bar showing young vs elderly healthy controls and pd.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    age_counts = stats_df.groupby(['age_group', 'diagnosis']).size().unstack(fill_value=0)
    
    categories = ['Young HC', 'Elderly HC', 'PD Patients']
    young_hc = len(stats_df[(stats_df['age_group'] == 'young') & (stats_df['diagnosis'] == 'hc')])
    elderly_hc = len(stats_df[(stats_df['age_group'] == 'elderly') & (stats_df['diagnosis'] == 'hc')])
    pd_patients = len(stats_df[stats_df['diagnosis'] == 'pd'])
    
    values = [young_hc, elderly_hc, pd_patients]
    colors = ['#27ae60', '#2ecc71', '#e74c3c']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.5)
    
    for bar, count in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Number of Subjects')
    ax.set_title('Subject Distribution by Group')
    ax.set_ylim(0, max(values) * 1.15)
    
    # add total annotation
    total = sum(values)
    ax.annotate(f'Total: {total} subjects', xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig4_dataset_summary_table(stats_df, save_path):
    """
    figure 4: dataset summary statistics table.
    
    visual table showing key dataset statistics.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    
    hc_df = stats_df[stats_df['diagnosis'] == 'hc']
    pd_df = stats_df[stats_df['diagnosis'] == 'pd']
    
    table_data = [
        ['Metric', 'Healthy Controls', "Parkinson's Disease", 'Total'],
        ['Subjects', str(len(hc_df)), str(len(pd_df)), str(len(stats_df))],
        ['Audio Samples', str(hc_df['n_samples'].sum()), str(pd_df['n_samples'].sum()), 
         str(stats_df['n_samples'].sum())],
        ['Samples/Subject (mean)', f"{hc_df['n_samples'].mean():.1f}", 
         f"{pd_df['n_samples'].mean():.1f}", f"{stats_df['n_samples'].mean():.1f}"],
        ['Samples/Subject (std)', f"{hc_df['n_samples'].std():.1f}", 
         f"{pd_df['n_samples'].std():.1f}", f"{stats_df['n_samples'].std():.1f}"],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # style header row
    for j in range(4):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # alternate row colors
    for i in range(1, 5):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax.set_title('Italian PVS Dataset Summary Statistics', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig5_preprocessing_pipeline(save_path):
    """
    figure 5: data preprocessing pipeline diagram.
    
    flowchart showing the preprocessing steps.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # define pipeline steps
    steps = [
        ('Raw Audio\n(Various formats)', '#3498db'),
        ('Load & Resample\n(16 kHz mono)', '#9b59b6'),
        ('Duration Filter\n(0.5-10s)', '#e74c3c'),
        ('Amplitude\nNormalization', '#f39c12'),
        ('Wav2Vec2\nReady', '#2ecc71'),
    ]
    
    n_steps = len(steps)
    box_width = 0.15
    box_height = 0.6
    spacing = (1 - n_steps * box_width) / (n_steps + 1)
    
    for i, (label, color) in enumerate(steps):
        x = spacing + i * (box_width + spacing)
        y = 0.2
        
        # draw box
        rect = plt.Rectangle((x, y), box_width, box_height, 
                             facecolor=color, edgecolor='black', 
                             linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        
        # add text
        ax.text(x + box_width/2, y + box_height/2, label, 
               ha='center', va='center', fontsize=9, fontweight='bold',
               color='white', wrap=True)
        
        # draw arrow to next step
        if i < n_steps - 1:
            arrow_x = x + box_width
            arrow_end = spacing + (i+1) * (box_width + spacing)
            ax.annotate('', xy=(arrow_end, y + box_height/2), 
                       xytext=(arrow_x, y + box_height/2),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Audio Preprocessing Pipeline', fontsize=12, y=0.95)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def main():
    """generate all phase 0 figures."""
    set_publication_style()
    
    print("collecting dataset statistics...")
    stats_df, samples_per_subject = collect_dataset_statistics()
    
    if len(stats_df) == 0:
        print("error: no dataset statistics collected")
        return
    
    print(f"found {len(stats_df)} subjects, {stats_df['n_samples'].sum()} total samples")
    
    figures_dir = project_root / 'results' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("\ngenerating phase 0 figures...")
    print("=" * 50)
    
    fig1_dataset_composition(stats_df, figures_dir / 'fig_p0_01_dataset_composition.pdf')
    fig1_dataset_composition(stats_df, figures_dir / 'fig_p0_01_dataset_composition.png')
    
    fig2_samples_per_subject(stats_df, figures_dir / 'fig_p0_02_samples_per_subject.pdf')
    fig2_samples_per_subject(stats_df, figures_dir / 'fig_p0_02_samples_per_subject.png')
    
    fig3_age_group_distribution(stats_df, figures_dir / 'fig_p0_03_age_group_distribution.pdf')
    fig3_age_group_distribution(stats_df, figures_dir / 'fig_p0_03_age_group_distribution.png')
    
    fig4_dataset_summary_table(stats_df, figures_dir / 'fig_p0_04_dataset_summary.pdf')
    fig4_dataset_summary_table(stats_df, figures_dir / 'fig_p0_04_dataset_summary.png')
    
    fig5_preprocessing_pipeline(figures_dir / 'fig_p0_05_preprocessing_pipeline.pdf')
    fig5_preprocessing_pipeline(figures_dir / 'fig_p0_05_preprocessing_pipeline.png')
    
    print("=" * 50)
    print(f"\nphase 0 figures saved to: {figures_dir}")
    print("\nfigure summary:")
    print("  fig_p0_01: dataset composition (subjects and samples)")
    print("  fig_p0_02: samples per subject distribution")
    print("  fig_p0_03: age group distribution")
    print("  fig_p0_04: dataset summary statistics table")
    print("  fig_p0_05: preprocessing pipeline diagram")


if __name__ == '__main__':
    main()
