"""
generate publication-quality figures for phase 1 clinical feature extraction.

documents the clinical feature extraction quality, distributions, and 
statistical properties with times new roman fonts and 300 dpi output.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

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


def load_clinical_features():
    """load extracted clinical features."""
    features_path = project_root / 'data' / 'clinical_features' / 'italian_pvs_features.csv'
    return pd.read_csv(features_path)


def fig1_f0_distribution(features_df, save_path):
    """
    figure 1: fundamental frequency (f0) distribution.
    
    violin plot showing f0_mean distribution for hc vs pd.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # f0 mean
    ax1 = axes[0]
    hc_f0 = features_df[features_df['label'] == 0]['f0_mean'].dropna()
    pd_f0 = features_df[features_df['label'] == 1]['f0_mean'].dropna()
    
    parts = ax1.violinplot([hc_f0, pd_f0], positions=[0, 1], showmeans=True, showmedians=True)
    colors = ['#2ecc71', '#e74c3c']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        if partname in parts:
            parts[partname].set_color('black')
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([r'Healthy', r"Parkinson's"])
    ax1.set_ylabel(r'$F_0$ Mean (Hz)')
    ax1.set_title(r'Fundamental Frequency Distribution')
    
    # statistical annotation
    t_stat, p_val = stats.ttest_ind(hc_f0, pd_f0)
    sig = r'$p < 0.001$' if p_val < 0.001 else r'$p = %.3f$' % p_val
    ax1.annotate(sig, xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=9)
    
    # f0 std
    ax2 = axes[1]
    hc_f0_std = features_df[features_df['label'] == 0]['f0_std'].dropna()
    pd_f0_std = features_df[features_df['label'] == 1]['f0_std'].dropna()
    
    parts2 = ax2.violinplot([hc_f0_std, pd_f0_std], positions=[0, 1], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts2['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        if partname in parts2:
            parts2[partname].set_color('black')
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([r'Healthy', r"Parkinson's"])
    ax2.set_ylabel(r'$F_0$ Standard Deviation (Hz)')
    ax2.set_title(r'$F_0$ Variability Distribution')
    
    t_stat2, p_val2 = stats.ttest_ind(hc_f0_std, pd_f0_std)
    sig2 = r'$p < 0.001$' if p_val2 < 0.001 else r'$p = %.3f$' % p_val2
    ax2.annotate(sig2, xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=9)
    
    fig.suptitle(r'Fundamental Frequency ($F_0$) Analysis', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig2_jitter_shimmer(features_df, save_path):
    """
    figure 2: jitter and shimmer distributions.
    
    key voice quality measures for pd detection.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    measures = [
        ('jitter_local', 'Jitter (local)', 'Jitter Local (%)'),
        ('jitter_rap', 'Jitter RAP', 'Jitter RAP (%)'),
        ('shimmer_local', 'Shimmer (local)', 'Shimmer Local (dB)'),
        ('shimmer_apq3', 'Shimmer APQ3', 'Shimmer APQ3 (dB)'),
    ]
    
    colors = ['#2ecc71', '#e74c3c']
    
    for idx, (feature, title, ylabel) in enumerate(measures):
        ax = axes[idx // 2, idx % 2]
        
        hc_data = features_df[features_df['label'] == 0][feature].dropna()
        pd_data = features_df[features_df['label'] == 1][feature].dropna()
        
        parts = ax.violinplot([hc_data, pd_data], positions=[0, 1], showmeans=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
            if partname in parts:
                parts[partname].set_color('black')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r'Healthy', r"Parkinson's"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # effect size
        pooled_std = np.sqrt((hc_data.std()**2 + pd_data.std()**2) / 2)
        cohens_d = (pd_data.mean() - hc_data.mean()) / pooled_std if pooled_std > 0 else 0
        ax.annotate(r"Cohen's $d$ = %.2f" % cohens_d, xy=(0.5, 0.95), xycoords='axes fraction', 
                   ha='center', fontsize=8)
    
    fig.suptitle(r'Jitter and Shimmer Analysis: Voice Quality Measures', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig3_hnr_distribution(features_df, save_path):
    """
    figure 3: harmonics-to-noise ratio (hnr) distribution.
    
    hnr is a key clinical biomarker for voice disorders.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    hc_hnr = features_df[features_df['label'] == 0]['hnr_mean'].dropna()
    pd_hnr = features_df[features_df['label'] == 1]['hnr_mean'].dropna()
    
    # kernel density plot
    sns.kdeplot(hc_hnr, ax=ax, color='#2ecc71', fill=True, alpha=0.5, 
                label=r'Healthy ($n$=%d, $\mu$=%.1f)' % (len(hc_hnr), hc_hnr.mean()))
    sns.kdeplot(pd_hnr, ax=ax, color='#e74c3c', fill=True, alpha=0.5, 
                label=r"Parkinson's ($n$=%d, $\mu$=%.1f)" % (len(pd_hnr), pd_hnr.mean()))
    
    # add vertical lines for means
    ax.axvline(hc_hnr.mean(), color='#27ae60', linestyle='--', linewidth=2)
    ax.axvline(pd_hnr.mean(), color='#c0392b', linestyle='--', linewidth=2)
    
    ax.set_xlabel(r'Harmonics-to-Noise Ratio (dB)')
    ax.set_ylabel(r'Density')
    ax.set_title(r"HNR Distribution: Healthy vs Parkinson's")
    ax.legend(loc='upper right')
    
    # statistical test
    t_stat, p_val = stats.ttest_ind(hc_hnr, pd_hnr)
    pooled_std = np.sqrt((hc_hnr.std()**2 + pd_hnr.std()**2) / 2)
    cohens_d = (pd_hnr.mean() - hc_hnr.mean()) / pooled_std if pooled_std > 0 else 0
    
    stats_text = r"$t$ = %.2f, $p < 0.001$" % t_stat + "\n" + r"Cohen's $d$ = %.2f" % cohens_d if p_val < 0.001 else \
                 r"$t$ = %.2f, $p$ = %.3f" % (t_stat, p_val) + "\n" + r"Cohen's $d$ = %.2f" % cohens_d
    ax.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
               ha='left', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig4_feature_extraction_quality(features_df, save_path):
    """
    figure 4: feature extraction quality assessment.
    
    shows completeness and validity of extracted features.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # missing data percentage
    ax1 = axes[0]
    feature_cols = [c for c in features_df.columns if c not in ['subject_id', 'label', 'path', 'diagnosis']]
    missing_pct = (features_df[feature_cols].isnull().sum() / len(features_df) * 100).sort_values(ascending=True)
    
    colors = plt.cm.RdYlGn_r(missing_pct / max(missing_pct.max(), 1))
    bars = ax1.barh(range(len(missing_pct)), missing_pct.values, color=colors, edgecolor='black', linewidth=0.3)
    ax1.set_yticks(range(len(missing_pct)))
    ax1.set_yticklabels([f.replace('_', ' ') for f in missing_pct.index], fontsize=7)
    ax1.set_xlabel(r'Missing Data (\%)')
    ax1.set_title(r'Feature Extraction Completeness')
    ax1.axvline(5, color='red', linestyle='--', alpha=0.7, label=r'5\% threshold')
    ax1.legend(loc='lower right', fontsize=8)
    
    # feature value ranges (boxplot for key features)
    ax2 = axes[1]
    key_features = ['f0_mean', 'jitter_local', 'shimmer_local', 'hnr_mean']
    available_features = [f for f in key_features if f in features_df.columns]
    
    # normalize for comparison
    normalized_data = []
    labels = []
    for feat in available_features:
        data = features_df[feat].dropna()
        normalized = (data - data.mean()) / data.std()
        normalized_data.append(normalized.values)
        labels.append(feat.replace('_', '\n'))
    
    bp = ax2.boxplot(normalized_data, tick_labels=labels, patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(available_features)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel(r'Normalized Value ($z$-score)')
    ax2.set_title(r'Key Feature Value Distributions')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    fig.suptitle(r'Clinical Feature Extraction Quality Assessment', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig5_formant_analysis(features_df, save_path):
    """
    figure 5: formant frequency analysis.
    
    f1-f4 formant distributions for vowel characterization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    formants = [
        ('f1_mean', r'$F_1$ (First Formant)'),
        ('f2_mean', r'$F_2$ (Second Formant)'),
        ('f3_mean', r'$F_3$ (Third Formant)'),
        ('f4_mean', r'$F_4$ (Fourth Formant)'),
    ]
    
    colors = ['#2ecc71', '#e74c3c']
    
    for idx, (feature, title) in enumerate(formants):
        ax = axes[idx // 2, idx % 2]
        
        if feature not in features_df.columns:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            continue
        
        hc_data = features_df[features_df['label'] == 0][feature].dropna()
        pd_data = features_df[features_df['label'] == 1][feature].dropna()
        
        if len(hc_data) == 0 or len(pd_data) == 0:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            continue
        
        # histogram
        bins = 30
        ax.hist(hc_data, bins=bins, alpha=0.6, label=r'Healthy', color=colors[0], 
                edgecolor='black', linewidth=0.3, density=True)
        ax.hist(pd_data, bins=bins, alpha=0.6, label=r"Parkinson's", color=colors[1], 
                edgecolor='black', linewidth=0.3, density=True)
        
        ax.axvline(hc_data.mean(), color='#27ae60', linestyle='--', linewidth=1.5)
        ax.axvline(pd_data.mean(), color='#c0392b', linestyle='--', linewidth=1.5)
        
        ax.set_xlabel(r'Frequency (Hz)')
        ax.set_ylabel(r'Density')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
    
    fig.suptitle(r'Formant Frequency Analysis', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def fig6_feature_summary_table(features_df, save_path):
    """
    figure 6: clinical feature summary statistics table.

    publication-quality latex booktabs table compiled to pdf.
    """
    import subprocess
    import tempfile

    # use same features as baseline (17 features)
    feature_cols = [
        'f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range',
        'voicing_fraction',
        'jitter_local', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp',
        'shimmer_local', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda',
        'hnr_mean', 'hnr_std'
    ]
    available_cols = [c for c in feature_cols if c in features_df.columns]

    hc_df = features_df[features_df['label'] == 0]
    pd_df = features_df[features_df['label'] == 1]

    # collect data for table
    rows = []
    for feat in available_cols:
        hc_vals = hc_df[feat].dropna()
        pd_vals = pd_df[feat].dropna()

        hc_mean, hc_std = hc_vals.mean(), hc_vals.std()
        pd_mean, pd_std = pd_vals.mean(), pd_vals.std()

        _, p_val = stats.ttest_ind(hc_vals, pd_vals)

        pooled_std = np.sqrt((hc_std**2 + pd_std**2) / 2)
        cohens_d = (pd_mean - hc_mean) / pooled_std if pooled_std > 0 else 0

        rows.append({
            'feature': feat.replace('_', ' '),
            'hc_mean': hc_mean, 'hc_std': hc_std,
            'pd_mean': pd_mean, 'pd_std': pd_std,
            'p_val': p_val, 'cohens_d': cohens_d
        })

    # generate latex code
    latex_code = r'''\documentclass[12pt]{article}
\usepackage{booktabs}
\usepackage{times}
\usepackage{array}
\usepackage[margin=0.5in]{geometry}
\pagestyle{empty}
\begin{document}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Feature} & \textbf{HC} $\boldsymbol{\mu}$ ($\boldsymbol{\sigma}$) & \textbf{PD} $\boldsymbol{\mu}$ ($\boldsymbol{\sigma}$) & \textbf{$p$-value} & \textbf{Cohen's $d$} \\
\midrule
'''

    for row in rows:
        feature_name = row['feature']
        hc_text = f"{row['hc_mean']:.4f} ({row['hc_std']:.4f})"
        pd_text = f"{row['pd_mean']:.4f} ({row['pd_std']:.4f})"

        # format p-value
        if row['p_val'] < 0.001:
            p_text = r'\textbf{$<$0.001}'
        elif row['p_val'] < 0.01:
            p_text = r'\textbf{$<$0.01}'
        elif row['p_val'] < 0.05:
            p_text = f"\\textbf{{{row['p_val']:.3f}}}"
        else:
            p_text = f"{row['p_val']:.3f}"

        d_text = f"{row['cohens_d']:.2f}"

        latex_code += f"{feature_name} & {hc_text} & {pd_text} & {p_text} & {d_text} \\\\\n"

    latex_code += r'''\bottomrule
\end{tabular}
\end{document}'''

    # compile with pdflatex
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / 'table.tex'
        pdf_path = Path(tmpdir) / 'table.pdf'

        # write latex file
        with open(tex_path, 'w') as f:
            f.write(latex_code)

        # compile to pdf
        try:
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', tmpdir, str(tex_path)],
                capture_output=True,
                timeout=30
            )

            # copy pdf to final location
            save_path = Path(save_path)
            if pdf_path.exists():
                if save_path.suffix == '.pdf':
                    subprocess.run(['cp', str(pdf_path), str(save_path)], check=True)
                    print(f"saved: {save_path}")
                elif save_path.suffix == '.png':
                    # convert pdf to png using imagemagick if available
                    try:
                        subprocess.run(
                            ['convert', '-density', '300', str(pdf_path), '-quality', '100', str(save_path)],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=30
                        )
                        print(f"saved: {save_path}")
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # fallback: just copy the pdf
                        pdf_save = save_path.with_suffix('.pdf')
                        subprocess.run(['cp', str(pdf_path), str(pdf_save)], check=True)
                        print(f"warning: imagemagick not available, saved as pdf: {pdf_save}")
            else:
                print(f"error: pdf not generated for {save_path}")
                print(f"pdflatex return code: {result.returncode}")
                print(f"pdflatex output (last 1000 chars):\n{result.stdout.decode()[-1000:]}")

        except FileNotFoundError:
            print("error: pdflatex not found. install texlive.")
        except Exception as e:
            print(f"unexpected error: {e}")


def main():
    """generate all phase 1 figures."""
    set_publication_style()
    
    print("loading clinical features...")
    features_df = load_clinical_features()
    print(f"loaded {len(features_df)} samples")
    
    figures_dir = project_root / 'results' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("\ngenerating phase 1 figures...")
    print("=" * 50)
    
    fig1_f0_distribution(features_df, figures_dir / 'fig_p1_01_f0_distribution.pdf')
    fig1_f0_distribution(features_df, figures_dir / 'fig_p1_01_f0_distribution.png')
    
    fig2_jitter_shimmer(features_df, figures_dir / 'fig_p1_02_jitter_shimmer.pdf')
    fig2_jitter_shimmer(features_df, figures_dir / 'fig_p1_02_jitter_shimmer.png')
    
    fig3_hnr_distribution(features_df, figures_dir / 'fig_p1_03_hnr_distribution.pdf')
    fig3_hnr_distribution(features_df, figures_dir / 'fig_p1_03_hnr_distribution.png')
    
    fig4_feature_extraction_quality(features_df, figures_dir / 'fig_p1_04_extraction_quality.pdf')
    fig4_feature_extraction_quality(features_df, figures_dir / 'fig_p1_04_extraction_quality.png')
    
    fig5_formant_analysis(features_df, figures_dir / 'fig_p1_05_formant_analysis.pdf')
    fig5_formant_analysis(features_df, figures_dir / 'fig_p1_05_formant_analysis.png')
    
    fig6_feature_summary_table(features_df, figures_dir / 'fig_p1_06_feature_summary.pdf')
    fig6_feature_summary_table(features_df, figures_dir / 'fig_p1_06_feature_summary.png')
    
    print("=" * 50)
    print(f"\nphase 1 figures saved to: {figures_dir}")
    print("\nfigure summary:")
    print("  fig_p1_01: f0 (fundamental frequency) distribution")
    print("  fig_p1_02: jitter and shimmer distributions")
    print("  fig_p1_03: hnr distribution")
    print("  fig_p1_04: feature extraction quality assessment")
    print("  fig_p1_05: formant frequency analysis")
    print("  fig_p1_06: clinical feature summary table")


if __name__ == '__main__':
    main()
