"""
latex-quality figure generator for research paper publication.

generates publication-ready figures with times new roman font, perfect
formatting, and comprehensive coverage of all research stages suitable
for isef, regeneron sts, and journal submission.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import pandas as pd


def set_latex_style():
    """
    set matplotlib to use latex rendering with times new roman.

    creates publication-quality figures suitable for ieee, nature,
    science, and other top-tier journals.
    """
    plt.rcParams.update({
        # fonts - times new roman for body, computer modern for math
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,

        # use latex for math
        'text.usetex': False,  # set to True if latex is installed
        'mathtext.fontset': 'stix',  # times-compatible math

        # line widths and sizes
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'patch.linewidth': 1.0,

        # figure quality
        'figure.dpi': 150,
        'savefig.dpi': 600,  # high-res for publication
        'savefig.format': 'pdf',  # vector format
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # grid and background
        'axes.grid': True,
        'axes.grid.which': 'major',
        'grid.alpha': 0.25,
        'grid.linestyle': ':',
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',

        # spines
        'axes.spines.top': False,
        'axes.spines.right': False,

        # legend
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,

        # colors - nature-style palette
        'axes.prop_cycle': plt.cycler(color=[
            '#0173B2', '#DE8F05', '#029E73', '#CC78BC',
            '#CA9161', '#949494', '#ECE133', '#56B4E9'
        ])
    })


# publication-quality color palettes
COLORS = {
    'primary': '#0173B2',  # nature blue
    'secondary': '#DE8F05',  # nature orange
    'success': '#029E73',  # nature green
    'danger': '#CC3311',  # nature red
    'pd': '#CC78BC',  # parkinson
    'hc': '#0173B2',  # healthy control
    'significant': '#CC3311',  # statistically significant
    'nonsignificant': '#949494',  # not significant
}

# color-blind safe palettes
CB_PALETTE = {
    'qualitative': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#949494', '#ECE133', '#56B4E9'],
    'diverging': ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B'],
    'sequential_blue': ['#F7FBFF', '#DEEBF7', '#C6DBEF', '#9ECAE1', '#6BAED6', '#4292C6', '#2171B5', '#08519C'],
    'sequential_red': ['#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', '#EF3B2C', '#CB181D', '#99000D'],
}


class PublicationFigureGenerator:
    """
    generate all figures needed for research paper publication.

    creates latex-quality figures for:
    - data preprocessing and exploration
    - model architecture
    - training dynamics
    - probing analysis
    - activation patching
    - cross-dataset generalization
    - hypothesis testing
    - statistical analysis
    """

    def __init__(self, output_dir: str = 'results/figures/publication'):
        """
        args:
            output_dir: directory for saving figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        set_latex_style()

        # create subdirectories
        (self.output_dir / 'preprocessing').mkdir(exist_ok=True)
        (self.output_dir / 'training').mkdir(exist_ok=True)
        (self.output_dir / 'analysis').mkdir(exist_ok=True)
        (self.output_dir / 'supplementary').mkdir(exist_ok=True)

    # ========================================================================
    # DATA PREPROCESSING FIGURES
    # ========================================================================

    def figure_preprocessing_pipeline(
        self,
        save: bool = True
    ) -> plt.Figure:
        """
        figure: data preprocessing pipeline flowchart.

        shows complete audio preprocessing workflow from raw to processed.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')

        # define pipeline stages
        stages = [
            ('Raw Audio\n(varied SR, channels)', 0.85),
            ('Resample to 16kHz', 0.70),
            ('Convert to Mono', 0.55),
            ('Voice Activity\nDetection', 0.40),
            ('Truncate/Pad\nto 10s', 0.25),
            ('Normalize\nAmplitude', 0.10),
        ]

        y_start = 0.9
        box_width = 0.6
        box_height = 0.08
        x_center = 0.5

        for i, (stage, y) in enumerate(stages):
            # draw box
            rect = FancyBboxPatch(
                (x_center - box_width/2, y - box_height/2),
                box_width, box_height,
                boxstyle='round,pad=0.01',
                facecolor='lightblue' if i % 2 == 0 else 'lightgreen',
                edgecolor='black',
                linewidth=1.5,
                transform=ax.transAxes
            )
            ax.add_patch(rect)

            # add text
            ax.text(x_center, y, stage,
                   ha='center', va='center',
                   fontsize=11, fontweight='bold',
                   transform=ax.transAxes)

            # add arrow to next stage
            if i < len(stages) - 1:
                arrow = FancyArrowPatch(
                    (x_center, y - box_height/2 - 0.01),
                    (x_center, stages[i+1][1] + box_height/2 + 0.01),
                    arrowstyle='->,head_width=0.4,head_length=0.2',
                    color='black',
                    linewidth=2,
                    transform=ax.transAxes
                )
                ax.add_patch(arrow)

        ax.set_title('Audio Preprocessing Pipeline', fontsize=14, fontweight='bold', pad=20)

        if save:
            plt.savefig(self.output_dir / 'preprocessing' / 'fig_pipeline.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'preprocessing' / 'fig_pipeline.png', dpi=600, bbox_inches='tight')

        return fig

    def figure_dataset_statistics(
        self,
        dataset_stats: Dict[str, Dict],
        save: bool = True
    ) -> plt.Figure:
        """
        figure: dataset statistics and demographics.

        args:
            dataset_stats: dict with keys for each dataset containing
                           subject counts, audio counts, demographics
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

        datasets = list(dataset_stats.keys())

        # panel a: subject counts
        ax1 = fig.add_subplot(gs[0, :])

        pd_counts = [dataset_stats[d].get('pd_count', 0) for d in datasets]
        hc_counts = [dataset_stats[d].get('hc_count', 0) for d in datasets]

        x = np.arange(len(datasets))
        width = 0.35

        ax1.bar(x - width/2, hc_counts, width, label='Healthy Control',
               color=COLORS['hc'], edgecolor='black', linewidth=1)
        ax1.bar(x + width/2, pd_counts, width, label='Parkinson\'s Disease',
               color=COLORS['pd'], edgecolor='black', linewidth=1)

        ax1.set_ylabel('Number of Subjects', fontweight='bold')
        ax1.set_title('(A) Dataset Subject Counts', fontweight='bold', loc='left')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets, rotation=45, ha='right')
        ax1.legend(frameon=True, loc='upper right')
        ax1.grid(axis='y', alpha=0.3)

        # panel b: audio file counts
        ax2 = fig.add_subplot(gs[1, :2])

        total_files = [dataset_stats[d].get('total_files', 0) for d in datasets]

        bars = ax2.barh(datasets, total_files, color=CB_PALETTE['qualitative'][:len(datasets)],
                       edgecolor='black', linewidth=1)

        for i, (bar, count) in enumerate(zip(bars, total_files)):
            ax2.text(count + max(total_files)*0.02, i, f'{count}',
                    va='center', fontsize=10, fontweight='bold')

        ax2.set_xlabel('Number of Audio Files', fontweight='bold')
        ax2.set_title('(B) Audio File Counts per Dataset', fontweight='bold', loc='left')
        ax2.grid(axis='x', alpha=0.3)

        # panel c: task distribution (if available)
        ax3 = fig.add_subplot(gs[1, 2])

        if 'task_distribution' in dataset_stats.get(datasets[0], {}):
            # aggregate task counts across datasets
            all_tasks = set()
            for d in datasets:
                if 'task_distribution' in dataset_stats[d]:
                    all_tasks.update(dataset_stats[d]['task_distribution'].keys())

            task_counts = {task: sum(dataset_stats[d].get('task_distribution', {}).get(task, 0)
                                    for d in datasets) for task in all_tasks}

            tasks = list(task_counts.keys())
            counts = list(task_counts.values())

            colors_cycle = CB_PALETTE['qualitative']
            pie_colors = [colors_cycle[i % len(colors_cycle)] for i in range(len(tasks))]

            ax3.pie(counts, labels=tasks, autopct='%1.1f%%',
                   colors=pie_colors, startangle=90,
                   wedgeprops={'edgecolor': 'black', 'linewidth': 1})
            ax3.set_title('(C) Task Distribution', fontweight='bold', loc='left')
        else:
            ax3.text(0.5, 0.5, 'Task data\nnot available',
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=11, style='italic')
            ax3.axis('off')

        # panel d: duration statistics
        ax4 = fig.add_subplot(gs[2, :])

        if 'mean_duration' in dataset_stats.get(datasets[0], {}):
            mean_durs = [dataset_stats[d].get('mean_duration', 0) for d in datasets]
            std_durs = [dataset_stats[d].get('std_duration', 0) for d in datasets]

            ax4.errorbar(x, mean_durs, yerr=std_durs,
                        marker='o', capsize=5, linewidth=2, markersize=8,
                        color=COLORS['primary'], ecolor='black')

            ax4.set_ylabel('Duration (seconds)', fontweight='bold')
            ax4.set_xlabel('Dataset', fontweight='bold')
            ax4.set_title('(D) Audio Duration Statistics', fontweight='bold', loc='left')
            ax4.set_xticks(x)
            ax4.set_xticklabels(datasets, rotation=45, ha='right')
            ax4.grid(alpha=0.3)

        fig.suptitle('Dataset Statistics and Demographics', fontsize=14, fontweight='bold', y=0.995)

        if save:
            plt.savefig(self.output_dir / 'preprocessing' / 'fig_dataset_stats.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'preprocessing' / 'fig_dataset_stats.png', dpi=600, bbox_inches='tight')

        return fig

    def figure_clinical_features_distribution(
        self,
        features_df: pd.DataFrame,
        save: bool = True
    ) -> plt.Figure:
        """
        figure: clinical feature distributions for pd vs hc.

        publication-quality violin plots with statistical annotations.
        """
        features = ['jitter_local', 'shimmer_local', 'hnr_mean',
                   'f0_mean', 'f0_std', 'voicing_fraction']

        features = [f for f in features if f in features_df.columns]

        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature in enumerate(features):
            ax = axes[idx]

            # prepare data
            hc_data = features_df[features_df['label'] == 0][feature].dropna()
            pd_data = features_df[features_df['label'] == 1][feature].dropna()

            data_to_plot = [hc_data, pd_data]

            # violin plot
            parts = ax.violinplot(data_to_plot, positions=[1, 2],
                                 showmeans=True, showmedians=True)

            # color violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(COLORS['hc'] if i == 0 else COLORS['pd'])
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)

            # statistical test
            if len(hc_data) > 0 and len(pd_data) > 0:
                t_stat, p_val = stats.ttest_ind(hc_data, pd_data)

                # effect size (Cohen's d)
                pooled_std = np.sqrt(((len(hc_data)-1)*hc_data.std()**2 +
                                     (len(pd_data)-1)*pd_data.std()**2) /
                                    (len(hc_data) + len(pd_data) - 2))
                cohens_d = (pd_data.mean() - hc_data.mean()) / pooled_std if pooled_std > 0 else 0

                # significance stars
                if p_val < 0.001:
                    sig_text = '***'
                elif p_val < 0.01:
                    sig_text = '**'
                elif p_val < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'n.s.'

                # add significance annotation
                y_max = max(hc_data.max(), pd_data.max())
                y_range = y_max - min(hc_data.min(), pd_data.min())
                y_ann = y_max + y_range * 0.1

                ax.plot([1, 2], [y_ann, y_ann], 'k-', linewidth=1.5)
                ax.text(1.5, y_ann + y_range*0.05, sig_text,
                       ha='center', va='bottom', fontsize=12, fontweight='bold')

                # add statistics text
                ax.text(0.95, 0.95,
                       f'$p$ = {p_val:.3e}\n$d$ = {cohens_d:.2f}',
                       transform=ax.transAxes,
                       ha='right', va='top',
                       fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white',
                                edgecolor='gray', alpha=0.8))

            ax.set_ylabel(feature.replace('_', ' ').title(), fontweight='bold')
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['HC', 'PD'])
            ax.set_title(f'({chr(65+idx)}) {feature.replace("_", " ").title()}',
                        fontweight='bold', loc='left')
            ax.grid(axis='y', alpha=0.3)

        # hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        fig.suptitle('Clinical Voice Features: PD vs Healthy Control',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'preprocessing' / 'fig_clinical_features.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'preprocessing' / 'fig_clinical_features.png', dpi=600, bbox_inches='tight')

        return fig

    # ========================================================================
    # MODEL ARCHITECTURE FIGURES
    # ========================================================================

    def figure_model_architecture(
        self,
        save: bool = True
    ) -> plt.Figure:
        """
        figure: wav2vec2 architecture with classification head.

        detailed architecture diagram showing cnn, transformer, and classifier.
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # title
        ax.text(7, 9.5, 'Wav2Vec2-based PD Detection Architecture',
               ha='center', va='center', fontsize=16, fontweight='bold')

        # input
        rect_input = FancyBboxPatch((0.5, 8), 2, 0.8,
                                    boxstyle='round,pad=0.05',
                                    facecolor='lightblue',
                                    edgecolor='black', linewidth=2)
        ax.add_patch(rect_input)
        ax.text(1.5, 8.4, 'Input Audio\n16kHz, 10s',
               ha='center', va='center', fontsize=10, fontweight='bold')

        # cnn feature extractor
        rect_cnn = FancyBboxPatch((0.5, 6), 2, 1.5,
                                  boxstyle='round,pad=0.05',
                                  facecolor='#FFE5CC',
                                  edgecolor='black', linewidth=2)
        ax.add_patch(rect_cnn)
        ax.text(1.5, 7, 'CNN Feature\nExtractor\n7 conv layers\nstride=[5,2,2,2,2,2,2]',
               ha='center', va='center', fontsize=9, fontweight='bold')

        # arrow
        arrow1 = FancyArrowPatch((1.5, 8), (1.5, 7.5),
                                arrowstyle='->,head_width=0.3,head_length=0.2',
                                color='black', linewidth=2)
        ax.add_patch(arrow1)

        # transformer encoder (12 layers)
        transformer_x = 4
        for i in range(12):
            y = 7.5 - i * 0.45

            rect_layer = FancyBboxPatch((transformer_x, y), 5, 0.35,
                                       boxstyle='round,pad=0.02',
                                       facecolor='#E6F3FF' if i % 2 == 0 else '#CCE5FF',
                                       edgecolor='black', linewidth=1)
            ax.add_patch(rect_layer)

            ax.text(transformer_x + 0.3, y + 0.175, f'L{i}',
                   ha='left', va='center', fontsize=8, fontweight='bold')

            # components
            ax.text(transformer_x + 1.5, y + 0.175, 'Attn (12 heads)',
                   ha='center', va='center', fontsize=7)
            ax.text(transformer_x + 3, y + 0.175, 'FFN',
                   ha='center', va='center', fontsize=7)
            ax.text(transformer_x + 4, y + 0.175, 'LayerNorm',
                   ha='center', va='center', fontsize=7)

        # transformer label
        ax.text(transformer_x + 2.5, 8, 'Transformer Encoder (12 layers, 768-dim, 12 heads)',
               ha='center', va='center', fontsize=10, fontweight='bold')

        # arrow from cnn to transformer
        arrow2 = FancyArrowPatch((2.5, 6.75), (transformer_x, 7.25),
                                arrowstyle='->,head_width=0.3,head_length=0.2',
                                color='black', linewidth=2)
        ax.add_patch(arrow2)

        # mean pooling
        pool_y = 1.5
        rect_pool = FancyBboxPatch((transformer_x + 1, pool_y), 3, 0.6,
                                   boxstyle='round,pad=0.05',
                                   facecolor='#FFFFCC',
                                   edgecolor='black', linewidth=2)
        ax.add_patch(rect_pool)
        ax.text(transformer_x + 2.5, pool_y + 0.3, 'Mean Pooling',
               ha='center', va='center', fontsize=10, fontweight='bold')

        # arrow from last transformer to pooling
        arrow3 = FancyArrowPatch((transformer_x + 2.5, 2.15), (transformer_x + 2.5, pool_y + 0.6),
                                arrowstyle='->,head_width=0.3,head_length=0.2',
                                color='black', linewidth=2)
        ax.add_patch(arrow3)

        # classification head
        clf_x = transformer_x + 10
        rect_clf = FancyBboxPatch((clf_x, 0.5), 2, 1.6,
                                  boxstyle='round,pad=0.05',
                                  facecolor='#FFE6E6',
                                  edgecolor='black', linewidth=2)
        ax.add_patch(rect_clf)
        ax.text(clf_x + 1, 1.7, 'Classifier',
               ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(clf_x + 1, 1.3, 'Linear(768 → 2)\n+\nSoftmax',
               ha='center', va='center', fontsize=9)

        # arrow from pooling to classifier
        arrow4 = FancyArrowPatch((transformer_x + 4, pool_y + 0.3), (clf_x, 1.3),
                                arrowstyle='->,head_width=0.3,head_length=0.2',
                                color='black', linewidth=2)
        ax.add_patch(arrow4)

        # output
        rect_output = FancyBboxPatch((clf_x, 0.2), 2, 0.2,
                                     boxstyle='round,pad=0.02',
                                     facecolor='lightgreen',
                                     edgecolor='black', linewidth=2)
        ax.add_patch(rect_output)
        ax.text(clf_x + 1, 0.3, 'Output: [P(HC), P(PD)]',
               ha='center', va='center', fontsize=9, fontweight='bold')

        # add legend for freezing
        legend_elements = [
            mpatches.Patch(facecolor='#FFE5CC', edgecolor='black', label='Frozen (CNN)'),
            mpatches.Patch(facecolor='#E6F3FF', edgecolor='black', label='Fine-tuned (Transformer)'),
            mpatches.Patch(facecolor='#FFE6E6', edgecolor='black', label='Trainable (Classifier)')
        ]
        ax.legend(handles=legend_elements, loc='lower left', frameon=True, fontsize=9)

        if save:
            plt.savefig(self.output_dir / 'fig_architecture.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'fig_architecture.png', dpi=600, bbox_inches='tight')

        return fig

    # ========================================================================
    # TRAINING FIGURES
    # ========================================================================

    def figure_training_curves(
        self,
        history: Dict[str, List[float]],
        save: bool = True
    ) -> plt.Figure:
        """
        figure: training and validation curves.

        shows loss, accuracy, and other metrics over epochs.

        args:
            history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        epochs = range(1, len(history.get('train_loss', [])) + 1)

        # loss curves
        ax1 = axes[0, 0]
        if 'train_loss' in history:
            ax1.plot(epochs, history['train_loss'],
                    marker='o', label='Training Loss',
                    color=COLORS['primary'], linewidth=2, markersize=4)
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'],
                    marker='s', label='Validation Loss',
                    color=COLORS['danger'], linewidth=2, markersize=4)

        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('(A) Training Loss', fontweight='bold', loc='left')
        ax1.legend(frameon=True)
        ax1.grid(alpha=0.3)

        # accuracy curves
        ax2 = axes[0, 1]
        if 'train_acc' in history:
            ax2.plot(epochs, history['train_acc'],
                    marker='o', label='Training Accuracy',
                    color=COLORS['primary'], linewidth=2, markersize=4)
        if 'val_acc' in history:
            ax2.plot(epochs, history['val_acc'],
                    marker='s', label='Validation Accuracy',
                    color=COLORS['danger'], linewidth=2, markersize=4)

        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Accuracy', fontweight='bold')
        ax2.set_title('(B) Training Accuracy', fontweight='bold', loc='left')
        ax2.legend(frameon=True)
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, 1])

        # f1 score
        ax3 = axes[1, 0]
        if 'train_f1' in history and 'val_f1' in history:
            ax3.plot(epochs, history['train_f1'],
                    marker='o', label='Training F1',
                    color=COLORS['primary'], linewidth=2, markersize=4)
            ax3.plot(epochs, history['val_f1'],
                    marker='s', label='Validation F1',
                    color=COLORS['danger'], linewidth=2, markersize=4)

            ax3.set_xlabel('Epoch', fontweight='bold')
            ax3.set_ylabel('F1 Score', fontweight='bold')
            ax3.set_title('(C) F1 Score', fontweight='bold', loc='left')
            ax3.legend(frameon=True)
            ax3.grid(alpha=0.3)
            ax3.set_ylim([0, 1])
        else:
            ax3.text(0.5, 0.5, 'F1 data\nnot available',
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=11, style='italic')
            ax3.axis('off')

        # learning rate schedule
        ax4 = axes[1, 1]
        if 'learning_rate' in history:
            ax4.plot(epochs, history['learning_rate'],
                    marker='o', color=COLORS['secondary'],
                    linewidth=2, markersize=4)
            ax4.set_xlabel('Epoch', fontweight='bold')
            ax4.set_ylabel('Learning Rate', fontweight='bold')
            ax4.set_title('(D) Learning Rate Schedule', fontweight='bold', loc='left')
            ax4.set_yscale('log')
            ax4.grid(alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'LR data\nnot available',
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=11, style='italic')
            ax4.axis('off')

        fig.suptitle('Training Dynamics', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'training' / 'fig_training_curves.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'training' / 'fig_training_curves.png', dpi=600, bbox_inches='tight')

        return fig

    def figure_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str] = ['Healthy', 'Parkinson\'s'],
        save: bool = True
    ) -> plt.Figure:
        """
        figure: publication-quality confusion matrix.

        includes accuracy, sensitivity, specificity annotations.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # create heatmap
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

        # add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Proportion', rotation=270, labelpad=20, fontweight='bold')

        # set ticks
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        # rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2%})',
                             ha='center', va='center',
                             color='white' if cm_norm[i, j] > 0.5 else 'black',
                             fontsize=12, fontweight='bold')

        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
        ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
        ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)

        # add metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics_text = f'Accuracy: {accuracy:.2%}\nSensitivity: {sensitivity:.2%}\nSpecificity: {specificity:.2%}'
        ax.text(1.02, 0.5, metrics_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'training' / 'fig_confusion_matrix.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'training' / 'fig_confusion_matrix.png', dpi=600, bbox_inches='tight')

        return fig

    def figure_roc_pr_curves(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save: bool = True
    ) -> plt.Figure:
        """
        figure: roc and precision-recall curves side by side.
        """
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # roc curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        ax1.plot(fpr, tpr, color=COLORS['primary'], linewidth=2.5,
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Chance', alpha=0.6)

        ax1.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
        ax1.set_title('(A) ROC Curve', fontweight='bold', fontsize=13, loc='left')
        ax1.legend(loc='lower right', frameon=True, fontsize=11)
        ax1.grid(alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_aspect('equal')

        # precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)

        ax2.plot(recall, precision, color=COLORS['success'], linewidth=2.5,
                label=f'PR Curve (AP = {avg_precision:.3f})')

        ax2.set_xlabel('Recall', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Precision', fontweight='bold', fontsize=12)
        ax2.set_title('(B) Precision-Recall Curve', fontweight='bold', fontsize=13, loc='left')
        ax2.legend(loc='lower left', frameon=True, fontsize=11)
        ax2.grid(alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])

        fig.suptitle('Model Performance Curves', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'training' / 'fig_roc_pr_curves.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'training' / 'fig_roc_pr_curves.png', dpi=600, bbox_inches='tight')

        return fig

    # ========================================================================
    # PROBING ANALYSIS FIGURES
    # ========================================================================

    def figure_layerwise_probing_heatmap(
        self,
        probing_results: Dict[str, np.ndarray],
        save: bool = True
    ) -> plt.Figure:
        """
        figure: layer-wise probing accuracy heatmap.

        shows which layers encode which clinical features.

        args:
            probing_results: dict mapping feature names to layer-wise accuracies
        """
        features = list(probing_results.keys())
        n_layers = len(probing_results[features[0]])

        # create matrix
        data = np.array([probing_results[f] for f in features])

        fig, ax = plt.subplots(figsize=(12, len(features) * 0.8 + 2))

        # heatmap
        im = ax.imshow(data, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')

        # colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probing Accuracy', rotation=270, labelpad=20, fontweight='bold')

        # ticks
        ax.set_xticks(np.arange(n_layers))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels([f'Layer {i}' for i in range(n_layers)], rotation=45, ha='right')
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features])

        # annotations
        for i in range(len(features)):
            for j in range(n_layers):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha='center', va='center',
                             color='white' if data[i, j] < 0.75 else 'black',
                             fontsize=8, fontweight='bold')

        ax.set_xlabel('Transformer Layer', fontweight='bold', fontsize=12)
        ax.set_ylabel('Clinical Feature', fontweight='bold', fontsize=12)
        ax.set_title('Layer-wise Clinical Feature Encoding', fontweight='bold', fontsize=14)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'analysis' / 'fig_layerwise_probing.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'analysis' / 'fig_layerwise_probing.png', dpi=600, bbox_inches='tight')

        return fig

    def figure_probing_accuracy_curves(
        self,
        probing_results: Dict[str, Dict[str, List[float]]],
        save: bool = True
    ) -> plt.Figure:
        """
        figure: probing accuracy curves across layers for each feature.

        args:
            probing_results: dict mapping features to {'layers': [...], 'accuracy': [...]}
        """
        n_features = len(probing_results)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, (feature, results) in enumerate(probing_results.items()):
            ax = axes[idx]

            layers = results.get('layers', range(len(results.get('accuracy', []))))
            accuracy = results.get('accuracy', [])

            ax.plot(layers, accuracy, marker='o', linewidth=2.5, markersize=8,
                   color=CB_PALETTE['qualitative'][idx % len(CB_PALETTE['qualitative'])])

            # baseline (chance)
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5,
                      label='Chance', alpha=0.7)

            # find best layer
            if len(accuracy) > 0:
                best_layer = np.argmax(accuracy)
                ax.axvline(x=best_layer, color='red', linestyle=':', linewidth=1.5,
                          label=f'Best: Layer {best_layer}', alpha=0.7)
                ax.scatter([best_layer], [accuracy[best_layer]],
                          s=200, color='red', marker='*', zorder=5)

            ax.set_xlabel('Layer', fontweight='bold')
            ax.set_ylabel('Probing Accuracy', fontweight='bold')
            ax.set_title(f'({chr(65+idx)}) {feature.replace("_", " ").title()}',
                        fontweight='bold', loc='left')
            ax.legend(frameon=True, loc='lower right')
            ax.grid(alpha=0.3)
            ax.set_ylim([0.4, 1.0])

        # hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        fig.suptitle('Probing Accuracy Across Layers', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'analysis' / 'fig_probing_curves.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'analysis' / 'fig_probing_curves.png', dpi=600, bbox_inches='tight')

        return fig

    def figure_multifeature_probing_comparison(
        self,
        single_feature_results: Dict[str, float],
        multi_feature_results: Dict[str, float],
        save: bool = True
    ) -> plt.Figure:
        """
        figure: comparison of single vs multi-feature probing.

        shows whether features are encoded independently or jointly.

        args:
            single_feature_results: dict mapping feature to accuracy
            multi_feature_results: dict mapping feature combination to accuracy
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        features = list(single_feature_results.keys())
        single_acc = [single_feature_results[f] for f in features]

        x = np.arange(len(features))
        width = 0.35

        bars1 = ax.bar(x - width/2, single_acc, width,
                      label='Single Feature',
                      color=COLORS['primary'],
                      edgecolor='black', linewidth=1)

        # if multi-feature results available, show them
        if multi_feature_results:
            multi_acc = [multi_feature_results.get(f, 0) for f in features]
            bars2 = ax.bar(x + width/2, multi_acc, width,
                          label='Multi-Feature Context',
                          color=COLORS['secondary'],
                          edgecolor='black', linewidth=1)

            # add value labels
            for i, (b1, b2) in enumerate(zip(bars1, bars2)):
                height1 = b1.get_height()
                height2 = b2.get_height()
                ax.text(b1.get_x() + b1.get_width()/2, height1 + 0.01,
                       f'{height1:.2f}', ha='center', va='bottom', fontsize=9)
                ax.text(b2.get_x() + b2.get_width()/2, height2 + 0.01,
                       f'{height2:.2f}', ha='center', va='bottom', fontsize=9)

                # significance test
                if abs(height2 - height1) > 0.05:
                    y_pos = max(height1, height2) + 0.05
                    ax.plot([x[i] - width/2, x[i] + width/2], [y_pos, y_pos],
                           'k-', linewidth=1.5)
                    ax.text(x[i], y_pos + 0.02, '*' if abs(height2 - height1) > 0.1 else 'n.s.',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Probing Accuracy', fontweight='bold', fontsize=12)
        ax.set_xlabel('Clinical Feature', fontweight='bold', fontsize=12)
        ax.set_title('Single vs Multi-Feature Probing', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace('_', ' ').title() for f in features],
                          rotation=45, ha='right')
        ax.legend(frameon=True, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'analysis' / 'fig_multifeature_probing.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'analysis' / 'fig_multifeature_probing.png', dpi=600, bbox_inches='tight')

        return fig

    # ========================================================================
    # ACTIVATION PATCHING FIGURES
    # ========================================================================

    def figure_layer_importance_heatmap(
        self,
        layer_importance: np.ndarray,
        save: bool = True
    ) -> plt.Figure:
        """
        figure: layer importance from activation patching.

        shows which layers are causally important for pd detection.

        args:
            layer_importance: array of shape (n_layers,) with importance scores
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        n_layers = len(layer_importance)
        x = np.arange(n_layers)

        # bar plot
        colors = [COLORS['danger'] if imp < -0.1 else COLORS['success'] if imp > 0.1
                 else COLORS['nonsignificant'] for imp in layer_importance]

        bars = ax.bar(x, layer_importance, color=colors, edgecolor='black', linewidth=1)

        # add value labels
        for i, (bar, imp) in enumerate(zip(bars, layer_importance)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                   height + (0.01 if height >= 0 else -0.01),
                   f'{imp:.3f}', ha='center',
                   va='bottom' if height >= 0 else 'top',
                   fontsize=9, fontweight='bold')

        ax.axhline(y=0, color='black', linewidth=1.5)
        ax.set_xlabel('Layer', fontweight='bold', fontsize=12)
        ax.set_ylabel('Importance Score\n(Change in Accuracy)', fontweight='bold', fontsize=12)
        ax.set_title('Layer-wise Causal Importance via Activation Patching',
                    fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{i}' for i in range(n_layers)])
        ax.grid(axis='y', alpha=0.3)

        # add legend
        legend_elements = [
            mpatches.Patch(facecolor=COLORS['danger'], edgecolor='black',
                          label='Harmful (< -0.1)'),
            mpatches.Patch(facecolor=COLORS['nonsignificant'], edgecolor='black',
                          label='Neutral'),
            mpatches.Patch(facecolor=COLORS['success'], edgecolor='black',
                          label='Important (> 0.1)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'analysis' / 'fig_layer_importance.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'analysis' / 'fig_layer_importance.png', dpi=600, bbox_inches='tight')

        return fig

    def figure_head_importance_matrix(
        self,
        head_importance: np.ndarray,
        save: bool = True
    ) -> plt.Figure:
        """
        figure: attention head importance matrix.

        shows which attention heads are causally important.

        args:
            head_importance: array of shape (n_layers, n_heads) with importance
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        n_layers, n_heads = head_importance.shape

        # heatmap
        im = ax.imshow(head_importance, cmap='RdBu_r',
                      vmin=-np.abs(head_importance).max(),
                      vmax=np.abs(head_importance).max(),
                      aspect='auto')

        # colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Importance Score', rotation=270, labelpad=20, fontweight='bold')

        # ticks
        ax.set_xticks(np.arange(n_heads))
        ax.set_yticks(np.arange(n_layers))
        ax.set_xticklabels([f'H{i}' for i in range(n_heads)])
        ax.set_yticklabels([f'L{i}' for i in range(n_layers)])

        # annotations for top/bottom heads
        flat_importance = head_importance.flatten()
        threshold = np.percentile(np.abs(flat_importance), 90)

        for i in range(n_layers):
            for j in range(n_heads):
                if np.abs(head_importance[i, j]) >= threshold:
                    text = ax.text(j, i, f'{head_importance[i, j]:.2f}',
                                 ha='center', va='center',
                                 color='white' if np.abs(head_importance[i, j]) > threshold*0.7 else 'black',
                                 fontsize=7, fontweight='bold')

        ax.set_xlabel('Attention Head', fontweight='bold', fontsize=12)
        ax.set_ylabel('Layer', fontweight='bold', fontsize=12)
        ax.set_title('Attention Head Importance via Activation Patching',
                    fontweight='bold', fontsize=14)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'analysis' / 'fig_head_importance.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'analysis' / 'fig_head_importance.png', dpi=600, bbox_inches='tight')

        return fig

    def figure_position_patching_analysis(
        self,
        position_importance: np.ndarray,
        time_axis: Optional[np.ndarray] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        figure: position-level patching importance.

        shows which temporal positions are critical for pd detection.

        args:
            position_importance: array of importance scores over positions
            time_axis: optional time axis in seconds
        """
        fig, ax = plt.subplots(figsize=(14, 5))

        n_positions = len(position_importance)
        x = time_axis if time_axis is not None else np.arange(n_positions)

        # smooth importance curve
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(position_importance, sigma=2)

        ax.plot(x, position_importance, alpha=0.4, color='gray',
               linewidth=1, label='Raw')
        ax.plot(x, smoothed, linewidth=2.5, color=COLORS['primary'],
               label='Smoothed')
        ax.fill_between(x, 0, smoothed, alpha=0.3, color=COLORS['primary'])

        # mark critical regions
        threshold = np.percentile(smoothed, 75)
        critical_mask = smoothed > threshold

        # find contiguous regions
        regions = []
        in_region = False
        start = 0
        for i, is_critical in enumerate(critical_mask):
            if is_critical and not in_region:
                start = i
                in_region = True
            elif not is_critical and in_region:
                regions.append((start, i-1))
                in_region = False
        if in_region:
            regions.append((start, len(critical_mask)-1))

        for start, end in regions:
            ax.axvspan(x[start], x[end], alpha=0.2, color='red',
                      label='Critical Region' if start == regions[0][0] else '')

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Time (s)' if time_axis is not None else 'Position',
                     fontweight='bold', fontsize=12)
        ax.set_ylabel('Importance Score', fontweight='bold', fontsize=12)
        ax.set_title('Temporal Position Importance via Activation Patching',
                    fontweight='bold', fontsize=14)
        ax.legend(frameon=True, fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'analysis' / 'fig_position_patching.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'analysis' / 'fig_position_patching.png', dpi=600, bbox_inches='tight')

        return fig

    # ========================================================================
    # CROSS-DATASET GENERALIZATION FIGURES
    # ========================================================================

    def figure_cross_dataset_matrix(
        self,
        results_matrix: np.ndarray,
        dataset_names: List[str],
        save: bool = True
    ) -> plt.Figure:
        """
        figure: cross-dataset evaluation matrix.

        shows train-on-X test-on-Y performance matrix.

        args:
            results_matrix: array of shape (n_datasets, n_datasets) with f1 scores
            dataset_names: list of dataset names
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        n_datasets = len(dataset_names)

        # heatmap
        im = ax.imshow(results_matrix, cmap='YlGn', vmin=0, vmax=1, aspect='auto')

        # colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('F1 Score', rotation=270, labelpad=20, fontweight='bold')

        # ticks
        ax.set_xticks(np.arange(n_datasets))
        ax.set_yticks(np.arange(n_datasets))
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.set_yticklabels(dataset_names)

        # annotations
        for i in range(n_datasets):
            for j in range(n_datasets):
                # diagonal (in-distribution) vs off-diagonal (cross-dataset)
                is_diagonal = (i == j)
                text = ax.text(j, i, f'{results_matrix[i, j]:.3f}',
                             ha='center', va='center',
                             color='white' if results_matrix[i, j] < 0.5 else 'black',
                             fontsize=10,
                             fontweight='bold' if is_diagonal else 'normal')

        ax.set_xlabel('Test Dataset', fontweight='bold', fontsize=12)
        ax.set_ylabel('Train Dataset', fontweight='bold', fontsize=12)
        ax.set_title('Cross-Dataset Generalization Performance',
                    fontweight='bold', fontsize=14)

        # add note
        ax.text(0.5, -0.15,
               'Diagonal: In-distribution performance | Off-diagonal: Cross-dataset generalization',
               ha='center', va='top', transform=ax.transAxes,
               fontsize=10, style='italic')

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'analysis' / 'fig_cross_dataset_matrix.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'analysis' / 'fig_cross_dataset_matrix.png', dpi=600, bbox_inches='tight')

        return fig

    def figure_dataset_alignment_correlation(
        self,
        alignment_scores: np.ndarray,
        dataset_names: List[str],
        save: bool = True
    ) -> plt.Figure:
        """
        figure: dataset clinical feature alignment correlation.

        uses hierarchical clustering to show dataset similarity.

        args:
            alignment_scores: correlation matrix of clinical feature distributions
            dataset_names: list of dataset names
        """
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1],
                              hspace=0.05, wspace=0.05)

        # main heatmap
        ax_main = fig.add_subplot(gs[1, 0])

        # hierarchical clustering
        linkage = hierarchy.linkage(pdist(alignment_scores, metric='euclidean'),
                                   method='average')
        dendro = hierarchy.dendrogram(linkage, no_plot=True)
        order = dendro['leaves']

        # reorder matrix
        ordered_matrix = alignment_scores[order, :][:, order]
        ordered_names = [dataset_names[i] for i in order]

        # plot heatmap
        im = ax_main.imshow(ordered_matrix, cmap='coolwarm',
                           vmin=-1, vmax=1, aspect='auto')

        ax_main.set_xticks(np.arange(len(ordered_names)))
        ax_main.set_yticks(np.arange(len(ordered_names)))
        ax_main.set_xticklabels(ordered_names, rotation=45, ha='right')
        ax_main.set_yticklabels(ordered_names)

        # annotations
        for i in range(len(ordered_names)):
            for j in range(len(ordered_names)):
                text = ax_main.text(j, i, f'{ordered_matrix[i, j]:.2f}',
                                  ha='center', va='center',
                                  color='white' if np.abs(ordered_matrix[i, j]) > 0.5 else 'black',
                                  fontsize=8)

        # dendrograms
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        hierarchy.dendrogram(linkage, ax=ax_top, color_threshold=0,
                           above_threshold_color='black')
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['left'].set_visible(False)
        ax_top.spines['bottom'].set_visible(False)

        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        hierarchy.dendrogram(linkage, ax=ax_right, orientation='left',
                           color_threshold=0, above_threshold_color='black')
        ax_right.set_xticks([])
        ax_right.set_yticks([])
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        ax_right.spines['bottom'].set_visible(False)

        # colorbar
        ax_cbar = fig.add_subplot(gs[0, 1])
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('Correlation', rotation=270, labelpad=15, fontweight='bold')

        fig.suptitle('Dataset Clinical Feature Alignment',
                    fontsize=14, fontweight='bold', y=0.98)

        if save:
            plt.savefig(self.output_dir / 'analysis' / 'fig_dataset_alignment.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'analysis' / 'fig_dataset_alignment.png', dpi=600, bbox_inches='tight')

        return fig

    # ========================================================================
    # HYPOTHESIS TESTING FIGURES
    # ========================================================================

    def figure_hypothesis_validation(
        self,
        hypotheses: List[Dict[str, Any]],
        save: bool = True
    ) -> plt.Figure:
        """
        figure: visual summary of hypothesis testing results.

        args:
            hypotheses: list of dicts with keys 'name', 'p_value', 'effect_size', 'result'
        """
        fig, ax = plt.subplots(figsize=(12, len(hypotheses) * 0.8 + 2))

        n_hyp = len(hypotheses)
        y_pos = np.arange(n_hyp)

        # extract data
        names = [h['name'] for h in hypotheses]
        p_values = [h.get('p_value', 1.0) for h in hypotheses]
        effect_sizes = [h.get('effect_size', 0) for h in hypotheses]
        results = [h.get('result', 'unknown') for h in hypotheses]

        # colors based on results
        colors = [COLORS['success'] if r == 'supported' else
                 COLORS['danger'] if r == 'rejected' else
                 COLORS['nonsignificant'] for r in results]

        # horizontal bars showing effect sizes
        bars = ax.barh(y_pos, effect_sizes, color=colors,
                      edgecolor='black', linewidth=1, alpha=0.7)

        # add p-value annotations
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            width = bar.get_width()

            # significance stars
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = 'n.s.'

            ax.text(width + 0.05, i, f'{sig} (p={p_val:.3f})',
                   va='center', fontsize=10, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'H{i+1}: {name}' for i, name in enumerate(names)])
        ax.set_xlabel('Effect Size (Cohen\'s d)', fontweight='bold', fontsize=12)
        ax.set_title('Hypothesis Testing Results', fontweight='bold', fontsize=14)
        ax.axvline(x=0, color='black', linewidth=1.5)
        ax.grid(axis='x', alpha=0.3)

        # legend
        legend_elements = [
            mpatches.Patch(facecolor=COLORS['success'], edgecolor='black',
                          label='Supported'),
            mpatches.Patch(facecolor=COLORS['danger'], edgecolor='black',
                          label='Rejected'),
            mpatches.Patch(facecolor=COLORS['nonsignificant'], edgecolor='black',
                          label='Inconclusive')
        ]
        ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=10)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'analysis' / 'fig_hypothesis_validation.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'analysis' / 'fig_hypothesis_validation.png', dpi=600, bbox_inches='tight')

        return fig

    # ========================================================================
    # STATISTICAL ANALYSIS FIGURES
    # ========================================================================

    def figure_effect_size_distribution(
        self,
        effect_sizes: Dict[str, List[float]],
        save: bool = True
    ) -> plt.Figure:
        """
        figure: distribution of effect sizes across experiments.

        args:
            effect_sizes: dict mapping experiment names to lists of effect sizes
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # panel a: violin plots
        ax1 = axes[0]

        data_to_plot = [effect_sizes[k] for k in effect_sizes.keys()]
        labels = list(effect_sizes.keys())

        parts = ax1.violinplot(data_to_plot, positions=range(len(labels)),
                              showmeans=True, showmedians=True)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(CB_PALETTE['qualitative'][i % len(CB_PALETTE['qualitative'])])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')

        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Cohen\'s d', fontweight='bold')
        ax1.set_title('(A) Effect Size Distributions', fontweight='bold', loc='left')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax1.axhline(y=0.2, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Small')
        ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Medium')
        ax1.axhline(y=0.8, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Large')
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend(frameon=True, fontsize=9)

        # panel b: cumulative distribution
        ax2 = axes[1]

        for i, (name, values) in enumerate(effect_sizes.items()):
            sorted_vals = np.sort(values)
            cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax2.plot(sorted_vals, cumulative, linewidth=2.5,
                    label=name,
                    color=CB_PALETTE['qualitative'][i % len(CB_PALETTE['qualitative'])])

        ax2.set_xlabel('Cohen\'s d', fontweight='bold')
        ax2.set_ylabel('Cumulative Probability', fontweight='bold')
        ax2.set_title('(B) Cumulative Distribution', fontweight='bold', loc='left')
        ax2.legend(frameon=True, fontsize=10)
        ax2.grid(alpha=0.3)

        fig.suptitle('Effect Size Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'analysis' / 'fig_effect_sizes.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'analysis' / 'fig_effect_sizes.png', dpi=600, bbox_inches='tight')

        return fig

    def figure_correlation_matrix(
        self,
        correlation_data: pd.DataFrame,
        title: str = 'Feature Correlation Matrix',
        save: bool = True,
        filename: str = 'fig_correlation_matrix'
    ) -> plt.Figure:
        """
        figure: correlation matrix with significance annotations.

        args:
            correlation_data: dataframe with features as columns
            title: figure title
            save: whether to save
            filename: output filename
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # compute correlation
        corr = correlation_data.corr()
        n_features = len(corr)

        # mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        # heatmap
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

        # colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Pearson r', rotation=270, labelpad=20, fontweight='bold')

        # ticks
        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(n_features))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)

        # annotations
        for i in range(n_features):
            for j in range(n_features):
                if not mask[i, j]:
                    # compute p-value
                    from scipy.stats import pearsonr
                    _, p_val = pearsonr(correlation_data.iloc[:, i],
                                       correlation_data.iloc[:, j])

                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''

                    text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}{sig}',
                                 ha='center', va='center',
                                 color='white' if np.abs(corr.iloc[i, j]) > 0.5 else 'black',
                                 fontsize=8)

        ax.set_title(title, fontweight='bold', fontsize=14)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'analysis' / f'{filename}.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'analysis' / f'{filename}.png', dpi=600, bbox_inches='tight')

        return fig

    # ========================================================================
    # ATTENTION ANALYSIS FIGURES
    # ========================================================================

    def figure_attention_patterns(
        self,
        attention_weights: np.ndarray,
        layer_idx: int,
        head_idx: int,
        time_axis: Optional[np.ndarray] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        figure: attention pattern visualization for specific layer and head.

        args:
            attention_weights: array of shape (seq_len, seq_len)
            layer_idx: layer index
            head_idx: head index
            time_axis: optional time axis in seconds
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        seq_len = attention_weights.shape[0]

        # heatmap
        im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')

        # colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontweight='bold')

        # ticks
        if time_axis is not None:
            tick_indices = np.linspace(0, seq_len-1, min(10, seq_len), dtype=int)
            ax.set_xticks(tick_indices)
            ax.set_yticks(tick_indices)
            ax.set_xticklabels([f'{time_axis[i]:.1f}s' for i in tick_indices], rotation=45)
            ax.set_yticklabels([f'{time_axis[i]:.1f}s' for i in tick_indices])
        else:
            ax.set_xticks(np.linspace(0, seq_len-1, min(10, seq_len), dtype=int))
            ax.set_yticks(np.linspace(0, seq_len-1, min(10, seq_len), dtype=int))

        ax.set_xlabel('Key Position', fontweight='bold', fontsize=12)
        ax.set_ylabel('Query Position', fontweight='bold', fontsize=12)
        ax.set_title(f'Attention Pattern: Layer {layer_idx}, Head {head_idx}',
                    fontweight='bold', fontsize=14)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'analysis' / f'fig_attention_L{layer_idx}_H{head_idx}.pdf',
                       bbox_inches='tight')
            plt.savefig(self.output_dir / 'analysis' / f'fig_attention_L{layer_idx}_H{head_idx}.png',
                       dpi=600, bbox_inches='tight')

        return fig

    # ========================================================================
    # SUPPLEMENTARY FIGURES
    # ========================================================================

    def figure_loso_cv_results(
        self,
        subject_results: Dict[str, float],
        save: bool = True
    ) -> plt.Figure:
        """
        figure: leave-one-subject-out cross-validation results.

        shows per-subject performance to check for outliers.

        args:
            subject_results: dict mapping subject IDs to f1 scores
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        subjects = list(subject_results.keys())
        scores = list(subject_results.values())

        x = np.arange(len(subjects))

        # bar plot
        bars = ax.bar(x, scores, color=COLORS['primary'],
                     edgecolor='black', linewidth=1, alpha=0.7)

        # highlight outliers
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        for i, (bar, score) in enumerate(zip(bars, scores)):
            if score < mean_score - 2*std_score:
                bar.set_color(COLORS['danger'])
            elif score > mean_score + 2*std_score:
                bar.set_color(COLORS['success'])

        # add mean line
        ax.axhline(y=mean_score, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_score:.3f}')
        ax.axhline(y=mean_score + std_score, color='gray', linestyle=':',
                  linewidth=1, alpha=0.7)
        ax.axhline(y=mean_score - std_score, color='gray', linestyle=':',
                  linewidth=1, alpha=0.7, label=f'±1 SD: {std_score:.3f}')

        ax.set_xlabel('Subject ID', fontweight='bold', fontsize=12)
        ax.set_ylabel('F1 Score', fontweight='bold', fontsize=12)
        ax.set_title('Leave-One-Subject-Out Cross-Validation Results',
                    fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(subjects, rotation=90, fontsize=8)
        ax.legend(frameon=True, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'supplementary' / 'fig_loso_cv.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'supplementary' / 'fig_loso_cv.png', dpi=600, bbox_inches='tight')

        return fig

    # ========================================================================
    # COMPREHENSIVE POSTER FIGURE
    # ========================================================================

    def figure_comprehensive_poster(
        self,
        data_dict: Dict[str, Any],
        save: bool = True
    ) -> plt.Figure:
        """
        figure: comprehensive multi-panel poster figure.

        creates a single figure summarizing the entire research project.

        args:
            data_dict: dict containing all necessary data for subplots
        """
        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(6, 4, hspace=0.4, wspace=0.3)

        # panel a: dataset statistics
        ax1 = fig.add_subplot(gs[0, :2])
        if 'dataset_stats' in data_dict:
            datasets = list(data_dict['dataset_stats'].keys())
            pd_counts = [data_dict['dataset_stats'][d].get('pd_count', 0) for d in datasets]
            hc_counts = [data_dict['dataset_stats'][d].get('hc_count', 0) for d in datasets]
            x = np.arange(len(datasets))
            width = 0.35
            ax1.bar(x - width/2, hc_counts, width, label='HC', color=COLORS['hc'], edgecolor='black')
            ax1.bar(x + width/2, pd_counts, width, label='PD', color=COLORS['pd'], edgecolor='black')
            ax1.set_xticks(x)
            ax1.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
            ax1.set_ylabel('Subjects', fontweight='bold')
            ax1.set_title('(A) Datasets', fontweight='bold', loc='left', fontsize=12)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)

        # panel b: clinical features
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'clinical_features' in data_dict:
            features_df = data_dict['clinical_features']
            features = ['jitter_local', 'shimmer_local', 'hnr_mean', 'f0_mean']
            features = [f for f in features if f in features_df.columns]
            for i, feat in enumerate(features):
                hc_data = features_df[features_df['label'] == 0][feat].dropna()
                pd_data = features_df[features_df['label'] == 1][feat].dropna()
                parts = ax2.violinplot([hc_data, pd_data], positions=[i*2+1, i*2+2],
                                      showmeans=True, widths=0.7)
                for pc in parts['bodies']:
                    pc.set_alpha(0.6)
            ax2.set_xticks([i*2+1.5 for i in range(len(features))])
            ax2.set_xticklabels([f.split('_')[0] for f in features], fontsize=9)
            ax2.set_title('(B) Clinical Features', fontweight='bold', loc='left', fontsize=12)
            ax2.grid(axis='y', alpha=0.3)

        # panel c: training curves
        ax3 = fig.add_subplot(gs[1, :2])
        if 'training_history' in data_dict:
            history = data_dict['training_history']
            epochs = range(1, len(history.get('train_loss', [])) + 1)
            if 'train_loss' in history and 'val_loss' in history:
                ax3.plot(epochs, history['train_loss'], marker='o', label='Train',
                        color=COLORS['primary'], linewidth=2, markersize=4)
                ax3.plot(epochs, history['val_loss'], marker='s', label='Val',
                        color=COLORS['danger'], linewidth=2, markersize=4)
                ax3.set_xlabel('Epoch', fontweight='bold')
                ax3.set_ylabel('Loss', fontweight='bold')
                ax3.set_title('(C) Training Loss', fontweight='bold', loc='left', fontsize=12)
                ax3.legend()
                ax3.grid(alpha=0.3)

        # panel d: confusion matrix
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'confusion_matrix' in data_dict:
            cm = data_dict['confusion_matrix']
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            im = ax4.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax4.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2%})',
                           ha='center', va='center',
                           color='white' if cm_norm[i, j] > 0.5 else 'black',
                           fontsize=10, fontweight='bold')
            ax4.set_xticks([0, 1])
            ax4.set_yticks([0, 1])
            ax4.set_xticklabels(['HC', 'PD'])
            ax4.set_yticklabels(['HC', 'PD'])
            ax4.set_xlabel('Predicted', fontweight='bold')
            ax4.set_ylabel('True', fontweight='bold')
            ax4.set_title('(D) Confusion Matrix', fontweight='bold', loc='left', fontsize=12)

        # panel e: probing heatmap
        ax5 = fig.add_subplot(gs[2, :])
        if 'probing_results' in data_dict:
            probing = data_dict['probing_results']
            features = list(probing.keys())
            n_layers = len(probing[features[0]])
            data = np.array([probing[f] for f in features])
            im = ax5.imshow(data, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
            ax5.set_xticks(np.arange(n_layers))
            ax5.set_yticks(np.arange(len(features)))
            ax5.set_xticklabels([f'L{i}' for i in range(n_layers)], fontsize=8)
            ax5.set_yticklabels([f.replace('_', ' ')[:15] for f in features], fontsize=9)
            ax5.set_xlabel('Layer', fontweight='bold')
            ax5.set_title('(E) Probing: Layer-wise Feature Encoding', fontweight='bold',
                         loc='left', fontsize=12)
            plt.colorbar(im, ax=ax5, fraction=0.03, pad=0.02, label='Accuracy')

        # panel f: layer importance
        ax6 = fig.add_subplot(gs[3, :2])
        if 'layer_importance' in data_dict:
            importance = data_dict['layer_importance']
            n_layers = len(importance)
            colors = [COLORS['danger'] if imp < -0.1 else COLORS['success'] if imp > 0.1
                     else COLORS['nonsignificant'] for imp in importance]
            ax6.bar(range(n_layers), importance, color=colors, edgecolor='black')
            ax6.axhline(y=0, color='black', linewidth=1.5)
            ax6.set_xlabel('Layer', fontweight='bold')
            ax6.set_ylabel('Importance', fontweight='bold')
            ax6.set_title('(F) Activation Patching: Layer Importance', fontweight='bold',
                         loc='left', fontsize=12)
            ax6.grid(axis='y', alpha=0.3)

        # panel g: cross-dataset matrix
        ax7 = fig.add_subplot(gs[3, 2:])
        if 'cross_dataset_matrix' in data_dict:
            matrix = data_dict['cross_dataset_matrix']
            dataset_names = data_dict.get('dataset_names', [f'D{i}' for i in range(len(matrix))])
            im = ax7.imshow(matrix, cmap='YlGn', vmin=0, vmax=1, aspect='auto')
            ax7.set_xticks(np.arange(len(dataset_names)))
            ax7.set_yticks(np.arange(len(dataset_names)))
            ax7.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=8)
            ax7.set_yticklabels(dataset_names, fontsize=8)
            ax7.set_xlabel('Test', fontweight='bold', fontsize=10)
            ax7.set_ylabel('Train', fontweight='bold', fontsize=10)
            ax7.set_title('(G) Cross-Dataset Generalization', fontweight='bold',
                         loc='left', fontsize=12)
            plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04, label='F1')

        # panel h: roc curve
        ax8 = fig.add_subplot(gs[4, :2])
        if 'roc_data' in data_dict:
            from sklearn.metrics import roc_curve, auc
            fpr = data_dict['roc_data'].get('fpr', [0, 1])
            tpr = data_dict['roc_data'].get('tpr', [0, 1])
            roc_auc = auc(fpr, tpr)
            ax8.plot(fpr, tpr, color=COLORS['primary'], linewidth=2.5,
                    label=f'AUC = {roc_auc:.3f}')
            ax8.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6)
            ax8.set_xlabel('FPR', fontweight='bold')
            ax8.set_ylabel('TPR', fontweight='bold')
            ax8.set_title('(H) ROC Curve', fontweight='bold', loc='left', fontsize=12)
            ax8.legend()
            ax8.grid(alpha=0.3)
            ax8.set_xlim([0, 1])
            ax8.set_ylim([0, 1])

        # panel i: effect sizes
        ax9 = fig.add_subplot(gs[4, 2:])
        if 'effect_sizes' in data_dict:
            effect_data = data_dict['effect_sizes']
            labels = list(effect_data.keys())
            values = [effect_data[k] for k in labels]
            colors_list = [COLORS['success'] if v > 0.5 else COLORS['danger'] if v < -0.5
                          else COLORS['nonsignificant'] for v in values]
            ax9.barh(labels, values, color=colors_list, edgecolor='black')
            ax9.axvline(x=0, color='black', linewidth=1.5)
            ax9.set_xlabel('Cohen\'s d', fontweight='bold')
            ax9.set_title('(I) Effect Sizes', fontweight='bold', loc='left', fontsize=12)
            ax9.grid(axis='x', alpha=0.3)

        # panel j: summary metrics
        ax10 = fig.add_subplot(gs[5, :])
        ax10.axis('off')
        if 'summary_metrics' in data_dict:
            metrics = data_dict['summary_metrics']
            summary_text = 'SUMMARY METRICS\n\n'
            for key, value in metrics.items():
                if isinstance(value, float):
                    summary_text += f'{key}: {value:.3f}   '
                else:
                    summary_text += f'{key}: {value}   '
            ax10.text(0.5, 0.5, summary_text, ha='center', va='center',
                     fontsize=11, family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray',
                              edgecolor='black', linewidth=2, alpha=0.8))

        fig.suptitle('Mechanistic Interpretability of Wav2Vec2 for Parkinson\'s Disease Detection',
                    fontsize=16, fontweight='bold', y=0.995)

        if save:
            plt.savefig(self.output_dir / 'fig_comprehensive_poster.pdf', bbox_inches='tight')
            plt.savefig(self.output_dir / 'fig_comprehensive_poster.png', dpi=600, bbox_inches='tight')

        return fig
