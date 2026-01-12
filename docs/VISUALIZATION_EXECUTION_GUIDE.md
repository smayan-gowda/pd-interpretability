# Visualization Enhancement Execution Guide

## Overview

This guide provides step-by-step instructions for executing all enhanced notebooks with comprehensive publication-grade visualizations. All notebooks in the GPU pipeline have been systematically upgraded with world-class visualizations suitable for international conferences (NeurIPS, ICML, ICLR).

**Total Visualizations Added**: 25+ publication-grade figures across 6 notebooks

**Enhancement Standards**:
- LaTeX rendering with Computer Modern fonts
- 300 DPI resolution for raster graphics
- PDF/PNG/SVG multi-format export
- Bootstrap confidence intervals (95% CI, 1000 iterations)
- Colorblind-accessible palettes (Okabe-Ito, Paul Tol)
- Statistical rigor (Shapiro-Wilk tests, effect sizes, significance tests)

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, A100, or better)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space for results and figures

### Software
```bash
# Core dependencies
python >= 3.9
pytorch >= 2.0
transformers >= 4.30
cuda >= 11.8

# Visualization libraries
matplotlib >= 3.5
seaborn >= 0.12
librosa >= 0.10
umap-learn >= 0.5  # Optional but recommended
```

### Environment Setup
```bash
# Activate environment
conda activate pd-interpretability

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Create results directory
mkdir -p results
```

---

## Execution Order

### Phase 01: Setup and Data ✅ Enhanced
**Notebook**: `notebooks/gpu/01_setup_and_data.ipynb`

**Enhancements Added**:
- `fig_p1_01_dataset_statistics.{pdf,png,svg}` - Comprehensive 7-panel dataset analysis

**Panels**:
1. Dataset overview table (demographics, sample counts)
2. HC waveform (healthy control)
3. PD waveform (Parkinson's disease)
4. Class distribution bar chart
5. HC mel-spectrogram
6. PD mel-spectrogram
7. Power spectral density comparison

**Execution**:
```bash
jupyter notebook notebooks/gpu/01_setup_and_data.ipynb
```

**Runtime**: ~5 minutes
**Outputs**: 1 figure (3 formats)
**Purpose**: Dataset characterization and baseline establishment

---

### Phase 03: Wav2Vec2 Training ✅ Enhanced
**Notebook**: `notebooks/gpu/03_train_wav2vec2.ipynb`

**Enhancements Added** (7 comprehensive visualizations):

1. **EMA-Smoothed Training Curves** (`fig_p3_01_*.{pdf,png,svg}`)
   - 4-panel: Loss, Accuracy, F1, AUC
   - Raw data (faint) + smoothed curves (bold, α=0.9)
   - Best validation epoch markers

2. **LOSO Distribution Plots** (`fig_p3_02_*.{pdf,png,svg}`)
   - Violin + box plots + scatter with jitter
   - Mean, median, 95% CI annotations
   - Per-metric statistics (Accuracy, F1, AUC)

3. **ROC/PR Curves with Bootstrap CI** (`fig_p3_03_*.{pdf,png,svg}`)
   - 2-panel: ROC-AUC and PR-AP curves
   - Bootstrap 95% CI bands (1000 iterations)
   - Baseline references (diagonal, prevalence)

4. **Per-Fold Confusion Matrices** (`fig_p3_04_*.{pdf,png,svg}`)
   - Grid layout (5 columns × n_rows)
   - Normalized + raw counts
   - Subject IDs + accuracy per fold

5. **Wav2Vec2 Architecture Diagram** (`fig_p3_05_*.{pdf,png,svg}`)
   - End-to-end model architecture
   - CNN feature extractor (7 layers)
   - 12 transformer layers with hyperparameters
   - Classification head details

6. **Learning Rate Schedule** (`fig_p3_06_*.{pdf,png,svg}`)
   - 2-panel: Full schedule + warmup zoom
   - Cosine annealing with linear warmup
   - Schedule parameters annotated

7. **Statistical Significance Tables** (`fig_p3_07_*.{pdf,png,svg}`)
   - Descriptive statistics (mean, SD, SEM, 95% CI)
   - Distributional statistics (Q1, Q3, IQR, CV)
   - Normality tests (Shapiro-Wilk)
   - Per-fold performance summary

**Execution**:
```bash
jupyter notebook notebooks/gpu/03_train_wav2vec2.ipynb
```

**Runtime**: ~8-12 hours (LOSO cross-validation)
**Outputs**: 7 figures × 3 formats = 21 files
**GPU Memory**: ~14GB peak

**Critical Notes**:
- LOSO training is computationally intensive
- Results saved to `results/loso_results.json`
- Best model saved for downstream tasks
- Execute cells sequentially, do not skip

---

### Phase 04: Activation Extraction ✅ Enhanced
**Notebook**: `notebooks/gpu/04_extract_activations.ipynb`

**Enhancements Added** (2 comprehensive visualizations):

1. **Activation Embeddings** (`fig_p4_01_*.{pdf,png,svg}`)
   - 6-panel: t-SNE and UMAP for layers [0, 6, 11]
   - PD vs HC separability visualization
   - 2D projections with class-specific colors/markers

2. **Activation Statistics** (`fig_p4_02_*.{pdf,png,svg}`)
   - 5-panel comprehensive analysis:
     - Layer-wise magnitude (mean ± SD)
     - PCA variance explained (50 components)
     - Activation distribution (PD vs HC)
     - Class separability (Euclidean distance)
     - Sparsity analysis (% near-zero activations)

**Execution**:
```bash
jupyter notebook notebooks/gpu/04_extract_activations.ipynb
```

**Runtime**: ~1-2 hours
**Outputs**: 2 figures × 3 formats = 6 files
**GPU Memory**: ~8GB

---

### Phase 05: Activation Patching ✅ Enhanced
**Notebook**: `notebooks/gpu/05_activation_patching.ipynb`

**Enhancements Added** (3 comprehensive visualizations):

1. **Patching Methodology Diagram** (`fig_p5_01_*.{pdf,png,svg}`)
   - Clean run (PD input)
   - Corrupted run (HC input)
   - Patched run (causal intervention)
   - Causal effect computation explained

2. **Attention Head Patching** (`fig_p5_02_*.{pdf,png,svg}`)
   - 12×12 heatmap (all layers × all heads)
   - Top-20 critical heads bar chart
   - Layer-wise aggregated effects
   - Critical heads marked with stars (effect > 0.7)

3. **Layer-Level Patching** (`fig_p5_03_*.{pdf,png,svg}`)
   - 3-panel analysis:
     - Patching vs ablation comparison
     - Selectivity scores per layer
     - Cumulative effect distribution

**Execution**:
```bash
jupyter notebook notebooks/gpu/05_activation_patching.ipynb
```

**Runtime**: ~3-4 hours
**Outputs**: 3 figures × 3 formats = 9 files
**GPU Memory**: ~12GB

**Note**: Patching experiments run 144 interventions (12 layers × 12 heads)

---

### Phase 07: Probing Experiments ✅ Enhanced
**Notebook**: `notebooks/gpu/07_probing_experiments.ipynb`

**Enhancements Added** (4 comprehensive visualizations):

1. **Probing Architecture Diagram** (`fig_p7_01_*.{pdf,png,svg}`)
   - Linear probe methodology
   - LOSO cross-validation protocol
   - Ridge regression with α=1.0
   - Layer-wise probing illustrated

2. **Clinical Feature Encoding** (`fig_p7_02_*.{pdf,png,svg}`)
   - 4-panel comprehensive analysis:
     - Layer-wise R² curves (all features)
     - Feature × Layer heatmap
     - Best layer per feature (bar chart)
     - R² distribution (violin plots)

3. **Control Task & Selectivity** (`fig_p7_03_*.{pdf,png,svg}`)
   - 3-panel validation:
     - Main vs control task comparison
     - Selectivity score heatmap
     - Layer-wise selectivity profiles

4. **Summary Tables** (`fig_p7_04_*.{pdf,png,svg}`)
   - Best encoding layer per feature
   - Layer-wise performance aggregation
   - Overall statistics (all pairs, best per feature)

**Execution**:
```bash
jupyter notebook notebooks/gpu/07_probing_experiments.ipynb
```

**Runtime**: ~2-3 hours
**Outputs**: 4 figures × 3 formats = 12 files
**GPU Memory**: ~6GB

**Clinical Features Probed**:
- Jitter (local, RAP)
- Shimmer (local, APQ3)
- HNR (harmonics-to-noise ratio)
- F0 statistics (mean, std)

---

### Phase 06: Cross-Dataset Generalization ✅ Enhanced (Previous Session)
**Notebook**: `notebooks/gpu/06_cross_dataset_generalization.ipynb`

**Enhancements** (7 publication-grade figures):
1. Cross-dataset performance matrices (3-panel)
2. Generalization gaps with error bars
3. Domain shift quantification (Wasserstein distance)
4. Layer-wise clinical encoding (4-panel)
5. Correlation analysis (alignment × generalization)
6. Training curves comparison (3 models)
7. Comprehensive dashboard (6-panel synthesis)

**Execution**:
```bash
jupyter notebook notebooks/gpu/06_cross_dataset_generalization.ipynb
```

**Runtime**: ~10-15 hours (LODO training)
**Outputs**: 7 figures × 3 formats = 21 files
**GPU Memory**: ~16GB

**Note**: Trains 3 dataset-specific models (Italian PVS, MDVR-KCL, Arkansas)

---

### Phase 08: Interpretable Prediction ✅ Enhanced (Previous Session)
**Notebook**: `notebooks/gpu/08_interpretable_prediction.ipynb`

**Existing Enhancements** (4 publication-grade figures):
1. Integrated dashboard
2. Layer information flow
3. Feature contributions
4. Interpretability pipeline

**Execution**:
```bash
jupyter notebook notebooks/gpu/08_interpretable_prediction.ipynb
```

**Runtime**: ~30 minutes
**Outputs**: 4 figures × 3 formats = 12 files

---

## Complete Execution Workflow

### Option A: Sequential Execution (Recommended)
Execute notebooks in dependency order:

```bash
# Day 1: Setup and Training (longest)
jupyter notebook notebooks/gpu/01_setup_and_data.ipynb         # 5 min
jupyter notebook notebooks/gpu/03_train_wav2vec2.ipynb         # 8-12 hours

# Day 2: Analysis Phase 1
jupyter notebook notebooks/gpu/04_extract_activations.ipynb    # 1-2 hours
jupyter notebook notebooks/gpu/07_probing_experiments.ipynb    # 2-3 hours

# Day 3: Analysis Phase 2
jupyter notebook notebooks/gpu/05_activation_patching.ipynb    # 3-4 hours
jupyter notebook notebooks/gpu/08_interpretable_prediction.ipynb  # 30 min

# Day 4-5: Cross-dataset (optional, if not done)
jupyter notebook notebooks/gpu/06_cross_dataset_generalization.ipynb  # 10-15 hours
```

### Option B: Parallel Execution (Advanced)
For users with multiple GPUs or compute nodes:

```bash
# GPU 0: Training
CUDA_VISIBLE_DEVICES=0 jupyter notebook notebooks/gpu/03_train_wav2vec2.ipynb

# GPU 1: Cross-dataset (if resources available)
CUDA_VISIBLE_DEVICES=1 jupyter notebook notebooks/gpu/06_cross_dataset_generalization.ipynb
```

---

## Verification and Quality Control

### After Each Notebook

1. **Check Figure Generation**:
```bash
ls -lh results/fig_p*_*.pdf  # Verify all PDFs generated
```

2. **Verify File Sizes**:
- PDF files should be 50-500 KB
- PNG files should be 500 KB - 3 MB
- SVG files should be 100-800 KB

3. **Visual Inspection**:
- Open each PDF in a viewer
- Verify LaTeX rendering (fonts, symbols)
- Check colorbar labels
- Ensure no text overlap
- Confirm legend readability

4. **Results Integrity**:
```bash
# Check saved results
ls -lh results/*.json
ls -lh results/*.pt
```

### Common Issues and Fixes

#### Issue: Out of Memory (OOM)
**Solution**:
```python
# Reduce batch size in notebook
batch_size = 2  # Instead of 4
grad_accum_steps = 8  # Instead of 4
```

#### Issue: UMAP not installed
**Solution**:
```bash
pip install umap-learn
# Or skip UMAP visualizations (t-SNE will still work)
```

#### Issue: LaTeX rendering fails
**Solution**:
```python
# Disable LaTeX if system doesn't have it
plt.rcParams['text.usetex'] = False
```

#### Issue: Librosa warnings
**Solution**:
```bash
# Update librosa
pip install --upgrade librosa
```

---

## Expected Outputs Summary

### By Phase

| Phase | Figures | Total Files | Disk Space |
|-------|---------|-------------|------------|
| 01    | 1       | 3           | ~2 MB      |
| 03    | 7       | 21          | ~15 MB     |
| 04    | 2       | 6           | ~8 MB      |
| 05    | 3       | 9           | ~10 MB     |
| 06    | 7       | 21          | ~18 MB     |
| 07    | 4       | 12          | ~12 MB     |
| 08    | 4       | 12          | ~10 MB     |
| **Total** | **28** | **84** | **~75 MB** |

### Results Directory Structure
```
results/
├── fig_p1_01_dataset_statistics.{pdf,png,svg}
├── fig_p3_01_training_curves.{pdf,png,svg}
├── fig_p3_02_loso_distributions.{pdf,png,svg}
├── fig_p3_03_roc_pr_curves.{pdf,png,svg}
├── fig_p3_04_confusion_matrices.{pdf,png,svg}
├── fig_p3_05_wav2vec2_architecture.{pdf,png,svg}
├── fig_p3_06_learning_rate_schedule.{pdf,png,svg}
├── fig_p3_07_statistical_tables.{pdf,png,svg}
├── fig_p4_01_activation_embeddings.{pdf,png,svg}
├── fig_p4_02_activation_statistics.{pdf,png,svg}
├── fig_p5_01_patching_methodology.{pdf,png,svg}
├── fig_p5_02_attention_head_patching.{pdf,png,svg}
├── fig_p5_03_layer_patching_ablation.{pdf,png,svg}
├── fig_p6_01_cross_dataset_matrices.{pdf,png,svg}
├── fig_p6_02_generalization_gaps.{pdf,png,svg}
├── fig_p6_03_domain_shift.{pdf,png,svg}
├── fig_p6_04_layerwise_encoding.{pdf,png,svg}
├── fig_p6_05_correlation_analysis.{pdf,png,svg}
├── fig_p6_06_training_curves.{pdf,png,svg}
├── fig_p6_07_comprehensive_dashboard.{pdf,png,svg}
├── fig_p7_01_probing_architecture.{pdf,png,svg}
├── fig_p7_02_clinical_encoding_comprehensive.{pdf,png,svg}
├── fig_p7_03_control_task_selectivity.{pdf,png,svg}
├── fig_p7_04_summary_tables.{pdf,png,svg}
├── fig_p8_01_integrated_dashboard.{pdf,png,svg}
├── fig_p8_02_layer_information_flow.{pdf,png,svg}
├── fig_p8_03_feature_contributions.{pdf,png,svg}
├── fig_p8_04_interpretability_pipeline.{pdf,png,svg}
├── loso_results.json
├── cross_dataset_results.json
├── probing_results.json
└── final_model.pt
```

---

## Publication and Presentation Use

### For Papers (LaTeX)
```latex
\usepackage{graphicx}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{results/fig_p3_01_training_curves.pdf}
  \caption{Training dynamics for LOSO cross-validation showing...}
  \label{fig:training_curves}
\end{figure}
```

### For Presentations (PowerPoint/Keynote)
- Use PNG files (300 DPI) for maximum compatibility
- SVG files for vector graphics in modern presentation software
- Figures designed for 16:9 aspect ratio

### For Posters
- PDF files recommended (infinite scaling)
- All figures already sized for poster panels
- Fonts readable at A0 poster size

---

## Estimated Total Execution Time

| Configuration | Total Time |
|---------------|------------|
| Single GPU (RTX 3090) | ~35-45 hours |
| Dual GPU (parallel) | ~20-25 hours |
| Cloud (A100 × 2) | ~15-20 hours |

**Breakdown**:
- Training (Phase 03): 40% of time
- Cross-dataset (Phase 06): 35% of time
- Analysis phases: 25% of time

---

## Git Integration

All enhancements have been committed with descriptive messages:

```bash
# View recent commits
git log --oneline --all -20

# Pull latest enhancements
git pull origin main

# Push results (optional, check .gitignore first)
git add results/
git commit -m "add generated visualization outputs"
git push origin main
```

**Note**: Large result files (>100MB) should use Git LFS or be excluded via `.gitignore`.

---

## Next Steps After Execution

1. **Verify All Outputs**: Check that all 84 files were generated
2. **Quality Review**: Visually inspect each figure
3. **Results Analysis**: Review JSON files for numerical results
4. **Paper/Abstract Drafting**: Use figures and results for manuscript
5. **Supplementary Materials**: Compile additional visualizations
6. **Archive**: Create timestamped backup of results/

---

## Support and Troubleshooting

For issues:
1. Check GPU memory: `nvidia-smi`
2. Review notebook error messages
3. Verify dependencies: `pip list | grep -E "torch|matplotlib|librosa"`
4. Check disk space: `df -h`

---

## Citation

If using these visualizations in publications:

```bibtex
@misc{pd-interpretability-2025,
  title={Interpretable Deep Learning for Parkinson's Disease Detection via Speech},
  author={[Your Name]},
  year={2025},
  note={Enhanced visualizations generated with Claude Sonnet 4.5}
}
```

---

**Document Version**: 1.0
**Last Updated**: 2026-01-12
**Enhancement Session**: Comprehensive Visualization Upgrade
**Total Enhancements**: 25+ publication-grade figures across 6 notebooks
