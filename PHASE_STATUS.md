# phase completion status (italian pvs only)

**last updated:** 2026-01-09

## ✓ completed phases

### phase 00b: preprocessing
- italian pvs dataset preprocessed and ready
- clinical features extracted ([data/clinical_features/italian_pvs_features.csv](data/clinical_features/italian_pvs_features.csv))
- activations cached for fast probing experiments

### phase 01: model training
- fine-tuned wav2vec2-base on italian pvs
- model saved at [results/final_model](results/final_model)
- baseline classification performance established

### phase 02: probing experiments (correlation analysis)
- notebook: [notebooks/cpu/05_probing_experiments.ipynb](notebooks/cpu/05_probing_experiments.ipynb)
- **layer-wise pd probing:** peak accuracy 0.981 at layer 8
- **clinical feature probing:**
  - jitter_local: r²=0.765 (layer 5)
  - shimmer_local: r²=0.742 (layer 2)
  - hnr_mean: r²=0.795 (layer 3)
  - f0_std: r²=0.616 (layer 0)
- **control validation:** random labels ~0.500 (validates probe selectivity)
- **selectivity:** all layers show significant selectivity (p < 0.0001)
- **figures:** publication-quality latex figures at [results/figures/](results/figures/)

## ⏳ next priority

### phase 03: activation patching (causation analysis)
- notebook: [notebooks/gpu/05_activation_patching.ipynb](notebooks/gpu/05_activation_patching.ipynb)
- status: ready to execute on google colab
- updates: matched professional standards of cpu notebook
- will establish: which layers/heads causally affect predictions

## 📊 preliminary results for abstract

**dataset:**
- 831 samples from 61 subjects (italian pvs)
- 437 pd / 394 healthy controls

**key findings:**
1. **layer-wise encoding:** clinical features (jitter, shimmer, hnr) are linearly decodable from early-to-middle transformer layers (0-5)
2. **peak discrimination:** layer 8 achieves 98.1% pd classification accuracy with loso cv
3. **clinical alignment:** model representations strongly correlate with established pd biomarkers (r² up to 0.795)
4. **probe selectivity:** all layers show significant selectivity over random labels (p < 0.0001)

**next step:** activation patching to establish causal relationships between encoded features and predictions

## 📁 results locations

- **data/results:** [results/probing/](results/probing/)
  - probing_results.json
  - activations_cache.pkl
  - layerwise_results_cache.pkl

- **figures:** [results/figures/](results/figures/)
  - fig_p2_01_layerwise_probing.pdf/png
  - fig_p2_02_clinical_feature_heatmap.pdf/png

## 🔄 multi-dataset strategy (deferred)

**current scope:** italian pvs only (sufficient for abstract)

**future expansion** (if time permits):
- neurovoz dataset: preprocessing notebook ready at [notebooks/gpu/00b_neurovoz_preprocessing.ipynb](notebooks/gpu/00b_neurovoz_preprocessing.ipynb)
- mpower dataset: awaiting access
- strategy: rerun same analysis pipeline on additional datasets

## 🎯 research quality standards

- ✓ publication-quality latex figures
- ✓ nested cross-validation with hyperparameter tuning
- ✓ leave-one-subject-out splitting (no data leakage)
- ✓ control task validation
- ✓ statistical significance testing
- ✓ lowercase/professional commit style
- ✓ comprehensive documentation

---

**status:** phases 00b-02 complete for italian pvs. ready for phase 03 (activation patching) on gpu.
