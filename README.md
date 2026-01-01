# Probing, Patching, and Predicting: Mechanistic Interpretability of Wav2Vec2 Representations for Clinically-Grounded Parkinson’s Disease (PD) Detection

## Overview

This project applies mechanistic interpretability techniques to understand how Wav2Vec2 speech models represent vocal biomarkers of Parkinson's Disease (PD). We combine probing classifiers, activation patching, and clinical feature analysis to bridge the gap between black-box deep learning and clinically interpretable insights.

## Project Structure

```
pd-interpretability/
├── notebooks/
│   ├── colab/                    # GPU-intensive notebooks for Google Colab
│   │   ├── 01_setup_and_data.ipynb      # Environment setup and data verification
│   │   ├── 02_train_wav2vec2.ipynb      # Model fine-tuning pipeline
│   │   └── 03_extract_activations.ipynb # Activation extraction for probing
│   └── local/                    # CPU-friendly analysis notebooks
│       └── 01_phase1_demonstration.ipynb
├── src/
│   ├── data/                     # Data loading and preprocessing
│   │   ├── datasets.py           # PD audio dataset classes
│   │   └── preprocessing.py      # Audio preprocessing utilities
│   ├── models/                   # Model training and probing
│   │   ├── classifier.py         # Wav2Vec2 fine-tuning for PD classification
│   │   ├── probes.py             # Linear probing classifiers
│   │   └── cross_validation.py   # LOSO and k-fold CV utilities
│   ├── interpretability/         # Core interpretability methods
│   │   ├── extraction.py         # Activation extraction from models
│   │   └── patching.py           # Activation patching experiments
│   ├── features/                 # Clinical feature analysis
│   │   └── clinical.py           # Clinical feature extraction
│   └── utils/                    # Utilities
│       ├── visualization.py      # Plotting and visualization
│       ├── analysis.py           # Statistical analysis
│       └── experiment.py         # Experiment tracking and logging
├── data/
│   ├── raw/                      # Original audio datasets
│   ├── processed/                # Preprocessed audio files
│   └── activations/              # Extracted model activations
├── results/
│   ├── checkpoints/              # Model checkpoints
│   ├── figures/                  # Generated figures
│   └── tables/                   # Results tables
├── tests/                        # Unit tests
├── configs/                      # Configuration files
└── docs/                         # Documentation
```

## Research Phases

### Phase 1-2: Data Collection and Preprocessing ✅

- Collected and organized Italian PVS, Arkansas, and MDVR-KCL datasets
- Implemented audio preprocessing with resampling and segmentation
- Created unified dataset interface

### Phase 3: Wav2Vec2 Fine-tuning (Current)

- Fine-tune `facebook/wav2vec2-base-960h` for PD classification
- Leave-One-Subject-Out (LOSO) cross-validation
- Target: 80-90% accuracy (clinical baseline: 75-85%)

### Phase 4: Activation Extraction (Current)

- Extract intermediate layer activations from fine-tuned model
- Store as memory-mapped arrays for efficient CPU access
- Extract attention weights for analysis

### Phase 5: Probing Experiments

- Train linear probes on each layer for PD classification
- Identify which layers encode diagnostic information
- Control experiments with permuted labels

### Phase 6: Activation Patching

- Minimal pairs analysis with PD vs. healthy speakers
- Measure logit difference recovery per layer
- Quantify causal importance of representations

### Phase 7: Clinical Analysis

- Correlate activations with clinical features (jitter, shimmer, etc.)
- Statistical comparison with effect sizes
- Interpret model behavior through clinical lens

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

For Google Colab:

```bash
pip install -r requirements-colab.txt
```

## Usage

### Training (Colab)

```python
from src.models import Wav2Vec2PDClassifier, create_training_args

classifier = Wav2Vec2PDClassifier(num_labels=2)
args = create_training_args(output_dir="./results/checkpoints")
# ... see notebooks/colab/02_train_wav2vec2.ipynb
```

### Extraction (Colab)

```python
from src.interpretability import Wav2Vec2ActivationExtractor

extractor = Wav2Vec2ActivationExtractor(model)
activations = extractor.extract_to_memmap(dataset, output_path="./data/activations")
# ... see notebooks/colab/03_extract_activations.ipynb
```

### Probing (Local)

```python
from src.models import LayerwiseProber

prober = LayerwiseProber(n_layers=12)
results = prober.run_layerwise_probing(activations, labels, subject_ids)
```

## Cross-Validation

We use Leave-One-Subject-Out (LOSO) cross-validation as the gold standard for medical data:

```python
from src.models import run_loso_cv

results = run_loso_cv(
    model_init_fn=lambda: Wav2Vec2PDClassifier(num_labels=2),
    train_dataset=dataset,
    subject_ids=subject_ids,
    training_args=args
)
print(f"LOSO Accuracy: {results.mean_metrics['accuracy']:.3f} ± {results.std_metrics['accuracy']:.3f}")
```

## Experiment Tracking

```python
from src.utils import ExperimentTracker, ExperimentConfig

config = ExperimentConfig(experiment_name="pd_wav2vec2_v1")
tracker = ExperimentTracker("pd_wav2vec2_v1", output_dir="./results", config=config)

tracker.start()
# ... training loop
tracker.log_metrics({'accuracy': 0.85, 'f1': 0.82}, epoch=1, phase='val')
tracker.finish()
```

## Testing

```bash
pytest tests/ -v
```

## Key References

1. Wav2Vec2: Baevski et al. (2020) - Self-supervised speech representation learning
2. Probing Classifiers: Belinkov (2022) - Probing neural network representations
3. Activation Patching: Vig et al. (2020) - Causal mediation analysis

## License

MIT License
