# project context: mechanistic interpretability of wav2vec2 for parkinson's disease detection

**project title**: "probing, patching, and predicting: mechanistic interpretability of wav2vec2 representations for clinically-grounded parkinson's disease detection"

**competition target**: isef grand prize / regeneron sts finalist

**timeline**: ~7 weeks total, abstract due in ~2 weeks (needs preliminary results)

**quality standard**: world-class, international-level research. every component must be publication-ready and scientifically rigorous.

---

## code style requirements

**critical**: follow these style guidelines for all code produced.

- all code, comments, docstrings, and variable names should be lowercase
- no emojis anywhere in code or comments
- minimal comments - code should be self-documenting through clear naming
- professional, research-oriented tone
- no excessive verbosity or filler phrases
- docstrings should be concise and technical
- avoid ai-generated patterns like "here's what this does" or "let me explain"
- naming convention: snake_case for everything (functions, variables, files)
- class names: PascalCase is acceptable but lowercase is also fine

**example of good style:**
```python
def extract_activations(model, audio_path, pooling='mean'):
    """extract transformer layer activations from audio file."""
    waveform = load_audio(audio_path, target_sr=16000)
    
    activations = {}
    for i, layer in enumerate(model.encoder.layers):
        activations[f'layer_{i}'] = get_layer_output(layer, waveform)
    
    if pooling == 'mean':
        return {k: v.mean(dim=1) for k, v in activations.items()}
    return activations
```

**example of bad style (avoid this):**
```python
def extract_activations(model, audio_path, pooling='mean'):
    """
    🎯 This function extracts activations from the model!
    
    Here's what it does:
    - First, we load the audio file
    - Then, we process it through each layer
    - Finally, we return the pooled activations
    
    Args:
        model: The model to use for extraction
        ...
    """
    # Load the audio file from the given path
    waveform = load_audio(audio_path, target_sr=16000)  # Loading audio here!
    ...
```

---

## 1. project overview

### 1.1 research question

what internal representations do speech transformers (wav2vec2) develop when fine-tuned for parkinson's disease detection, and which learned features causally determine pd/healthy classification?

### 1.2 core methodology (three pillars)

1. **probing**: train linear classifiers on intermediate layer activations to determine where clinical features (jitter, shimmer, hnr) are encoded
2. **patching**: use activation patching to establish which internal features causally affect predictions
3. **predicting**: demonstrate that models with clinically-aligned representations generalize better across datasets

### 1.3 why this matters

- current pd detection models are black boxes
- doctors don't trust ai they can't understand
- this project bridges clinical speech pathology and deep learning
- first application of mechanistic interpretability to speech-based disease detection

### 1.4 hypotheses

**hypothesis 1 (layer-wise clinical encoding):**
clinical voice biomarkers (jitter, shimmer, hnr, f0 variability) are linearly decodable from specific transformer layers, with prosodic features in middle layers (5-8) and phonatory features in early layers (2-4).

**hypothesis 2 (causal feature dependency):**
the model's pd classification depends causally on internal representations that correlate with established clinical biomarkers, not on spurious dataset artifacts.

**hypothesis 3 (generalization prediction):**
models with internal representations more aligned to clinical biomarkers will generalize better across datasets.

---

## 2. technical specifications

### 2.1 core libraries

```
torch>=2.1.0
torchaudio>=2.1.0
transformers==4.36.0
datasets==2.16.0
nnsight>=0.2.0
captum>=0.7.0
librosa==0.10.1
praat-parselmouth>=0.4.3
opensmile>=2.5.0
scikit-learn==1.3.2
matplotlib>=3.7.0
seaborn>=0.12.0
```

### 2.2 model architecture

base model: `facebook/wav2vec2-base-960h`
- 12 transformer layers
- hidden size: 768
- 12 attention heads per layer
- ~95m parameters

fine-tuning approach: add classification head on top of wav2vec2 using `Wav2Vec2ForSequenceClassification` with `num_labels=2` (pd vs healthy).

### 2.3 critical: use nnsight, not transformerlens

transformerlens is designed for gpt-2 style decoder-only text models. wav2vec2 is an encoder-only audio model with cnn feature extractors. they are incompatible.

use nnsight for activation extraction and patching. it works with any huggingface model.

alternative: manual pytorch hooks if nnsight has compatibility issues.

the approach:
- register forward hooks on each transformer layer
- cache activations during forward pass
- for patching: create hook that replaces activation with cached clean activation

---

## 3. project structure

```
pd-interpretability/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py          # dataset classes for each corpus
│   │   ├── preprocessing.py     # audio preprocessing utilities
│   │   └── augmentation.py      # data augmentation if needed
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py        # fine-tuning wrapper
│   │   └── probes.py            # linear probing classifiers
│   ├── interpretability/
│   │   ├── __init__.py
│   │   ├── extraction.py        # extract activations from all layers
│   │   ├── probing.py           # probing classifier experiments
│   │   ├── patching.py          # activation patching experiments
│   │   └── analysis.py          # statistical analysis utilities
│   ├── features/
│   │   ├── __init__.py
│   │   └── clinical.py          # parselmouth feature extraction
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py     # plotting utilities
│       ├── metrics.py           # evaluation metrics
│       └── io.py                # file i/o utilities
├── notebooks/
│   ├── local/                   # vs code notebooks
│   └── colab/                   # gpu notebooks
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/                     # original datasets
│   ├── processed/               # preprocessed audio
│   ├── clinical_features/       # extracted clinical features
│   └── activations/             # cached model activations
├── results/
│   ├── figures/
│   ├── tables/
│   └── checkpoints/
├── tests/
├── requirements.txt
└── README.md
```

---

## 4. dataset specifications

### 4.1 dataset overview

| dataset | subjects | audio files | language | status |
|---------|----------|-------------|----------|--------|
| italian pvs | 65 (28 pd, 37 hc) | 831 | italian | ready |
| mdvr-kcl | 37 (16 pd, 21 hc) | ~74 | english | ready |
| arkansas | ~81 (40 pd, 41 hc) | 81 | english | ready |
| neurovoz | 108 (54 pd, 54 hc) | 2,903 | spanish | pending |
| pc-gita | 100 (50 pd, 50 hc) | ~6,300 | spanish | pending |
| ewa-db | ~375 (175 pd, ~200 hc subset) | ~5,000 | slovak | pending |

### 4.2 dataset class requirements

each dataset class must:
1. load audio files with consistent sampling rate (16khz for wav2vec2)
2. parse metadata (subject id, diagnosis, age, sex, updrs if available)
3. support task filtering (e.g., only sustained vowels)
4. implement proper train/val/test splits (subject-wise, not sample-wise to prevent data leakage)
5. return pytorch-compatible format with keys: `input_values`, `label`, `subject_id`, `path`

### 4.3 base dataset pattern

create a base class `BasePDDataset` that:
- takes `root_dir`, `task`, `target_sr=16000`, `max_duration=10.0`
- implements `__len__` and `__getitem__`
- handles resampling, mono conversion, truncation/padding
- provides `get_subject_split()` method for proper cross-validation

then create subclasses for each dataset (ItalianPVSDataset, MDVRKCLDataset, etc.) that implement `_load_samples()` to parse that dataset's specific structure.

---

## 5. implementation phases

### phase 1: data infrastructure

**deliverables:**
- `src/data/datasets.py` - all dataset classes
- `src/data/preprocessing.py` - audio preprocessing utilities
- `src/features/clinical.py` - parselmouth feature extraction (jitter, shimmer, hnr, f0)
- unit tests for data loading

**clinical features to extract:**
- f0_mean, f0_std (pitch)
- jitter_local, jitter_rap, jitter_ppq5
- shimmer_local, shimmer_apq3, shimmer_apq5
- hnr (harmonics-to-noise ratio)

use parselmouth (python interface to praat) for extraction.

### phase 2: clinical baseline

**deliverables:**
- extract clinical features for all samples
- train svm/random forest baseline using clinical features only
- document baseline accuracy (target: 70-85%)
- use leave-one-subject-out cross-validation

this proves clinical features contain pd information and provides comparison point.

### phase 3: wav2vec2 fine-tuning (colab gpu)

**deliverables:**
- `src/models/classifier.py` - fine-tuning code
- fine-tuned model checkpoint
- training curves and evaluation metrics
- model accuracy (target: 80-90%)

approach:
- use `Wav2Vec2ForSequenceClassification`
- freeze cnn feature encoder (recommended for small datasets)
- use gradient checkpointing and fp16 for memory efficiency
- batch size 8, gradient accumulation 4 (effective 32)
- learning rate 1e-4, warmup 10%

### phase 4: activation extraction (colab gpu)

**deliverables:**
- `src/interpretability/extraction.py` - extraction code
- activations saved for all samples, all 12 layers
- saved as numpy memmap for efficient cpu access later

approach:
- register forward hooks on each transformer layer
- run inference on all samples
- pool activations (mean over time dimension) to get [n_samples, n_layers, hidden_size]
- save as memmap with metadata json

### phase 5: probing experiments (vs code cpu)

**deliverables:**
- `src/interpretability/probing.py` - probing classifier code
- layer-wise pd classification probing results
- clinical feature probing results (where is jitter encoded? shimmer? hnr?)
- publication-quality figures
- **abstract ready** with preliminary results

approach:
- for each layer, train logistic regression to predict pd/hc from that layer's activations
- use leave-one-subject-out cross-validation
- for clinical feature probing: train ridge regression to predict each clinical feature from each layer's activations
- identify which layers encode which features

key outputs:
- accuracy vs layer curve (should show middle layers are most discriminative)
- clinical feature encoding heatmap (layers x features, colored by r² score)

### phase 6: activation patching

**deliverables:**
- `src/interpretability/patching.py` - patching experiment code
- layer-level patching results (which layers are causal?)
- attention head patching results (optional, more granular)
- figures showing causal importance

approach:
- create minimal pairs: matched pd and hc samples (same task, similar characteristics)
- for each pair:
  - get clean (hc) activations
  - run corrupted (pd) input, patch in clean activations at each layer
  - measure how much prediction recovers toward clean prediction
- metric: logit difference recovered (0% = no effect, 100% = full recovery)

### phase 7: analysis & documentation

**deliverables:**
- complete statistical analysis
- all publication-quality figures
- research paper draft (for arxiv/workshop submission)
- isef poster design
- presentation preparation

---

## 6. quality standards

### 6.1 code quality
- all functions have concise docstrings
- type hints for function signatures
- unit tests for critical functions
- consistent lowercase snake_case naming
- no hardcoded paths - use config or arguments
- maximum function length ~50 lines

### 6.2 reproducibility
- all random seeds set and documented
- requirements.txt with pinned versions
- config files for experiments
- save hyperparameters with results

### 6.3 statistical rigor
- always use leave-one-subject-out cross-validation for medical data
- report mean ± std for all metrics
- correct for multiple comparisons when testing many layers
- report effect sizes

### 6.4 visualization
- dpi 300 for saved figures
- font size minimum 12pt
- color-blind friendly palettes (viridis)
- include error bars/confidence intervals

---

## 7. expected results

### baseline performance
- clinical features svm: 75-85% accuracy
- wav2vec2 fine-tuned: 80-90% accuracy

### probing results (hypothesis 1)
- middle layers (5-8) show highest pd classification probing accuracy
- early layers (2-4) show highest jitter/shimmer encoding
- later layers (9-11) show most task-specific features

### patching results (hypothesis 2)
- patching middle layers recovers 50-80% of logit difference
- early/late layers contribute less

### for abstract (~2 weeks)
minimum needed:
1. fine-tuned model accuracy
2. one probing figure showing layer-wise accuracy curve
3. one sentence about where pd information is encoded

---

## 8. common issues

### nnsight compatibility
if nnsight doesn't work with wav2vec2, use manual pytorch hooks:
- `module.register_forward_hook(fn)` to capture outputs
- create hook that stores `output[0].detach()` in a dict

### out of memory on colab
- enable gradient checkpointing: `model.gradient_checkpointing_enable()`
- reduce batch size, increase gradient accumulation
- use fp16: `TrainingArguments(fp16=True)`
- process activations in batches

### poor probing accuracy
- check model is properly fine-tuned (training loss should decrease)
- ensure subject-wise splits (no data leakage)
- try different clinical features

### dataset loading errors
- verify audio readable by torchaudio
- handle corrupted files gracefully with try/except
- check sampling rate conversion

---

## 9. key references

these papers inform the methodology. understanding them helps implement correctly:

- **wav2vec2**: baevski et al. 2020 - the base model architecture
- **probing classifiers**: belinkov 2022 - methodology for probing
- **activation patching**: meng et al. 2022 (rome paper) - patching methodology
- **pc-gita**: orozco-arroyave et al. 2014 - pd voice detection baseline
- **speech mi**: recent arxiv papers on mechanistic interpretability of whisper/audio models

---

## 10. what to build

start with phase 1. implement in this order:

1. `src/data/datasets.py` - base class + italian pvs dataset class
2. `src/data/preprocessing.py` - audio loading utilities
3. `src/features/clinical.py` - parselmouth feature extraction
4. tests to verify data loading works
5. proceed to phase 2 (clinical baseline)

do not implement phases 3-4 yet - those require colab gpu and will be done after phases 1-2 are complete and tested.
