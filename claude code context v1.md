# COMPREHENSIVE PROJECT CONTEXT: Mechanistic Interpretability of Wav2Vec2 for Parkinson's Disease Detection

**Project Title**: "Probing, Patching, and Predicting: Mechanistic Interpretability of Wav2Vec2 Representations for Clinically-Grounded Parkinson's Disease Detection"

**Competition Target**: ISEF Grand Prize / Regeneron STS Finalist

**Timeline**: ~7 weeks total, abstract due in ~2 weeks (needs preliminary results)

**Quality Standard**: World-class, international-level research. Every component must be publication-ready and scientifically rigorous.

---

## 1. PROJECT OVERVIEW

### 1.1 Research Question
**What internal representations do speech transformers (Wav2Vec2) develop when fine-tuned for Parkinson's disease detection, and which learned features causally determine PD/healthy classification?**

### 1.2 Core Methodology (Three Pillars)

1. **PROBING**: Train linear classifiers on intermediate layer activations to determine WHERE clinical features (jitter, shimmer, HNR) are encoded
2. **PATCHING**: Use activation patching to establish WHICH internal features CAUSALLY affect predictions
3. **PREDICTING**: Demonstrate that models with clinically-aligned representations generalize better across datasets

### 1.3 Why This Matters
- Current PD detection models are black boxes
- Doctors don't trust AI they can't understand
- This project bridges clinical speech pathology and deep learning
- First application of mechanistic interpretability to speech-based disease detection

### 1.4 Key Hypotheses

**Hypothesis 1 (Layer-wise Clinical Encoding):**
Clinical voice biomarkers (jitter, shimmer, HNR, F0 variability) are linearly decodable from specific transformer layers, with prosodic features in middle layers (5-8) and phonatory features in early layers (2-4).

**Hypothesis 2 (Causal Feature Dependency):**
The model's PD classification depends causally on internal representations that correlate with established clinical biomarkers, not on spurious dataset artifacts.

**Hypothesis 3 (Generalization Prediction):**
Models with internal representations more aligned to clinical biomarkers will generalize better across datasets.

---

## 2. TECHNICAL SPECIFICATIONS

### 2.1 Core Libraries and Versions

```python
# Core ML
torch>=2.1.0
torchaudio>=2.1.0
transformers==4.36.0
datasets==2.16.0

# CRITICAL: Interpretability (NOT TransformerLens - it doesn't support audio models)
nnsight>=0.2.0  # Use this for activation extraction and patching
captum>=0.7.0   # For attribution methods

# Audio Processing
librosa==0.10.1
praat-parselmouth>=0.4.3  # Clinical features (jitter, shimmer, HNR)
opensmile>=2.5.0          # eGeMAPS features
audiofile>=1.3.0
soundfile>=0.12.0

# ML Utilities
scikit-learn==1.3.2
scipy>=1.11.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
```

### 2.2 Model Architecture

**Base Model**: `facebook/wav2vec2-base-960h`
- 12 transformer layers
- Hidden size: 768
- 12 attention heads per layer
- ~95M parameters

**Fine-tuning Approach**: Add classification head on top of Wav2Vec2
```python
from transformers import Wav2Vec2ForSequenceClassification

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h",
    num_labels=2,  # PD vs Healthy
    problem_type="single_label_classification"
)
```

### 2.3 Critical Technical Note: Use NNsight, NOT TransformerLens

TransformerLens is designed for GPT-2 style decoder-only text models. Wav2Vec2 is an encoder-only audio model with CNN feature extractors. **They are incompatible.**

**Use NNsight instead:**
```python
from nnsight import LanguageModel  # Works with any HuggingFace model
import torch

# Wrap model with NNsight
model = LanguageModel("facebook/wav2vec2-base-960h", device_map="auto")

# Extract activations
with model.trace(audio_input):
    # Access any layer's output
    layer_6_output = model.wav2vec2.encoder.layers[6].output[0].save()

# Activation patching
with model.trace(corrupted_input):
    # Patch clean activations into corrupted run
    model.wav2vec2.encoder.layers[6].output[0][:] = clean_activations
    patched_output = model.classifier.output.save()
```

### 2.4 Alternative: Manual PyTorch Hooks

If NNsight has issues, use raw PyTorch hooks:
```python
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output[0].detach()
    return hook

# Register hooks
for i, layer in enumerate(model.wav2vec2.encoder.layers):
    layer.register_forward_hook(get_activation(f'layer_{i}'))

# Forward pass populates activations dict
_ = model(input_values)
```

---

## 3. PROJECT STRUCTURE

```
pd-interpretability/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py          # Dataset classes for each corpus
│   │   ├── preprocessing.py     # Audio preprocessing utilities
│   │   └── augmentation.py      # Data augmentation (if needed)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── wav2vec2_classifier.py  # Fine-tuning wrapper
│   │   └── probes.py               # Linear probing classifiers
│   ├── interpretability/
│   │   ├── __init__.py
│   │   ├── activation_extraction.py  # Extract activations from all layers
│   │   ├── probing.py                # Probing classifier experiments
│   │   ├── patching.py               # Activation patching experiments
│   │   └── analysis.py               # Statistical analysis utilities
│   ├── features/
│   │   ├── __init__.py
│   │   ├── clinical.py           # Parselmouth feature extraction
│   │   └── opensmile_features.py # eGeMAPS features
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py      # Plotting utilities
│       ├── metrics.py            # Evaluation metrics
│       └── io.py                 # File I/O utilities
├── notebooks/
│   ├── local/                    # VS Code notebooks
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_clinical_baseline.ipynb
│   │   ├── 03_probing_analysis.ipynb
│   │   └── 04_visualization.ipynb
│   └── colab/                    # GPU notebooks
│       ├── 00_colab_setup.ipynb
│       ├── 01_finetune_wav2vec2.ipynb
│       └── 02_extract_activations.ipynb
├── configs/
│   └── experiment_config.yaml
├── data/
│   ├── raw/                      # Original datasets
│   │   ├── italian_pvs/
│   │   ├── mdvr_kcl/
│   │   ├── neurovoz/
│   │   ├── pc_gita/
│   │   └── ewa_db/
│   ├── processed/                # Preprocessed audio
│   ├── clinical_features/        # Extracted clinical features
│   └── activations/              # Cached model activations
├── results/
│   ├── figures/
│   ├── tables/
│   └── checkpoints/
├── tests/
│   ├── test_data_loading.py
│   ├── test_feature_extraction.py
│   └── test_probing.py
├── requirements.txt
├── requirements-colab.txt
└── README.md
```

---

## 4. DATASET SPECIFICATIONS

### 4.1 Dataset Overview

| Dataset | Subjects | Audio Files | Language | Tasks | Status |
|---------|----------|-------------|----------|-------|--------|
| Italian PVS | 65 (28 PD, 37 HC) | 831 | Italian | Sustained vowels, reading, spontaneous | **READY** |
| MDVR-KCL | 37 (16 PD, 21 HC) | ~74 | English | Read speech, dialogue | **READY** |
| Arkansas | ~81 (40 PD, 41 HC) | 81 | English | Various | **READY** |
| NeuroVoz | 108 (54 PD, 54 HC) | 2,903 | Spanish | Vowels, DDK, monologue | Pending |
| PC-GITA | 100 (50 PD, 50 HC) | ~6,300 | Spanish | Vowels, DDK, sentences | Pending |
| EWA-DB | ~375 (175 PD, ~200 HC subset) | ~5,000 | Slovak | Vowels, DDK, naming | Pending |

### 4.2 Dataset Loading Requirements

Each dataset class must:
1. Load audio files with consistent sampling rate (16kHz for Wav2Vec2)
2. Parse metadata (subject ID, diagnosis, age, sex, UPDRS if available)
3. Support task filtering (e.g., only sustained vowels)
4. Implement proper train/val/test splits (subject-wise, not sample-wise)
5. Return PyTorch-compatible format

### 4.3 Italian PVS Dataset Structure
```
italian_pvs/
├── metadata.xlsx           # Subject info, diagnosis, demographics
├── PD/                     # Parkinson's patients
│   ├── PD001/
│   │   ├── vowel_a.wav
│   │   ├── vowel_e.wav
│   │   ├── reading.wav
│   │   └── spontaneous.wav
│   └── ...
└── HC/                     # Healthy controls
    ├── HC001/
    └── ...
```

### 4.4 Dataset Class Template

```python
# src/data/datasets.py

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd

class BasePDDataset(Dataset):
    """Base class for PD speech datasets."""
    
    def __init__(
        self,
        root_dir: str,
        task: str = "vowel_a",  # Filter by task
        target_sr: int = 16000,  # Wav2Vec2 expects 16kHz
        max_duration: float = 10.0,  # Max audio length in seconds
        transform=None
    ):
        self.root_dir = Path(root_dir)
        self.task = task
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.transform = transform
        
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load sample metadata. Override in subclasses."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(sample['path'])
        
        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Truncate or pad
        max_samples = int(self.max_duration * self.target_sr)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            padding = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Apply transform if any
        if self.transform:
            waveform = self.transform(waveform)
        
        return {
            'input_values': waveform.squeeze(0),
            'label': sample['label'],  # 0 = HC, 1 = PD
            'subject_id': sample['subject_id'],
            'path': str(sample['path'])
        }
    
    def get_subject_split(
        self, 
        test_size: float = 0.2, 
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[List[int], List[int], List[int]]:
        """Split by subject to prevent data leakage."""
        from sklearn.model_selection import train_test_split
        
        subjects = list(set(s['subject_id'] for s in self.samples))
        labels = [self.samples[self._get_first_sample_idx(s)]['label'] for s in subjects]
        
        # First split: train+val vs test
        train_val_subjects, test_subjects = train_test_split(
            subjects, test_size=test_size, stratify=labels, random_state=random_state
        )
        
        # Second split: train vs val
        train_val_labels = [labels[subjects.index(s)] for s in train_val_subjects]
        train_subjects, val_subjects = train_test_split(
            train_val_subjects, test_size=val_size/(1-test_size), 
            stratify=train_val_labels, random_state=random_state
        )
        
        # Convert to indices
        train_idx = [i for i, s in enumerate(self.samples) if s['subject_id'] in train_subjects]
        val_idx = [i for i, s in enumerate(self.samples) if s['subject_id'] in val_subjects]
        test_idx = [i for i, s in enumerate(self.samples) if s['subject_id'] in test_subjects]
        
        return train_idx, val_idx, test_idx
    
    def _get_first_sample_idx(self, subject_id: str) -> int:
        for i, s in enumerate(self.samples):
            if s['subject_id'] == subject_id:
                return i
        raise ValueError(f"Subject {subject_id} not found")


class ItalianPVSDataset(BasePDDataset):
    """Italian Parkinson's Voice and Speech dataset."""
    
    def _load_samples(self) -> List[Dict]:
        samples = []
        
        # Load metadata
        metadata_path = self.root_dir / "metadata.xlsx"
        if metadata_path.exists():
            metadata = pd.read_excel(metadata_path)
        else:
            metadata = None
        
        # Iterate through PD and HC folders
        for diagnosis, label in [("PD", 1), ("HC", 0)]:
            diagnosis_dir = self.root_dir / diagnosis
            if not diagnosis_dir.exists():
                continue
                
            for subject_dir in diagnosis_dir.iterdir():
                if not subject_dir.is_dir():
                    continue
                
                subject_id = subject_dir.name
                
                # Find audio files matching task
                for audio_file in subject_dir.glob("*.wav"):
                    if self.task and self.task not in audio_file.stem.lower():
                        continue
                    
                    samples.append({
                        'path': audio_file,
                        'label': label,
                        'subject_id': subject_id,
                        'task': audio_file.stem
                    })
        
        return samples
```

---

## 5. IMPLEMENTATION PHASES

### PHASE 1: Data Infrastructure (Days 1-3)

**Deliverables:**
- [ ] `src/data/datasets.py` - All dataset classes
- [ ] `src/data/preprocessing.py` - Audio preprocessing utilities
- [ ] `src/features/clinical.py` - Parselmouth feature extraction
- [ ] Unit tests for data loading

**Clinical Feature Extraction:**
```python
# src/features/clinical.py

import parselmouth
from parselmouth.praat import call
import numpy as np
from typing import Dict

def extract_clinical_features(audio_path: str) -> Dict[str, float]:
    """Extract clinical voice features using Parselmouth/Praat."""
    
    sound = parselmouth.Sound(audio_path)
    
    # Pitch analysis
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
    f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    
    # Point process for jitter/shimmer
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
    
    # Jitter variants
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    
    # Shimmer variants
    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # Harmonics-to-Noise Ratio
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    return {
        'f0_mean': f0_mean,
        'f0_std': f0_std,
        'jitter_local': jitter_local,
        'jitter_rap': jitter_rap,
        'jitter_ppq5': jitter_ppq5,
        'shimmer_local': shimmer_local,
        'shimmer_apq3': shimmer_apq3,
        'shimmer_apq5': shimmer_apq5,
        'hnr': hnr
    }
```

### PHASE 2: Clinical Baseline (Days 4-5)

**Deliverables:**
- [ ] Extract clinical features for all samples
- [ ] Train SVM/Random Forest baseline using clinical features only
- [ ] Document baseline accuracy (target: 70-85%)
- [ ] This proves clinical features contain PD information

```python
# notebooks/local/02_clinical_baseline.ipynb

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load clinical features (extracted in Phase 1)
features_df = pd.read_csv('data/clinical_features/italian_pvs_features.csv')
X = features_df.drop(['subject_id', 'label', 'path'], axis=1).values
y = features_df['label'].values
groups = features_df['subject_id'].values

# Leave-One-Subject-Out cross-validation (gold standard for medical ML)
logo = LeaveOneGroupOut()
scaler = StandardScaler()

# SVM baseline
svm = SVC(kernel='rbf', C=1.0)
svm_scores = cross_val_score(svm, scaler.fit_transform(X), y, cv=logo, groups=groups)
print(f"SVM Accuracy: {svm_scores.mean():.3f} ± {svm_scores.std():.3f}")

# Random Forest baseline
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf, X, y, cv=logo, groups=groups)
print(f"RF Accuracy: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")
```

### PHASE 3: Wav2Vec2 Fine-tuning (Days 6-8) - COLAB GPU

**Deliverables:**
- [ ] `src/models/wav2vec2_classifier.py` - Fine-tuning code
- [ ] Fine-tuned model checkpoint saved to Drive
- [ ] Training curves and evaluation metrics
- [ ] Model accuracy (target: 80-90%)

```python
# src/models/wav2vec2_classifier.py

from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer
)
from typing import Dict
import torch

def create_model(num_labels: int = 2, freeze_feature_encoder: bool = True):
    """Create Wav2Vec2 classification model."""
    
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base-960h",
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    # Freeze CNN feature encoder (recommended for small datasets)
    if freeze_feature_encoder:
        model.freeze_feature_encoder()
    
    return model

def get_training_args(output_dir: str, num_epochs: int = 10) -> TrainingArguments:
    """Get training arguments optimized for small medical datasets."""
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch size = 32
        warmup_ratio=0.1,
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=True,  # Mixed precision for T4 GPU
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        gradient_checkpointing=True,  # Save memory
    )

class DataCollator:
    """Collate function for Wav2Vec2."""
    
    def __init__(self, feature_extractor: Wav2Vec2FeatureExtractor, max_length: int = 160000):
        self.feature_extractor = feature_extractor
        self.max_length = max_length
    
    def __call__(self, batch):
        input_values = [item['input_values'].numpy() for item in batch]
        labels = torch.tensor([item['label'] for item in batch])
        
        # Pad/truncate to same length
        processed = self.feature_extractor(
            input_values,
            sampling_rate=16000,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        processed['labels'] = labels
        return processed
```

### PHASE 4: Activation Extraction (Days 9-10) - COLAB GPU

**Deliverables:**
- [ ] `src/interpretability/activation_extraction.py` - Extraction code
- [ ] Activations saved for all samples, all layers
- [ ] Saved as numpy memmap for efficient access

```python
# src/interpretability/activation_extraction.py

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm import tqdm

class Wav2Vec2ActivationExtractor:
    """Extract activations from all layers of Wav2Vec2."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda"
    ):
        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        # Load model
        if checkpoint_path:
            # Load fine-tuned model
            from transformers import Wav2Vec2ForSequenceClassification
            full_model = Wav2Vec2ForSequenceClassification.from_pretrained(checkpoint_path)
            self.model = full_model.wav2vec2
        else:
            self.model = Wav2Vec2Model.from_pretrained(model_name)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        self.num_layers = len(self.model.encoder.layers)
        self.hidden_size = self.model.config.hidden_size
        
        # Storage for activations
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        
        def get_activation(name):
            def hook(module, input, output):
                # output is tuple, first element is hidden states
                self.activations[name] = output[0].detach().cpu()
            return hook
        
        # Hook into each transformer layer
        for i, layer in enumerate(self.model.encoder.layers):
            layer.register_forward_hook(get_activation(f'layer_{i}'))
        
        # Also capture CNN feature extractor output
        self.model.feature_extractor.register_forward_hook(
            get_activation('cnn_features')
        )
    
    @torch.no_grad()
    def extract(
        self,
        audio_path: str,
        pooling: str = 'mean'  # 'mean', 'max', 'cls', or 'none'
    ) -> Dict[str, np.ndarray]:
        """Extract activations from a single audio file."""
        
        import torchaudio
        
        # Load and preprocess audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        
        # Process through feature extractor
        inputs = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Forward pass
        self.activations = {}
        _ = self.model(inputs.input_values.to(self.device))
        
        # Pool activations
        result = {}
        for name, act in self.activations.items():
            if pooling == 'mean':
                result[name] = act.mean(dim=1).squeeze(0).numpy()  # [hidden_size]
            elif pooling == 'max':
                result[name] = act.max(dim=1).values.squeeze(0).numpy()
            elif pooling == 'cls':
                result[name] = act[:, 0, :].squeeze(0).numpy()  # First token
            else:  # 'none'
                result[name] = act.squeeze(0).numpy()  # [seq_len, hidden_size]
        
        return result
    
    def extract_batch_to_memmap(
        self,
        audio_paths: List[str],
        output_path: str,
        pooling: str = 'mean'
    ) -> np.ndarray:
        """Extract activations for many files, save to memmap."""
        
        n_samples = len(audio_paths)
        
        # Determine shape from first sample
        first_result = self.extract(audio_paths[0], pooling=pooling)
        n_layers = len([k for k in first_result.keys() if k.startswith('layer_')])
        hidden_size = first_result['layer_0'].shape[-1]
        
        # Create memmap
        shape = (n_samples, n_layers, hidden_size)
        activations = np.memmap(output_path, dtype='float32', mode='w+', shape=shape)
        
        # Extract all
        for i, path in enumerate(tqdm(audio_paths, desc="Extracting activations")):
            try:
                result = self.extract(path, pooling=pooling)
                for layer_idx in range(n_layers):
                    activations[i, layer_idx, :] = result[f'layer_{layer_idx}']
            except Exception as e:
                print(f"Error processing {path}: {e}")
                activations[i, :, :] = 0
        
        # Flush to disk
        activations.flush()
        
        # Save metadata
        import json
        metadata = {
            'shape': list(shape),
            'n_samples': n_samples,
            'n_layers': n_layers,
            'hidden_size': hidden_size,
            'pooling': pooling,
            'audio_paths': audio_paths
        }
        with open(output_path.replace('.dat', '_metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        return activations
```

### PHASE 5: Probing Experiments (Days 11-14) - VS CODE CPU

**Deliverables:**
- [ ] `src/interpretability/probing.py` - Probing classifier code
- [ ] Layer-wise PD classification probing results
- [ ] Clinical feature probing results (where is jitter encoded?)
- [ ] Publication-quality figures
- [ ] **ABSTRACT READY** with preliminary results

```python
# src/interpretability/probing.py

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class ProbingClassifier:
    """Linear probing classifier for interpretability analysis."""
    
    def __init__(self, task: str = 'classification'):
        """
        Args:
            task: 'classification' for PD/HC, 'regression' for clinical features
        """
        self.task = task
        self.scaler = StandardScaler()
        
        if task == 'classification':
            self.model = LogisticRegression(max_iter=1000, C=1.0)
        else:
            self.model = Ridge(alpha=1.0)
    
    def evaluate_layer(
        self,
        activations: np.ndarray,  # [n_samples, hidden_size]
        labels: np.ndarray,       # [n_samples] or [n_samples, n_features]
        groups: np.ndarray,       # [n_samples] - subject IDs for LOSO
    ) -> Dict[str, float]:
        """Evaluate probing accuracy with LOSO cross-validation."""
        
        X = self.scaler.fit_transform(activations)
        
        logo = LeaveOneGroupOut()
        
        if self.task == 'classification':
            scores = cross_val_score(self.model, X, labels, cv=logo, groups=groups)
            return {
                'accuracy': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
        else:
            # For regression (clinical features)
            from sklearn.metrics import r2_score, make_scorer
            r2_scorer = make_scorer(r2_score)
            scores = cross_val_score(self.model, X, labels, cv=logo, groups=groups, scoring=r2_scorer)
            return {
                'r2': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }


def run_layerwise_probing(
    activations_path: str,
    labels: np.ndarray,
    groups: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """Run probing analysis across all layers."""
    
    # Load activations memmap
    import json
    with open(activations_path.replace('.dat', '_metadata.json')) as f:
        metadata = json.load(f)
    
    activations = np.memmap(
        activations_path, 
        dtype='float32', 
        mode='r', 
        shape=tuple(metadata['shape'])
    )
    
    results = {}
    prober = ProbingClassifier(task='classification')
    
    for layer_idx in range(metadata['n_layers']):
        layer_acts = activations[:, layer_idx, :]
        results[layer_idx] = prober.evaluate_layer(layer_acts, labels, groups)
        print(f"Layer {layer_idx}: {results[layer_idx]['accuracy']:.3f} ± {results[layer_idx]['std']:.3f}")
    
    return results


def run_clinical_feature_probing(
    activations_path: str,
    clinical_features: np.ndarray,  # [n_samples, n_clinical_features]
    feature_names: List[str],
    groups: np.ndarray
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Probe each clinical feature at each layer."""
    
    import json
    with open(activations_path.replace('.dat', '_metadata.json')) as f:
        metadata = json.load(f)
    
    activations = np.memmap(
        activations_path,
        dtype='float32',
        mode='r',
        shape=tuple(metadata['shape'])
    )
    
    results = {name: {} for name in feature_names}
    
    for feat_idx, feat_name in enumerate(feature_names):
        prober = ProbingClassifier(task='regression')
        feat_values = clinical_features[:, feat_idx]
        
        # Skip if constant or NaN
        if np.std(feat_values) < 1e-6 or np.any(np.isnan(feat_values)):
            continue
        
        for layer_idx in range(metadata['n_layers']):
            layer_acts = activations[:, layer_idx, :]
            results[feat_name][layer_idx] = prober.evaluate_layer(
                layer_acts, feat_values, groups
            )
    
    return results


def plot_layerwise_probing(
    results: Dict[int, Dict[str, float]],
    title: str = "Layer-wise PD Classification Probing Accuracy",
    save_path: str = None
):
    """Create publication-quality probing accuracy plot."""
    
    layers = sorted(results.keys())
    accuracies = [results[l]['accuracy'] for l in layers]
    stds = [results[l]['std'] for l in layers]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(layers, accuracies, yerr=stds, marker='o', capsize=5, 
                linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance')
    
    ax.set_xlabel('Transformer Layer', fontsize=12)
    ax.set_ylabel('Probing Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(layers)
    ax.set_ylim([0.4, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_clinical_feature_heatmap(
    results: Dict[str, Dict[int, Dict[str, float]]],
    save_path: str = None
):
    """Create heatmap of clinical feature encoding across layers."""
    
    feature_names = list(results.keys())
    layers = sorted(list(results[feature_names[0]].keys()))
    
    # Build matrix
    matrix = np.zeros((len(feature_names), len(layers)))
    for i, feat in enumerate(feature_names):
        for j, layer in enumerate(layers):
            if layer in results[feat]:
                matrix[i, j] = results[feat][layer].get('r2', 0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(
        matrix, 
        xticklabels=layers, 
        yticklabels=feature_names,
        cmap='viridis',
        annot=True,
        fmt='.2f',
        ax=ax
    )
    
    ax.set_xlabel('Transformer Layer', fontsize=12)
    ax.set_ylabel('Clinical Feature', fontsize=12)
    ax.set_title('Clinical Feature Encoding Across Layers (R² Score)', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
```

### PHASE 6: Activation Patching (Days 15-21)

**Deliverables:**
- [ ] `src/interpretability/patching.py` - Patching experiment code
- [ ] Layer-level patching results (which layers are causal?)
- [ ] Attention head patching results
- [ ] Figures showing causal importance

```python
# src/interpretability/patching.py

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import Wav2Vec2ForSequenceClassification
from tqdm import tqdm

class ActivationPatcher:
    """Perform activation patching experiments."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda"
    ):
        self.device = device
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        self.model = self.model.to(device)
        self.model.eval()
        
        self.num_layers = len(self.model.wav2vec2.encoder.layers)
        self.cached_activations = {}
    
    def _get_patch_hook(self, source_activation: torch.Tensor):
        """Create hook that patches in source activation."""
        def hook(module, input, output):
            # output is (hidden_states, ...), we patch hidden_states
            patched = (source_activation.to(output[0].device),) + output[1:]
            return patched
        return hook
    
    @torch.no_grad()
    def get_clean_activations(
        self,
        input_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Run clean forward pass and cache all activations."""
        
        activations = {}
        hooks = []
        
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output[0].clone()
            return hook
        
        # Register hooks
        for i, layer in enumerate(self.model.wav2vec2.encoder.layers):
            h = layer.register_forward_hook(get_activation(f'layer_{i}'))
            hooks.append(h)
        
        # Forward pass
        input_values = input_values.to(self.device)
        outputs = self.model(input_values)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        activations['logits'] = outputs.logits.clone()
        return activations
    
    @torch.no_grad()
    def patch_layer(
        self,
        corrupted_input: torch.Tensor,
        clean_activation: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Run forward pass with one layer patched."""
        
        handle = self.model.wav2vec2.encoder.layers[layer_idx].register_forward_hook(
            self._get_patch_hook(clean_activation)
        )
        
        corrupted_input = corrupted_input.to(self.device)
        outputs = self.model(corrupted_input)
        
        handle.remove()
        
        return outputs.logits
    
    def run_layer_patching_experiment(
        self,
        clean_inputs: List[torch.Tensor],
        corrupted_inputs: List[torch.Tensor],
        clean_labels: List[int]
    ) -> Dict[int, Dict[str, float]]:
        """
        Run layer-wise patching experiment.
        
        For each (clean, corrupted) pair:
        1. Get clean activations and logits
        2. Get corrupted logits (baseline)
        3. For each layer, patch clean activation into corrupted run
        4. Measure how much prediction recovers toward clean
        """
        
        results = {i: {'logit_diff_recovered': []} for i in range(self.num_layers)}
        results['baseline'] = {'accuracy': []}
        results['clean'] = {'accuracy': []}
        
        for clean_input, corrupted_input, clean_label in tqdm(
            zip(clean_inputs, corrupted_inputs, clean_labels),
            total=len(clean_inputs),
            desc="Patching experiments"
        ):
            # Get clean forward pass
            clean_acts = self.get_clean_activations(clean_input.unsqueeze(0))
            clean_logits = clean_acts['logits']
            clean_pred = clean_logits.argmax(dim=-1).item()
            results['clean']['accuracy'].append(clean_pred == clean_label)
            
            # Get corrupted forward pass
            corrupted_input = corrupted_input.unsqueeze(0).to(self.device)
            corrupted_logits = self.model(corrupted_input).logits
            corrupted_pred = corrupted_logits.argmax(dim=-1).item()
            results['baseline']['accuracy'].append(corrupted_pred == clean_label)
            
            # Calculate logit difference
            clean_logit_diff = (clean_logits[0, clean_label] - clean_logits[0, 1-clean_label]).item()
            corrupted_logit_diff = (corrupted_logits[0, clean_label] - corrupted_logits[0, 1-clean_label]).item()
            total_diff = clean_logit_diff - corrupted_logit_diff
            
            # Patch each layer
            for layer_idx in range(self.num_layers):
                patched_logits = self.patch_layer(
                    corrupted_input,
                    clean_acts[f'layer_{layer_idx}'],
                    layer_idx
                )
                patched_logit_diff = (patched_logits[0, clean_label] - patched_logits[0, 1-clean_label]).item()
                
                # How much of the logit difference did we recover?
                if abs(total_diff) > 1e-6:
                    recovery = (patched_logit_diff - corrupted_logit_diff) / total_diff
                else:
                    recovery = 0.0
                
                results[layer_idx]['logit_diff_recovered'].append(recovery)
        
        # Aggregate
        summary = {}
        for layer_idx in range(self.num_layers):
            recoveries = results[layer_idx]['logit_diff_recovered']
            summary[layer_idx] = {
                'mean_recovery': np.mean(recoveries),
                'std_recovery': np.std(recoveries),
                'median_recovery': np.median(recoveries)
            }
        
        summary['baseline_accuracy'] = np.mean(results['baseline']['accuracy'])
        summary['clean_accuracy'] = np.mean(results['clean']['accuracy'])
        
        return summary


def create_minimal_pairs(
    dataset,
    n_pairs: int = 50
) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Create minimal pairs for patching experiments.
    
    Match PD and HC samples by:
    - Similar duration
    - Same task type
    - Similar age/sex if available
    """
    
    pd_samples = [s for s in dataset.samples if s['label'] == 1]
    hc_samples = [s for s in dataset.samples if s['label'] == 0]
    
    pairs = []
    used_hc = set()
    
    for pd_sample in pd_samples[:n_pairs]:
        # Find best matching HC
        best_match = None
        best_score = float('inf')
        
        for hc_sample in hc_samples:
            if hc_sample['path'] in used_hc:
                continue
            
            # Simple matching by task
            if pd_sample.get('task') == hc_sample.get('task'):
                score = 0  # Perfect task match
            else:
                score = 1
            
            if score < best_score:
                best_score = score
                best_match = hc_sample
        
        if best_match:
            used_hc.add(best_match['path'])
            
            # Load audio
            pd_data = dataset[dataset.samples.index(pd_sample)]
            hc_data = dataset[dataset.samples.index(best_match)]
            
            # PD is "corrupted", HC is "clean" (we want to see if patching HC info helps)
            pairs.append((
                hc_data['input_values'],  # clean
                pd_data['input_values'],   # corrupted
                0  # clean label (HC)
            ))
    
    return pairs
```

### PHASE 7: Analysis & Documentation (Days 22-35)

**Deliverables:**
- [ ] Complete statistical analysis
- [ ] All publication-quality figures
- [ ] Research paper draft (for arXiv/workshop)
- [ ] ISEF poster design
- [ ] Presentation preparation

---

## 6. QUALITY STANDARDS

### 6.1 Code Quality
- All functions must have docstrings with Args, Returns, Examples
- Type hints for all function signatures
- Unit tests for critical functions (data loading, feature extraction)
- Consistent naming: `snake_case` for functions/variables, `PascalCase` for classes
- Maximum function length: 50 lines
- No hardcoded paths - use config files or arguments

### 6.2 Reproducibility
- All random seeds must be set and documented
- Requirements.txt with pinned versions
- Config files for all experiments
- Save all hyperparameters with results
- Git commit hash recorded with each experiment

### 6.3 Statistical Rigor
- Always use Leave-One-Subject-Out cross-validation for medical data
- Report mean ± std for all metrics
- Use appropriate statistical tests (paired t-test, Wilcoxon)
- Correct for multiple comparisons when testing many layers
- Report effect sizes, not just p-values

### 6.4 Visualization Standards
- DPI: 300 for all saved figures
- Font size: minimum 12pt for labels
- Color-blind friendly palettes (viridis, colorbrewer)
- Include error bars/confidence intervals
- Clear axis labels with units

---

## 7. EXPECTED RESULTS

### 7.1 Baseline Performance
- Clinical features SVM: 75-85% accuracy
- Wav2Vec2 fine-tuned: 80-90% accuracy

### 7.2 Probing Results (Hypothesis 1)
- Expect: Middle layers (5-8) show highest PD classification probing accuracy
- Expect: Early layers (2-4) show highest jitter/shimmer encoding
- Expect: Later layers (9-11) show most task-specific features

### 7.3 Patching Results (Hypothesis 2)
- Expect: Patching middle layers recovers 50-80% of logit difference
- Expect: Early/late layers contribute less
- Expect: Specific attention heads are disproportionately important

### 7.4 For Abstract (2 weeks)
You need at minimum:
1. Fine-tuned model accuracy (e.g., "85% on Italian PVS")
2. One probing figure showing layer-wise accuracy curve
3. One sentence about where PD information is encoded

---

## 8. COMMON ISSUES AND SOLUTIONS

### Issue: NNsight doesn't work with Wav2Vec2
**Solution**: Use manual PyTorch hooks (code provided in Section 2.4)

### Issue: Out of memory on Colab
**Solution**: 
- Enable gradient checkpointing
- Reduce batch size, increase gradient accumulation
- Use fp16 mixed precision
- Process activations in batches

### Issue: Poor probing accuracy at all layers
**Possible causes**:
- Model not properly fine-tuned (check training curves)
- Data leakage in cross-validation (ensure subject-wise splits)
- Features not discriminative (try different clinical features)

### Issue: Dataset loading errors
**Solutions**:
- Check audio file format (must be readable by torchaudio)
- Verify sampling rate conversion
- Handle corrupted files gracefully

---

## 9. CONTACT AND RESOURCES

### Key Papers to Cite
1. Wav2Vec2: Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (NeurIPS 2020)
2. Mechanistic Interpretability Survey: Räuker et al., "Toward Transparent AI" (2023)
3. Probing Classifiers: Belinkov, "Probing Classifiers" (TACL 2022)
4. Activation Patching: Meng et al., "Locating and Editing Factual Associations" (NeurIPS 2022)
5. PD Voice Detection: Orozco-Arroyave et al., PC-GITA paper (LREC 2014)

### Useful Resources
- NNsight documentation: https://nnsight.net/
- TransformerLens (for reference): https://github.com/neelnanda-io/TransformerLens
- ARENA 3.0 MI tutorials: https://arena3-chapter1-transformer-interp.streamlit.app/
- Neel Nanda's MI guide: https://www.neelnanda.io/mechanistic-interpretability/getting-started

---

**This document contains everything needed to build this project. Start with Phase 1 (data infrastructure) and proceed sequentially. Good luck!**
