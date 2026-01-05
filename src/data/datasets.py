"""
parkinson's disease speech dataset classes.

implements pytorch dataset classes for multiple pd voice corpora with
standardized interfaces for mechanistic interpretability research.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import warnings

import torch
from torch.utils.data import Dataset, Subset
import torchaudio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class BasePDDataset(Dataset, ABC):
    """
    base class for parkinson's disease speech datasets.

    provides standardized interface for audio loading, preprocessing, and
    subject-wise splitting to prevent data leakage in cross-validation.

    all audio is resampled to 16khz mono for wav2vec2 compatibility.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        task: Optional[Union[str, List[str]]] = None,
        target_sr: int = 16000,
        max_duration: float = 10.0,
        min_duration: float = 0.5,
        normalize_audio: bool = True,
        return_path: bool = True,
        cache_audio: bool = False
    ):
        """
        args:
            root_dir: path to dataset root directory
            task: task name filter (e.g., 'vowel_a') or list of tasks. none = all tasks
            target_sr: target sampling rate in hz
            max_duration: maximum audio duration in seconds
            min_duration: minimum audio duration in seconds (for quality control)
            normalize_audio: whether to normalize audio amplitude
            return_path: whether to include file path in returned dict
            cache_audio: whether to cache loaded audio in memory (faster but uses ram)
        """
        self.root_dir = Path(root_dir)
        self.task = [task] if isinstance(task, str) else task
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.normalize_audio = normalize_audio
        self.return_path = return_path
        self.cache_audio = cache_audio

        self.max_samples = int(max_duration * target_sr)
        self.min_samples = int(min_duration * target_sr)

        self._audio_cache = {} if cache_audio else None

        if not self.root_dir.exists():
            raise FileNotFoundError(f"dataset root directory not found: {self.root_dir}")

        self.samples = self._load_samples()

        if len(self.samples) == 0:
            warnings.warn(f"no samples found in {self.root_dir} with task filter {self.task}")

        self._validate_samples()

    @abstractmethod
    def _load_samples(self) -> List[Dict]:
        """
        load dataset-specific sample metadata.

        must return list of dicts with keys:
        - path: Path object to audio file
        - label: int (0 for healthy control, 1 for pd)
        - subject_id: str (unique subject identifier)
        - task: str (task name, e.g., 'vowel_a')
        - additional dataset-specific metadata
        """
        raise NotImplementedError

    def _validate_samples(self):
        """validate that loaded samples have required fields."""
        required_keys = {'path', 'label', 'subject_id', 'task'}

        for i, sample in enumerate(self.samples[:5]):
            missing = required_keys - set(sample.keys())
            if missing:
                raise ValueError(
                    f"sample {i} missing required keys: {missing}. "
                    f"check _load_samples() implementation."
                )

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def n_subjects(self) -> int:
        """return number of unique subjects in dataset."""
        return len(set(s['subject_id'] for s in self.samples))

    @property
    def subject_ids(self) -> List[str]:
        """return list of unique subject ids."""
        return list(set(s['subject_id'] for s in self.samples))

    def __getitem__(self, idx: int) -> Dict:
        """
        load and preprocess audio sample.

        returns:
            dict with keys:
            - input_values: torch.tensor [n_samples] mono audio at target_sr
            - label: int (0=hc, 1=pd)
            - subject_id: str
            - task: str
            - path: str (if return_path=true)
            - additional metadata from sample dict
        """
        sample = self.samples[idx]

        if self.cache_audio and str(sample['path']) in self._audio_cache:
            waveform = self._audio_cache[str(sample['path'])]
        else:
            waveform = self._load_and_preprocess_audio(sample['path'])
            if self.cache_audio:
                self._audio_cache[str(sample['path'])] = waveform

        result = {
            'input_values': waveform,
            'label': sample['label'],
            'subject_id': sample['subject_id'],
            'task': sample['task']
        }

        if self.return_path:
            result['path'] = str(sample['path'])

        for key in sample:
            if key not in result and key != 'path':
                result[key] = sample[key]

        return result

    def _load_and_preprocess_audio(self, path: Path) -> torch.Tensor:
        """load audio file and apply preprocessing pipeline."""
        try:
            waveform, sr = torchaudio.load(path)
        except Exception as e:
            raise RuntimeError(f"failed to load audio from {path}: {e}")

        waveform = self._convert_to_mono(waveform)
        waveform = self._resample(waveform, sr)
        waveform = self._truncate_or_pad(waveform)

        if self.normalize_audio:
            waveform = self._normalize(waveform)

        return waveform.squeeze(0)

    def _convert_to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """convert stereo to mono by averaging channels."""
        if waveform.shape[0] > 1:
            return waveform.mean(dim=0, keepdim=True)
        return waveform

    def _resample(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """resample to target sampling rate."""
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        return waveform

    def _truncate_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        """truncate or zero-pad to max_duration."""
        current_samples = waveform.shape[1]

        if current_samples > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        elif current_samples < self.max_samples:
            padding = self.max_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform

    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """normalize audio amplitude to [-1, 1] range."""
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val
        return waveform

    def get_subject_split(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[Subset, Subset, Subset]:
        """
        create train/val/test split by subject to prevent data leakage.

        ensures no subject appears in multiple splits, which is critical for
        proper evaluation in medical machine learning.

        args:
            test_size: fraction of subjects for test set
            val_size: fraction of subjects for validation set
            random_state: random seed for reproducibility
            stratify: whether to stratify split by diagnosis label

        returns:
            (train_dataset, val_dataset, test_dataset) as subset objects
        """
        subject_ids = list(set(s['subject_id'] for s in self.samples))

        if stratify:
            subject_labels = [
                self.samples[self._get_first_sample_idx(sid)]['label']
                for sid in subject_ids
            ]
        else:
            subject_labels = None

        train_val_subjects, test_subjects = train_test_split(
            subject_ids,
            test_size=test_size,
            stratify=subject_labels,
            random_state=random_state
        )

        # handle case where val_size is 0
        if val_size == 0 or val_size is None:
            train_subjects = train_val_subjects
            val_subjects = []
        else:
            if stratify:
                train_val_labels = [
                    self.samples[self._get_first_sample_idx(sid)]['label']
                    for sid in train_val_subjects
                ]
            else:
                train_val_labels = None

            train_subjects, val_subjects = train_test_split(
                train_val_subjects,
                test_size=val_size / (1 - test_size),
                stratify=train_val_labels,
                random_state=random_state
            )

        train_indices = [
            i for i, s in enumerate(self.samples)
            if s['subject_id'] in train_subjects
        ]
        val_indices = [
            i for i, s in enumerate(self.samples)
            if s['subject_id'] in val_subjects
        ]
        test_indices = [
            i for i, s in enumerate(self.samples)
            if s['subject_id'] in test_subjects
        ]

        return (
            Subset(self, train_indices),
            Subset(self, val_indices),
            Subset(self, test_indices)
        )

    def _get_first_sample_idx(self, subject_id: str) -> int:
        """get index of first sample for given subject."""
        for i, s in enumerate(self.samples):
            if s['subject_id'] == subject_id:
                return i
        raise ValueError(f"subject {subject_id} not found")

    def get_label_distribution(self) -> Dict[str, int]:
        """get distribution of labels in dataset."""
        labels = [s['label'] for s in self.samples]
        return {
            'healthy_control': sum(1 for l in labels if l == 0),
            'parkinson': sum(1 for l in labels if l == 1),
            'total': len(labels)
        }

    def get_task_distribution(self) -> Dict[str, int]:
        """get distribution of tasks in dataset."""
        tasks = [s['task'] for s in self.samples]
        task_counts = {}
        for task in tasks:
            task_counts[task] = task_counts.get(task, 0) + 1
        return task_counts

    def get_subject_count(self) -> int:
        """get number of unique subjects."""
        return len(set(s['subject_id'] for s in self.samples))


class ItalianPVSDataset(BasePDDataset):
    """
    italian parkinson's voice and speech dataset.

    source: ieee dataport / figshare
    actual structure (real data):
        root_dir/
        ├── 15 young healthy control/
        │   ├── SubjectName/
        │   │   ├── VA1...wav (vowel a)
        │   │   ├── VE1...wav (vowel e)
        │   │   ├── VI1...wav (vowel i)
        │   │   ├── VO1...wav (vowel o)
        │   │   ├── VU1...wav (vowel u)
        │   │   ├── PR1...wav (prose reading)
        │   │   ├── FB1...wav (freephone/spontaneous)
        │   │   └── ...
        │   └── ...
        ├── 22 elderly healthy control/
        │   └── ...
        └── 28 people with parkinson's disease/
            ├── 1-5/
            │   ├── SubjectName/
            │   │   └── ...
            │   └── ...
            └── ...

    total: 65 subjects (28 pd, 37 hc), 831 audio files
    tasks: sustained vowels (a, e, i, o, u), reading (pr), spontaneous (fb)

    file naming convention:
        - VA = vowel a
        - VE = vowel e  
        - VI = vowel i
        - VO = vowel o
        - VU = vowel u
        - PR = prose reading
        - FB = freephone/spontaneous
        - B = breathing/other
        - D = diadochokinesis
    """

    DIAGNOSIS_FOLDERS = {
        '15 young healthy control': ('HC_young', 0),
        '22 elderly healthy control': ('HC_elderly', 0),
        '28 people with parkinson\'s disease': ('PD', 1),
        'PD': ('PD', 1),
        'HC': ('HC', 0),
    }

    TASK_PREFIXES = {
        'VA': 'vowel_a',
        'VE': 'vowel_e',
        'VI': 'vowel_i',
        'VO': 'vowel_o',
        'VU': 'vowel_u',
        'PR': 'reading',
        'FB': 'spontaneous',
        'B': 'breathing',
        'D': 'diadochokinesis',
    }

    def _load_samples(self) -> List[Dict]:
        """load italian pvs samples with metadata."""
        samples = []

        for folder_name, (diagnosis, label) in self.DIAGNOSIS_FOLDERS.items():
            diagnosis_dir = self.root_dir / folder_name

            if not diagnosis_dir.exists():
                continue

            samples.extend(self._load_from_diagnosis_folder(
                diagnosis_dir, diagnosis, label
            ))

        if len(samples) == 0:
            for subfolder in self.root_dir.iterdir():
                if not subfolder.is_dir() or subfolder.name.startswith('.'):
                    continue

                is_pd = 'parkinson' in subfolder.name.lower() or 'pd' in subfolder.name.lower()
                label = 1 if is_pd else 0
                diagnosis = 'PD' if is_pd else 'HC'

                samples.extend(self._load_from_diagnosis_folder(
                    subfolder, diagnosis, label
                ))

        return samples

    def _load_from_diagnosis_folder(
        self,
        folder: Path,
        diagnosis: str,
        label: int
    ) -> List[Dict]:
        """load samples from a diagnosis folder, handling nested structures."""
        samples = []

        for item in sorted(folder.iterdir()):
            if not item.is_dir() or item.name.startswith('.'):
                continue

            if item.name.startswith(('1-', '6-', '11-', '17-')) or item.name[0].isdigit():
                for subject_dir in sorted(item.iterdir()):
                    if subject_dir.is_dir() and not subject_dir.name.startswith('.'):
                        samples.extend(self._load_subject_folder(
                            subject_dir, diagnosis, label
                        ))
            else:
                samples.extend(self._load_subject_folder(item, diagnosis, label))

        return samples

    def _load_subject_folder(
        self,
        subject_dir: Path,
        diagnosis: str,
        label: int
    ) -> List[Dict]:
        """load all audio files from a subject folder."""
        samples = []
        subject_id = f"{diagnosis}_{subject_dir.name.replace(' ', '_')}"

        for audio_file in sorted(subject_dir.glob("*.wav")):
            if audio_file.name.startswith('.'):
                continue

            task_name = self._parse_task_name(audio_file.stem)

            if self.task is not None and task_name not in self.task:
                continue

            samples.append({
                'path': audio_file,
                'label': label,
                'subject_id': subject_id,
                'task': task_name,
                'diagnosis': diagnosis
            })

        for audio_file in sorted(subject_dir.glob("*.txt")):
            pass

        return samples

    def _load_metadata(self, path: Path) -> Optional[pd.DataFrame]:
        """load metadata excel file if exists."""
        if path.exists():
            try:
                return pd.read_excel(path)
            except Exception as e:
                warnings.warn(f"failed to load metadata from {path}: {e}")
        return None

    def _get_subject_metadata(
        self,
        metadata_df: Optional[pd.DataFrame],
        subject_id: str
    ) -> Optional[Dict]:
        """extract metadata for specific subject."""
        if metadata_df is None:
            return None

        subject_row = metadata_df[metadata_df['subject_id'] == subject_id]
        if len(subject_row) == 0:
            return None

        meta = subject_row.iloc[0].to_dict()
        meta = {k: v for k, v in meta.items() if pd.notna(v)}
        return meta

    def _parse_task_name(self, filename: str) -> str:
        """extract task name from filename using prefix codes."""
        filename_upper = filename.upper()

        for prefix, task in self.TASK_PREFIXES.items():
            if filename_upper.startswith(prefix):
                return task

        filename_lower = filename.lower()

        if 'vowel_a' in filename_lower or filename_lower == 'a':
            return 'vowel_a'
        elif 'vowel_e' in filename_lower or filename_lower == 'e':
            return 'vowel_e'
        elif 'vowel_i' in filename_lower or filename_lower == 'i':
            return 'vowel_i'
        elif 'vowel_o' in filename_lower or filename_lower == 'o':
            return 'vowel_o'
        elif 'vowel_u' in filename_lower or filename_lower == 'u':
            return 'vowel_u'
        elif 'reading' in filename_lower or 'prose' in filename_lower:
            return 'reading'
        elif 'spontaneous' in filename_lower or 'monologue' in filename_lower or 'free' in filename_lower:
            return 'spontaneous'
        else:
            return 'unknown'


class MDVRKCLDataset(BasePDDataset):
    """
    mdvr-kcl parkinson's speech dataset.

    source: king's college london / zenodo
    actual structure:
        root_dir/
        ├── ReadText/
        │   ├── HC/
        │   │   ├── ID01_hc_2_0_0.wav
        │   │   └── ...
        │   └── PD/
        │       ├── ID02_pd_2_0_0.wav
        │       └── ...
        └── SpontaneousDialogue/
            ├── HC/
            └── PD/

    filename format: ID{num}_{hc|pd}_{n}_{n}_{n}.wav

    total: 37 subjects (16 pd, 21 hc), ~74 audio files
    tasks: reading passage (ReadText), spontaneous dialogue
    """

    TASK_FOLDERS = {
        'ReadText': 'reading',
        'SpontaneousDialogue': 'spontaneous',
    }

    def _load_samples(self) -> List[Dict]:
        """load mdvr-kcl samples from actual directory structure."""
        samples = []

        for task_folder, task_name in self.TASK_FOLDERS.items():
            task_dir = self.root_dir / task_folder

            if not task_dir.exists():
                continue

            for diagnosis in ['HC', 'PD']:
                diagnosis_dir = task_dir / diagnosis
                label = 1 if diagnosis == 'PD' else 0

                if not diagnosis_dir.exists():
                    continue

                for audio_file in sorted(diagnosis_dir.glob("*.wav")):
                    if audio_file.name.startswith('.'):
                        continue

                    subject_id = self._extract_subject_id(audio_file.stem, diagnosis)

                    if self.task is not None and task_name not in self.task:
                        continue

                    sample = {
                        'path': audio_file,
                        'label': label,
                        'subject_id': subject_id,
                        'task': task_name,
                        'diagnosis': diagnosis
                    }

                    parsed = self._parse_filename_metadata(audio_file.stem)
                    if parsed:
                        sample.update(parsed)

                    samples.append(sample)

        if len(samples) == 0:
            samples = self._load_legacy_format()

        return samples

    def _extract_subject_id(self, filename: str, diagnosis: str) -> str:
        """extract subject id from filename."""
        parts = filename.split('_')
        if len(parts) >= 1 and parts[0].startswith('ID'):
            return f"{diagnosis}_{parts[0]}"
        return f"{diagnosis}_{filename}"

    def _parse_filename_metadata(self, filename: str) -> Optional[Dict]:
        """parse metadata from mdvr-kcl filename format."""
        parts = filename.split('_')

        if len(parts) >= 5:
            try:
                return {
                    'hy_stage': int(parts[2]),
                    'updrs_speech': int(parts[3]),
                    'updrs_facial': int(parts[4])
                }
            except (ValueError, IndexError):
                pass

        return None

    def _load_legacy_format(self) -> List[Dict]:
        """fallback loader for original expected format."""
        samples = []

        metadata_path = self.root_dir / "metadata.csv"
        metadata_df = self._load_metadata_csv(metadata_path)

        audio_dir = self.root_dir / "audio"
        if not audio_dir.exists():
            audio_dir = self.root_dir

        for audio_file in sorted(audio_dir.glob("*.wav")):
            parsed = self._parse_filename(audio_file.stem)
            if parsed is None:
                continue

            diagnosis, subject_num, task_name = parsed
            label = 1 if diagnosis == "PD" else 0
            subject_id = f"{diagnosis}_{subject_num:03d}"

            if self.task is not None and task_name not in self.task:
                continue

            sample = {
                'path': audio_file,
                'label': label,
                'subject_id': subject_id,
                'task': task_name,
                'diagnosis': diagnosis
            }

            if metadata_df is not None:
                subject_meta = self._get_subject_metadata_csv(
                    metadata_df, subject_id
                )
                if subject_meta:
                    sample.update(subject_meta)

            samples.append(sample)

        return samples

    def _load_metadata_csv(self, path: Path) -> Optional[pd.DataFrame]:
        """load metadata csv file."""
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as e:
                warnings.warn(f"failed to load metadata from {path}: {e}")
        return None

    def _get_subject_metadata_csv(
        self,
        metadata_df: pd.DataFrame,
        subject_id: str
    ) -> Optional[Dict]:
        """extract metadata for subject from csv."""
        subject_row = metadata_df[metadata_df['subject_id'] == subject_id]
        if len(subject_row) == 0:
            return None

        meta = subject_row.iloc[0].to_dict()
        return {k: v for k, v in meta.items() if pd.notna(v)}

    def _parse_filename(self, filename: str) -> Optional[Tuple[str, int, str]]:
        """
        parse mdvr-kcl filename format.

        expected format: {PD|HC}_{num}_{task}
        returns: (diagnosis, subject_number, task_name)
        """
        parts = filename.split('_')
        if len(parts) < 3:
            return None

        diagnosis = parts[0]
        try:
            subject_num = int(parts[1])
        except ValueError:
            return None

        task_name = '_'.join(parts[2:])

        task_name_lower = task_name.lower()
        if 'reading' in task_name_lower:
            task_name = 'reading'
        elif 'monologue' in task_name_lower or 'spontaneous' in task_name_lower:
            task_name = 'monologue'

        return diagnosis, subject_num, task_name


class ArkansasDataset(BasePDDataset):
    """
    arkansas parkinson's disease speech dataset (figshare).

    actual structure:
        root_dir/
        ├── Demographics_age_sex.xlsx
        ├── HC_AH/
        │   ├── subject_001.wav
        │   └── ...
        └── PD_AH/
            ├── subject_001.wav
            └── ...

    alternate structure (generic):
        root_dir/
        ├── PD/
        └── HC/

    total: ~81 subjects (40 pd, 41 hc)
    tasks: typically sustained vowel /a/ (ah)
    """

    DIAGNOSIS_FOLDERS = {
        'PD_AH': ('PD', 1),
        'HC_AH': ('HC', 0),
        'PD': ('PD', 1),
        'HC': ('HC', 0),
    }

    def _load_samples(self) -> List[Dict]:
        """load arkansas dataset samples from actual structure."""
        samples = []

        metadata_path = self.root_dir / "Demographics_age_sex.xlsx"
        metadata_df = self._load_metadata_excel(metadata_path)

        for folder_name, (diagnosis, label) in self.DIAGNOSIS_FOLDERS.items():
            diagnosis_dir = self.root_dir / folder_name

            if not diagnosis_dir.exists():
                continue

            for audio_file in sorted(diagnosis_dir.glob("*.wav")):
                if audio_file.name.startswith('.'):
                    continue

                subject_id = f"{diagnosis}_{audio_file.stem}"
                task_name = 'vowel_ah'

                if self.task is not None and task_name not in self.task:
                    continue

                sample = {
                    'path': audio_file,
                    'label': label,
                    'subject_id': subject_id,
                    'task': task_name,
                    'diagnosis': diagnosis
                }

                if metadata_df is not None:
                    meta = self._get_metadata_for_file(metadata_df, audio_file.stem)
                    if meta:
                        sample.update(meta)

                samples.append(sample)

        return samples

    def _load_metadata_excel(self, path: Path) -> Optional[pd.DataFrame]:
        """load metadata from excel file."""
        if path.exists():
            try:
                return pd.read_excel(path)
            except Exception as e:
                warnings.warn(f"failed to load metadata from {path}: {e}")
        return None

    def _load_metadata_csv(self, path: Path) -> Optional[pd.DataFrame]:
        """load metadata csv."""
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as e:
                warnings.warn(f"failed to load metadata: {e}")
        return None

    def _get_metadata_for_file(
        self,
        metadata_df: pd.DataFrame,
        filename: str
    ) -> Optional[Dict]:
        """get metadata for specific file."""
        possible_columns = ['filename', 'file', 'id', 'subject', 'name']

        for col in possible_columns:
            if col in metadata_df.columns:
                file_row = metadata_df[metadata_df[col].astype(str) == filename]
                if len(file_row) > 0:
                    meta = file_row.iloc[0].to_dict()
                    return {k: v for k, v in meta.items() if pd.notna(v)}

        return None

    def _infer_task(self, filename: str) -> str:
        """infer task from filename."""
        filename_lower = filename.lower()

        if 'ah' in filename_lower or 'vowel' in filename_lower or 'sustained' in filename_lower:
            return 'vowel_ah'
        elif 'reading' in filename_lower:
            return 'reading'
        else:
            return 'vowel_ah'


class NeuroVozDataset(BasePDDataset):
    """
    neurovoz spanish parkinson's speech dataset.

    actual structure:
        root_dir/ (should point to data/raw/neurovoz)
        ├── audios/  (FLAT directory with all WAV files)
        │   ├── HC_A1_0034.wav  (naming: {HC|PD}_{TASK}_{ID}.wav)
        │   ├── PD_FREE_0023.wav
        │   └── ...
        └── metadata/
            ├── metadata_hc.csv  (54 HC subjects with clinical data)
            └── metadata_pd.csv  (54 PD subjects with UPDRS, H-Y, symptoms)

    total: 108 subjects (54 pd, 54 hc), ~2,900 audio files
    tasks: sustained vowels (A1-A3, E1-E3, I1-I3, O1-O3, U1-U3),
           spanish words (CALLE, CARMEN, DIABLO, etc.), FREE speech

    metadata includes: UPDRS scale, Hoehn & Yahr stadium, disease duration,
                      symptoms (tremor, dysphagia, hypophonic voice), demographics
    """

    def _load_samples(self) -> List[Dict]:
        """load neurovoz samples from flat directory with metadata integration."""
        samples = []

        # determine audio directory (can pass neurovoz root or audios directly)
        if self.root_dir.name == 'audios':
            audio_dir = self.root_dir
            metadata_dir = self.root_dir.parent / 'metadata'
        else:
            audio_dir = self.root_dir / 'audios'
            metadata_dir = self.root_dir / 'metadata'

        if not audio_dir.exists():
            raise FileNotFoundError(f"audios directory not found: {audio_dir}")

        # load metadata CSVs
        metadata = self._load_metadata_files(metadata_dir)

        # load audio files from flat audios directory
        for audio_file in sorted(audio_dir.glob("*.wav")):
            if audio_file.name.startswith('.') or audio_file.name.startswith('._'):
                continue

            # parse filename: {HC|PD}_{TASK}_{ID}.wav
            parsed = self._parse_filename(audio_file.stem)
            if parsed is None:
                continue  # skip unparseable files silently

            diagnosis, task, subject_num = parsed
            label = 1 if diagnosis == 'PD' else 0
            subject_id = f"{diagnosis}_{subject_num:04d}"

            # filter by task if specified
            if self.task is not None and task not in self.task:
                continue

            sample = {
                'path': audio_file,
                'label': label,
                'subject_id': subject_id,
                'task': task,
                'diagnosis': diagnosis,
                'subject_number': subject_num
            }

            # merge with metadata if available
            if subject_id in metadata:
                sample.update(metadata[subject_id])

            samples.append(sample)

        return samples

    def _load_metadata_files(self, metadata_dir: Path) -> Dict:
        """load and parse metadata CSV files."""
        metadata = {}

        if not metadata_dir.exists():
            warnings.warn(f"metadata directory not found: {metadata_dir}")
            return metadata

        # load HC metadata
        hc_file = metadata_dir / 'metadata_hc.csv'
        if hc_file.exists():
            try:
                df = pd.read_csv(hc_file, encoding='utf-8')
                for _, row in df.iterrows():
                    if pd.notna(row.get('ID')):
                        subject_id = f"HC_{int(row['ID']):04d}"
                        if subject_id not in metadata:
                            metadata[subject_id] = self._parse_metadata_row(row)
            except Exception as e:
                warnings.warn(f"failed to load HC metadata: {e}")

        # load PD metadata
        pd_file = metadata_dir / 'metadata_pd.csv'
        if pd_file.exists():
            try:
                df = pd.read_csv(pd_file, encoding='utf-8')
                for _, row in df.iterrows():
                    if pd.notna(row.get('ID')):
                        subject_id = f"PD_{int(row['ID']):04d}"
                        if subject_id not in metadata:
                            metadata[subject_id] = self._parse_metadata_row(row)
            except Exception as e:
                warnings.warn(f"failed to load PD metadata: {e}")

        return metadata

    def _parse_metadata_row(self, row: pd.Series) -> Dict:
        """extract relevant metadata fields from CSV row."""
        meta = {}

        # demographics
        if 'Age' in row and pd.notna(row['Age']):
            meta['age'] = float(row['Age'])
        if 'Sex' in row and pd.notna(row['Sex']):
            meta['sex'] = int(row['Sex'])  # 0=female, 1=male typically

        # clinical scores
        if 'UPDRS scale' in row and pd.notna(row['UPDRS scale']):
            meta['updrs'] = float(row['UPDRS scale'])
        if 'H-Y Stadium' in row and pd.notna(row['H-Y Stadium']):
            meta['hy_stadium'] = float(row['H-Y Stadium'])
        if 'Time Disease (years)' in row and pd.notna(row['Time Disease (years)']):
            meta['disease_duration'] = float(row['Time Disease (years)'])

        # symptoms (binary flags)
        symptom_cols = ['Vocal tremor', 'Cephalic tremor', 'Mandibular tremor',
                       'Sialorrhoea', 'Dysphagia', 'Hypophonic voice']
        for col in symptom_cols:
            if col in row and pd.notna(row[col]):
                meta[col.lower().replace(' ', '_')] = int(row[col])

        return meta

    def _parse_filename(self, filename: str) -> Optional[Tuple[str, str, int]]:
        """
        parse neurovoz filename format.

        expected: {HC|PD}_{TASK}_{ID} or {HC|PD}_{WORD1}_{WORD2}_{ID}
        example: HC_A1_0034 -> ('HC', 'vowel_a1', 34)
                 HC_PAN_VINO_0034 -> ('HC', 'pan_vino', 34)

        returns: (diagnosis, task_name, subject_number)
        """
        parts = filename.split('_')
        if len(parts) < 3:
            return None

        diagnosis = parts[0]  # HC or PD
        if diagnosis not in ['HC', 'PD']:
            return None

        # try to parse subject_num from last part
        try:
            subject_num = int(parts[-1])
        except ValueError:
            return None

        # everything between diagnosis and subject_num is the task
        # join with underscore for multi-word tasks
        task_raw = '_'.join(parts[1:-1])
        task_name = self._parse_task_name(task_raw)

        return diagnosis, task_name, subject_num

    def _parse_task_name(self, task: str) -> str:
        """parse task name from filename."""
        task_upper = task.upper()

        # sustained vowels
        vowel_map = {
            'A1': 'vowel_a1', 'A2': 'vowel_a2', 'A3': 'vowel_a3',
            'E1': 'vowel_e1', 'E2': 'vowel_e2', 'E3': 'vowel_e3',
            'I1': 'vowel_i1', 'I2': 'vowel_i2', 'I3': 'vowel_i3',
            'O1': 'vowel_o1', 'O2': 'vowel_o2', 'O3': 'vowel_o3',
            'U1': 'vowel_u1', 'U2': 'vowel_u2', 'U3': 'vowel_u3',
        }

        if task_upper in vowel_map:
            return vowel_map[task_upper]

        # special tasks
        if task_upper == 'FREE':
            return 'free_speech'

        # spanish words - keep as-is but lowercase
        return task.lower()


class PCGITADataset(BasePDDataset):
    """
    pc-gita colombian spanish parkinson's speech dataset.

    structure:
        root_dir/
        ├── metadata.txt or metadata.csv
        ├── PD/
        │   ├── subject_001/
        │   │   ├── vowel_a.wav
        │   │   ├── vowel_e.wav
        │   │   ├── ddk_pataka.wav
        │   │   ├── sentence_001.wav
        │   │   └── ...
        │   └── ...
        └── HC/

    total: 100 subjects (50 pd, 50 hc), ~6,300 audio files
    tasks: sustained vowels, ddk, sentences, reading
    """

    def _load_samples(self) -> List[Dict]:
        """load pc-gita samples."""
        samples = []

        for diagnosis, label in [("PD", 1), ("HC", 0)]:
            diagnosis_dir = self.root_dir / diagnosis

            if not diagnosis_dir.exists():
                warnings.warn(f"directory not found: {diagnosis_dir}")
                continue

            for subject_dir in sorted(diagnosis_dir.iterdir()):
                if not subject_dir.is_dir():
                    continue

                subject_id = f"{diagnosis}_{subject_dir.name}"

                for audio_file in sorted(subject_dir.glob("*.wav")):
                    task_name = self._parse_pcgita_task(audio_file.stem)

                    if self.task is not None and task_name not in self.task:
                        continue

                    samples.append({
                        'path': audio_file,
                        'label': label,
                        'subject_id': subject_id,
                        'task': task_name,
                        'diagnosis': diagnosis
                    })

        return samples

    def _parse_pcgita_task(self, filename: str) -> str:
        """parse pc-gita task name."""
        filename_lower = filename.lower()

        if 'vowel' in filename_lower:
            for vowel in ['a', 'e', 'i', 'o', 'u']:
                if f'vowel_{vowel}' in filename_lower or f'_{vowel}' in filename_lower:
                    return f'vowel_{vowel}'

        if 'ddk' in filename_lower or 'pataka' in filename_lower:
            return 'ddk_pataka'

        if 'sentence' in filename_lower:
            return 'sentence'

        if 'reading' in filename_lower:
            return 'reading'

        return filename


class EWADBDataset(BasePDDataset):
    """
    ewa-db slovak parkinson's speech dataset.

    structure:
        root_dir/
        ├── metadata.csv
        ├── PD/
        │   ├── subject_001/
        │   │   ├── vowel_a.wav
        │   │   ├── ddk_pa.wav
        │   │   ├── naming.wav
        │   │   └── ...
        │   └── ...
        └── HC/

    total: ~375 subjects (175 pd, ~200 hc subset), ~5,000 audio files
    tasks: sustained vowels, ddk, picture naming
    """

    def _load_samples(self) -> List[Dict]:
        """load ewa-db samples."""
        samples = []

        for diagnosis, label in [("PD", 1), ("HC", 0)]:
            diagnosis_dir = self.root_dir / diagnosis

            if not diagnosis_dir.exists():
                warnings.warn(f"directory not found: {diagnosis_dir}")
                continue

            for subject_dir in sorted(diagnosis_dir.iterdir()):
                if not subject_dir.is_dir():
                    continue

                subject_id = f"{diagnosis}_{subject_dir.name}"

                for audio_file in sorted(subject_dir.glob("*.wav")):
                    task_name = self._parse_ewadb_task(audio_file.stem)

                    if self.task is not None and task_name not in self.task:
                        continue

                    samples.append({
                        'path': audio_file,
                        'label': label,
                        'subject_id': subject_id,
                        'task': task_name,
                        'diagnosis': diagnosis
                    })

        return samples

    def _parse_ewadb_task(self, filename: str) -> str:
        """parse ewa-db task name."""
        filename_lower = filename.lower()

        if 'vowel' in filename_lower:
            for vowel in ['a', 'e', 'i', 'o', 'u']:
                if f'_{vowel}' in filename_lower or vowel == filename_lower:
                    return f'vowel_{vowel}'

        if 'ddk' in filename_lower:
            if 'pa' in filename_lower:
                return 'ddk_pa'
            elif 'ta' in filename_lower:
                return 'ddk_ta'
            elif 'ka' in filename_lower:
                return 'ddk_ka'
            return 'ddk'

        if 'naming' in filename_lower or 'picture' in filename_lower:
            return 'naming'

        return filename


def create_combined_dataset(
    datasets: List[BasePDDataset],
    balance_datasets: bool = False
) -> 'CombinedPDDataset':
    """
    combine multiple pd datasets into single dataset.

    args:
        datasets: list of dataset instances to combine
        balance_datasets: whether to balance sample count across datasets

    returns:
        combined dataset instance
    """
    return CombinedPDDataset(datasets, balance_datasets)


class CombinedPDDataset(Dataset):
    """
    combines multiple pd datasets for cross-dataset training.

    maintains source dataset information for cross-dataset generalization
    analysis.
    """

    def __init__(
        self,
        datasets: List[BasePDDataset],
        balance_datasets: bool = False
    ):
        """
        args:
            datasets: list of dataset instances
            balance_datasets: whether to downsample to match smallest dataset
        """
        self.datasets = datasets
        self.balance_datasets = balance_datasets

        self._build_index()

    def _build_index(self):
        """build index mapping global index to (dataset_idx, local_idx)."""
        self.index_map = []

        if self.balance_datasets:
            min_size = min(len(ds) for ds in self.datasets)
            for ds_idx, dataset in enumerate(self.datasets):
                indices = np.random.choice(
                    len(dataset), min_size, replace=False
                )
                for local_idx in indices:
                    self.index_map.append((ds_idx, local_idx))
        else:
            for ds_idx, dataset in enumerate(self.datasets):
                for local_idx in range(len(dataset)):
                    self.index_map.append((ds_idx, local_idx))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict:
        """get item from appropriate source dataset."""
        ds_idx, local_idx = self.index_map[idx]
        item = self.datasets[ds_idx][local_idx]
        item['source_dataset'] = ds_idx
        return item
