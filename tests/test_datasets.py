"""
unit tests for pd speech dataset classes.

tests dataset loading, preprocessing, and subject-wise splitting.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import torchaudio
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import (
    BasePDDataset,
    ItalianPVSDataset,
    MDVRKCLDataset,
    ArkansasDataset,
    CombinedPDDataset
)


@pytest.fixture
def temp_audio_file():
    """create temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sr = 16000
        duration = 1.0
        waveform = torch.randn(1, int(sr * duration))
        torchaudio.save(f.name, waveform, sr)
        yield Path(f.name)
        Path(f.name).unlink()


@pytest.fixture
def mock_italian_pvs_dataset(tmp_path):
    """create mock italian pvs dataset structure."""
    root = tmp_path / "italian_pvs"
    root.mkdir()

    for diagnosis, label in [("PD", 1), ("HC", 0)]:
        diag_dir = root / diagnosis
        diag_dir.mkdir()

        for i in range(3):
            subject_dir = diag_dir / f"{diagnosis}{i:03d}"
            subject_dir.mkdir()

            for vowel in ['a', 'e', 'i']:
                audio_path = subject_dir / f"vowel_{vowel}.wav"
                sr = 16000
                waveform = torch.randn(1, sr * 2)
                torchaudio.save(str(audio_path), waveform, sr)

    return root


class TestBasePDDataset:
    """test base dataset functionality."""

    def test_initialization_invalid_path(self):
        """test that invalid path raises error."""
        with pytest.raises(FileNotFoundError):
            class TestDataset(BasePDDataset):
                def _load_samples(self):
                    return []

            TestDataset(root_dir="/nonexistent/path")

    def test_audio_resampling(self, tmp_path):
        """test audio resampling to target sr."""
        audio_path = tmp_path / "test.wav"
        original_sr = 44100
        target_sr = 16000
        waveform = torch.randn(1, original_sr * 2)
        torchaudio.save(str(audio_path), waveform, original_sr)

        class TestDataset(BasePDDataset):
            def _load_samples(self):
                return [{
                    'path': audio_path,
                    'label': 0,
                    'subject_id': 'test',
                    'task': 'test'
                }]

        dataset = TestDataset(root_dir=tmp_path, target_sr=target_sr)
        sample = dataset[0]

        expected_length = target_sr * 2
        assert abs(len(sample['input_values']) - expected_length) < 100

    def test_mono_conversion(self, tmp_path):
        """test stereo to mono conversion."""
        audio_path = tmp_path / "stereo.wav"
        sr = 16000
        waveform = torch.randn(2, sr * 2)
        torchaudio.save(str(audio_path), waveform, sr)

        class TestDataset(BasePDDataset):
            def _load_samples(self):
                return [{
                    'path': audio_path,
                    'label': 0,
                    'subject_id': 'test',
                    'task': 'test'
                }]

        dataset = TestDataset(root_dir=tmp_path)
        sample = dataset[0]

        assert sample['input_values'].dim() == 1

    def test_truncation(self, tmp_path):
        """test audio truncation to max duration."""
        audio_path = tmp_path / "long.wav"
        sr = 16000
        long_duration = 15.0
        waveform = torch.randn(1, int(sr * long_duration))
        torchaudio.save(str(audio_path), waveform, sr)

        class TestDataset(BasePDDataset):
            def _load_samples(self):
                return [{
                    'path': audio_path,
                    'label': 0,
                    'subject_id': 'test',
                    'task': 'test'
                }]

        max_duration = 10.0
        dataset = TestDataset(root_dir=tmp_path, max_duration=max_duration)
        sample = dataset[0]

        assert len(sample['input_values']) == int(sr * max_duration)

    def test_padding(self, tmp_path):
        """test audio padding to max duration."""
        audio_path = tmp_path / "short.wav"
        sr = 16000
        short_duration = 1.0
        waveform = torch.randn(1, int(sr * short_duration))
        torchaudio.save(str(audio_path), waveform, sr)

        class TestDataset(BasePDDataset):
            def _load_samples(self):
                return [{
                    'path': audio_path,
                    'label': 0,
                    'subject_id': 'test',
                    'task': 'test'
                }]

        max_duration = 5.0
        dataset = TestDataset(root_dir=tmp_path, max_duration=max_duration)
        sample = dataset[0]

        assert len(sample['input_values']) == int(sr * max_duration)

    def test_normalization(self, tmp_path):
        """test audio normalization."""
        audio_path = tmp_path / "loud.wav"
        sr = 16000
        waveform = torch.randn(1, sr * 2) * 10
        torchaudio.save(str(audio_path), waveform, sr)

        class TestDataset(BasePDDataset):
            def _load_samples(self):
                return [{
                    'path': audio_path,
                    'label': 0,
                    'subject_id': 'test',
                    'task': 'test'
                }]

        dataset = TestDataset(root_dir=tmp_path, normalize_audio=True)
        sample = dataset[0]

        assert sample['input_values'].abs().max() <= 1.0

    def test_subject_split(self, tmp_path):
        """test subject-wise train/val/test split."""
        class TestDataset(BasePDDataset):
            def _load_samples(self):
                samples = []
                for subj in range(10):
                    for i in range(5):
                        audio_path = tmp_path / f"s{subj}_{i}.wav"
                        waveform = torch.randn(1, 16000)
                        torchaudio.save(str(audio_path), waveform, 16000)

                        samples.append({
                            'path': audio_path,
                            'label': subj % 2,
                            'subject_id': f'subject_{subj}',
                            'task': 'test'
                        })
                return samples

        dataset = TestDataset(root_dir=tmp_path)
        train, val, test = dataset.get_subject_split(
            test_size=0.2, val_size=0.1, random_state=42
        )

        assert len(train) + len(val) + len(test) == len(dataset)

        train_subjects = set(dataset[i]['subject_id'] for i in train.indices)
        val_subjects = set(dataset[i]['subject_id'] for i in val.indices)
        test_subjects = set(dataset[i]['subject_id'] for i in test.indices)

        assert len(train_subjects & val_subjects) == 0
        assert len(train_subjects & test_subjects) == 0
        assert len(val_subjects & test_subjects) == 0

    def test_label_distribution(self, tmp_path):
        """test label distribution computation."""
        class TestDataset(BasePDDataset):
            def _load_samples(self):
                return [
                    {'path': tmp_path / 'a.wav', 'label': 0, 'subject_id': 's1', 'task': 't'},
                    {'path': tmp_path / 'b.wav', 'label': 0, 'subject_id': 's2', 'task': 't'},
                    {'path': tmp_path / 'c.wav', 'label': 1, 'subject_id': 's3', 'task': 't'},
                ]

        dataset = TestDataset(root_dir=tmp_path)
        dist = dataset.get_label_distribution()

        assert dist['healthy_control'] == 2
        assert dist['parkinson'] == 1
        assert dist['total'] == 3


class TestItalianPVSDataset:
    """test italian pvs dataset."""

    def test_dataset_loading(self, mock_italian_pvs_dataset):
        """test loading italian pvs dataset."""
        dataset = ItalianPVSDataset(
            root_dir=mock_italian_pvs_dataset,
            task='vowel_a'
        )

        assert len(dataset) > 0

        sample = dataset[0]
        assert 'input_values' in sample
        assert 'label' in sample
        assert 'subject_id' in sample
        assert 'task' in sample
        assert sample['task'] == 'vowel_a'

    def test_task_filtering(self, mock_italian_pvs_dataset):
        """test filtering by task."""
        dataset_all = ItalianPVSDataset(
            root_dir=mock_italian_pvs_dataset,
            task=None
        )

        dataset_a = ItalianPVSDataset(
            root_dir=mock_italian_pvs_dataset,
            task='vowel_a'
        )

        assert len(dataset_a) < len(dataset_all)

        for sample in [dataset_a[i] for i in range(len(dataset_a))]:
            assert sample['task'] == 'vowel_a'

    def test_subject_count(self, mock_italian_pvs_dataset):
        """test subject counting."""
        dataset = ItalianPVSDataset(root_dir=mock_italian_pvs_dataset)

        subject_count = dataset.get_subject_count()
        assert subject_count == 6


class TestCombinedPDDataset:
    """test combined dataset functionality."""

    def test_combining_datasets(self, tmp_path):
        """test combining multiple datasets."""
        class Dataset1(BasePDDataset):
            def _load_samples(self):
                samples = []
                for i in range(5):
                    path = tmp_path / f"d1_{i}.wav"
                    waveform = torch.randn(1, 16000)
                    torchaudio.save(str(path), waveform, 16000)
                    samples.append({
                        'path': path,
                        'label': 0,
                        'subject_id': f's1_{i}',
                        'task': 'test'
                    })
                return samples

        class Dataset2(BasePDDataset):
            def _load_samples(self):
                samples = []
                for i in range(3):
                    path = tmp_path / f"d2_{i}.wav"
                    waveform = torch.randn(1, 16000)
                    torchaudio.save(str(path), waveform, 16000)
                    samples.append({
                        'path': path,
                        'label': 1,
                        'subject_id': f's2_{i}',
                        'task': 'test'
                    })
                return samples

        ds1 = Dataset1(root_dir=tmp_path)
        ds2 = Dataset2(root_dir=tmp_path)

        combined = CombinedPDDataset([ds1, ds2], balance_datasets=False)

        assert len(combined) == len(ds1) + len(ds2)

    def test_balanced_combining(self, tmp_path):
        """test balanced dataset combination."""
        class Dataset1(BasePDDataset):
            def _load_samples(self):
                samples = []
                for i in range(10):
                    path = tmp_path / f"d1_{i}.wav"
                    waveform = torch.randn(1, 16000)
                    torchaudio.save(str(path), waveform, 16000)
                    samples.append({
                        'path': path,
                        'label': 0,
                        'subject_id': f's1_{i}',
                        'task': 'test'
                    })
                return samples

        class Dataset2(BasePDDataset):
            def _load_samples(self):
                samples = []
                for i in range(5):
                    path = tmp_path / f"d2_{i}.wav"
                    waveform = torch.randn(1, 16000)
                    torchaudio.save(str(path), waveform, 16000)
                    samples.append({
                        'path': path,
                        'label': 1,
                        'subject_id': f's2_{i}',
                        'task': 'test'
                    })
                return samples

        ds1 = Dataset1(root_dir=tmp_path)
        ds2 = Dataset2(root_dir=tmp_path)

        combined = CombinedPDDataset([ds1, ds2], balance_datasets=True)

        assert len(combined) == 2 * min(len(ds1), len(ds2))

    def test_source_dataset_tracking(self, tmp_path):
        """test that source dataset is tracked."""
        class Dataset1(BasePDDataset):
            def _load_samples(self):
                path = tmp_path / "d1.wav"
                waveform = torch.randn(1, 16000)
                torchaudio.save(str(path), waveform, 16000)
                return [{
                    'path': path,
                    'label': 0,
                    'subject_id': 's1',
                    'task': 'test'
                }]

        ds1 = Dataset1(root_dir=tmp_path)
        combined = CombinedPDDataset([ds1], balance_datasets=False)

        sample = combined[0]
        assert 'source_dataset' in sample
        assert sample['source_dataset'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
