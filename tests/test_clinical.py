"""
unit tests for clinical feature extraction.

tests parselmouth-based extraction of jitter, shimmer, hnr, and formants.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import torch
import torchaudio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.clinical import (
    ClinicalFeatureExtractor,
    extract_clinical_features,
    batch_extract_features,
    get_clinical_feature_names,
    get_pd_discriminative_features,
    create_binary_clinical_labels,
    compute_clinical_alignment_score
)


@pytest.fixture
def synthetic_vowel_audio():
    """create synthetic vowel-like audio for testing."""
    sr = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sr * duration))

    f0 = 150.0
    fundamental = torch.sin(2 * np.pi * f0 * t)

    harmonics = (
        0.5 * torch.sin(2 * np.pi * 2 * f0 * t) +
        0.3 * torch.sin(2 * np.pi * 3 * f0 * t) +
        0.2 * torch.sin(2 * np.pi * 4 * f0 * t)
    )

    waveform = (fundamental + harmonics) * 0.5

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        torchaudio.save(f.name, waveform.unsqueeze(0), sr)
        yield Path(f.name)
        Path(f.name).unlink()


class TestClinicalFeatureExtractor:
    """test clinical feature extractor class."""

    def test_initialization(self):
        """test extractor initialization."""
        extractor = ClinicalFeatureExtractor(
            f0_min=75.0,
            f0_max=600.0
        )

        assert extractor.f0_min == 75.0
        assert extractor.f0_max == 600.0

    def test_extract_features(self, synthetic_vowel_audio):
        """test feature extraction from audio file."""
        extractor = ClinicalFeatureExtractor()

        features = extractor.extract(synthetic_vowel_audio)

        assert isinstance(features, dict)
        assert len(features) > 0

    def test_pitch_features(self, synthetic_vowel_audio):
        """test pitch feature extraction."""
        extractor = ClinicalFeatureExtractor(f0_min=100.0, f0_max=300.0)

        features = extractor.extract(synthetic_vowel_audio)

        assert 'f0_mean' in features
        assert 'f0_std' in features
        assert 'f0_min' in features
        assert 'f0_max' in features
        assert 'f0_median' in features
        assert 'f0_range' in features

        if not np.isnan(features['f0_mean']):
            assert features['f0_mean'] > 0

    def test_jitter_features(self, synthetic_vowel_audio):
        """test jitter feature extraction."""
        extractor = ClinicalFeatureExtractor()

        features = extractor.extract(synthetic_vowel_audio)

        assert 'jitter_local' in features
        assert 'jitter_rap' in features
        assert 'jitter_ppq5' in features
        assert 'jitter_ddp' in features

    def test_shimmer_features(self, synthetic_vowel_audio):
        """test shimmer feature extraction."""
        extractor = ClinicalFeatureExtractor()

        features = extractor.extract(synthetic_vowel_audio)

        assert 'shimmer_local' in features
        assert 'shimmer_apq3' in features
        assert 'shimmer_apq5' in features
        assert 'shimmer_apq11' in features
        assert 'shimmer_dda' in features

    def test_hnr_features(self, synthetic_vowel_audio):
        """test hnr feature extraction."""
        extractor = ClinicalFeatureExtractor()

        features = extractor.extract(synthetic_vowel_audio)

        assert 'hnr_mean' in features
        assert 'hnr_std' in features

    def test_formant_features(self, synthetic_vowel_audio):
        """test formant feature extraction."""
        extractor = ClinicalFeatureExtractor()

        features = extractor.extract(synthetic_vowel_audio)

        assert 'f1_mean' in features
        assert 'f1_std' in features
        assert 'f2_mean' in features
        assert 'f2_std' in features
        assert 'f3_mean' in features
        assert 'f3_std' in features
        assert 'f4_mean' in features
        assert 'f4_std' in features

    def test_duration_features(self, synthetic_vowel_audio):
        """test duration feature extraction."""
        extractor = ClinicalFeatureExtractor()

        features = extractor.extract(synthetic_vowel_audio)

        assert 'total_duration' in features
        assert 'voiced_duration' in features
        assert 'unvoiced_duration' in features

        if not np.isnan(features['total_duration']):
            assert features['total_duration'] > 0

    def test_nan_handling(self):
        """test handling of problematic audio."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sr = 16000
            silence = torch.zeros(1, sr)
            torchaudio.save(f.name, silence, sr)

            extractor = ClinicalFeatureExtractor()

            features = extractor.extract(f.name)

            assert isinstance(features, dict)

            Path(f.name).unlink()


class TestConvenienceFunctions:
    """test convenience functions."""

    def test_extract_clinical_features(self, synthetic_vowel_audio):
        """test convenience function."""
        features = extract_clinical_features(
            synthetic_vowel_audio,
            f0_min=75.0,
            f0_max=600.0
        )

        assert isinstance(features, dict)
        assert len(features) > 0

    def test_batch_extract_features(self):
        """test batch extraction."""
        audio_files = []

        for i in range(3):
            sr = 16000
            t = torch.linspace(0, 1, sr)
            f0 = 150.0 + i * 50.0
            waveform = torch.sin(2 * np.pi * f0 * t).unsqueeze(0)

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                torchaudio.save(f.name, waveform, sr)
                audio_files.append(Path(f.name))

        features_list = batch_extract_features(
            audio_files,
            verbose=False
        )

        assert len(features_list) == 3

        for features in features_list:
            if features is not None:
                assert isinstance(features, dict)

        for f in audio_files:
            f.unlink()

    def test_get_clinical_feature_names(self):
        """test getting feature names."""
        names = get_clinical_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0
        assert 'jitter_local' in names
        assert 'shimmer_local' in names
        assert 'hnr_mean' in names
        assert 'f0_mean' in names

    def test_get_pd_discriminative_features(self):
        """test getting pd discriminative features."""
        features = get_pd_discriminative_features()

        assert isinstance(features, list)
        assert len(features) > 0
        assert all(f in get_clinical_feature_names() for f in features)


class TestBinaryLabels:
    """test binary label creation."""

    def test_create_binary_labels_default_thresholds(self):
        """test binary label creation with default thresholds."""
        features = {
            'jitter_local': 0.015,
            'shimmer_local': 0.040,
            'hnr_mean': 12.0,
            'f0_std': 8.0
        }

        binary_labels = create_binary_clinical_labels(features)

        assert isinstance(binary_labels, dict)
        assert 'jitter_local_binary' in binary_labels
        assert 'shimmer_local_binary' in binary_labels
        assert 'hnr_binary' in binary_labels

    def test_create_binary_labels_custom_thresholds(self):
        """test binary labels with custom thresholds."""
        features = {
            'jitter_local': 0.008,
            'shimmer_local': 0.025
        }

        thresholds = {
            'jitter_local': 0.01,
            'shimmer_local': 0.03
        }

        binary_labels = create_binary_clinical_labels(features, thresholds)

        assert binary_labels['jitter_local_binary'] == 0
        assert binary_labels['shimmer_local_binary'] == 0

    def test_create_binary_labels_abnormal(self):
        """test binary labels for abnormal values."""
        features = {
            'jitter_local': 0.025,
            'shimmer_local': 0.050,
            'hnr_mean': 10.0
        }

        binary_labels = create_binary_clinical_labels(features)

        assert binary_labels['jitter_local_binary'] == 1
        assert binary_labels['shimmer_local_binary'] == 1
        assert binary_labels['hnr_binary'] == 1

    def test_create_binary_labels_nan_handling(self):
        """test binary labels with nan values."""
        features = {
            'jitter_local': np.nan,
            'shimmer_local': 0.040
        }

        binary_labels = create_binary_clinical_labels(features)

        assert 'jitter_local_binary' not in binary_labels
        assert 'shimmer_local_binary' in binary_labels


class TestClinicalAlignmentScore:
    """test clinical alignment score computation."""

    def test_compute_alignment_score_basic(self):
        """test basic alignment score computation."""
        n_samples = 50
        n_dims = 128
        n_features = 5

        model_features = np.random.randn(n_samples, n_dims)

        clinical_features = np.random.randn(n_samples, n_features)

        feature_names = ['jitter', 'shimmer', 'hnr', 'f0_mean', 'f0_std']

        score = compute_clinical_alignment_score(
            model_features,
            clinical_features,
            feature_names
        )

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_compute_alignment_score_with_nans(self):
        """test alignment score with missing values."""
        n_samples = 50
        n_dims = 128
        n_features = 3

        model_features = np.random.randn(n_samples, n_dims)

        clinical_features = np.random.randn(n_samples, n_features)
        clinical_features[:10, 0] = np.nan

        feature_names = ['jitter', 'shimmer', 'hnr']

        score = compute_clinical_alignment_score(
            model_features,
            clinical_features,
            feature_names
        )

        assert isinstance(score, float)

    def test_compute_alignment_score_correlated(self):
        """test alignment score with correlated features."""
        n_samples = 100
        n_dims = 10

        base_signal = np.random.randn(n_samples, 1)

        model_features = np.concatenate([
            base_signal,
            np.random.randn(n_samples, n_dims - 1)
        ], axis=1)

        clinical_features = base_signal + 0.1 * np.random.randn(n_samples, 1)

        score = compute_clinical_alignment_score(
            model_features,
            clinical_features,
            ['feature1']
        )

        assert score > 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
