"""
unit tests for audio preprocessing utilities.

tests vad, filtering, normalization, and quality checks.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import torchaudio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    load_audio,
    normalize_audio,
    remove_dc_offset,
    apply_preemphasis,
    detect_voice_activity,
    remove_silence,
    segment_audio,
    apply_bandpass_filter,
    apply_highpass_filter,
    compute_snr,
    check_audio_quality,
    pad_or_truncate,
    extract_fundamental_frequency,
    compute_zcr,
    apply_window_function,
    AudioPreprocessor
)


@pytest.fixture
def sample_audio():
    """create sample audio tensor."""
    sr = 16000
    duration = 2.0
    t = torch.linspace(0, duration, int(sr * duration))
    frequency = 440.0
    waveform = torch.sin(2 * np.pi * frequency * t).unsqueeze(0)
    return waveform, sr


@pytest.fixture
def temp_audio_file(sample_audio):
    """create temporary audio file."""
    waveform, sr = sample_audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        torchaudio.save(f.name, waveform, sr)
        yield Path(f.name)
        Path(f.name).unlink()


class TestAudioLoading:
    """test audio loading functions."""

    def test_load_audio_basic(self, temp_audio_file):
        """test basic audio loading."""
        waveform, sr = load_audio(temp_audio_file, target_sr=16000)

        assert isinstance(waveform, torch.Tensor)
        assert sr == 16000
        assert waveform.shape[0] == 1

    def test_load_audio_resampling(self, temp_audio_file):
        """test resampling during load."""
        waveform, sr = load_audio(temp_audio_file, target_sr=8000)

        assert sr == 8000

    def test_load_audio_mono_conversion(self):
        """test stereo to mono conversion."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sr = 16000
            stereo = torch.randn(2, sr * 2)
            torchaudio.save(f.name, stereo, sr)

            waveform, _ = load_audio(f.name, mono=True)

            assert waveform.shape[0] == 1

            Path(f.name).unlink()


class TestNormalization:
    """test normalization functions."""

    def test_normalize_audio(self):
        """test audio normalization."""
        waveform = torch.randn(1, 16000) * 10

        normalized = normalize_audio(waveform, target_level=-20.0)

        assert normalized.abs().max() <= 1.0

    def test_remove_dc_offset(self):
        """test dc offset removal."""
        waveform = torch.randn(1, 16000) + 0.5

        centered = remove_dc_offset(waveform)

        assert abs(centered.mean().item()) < 0.01

    def test_apply_preemphasis(self):
        """test preemphasis filter."""
        waveform = torch.randn(1, 16000)

        filtered = apply_preemphasis(waveform, coeff=0.97)

        assert filtered.shape == waveform.shape
        assert not torch.allclose(filtered, waveform)


class TestVoiceActivityDetection:
    """test voice activity detection."""

    def test_detect_voice_activity_signal(self):
        """test vad with clean signal."""
        sr = 16000
        duration = 1.0

        t = torch.linspace(0, duration, int(sr * duration))
        signal = torch.sin(2 * np.pi * 440 * t)

        vad_mask = detect_voice_activity(signal, sr, energy_threshold=0.001)

        assert vad_mask.sum() > 0

    def test_detect_voice_activity_silence(self):
        """test vad with silence."""
        sr = 16000
        silence = torch.zeros(sr * 1)

        vad_mask = detect_voice_activity(silence, sr)

        assert vad_mask.sum() == 0

    def test_remove_silence(self):
        """test silence removal."""
        sr = 16000

        silence = torch.zeros(1, sr)
        signal = torch.randn(1, sr)
        audio = torch.cat([silence, signal, silence], dim=1)

        trimmed = remove_silence(audio, sr, vad_threshold=0.001)

        assert trimmed.shape[1] < audio.shape[1]


class TestSegmentation:
    """test audio segmentation."""

    def test_segment_audio(self):
        """test audio segmentation into windows."""
        sr = 16000
        duration = 10.0
        waveform = torch.randn(1, int(sr * duration))

        segments = segment_audio(
            waveform, sr,
            segment_duration=3.0,
            overlap=0.5
        )

        assert len(segments) > 1

        for seg in segments:
            assert seg.shape[-1] <= int(sr * 3.0)

    def test_segment_audio_min_duration(self):
        """test minimum segment duration."""
        sr = 16000
        waveform = torch.randn(1, sr * 5)

        segments = segment_audio(
            waveform, sr,
            segment_duration=3.0,
            overlap=0.0,
            min_segment_duration=2.0
        )

        for seg in segments:
            assert seg.shape[-1] >= int(sr * 2.0)


class TestFiltering:
    """test filtering functions."""

    def test_bandpass_filter(self):
        """test bandpass filtering."""
        sr = 16000
        waveform = torch.randn(1, sr * 2)

        filtered = apply_bandpass_filter(
            waveform, sr,
            lowcut=80.0,
            highcut=8000.0
        )

        assert filtered.shape == waveform.shape

    def test_highpass_filter(self):
        """test highpass filtering."""
        sr = 16000
        waveform = torch.randn(1, sr * 2)

        filtered = apply_highpass_filter(waveform, sr, cutoff=80.0)

        assert filtered.shape == waveform.shape


class TestQualityMetrics:
    """test quality assessment functions."""

    def test_compute_snr_signal(self):
        """test snr computation with signal."""
        sr = 16000
        t = torch.linspace(0, 1, sr)
        signal = torch.sin(2 * np.pi * 440 * t)

        snr = compute_snr(signal, sr)

        assert snr > 0

    def test_compute_snr_noise(self):
        """test snr computation with noise."""
        sr = 16000
        noise = torch.randn(sr) * 0.01

        snr = compute_snr(noise, sr)

        assert isinstance(snr, float)

    def test_check_audio_quality_good(self):
        """test quality check with good audio."""
        sr = 16000
        t = torch.linspace(0, 1, sr)
        waveform = torch.sin(2 * np.pi * 440 * t) * 0.5

        passes, metrics = check_audio_quality(waveform, sr)

        assert 'snr_db' in metrics
        assert 'clipping_ratio' in metrics
        assert 'rms' in metrics

    def test_check_audio_quality_clipped(self):
        """test quality check with clipped audio."""
        sr = 16000
        waveform = torch.ones(sr)

        passes, metrics = check_audio_quality(waveform, sr)

        assert metrics['clipping_ratio'] > 0.9


class TestPadTruncate:
    """test padding and truncation."""

    def test_pad_or_truncate_pad(self):
        """test padding short audio."""
        waveform = torch.randn(1, 1000)
        target_length = 2000

        result = pad_or_truncate(waveform, target_length)

        assert result.shape[-1] == target_length

    def test_pad_or_truncate_truncate(self):
        """test truncating long audio."""
        waveform = torch.randn(1, 5000)
        target_length = 2000

        result = pad_or_truncate(waveform, target_length)

        assert result.shape[-1] == target_length

    def test_pad_or_truncate_exact(self):
        """test exact length audio."""
        waveform = torch.randn(1, 2000)
        target_length = 2000

        result = pad_or_truncate(waveform, target_length)

        assert result.shape[-1] == target_length
        assert torch.allclose(result, waveform)


class TestFeatureExtraction:
    """test feature extraction functions."""

    def test_extract_fundamental_frequency(self):
        """test f0 extraction."""
        sr = 16000
        duration = 1.0
        t = torch.linspace(0, duration, int(sr * duration))
        f0 = 220.0
        waveform = torch.sin(2 * np.pi * f0 * t)

        f0_contour = extract_fundamental_frequency(
            waveform, sr,
            min_f0=100.0,
            max_f0=400.0
        )

        assert len(f0_contour) > 0
        assert isinstance(f0_contour, np.ndarray)

    def test_compute_zcr(self):
        """test zero crossing rate computation."""
        waveform = torch.randn(16000)

        zcr = compute_zcr(waveform, frame_length=2048)

        assert len(zcr) > 0
        assert all(z >= 0 for z in zcr)

    def test_apply_window_function_hann(self):
        """test hann window application."""
        waveform = torch.ones(1000)

        windowed = apply_window_function(waveform, window_type='hann')

        assert windowed.shape == waveform.shape
        assert windowed[0] < waveform[0]
        assert windowed[-1] < waveform[-1]

    def test_apply_window_function_hamming(self):
        """test hamming window application."""
        waveform = torch.ones(1000)

        windowed = apply_window_function(waveform, window_type='hamming')

        assert windowed.shape == waveform.shape


class TestAudioPreprocessor:
    """test audio preprocessor pipeline."""

    def test_preprocessor_basic(self):
        """test basic preprocessing pipeline."""
        sr = 16000
        waveform = torch.randn(1, sr * 2)

        preprocessor = AudioPreprocessor(
            target_sr=16000,
            remove_silence=False,
            apply_filters=True,
            normalize=True
        )

        processed, metrics = preprocessor(waveform, sr)

        assert isinstance(processed, torch.Tensor)
        assert isinstance(metrics, dict)

    def test_preprocessor_quality_check(self):
        """test preprocessor with quality check."""
        sr = 16000
        t = torch.linspace(0, 2, sr * 2)
        waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)

        preprocessor = AudioPreprocessor(check_quality=True)

        processed, metrics = preprocessor(waveform, sr)

        assert 'passes_quality' in metrics
        assert 'snr_db' in metrics

    def test_preprocessor_resampling(self):
        """test preprocessor resampling."""
        sr = 44100
        waveform = torch.randn(1, sr * 2)

        preprocessor = AudioPreprocessor(target_sr=16000)

        processed, metrics = preprocessor(waveform, sr)

        expected_length = int(16000 * 2)
        assert abs(processed.shape[-1] - expected_length) < 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
