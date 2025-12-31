"""
audio preprocessing utilities for parkinson's disease speech analysis.

provides research-grade preprocessing functions including voice activity
detection, noise reduction, segmentation, and quality control.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Union
import warnings

import torch
import torchaudio
import numpy as np
from scipy import signal
from scipy.ndimage import binary_dilation


def load_audio(
    path: Union[str, Path],
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    load audio file with preprocessing.

    args:
        path: path to audio file
        target_sr: target sampling rate
        mono: convert to mono if true
        normalize: normalize amplitude to [-1, 1]

    returns:
        (waveform, sample_rate) where waveform is [channels, samples]
    """
    waveform, sr = torchaudio.load(path)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    if normalize:
        waveform = normalize_audio(waveform)

    return waveform, sr


def normalize_audio(waveform: torch.Tensor, target_level: float = -20.0) -> torch.Tensor:
    """
    normalize audio to target rms level in db.

    args:
        waveform: audio tensor [channels, samples]
        target_level: target rms level in db

    returns:
        normalized waveform
    """
    rms = torch.sqrt(torch.mean(waveform ** 2))

    if rms > 0:
        current_level_db = 20 * torch.log10(rms)
        gain_db = target_level - current_level_db
        gain = 10 ** (gain_db / 20)
        waveform = waveform * gain

    waveform = torch.clamp(waveform, -1.0, 1.0)

    return waveform


def remove_dc_offset(waveform: torch.Tensor) -> torch.Tensor:
    """
    remove dc offset by subtracting mean.

    args:
        waveform: audio tensor [channels, samples]

    returns:
        centered waveform
    """
    return waveform - waveform.mean(dim=-1, keepdim=True)


def apply_preemphasis(
    waveform: torch.Tensor,
    coeff: float = 0.97
) -> torch.Tensor:
    """
    apply preemphasis filter to enhance high frequencies.

    commonly used in speech processing to improve snr of higher formants.

    args:
        waveform: audio tensor [channels, samples]
        coeff: preemphasis coefficient (typically 0.95-0.97)

    returns:
        filtered waveform
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    output = torch.zeros_like(waveform)
    output[:, 0] = waveform[:, 0]
    output[:, 1:] = waveform[:, 1:] - coeff * waveform[:, :-1]

    return output.squeeze() if output.shape[0] == 1 else output


def detect_voice_activity(
    waveform: torch.Tensor,
    sr: int,
    frame_duration: float = 0.025,
    energy_threshold: float = 0.01,
    min_duration: float = 0.1
) -> torch.Tensor:
    """
    detect voice activity using energy-based vad.

    args:
        waveform: audio tensor [samples] or [channels, samples]
        sr: sampling rate
        frame_duration: frame length in seconds
        energy_threshold: energy threshold relative to max
        min_duration: minimum voiced segment duration in seconds

    returns:
        binary mask [samples] where 1 = voice, 0 = silence
    """
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    frame_length = int(frame_duration * sr)
    hop_length = frame_length // 2

    energy = torch.zeros(len(waveform))
    for i in range(0, len(waveform) - frame_length, hop_length):
        frame = waveform[i:i + frame_length]
        frame_energy = torch.sum(frame ** 2) / len(frame)

        energy[i:i + frame_length] = frame_energy

    threshold = energy.max() * energy_threshold
    vad_mask = (energy > threshold).float()

    min_samples = int(min_duration * sr)
    vad_mask = smooth_vad_mask(vad_mask.numpy(), min_samples)

    return torch.from_numpy(vad_mask)


def smooth_vad_mask(mask: np.ndarray, min_samples: int) -> np.ndarray:
    """
    smooth vad mask using morphological operations.

    removes isolated voiced/unvoiced segments shorter than min_samples.

    args:
        mask: binary vad mask
        min_samples: minimum segment length

    returns:
        smoothed mask
    """
    struct = np.ones(min_samples)
    mask = binary_dilation(mask, structure=struct)

    return mask.astype(np.float32)


def remove_silence(
    waveform: torch.Tensor,
    sr: int,
    vad_threshold: float = 0.01,
    padding: float = 0.1
) -> torch.Tensor:
    """
    remove silence from audio using vad.

    args:
        waveform: audio tensor [samples] or [channels, samples]
        sr: sampling rate
        vad_threshold: energy threshold for vad
        padding: seconds of padding to keep around voiced segments

    returns:
        trimmed waveform
    """
    is_stereo = waveform.dim() == 2

    if is_stereo:
        vad_input = waveform.mean(dim=0)
    else:
        vad_input = waveform

    vad_mask = detect_voice_activity(vad_input, sr, energy_threshold=vad_threshold)

    voiced_indices = torch.where(vad_mask > 0.5)[0]

    if len(voiced_indices) == 0:
        warnings.warn("no voice activity detected, returning original audio")
        return waveform

    start_idx = max(0, voiced_indices[0] - int(padding * sr))
    end_idx = min(len(vad_mask), voiced_indices[-1] + int(padding * sr))

    if is_stereo:
        return waveform[:, start_idx:end_idx]
    else:
        return waveform[start_idx:end_idx]


def segment_audio(
    waveform: torch.Tensor,
    sr: int,
    segment_duration: float = 3.0,
    overlap: float = 0.5,
    min_segment_duration: float = 1.0
) -> List[torch.Tensor]:
    """
    segment long audio into overlapping windows.

    useful for processing sustained vowels or long recordings.

    args:
        waveform: audio tensor [samples] or [channels, samples]
        sr: sampling rate
        segment_duration: target segment length in seconds
        overlap: overlap fraction (0-1)
        min_segment_duration: minimum segment duration to keep

    returns:
        list of audio segments
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    segment_samples = int(segment_duration * sr)
    hop_samples = int(segment_samples * (1 - overlap))
    min_samples = int(min_segment_duration * sr)

    segments = []
    total_samples = waveform.shape[1]

    for start in range(0, total_samples - min_samples, hop_samples):
        end = min(start + segment_samples, total_samples)
        segment = waveform[:, start:end]

        if segment.shape[1] >= min_samples:
            segments.append(segment.squeeze(0) if segment.shape[0] == 1 else segment)

        if end >= total_samples:
            break

    return segments


def apply_bandpass_filter(
    waveform: torch.Tensor,
    sr: int,
    lowcut: float = 80.0,
    highcut: float = 8000.0,
    order: int = 5
) -> torch.Tensor:
    """
    apply butterworth bandpass filter.

    removes frequencies outside typical speech range.

    args:
        waveform: audio tensor [samples] or [channels, samples]
        sr: sampling rate
        lowcut: lower cutoff frequency in hz
        highcut: upper cutoff frequency in hz
        order: filter order

    returns:
        filtered waveform
    """
    nyquist = sr / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = signal.butter(order, [low, high], btype='band')

    if waveform.dim() == 1:
        filtered = signal.filtfilt(b, a, waveform.numpy())
        return torch.from_numpy(filtered).float()
    else:
        filtered = np.zeros_like(waveform.numpy())
        for ch in range(waveform.shape[0]):
            filtered[ch] = signal.filtfilt(b, a, waveform[ch].numpy())
        return torch.from_numpy(filtered).float()


def apply_highpass_filter(
    waveform: torch.Tensor,
    sr: int,
    cutoff: float = 80.0,
    order: int = 5
) -> torch.Tensor:
    """
    apply highpass filter to remove low-frequency noise.

    args:
        waveform: audio tensor
        sr: sampling rate
        cutoff: cutoff frequency in hz
        order: filter order

    returns:
        filtered waveform
    """
    nyquist = sr / 2
    normal_cutoff = cutoff / nyquist

    b, a = signal.butter(order, normal_cutoff, btype='high')

    if waveform.dim() == 1:
        filtered = signal.filtfilt(b, a, waveform.numpy())
        return torch.from_numpy(filtered).float()
    else:
        filtered = np.zeros_like(waveform.numpy())
        for ch in range(waveform.shape[0]):
            filtered[ch] = signal.filtfilt(b, a, waveform[ch].numpy())
        return torch.from_numpy(filtered).float()


def compute_snr(
    waveform: torch.Tensor,
    sr: int,
    frame_duration: float = 0.025
) -> float:
    """
    estimate signal-to-noise ratio.

    uses energy difference between high and low energy frames.

    args:
        waveform: audio tensor [samples]
        sr: sampling rate
        frame_duration: frame length in seconds

    returns:
        estimated snr in db
    """
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    frame_length = int(frame_duration * sr)
    hop_length = frame_length // 2

    energies = []
    for i in range(0, len(waveform) - frame_length, hop_length):
        frame = waveform[i:i + frame_length]
        energy = torch.sum(frame ** 2) / len(frame)
        energies.append(energy.item())

    energies = sorted(energies)

    signal_energy = np.mean(energies[-len(energies)//4:])
    noise_energy = np.mean(energies[:len(energies)//4])

    if noise_energy > 0:
        snr = 10 * np.log10(signal_energy / noise_energy)
    else:
        snr = float('inf')

    return snr


def check_audio_quality(
    waveform: torch.Tensor,
    sr: int,
    min_snr: float = 5.0,
    max_clipping_ratio: float = 0.01
) -> Tuple[bool, Dict[str, float]]:
    """
    perform quality checks on audio.

    args:
        waveform: audio tensor
        sr: sampling rate
        min_snr: minimum acceptable snr in db
        max_clipping_ratio: maximum acceptable fraction of clipped samples

    returns:
        (passes_quality_check, metrics_dict)
    """
    if waveform.dim() == 2:
        waveform_mono = waveform.mean(dim=0)
    else:
        waveform_mono = waveform

    snr = compute_snr(waveform_mono, sr)

    clipped = torch.abs(waveform_mono) > 0.99
    clipping_ratio = clipped.float().mean().item()

    rms = torch.sqrt(torch.mean(waveform_mono ** 2)).item()

    metrics = {
        'snr_db': snr,
        'clipping_ratio': clipping_ratio,
        'rms': rms,
        'duration': len(waveform_mono) / sr
    }

    passes = (snr >= min_snr) and (clipping_ratio <= max_clipping_ratio)

    return passes, metrics


def pad_or_truncate(
    waveform: torch.Tensor,
    target_length: int,
    pad_mode: str = 'constant'
) -> torch.Tensor:
    """
    pad or truncate audio to exact length.

    args:
        waveform: audio tensor [samples] or [channels, samples]
        target_length: target number of samples
        pad_mode: padding mode ('constant', 'reflect', 'replicate')

    returns:
        processed waveform of length target_length
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    current_length = waveform.shape[1]

    if current_length > target_length:
        waveform = waveform[:, :target_length]
    elif current_length < target_length:
        padding = target_length - current_length
        if pad_mode == 'constant':
            waveform = torch.nn.functional.pad(waveform, (0, padding), value=0)
        elif pad_mode == 'reflect':
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='reflect')
        elif pad_mode == 'replicate':
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='replicate')

    return waveform.squeeze(0) if waveform.shape[0] == 1 else waveform


def extract_fundamental_frequency(
    waveform: torch.Tensor,
    sr: int,
    min_f0: float = 75.0,
    max_f0: float = 600.0
) -> np.ndarray:
    """
    extract fundamental frequency contour using autocorrelation.

    args:
        waveform: audio tensor [samples]
        sr: sampling rate
        min_f0: minimum f0 in hz
        max_f0: maximum f0 in hz

    returns:
        f0 contour as numpy array
    """
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    waveform_np = waveform.numpy()

    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    min_lag = int(sr / max_f0)
    max_lag = int(sr / min_f0)

    f0_contour = []

    for i in range(0, len(waveform_np) - frame_length, hop_length):
        frame = waveform_np[i:i + frame_length]

        frame = frame - np.mean(frame)

        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]

        corr = corr[min_lag:max_lag]

        if len(corr) > 0:
            peak_idx = np.argmax(corr) + min_lag
            f0 = sr / peak_idx
            f0_contour.append(f0)
        else:
            f0_contour.append(0.0)

    return np.array(f0_contour)


def compute_zcr(waveform: torch.Tensor, frame_length: int = 2048) -> np.ndarray:
    """
    compute zero crossing rate over time.

    args:
        waveform: audio tensor [samples]
        frame_length: frame size

    returns:
        zcr values per frame
    """
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    waveform_np = waveform.numpy()
    hop_length = frame_length // 2

    zcr = []
    for i in range(0, len(waveform_np) - frame_length, hop_length):
        frame = waveform_np[i:i + frame_length]
        zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
        zcr.append(zero_crossings / len(frame))

    return np.array(zcr)


def apply_window_function(
    waveform: torch.Tensor,
    window_type: str = 'hann'
) -> torch.Tensor:
    """
    apply window function to reduce spectral leakage.

    args:
        waveform: audio tensor [samples]
        window_type: 'hann', 'hamming', or 'blackman'

    returns:
        windowed waveform
    """
    if waveform.dim() == 2:
        raise ValueError("waveform must be 1d for windowing")

    n_samples = len(waveform)

    if window_type == 'hann':
        window = torch.hann_window(n_samples)
    elif window_type == 'hamming':
        window = torch.hamming_window(n_samples)
    elif window_type == 'blackman':
        window = torch.blackman_window(n_samples)
    else:
        raise ValueError(f"unknown window type: {window_type}")

    return waveform * window


class AudioPreprocessor:
    """
    audio preprocessing pipeline for consistent processing.

    applies sequence of preprocessing steps with configurable parameters.
    """

    def __init__(
        self,
        target_sr: int = 16000,
        remove_silence: bool = True,
        apply_filters: bool = True,
        normalize: bool = True,
        check_quality: bool = True
    ):
        """
        args:
            target_sr: target sampling rate
            remove_silence: whether to apply vad and remove silence
            apply_filters: whether to apply bandpass filtering
            normalize: whether to normalize amplitude
            check_quality: whether to check audio quality
        """
        self.target_sr = target_sr
        self.remove_silence_flag = remove_silence
        self.apply_filters_flag = apply_filters
        self.normalize_flag = normalize
        self.check_quality_flag = check_quality

    def __call__(
        self,
        waveform: torch.Tensor,
        sr: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        apply preprocessing pipeline.

        args:
            waveform: input audio
            sr: input sampling rate

        returns:
            (processed_waveform, quality_metrics)
        """
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        if self.apply_filters_flag:
            waveform = apply_bandpass_filter(waveform, self.target_sr)

        waveform = remove_dc_offset(waveform)

        if self.remove_silence_flag:
            waveform = remove_silence(waveform, self.target_sr)

        if self.normalize_flag:
            waveform = normalize_audio(waveform)

        quality_metrics = {}
        if self.check_quality_flag:
            passes, metrics = check_audio_quality(waveform, self.target_sr)
            quality_metrics = metrics
            quality_metrics['passes_quality'] = passes

        return waveform, quality_metrics
