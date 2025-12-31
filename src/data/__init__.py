"""data loading and preprocessing modules for pd speech analysis."""

from .datasets import (
    BasePDDataset,
    ItalianPVSDataset,
    MDVRKCLDataset,
    ArkansasDataset,
    NeuroVozDataset,
    PCGITADataset,
    EWADBDataset,
    CombinedPDDataset,
    create_combined_dataset
)

from .preprocessing import (
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

__all__ = [
    'BasePDDataset',
    'ItalianPVSDataset',
    'MDVRKCLDataset',
    'ArkansasDataset',
    'NeuroVozDataset',
    'PCGITADataset',
    'EWADBDataset',
    'CombinedPDDataset',
    'create_combined_dataset',
    'load_audio',
    'normalize_audio',
    'remove_dc_offset',
    'apply_preemphasis',
    'detect_voice_activity',
    'remove_silence',
    'segment_audio',
    'apply_bandpass_filter',
    'apply_highpass_filter',
    'compute_snr',
    'check_audio_quality',
    'pad_or_truncate',
    'extract_fundamental_frequency',
    'compute_zcr',
    'apply_window_function',
    'AudioPreprocessor'
]
