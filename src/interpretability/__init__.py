"""mechanistic interpretability tools for wav2vec2."""

from .extraction import (
    Wav2Vec2ActivationExtractor,
    AttentionExtractor,
    extract_activations_from_dataset,
    load_activations_memmap
)

from .patching import (
    ActivationPatcher,
    AttentionHeadPatcher,
    create_minimal_pairs,
    create_mfcc_matched_pairs,
    compute_mfcc_distance,
    compute_patching_importance,
    compute_causal_contribution
)

__all__ = [
    'Wav2Vec2ActivationExtractor',
    'AttentionExtractor',
    'extract_activations_from_dataset',
    'load_activations_memmap',
    'ActivationPatcher',
    'AttentionHeadPatcher',
    'create_minimal_pairs',
    'create_mfcc_matched_pairs',
    'compute_mfcc_distance',
    'compute_patching_importance',
    'compute_causal_contribution'
]
