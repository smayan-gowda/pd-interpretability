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
    PatchingResult,
    HeadImportanceRanking,
    ClinicalStratifiedPatcher,
    PathPatchingAnalyzer,
    get_activation_patching_hook,
    get_mean_ablation_hook,
    create_minimal_pairs,
    create_mfcc_matched_pairs,
    compute_mfcc_distance,
    compute_patching_importance,
    compute_causal_contribution
)

from .prediction_interface import (
    InterpretablePrediction,
    InterpretablePredictionInterface,
    FeatureContribution,
    create_interpretable_interface,
    CLINICAL_INTERPRETATIONS
)

__all__ = [
    # extraction
    'Wav2Vec2ActivationExtractor',
    'AttentionExtractor',
    'extract_activations_from_dataset',
    'load_activations_memmap',
    # patching core
    'ActivationPatcher',
    'AttentionHeadPatcher',
    'PatchingResult',
    'HeadImportanceRanking',
    # advanced patching
    'ClinicalStratifiedPatcher',
    'PathPatchingAnalyzer',
    # hook functions
    'get_activation_patching_hook',
    'get_mean_ablation_hook',
    # utilities
    'create_minimal_pairs',
    'create_mfcc_matched_pairs',
    'compute_mfcc_distance',
    'compute_patching_importance',
    'compute_causal_contribution',
    # interpretable prediction interface (phase 5 synthesis)
    'InterpretablePrediction',
    'InterpretablePredictionInterface',
    'FeatureContribution',
    'create_interpretable_interface',
    'CLINICAL_INTERPRETATIONS'
]
