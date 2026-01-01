"""model training and probing modules."""

from .classifier import (
    Wav2Vec2PDClassifier,
    DataCollatorWithPadding,
    PDClassifierTrainer,
    compute_metrics,
    create_training_args,
    evaluate_model_on_dataset
)

from .probes import (
    LinearProbe,
    LayerwiseProber,
    MultiFeatureProber,
    ControlTaskProber,
    compute_selectivity_score,
    permutation_test_probe,
    create_control_labels
)

from .cross_validation import (
    CVResults,
    CrossValidationTrainer,
    run_loso_cv,
    run_stratified_kfold_cv
)

__all__ = [
    # classifier
    'Wav2Vec2PDClassifier',
    'DataCollatorWithPadding',
    'PDClassifierTrainer',
    'compute_metrics',
    'create_training_args',
    'evaluate_model_on_dataset',
    # probes
    'LinearProbe',
    'LayerwiseProber',
    'MultiFeatureProber',
    'ControlTaskProber',
    'compute_selectivity_score',
    'permutation_test_probe',
    'create_control_labels',
    # cross validation
    'CVResults',
    'CrossValidationTrainer',
    'run_loso_cv',
    'run_stratified_kfold_cv'
]
