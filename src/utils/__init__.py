"""
utils subpackage for visualization and analysis.

exports visualization and statistical analysis utilities.
"""

from .visualization import (
    plot_layerwise_probing,
    plot_clinical_feature_heatmap,
    plot_patching_results,
    plot_clinical_comparison,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_attention_heatmap,
    create_results_dashboard
)

from .analysis import (
    compute_effect_size_cohens_d,
    compute_effect_size_hedges_g,
    compute_effect_size_glass_delta,
    interpret_effect_size,
    bootstrap_confidence_interval,
    paired_comparison_test,
    independent_comparison_test,
    compute_correlation,
    multiple_comparison_correction,
    compare_layers_probing,
    analyze_patching_results,
    compute_clinical_alignment_statistics,
    generate_results_summary
)

__all__ = [
    # visualization
    'plot_layerwise_probing',
    'plot_clinical_feature_heatmap',
    'plot_patching_results',
    'plot_clinical_comparison',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_attention_heatmap',
    'create_results_dashboard',
    # analysis
    'compute_effect_size_cohens_d',
    'compute_effect_size_hedges_g',
    'compute_effect_size_glass_delta',
    'interpret_effect_size',
    'bootstrap_confidence_interval',
    'paired_comparison_test',
    'independent_comparison_test',
    'compute_correlation',
    'multiple_comparison_correction',
    'compare_layers_probing',
    'analyze_patching_results',
    'compute_clinical_alignment_statistics',
    'generate_results_summary',
    # experiment tracking
    'ExperimentConfig',
    'MetricsLogger',
    'ExperimentTracker',
    'create_experiment_id',
    'load_experiment',
    'list_experiments'
]

# experiment tracking
from .experiment import (
    ExperimentConfig,
    MetricsLogger,
    ExperimentTracker,
    create_experiment_id,
    load_experiment,
    list_experiments
)
