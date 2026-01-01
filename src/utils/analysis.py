"""
statistical analysis utilities for mechanistic interpretability.

provides research-grade statistical testing, effect size calculations,
and confidence intervals for probing and patching experiments.
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, mannwhitneyu, wilcoxon,
    pearsonr, spearmanr, bootstrap
)


def compute_effect_size_cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled: bool = True
) -> float:
    """
    compute cohen's d effect size between two groups.

    args:
        group1: first group values
        group2: second group values
        pooled: whether to use pooled standard deviation

    returns:
        cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    if pooled:
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std
    else:
        d = (mean1 - mean2) / np.std(group2, ddof=1)
    
    return d


def compute_effect_size_glass_delta(
    experimental: np.ndarray,
    control: np.ndarray
) -> float:
    """
    compute glass's delta effect size.

    uses control group std as denominator.

    args:
        experimental: experimental group values
        control: control group values

    returns:
        glass's delta
    """
    return (np.mean(experimental) - np.mean(control)) / np.std(control, ddof=1)


def compute_effect_size_hedges_g(
    group1: np.ndarray,
    group2: np.ndarray
) -> float:
    """
    compute hedges' g (bias-corrected cohen's d).

    args:
        group1: first group values
        group2: second group values

    returns:
        hedges' g
    """
    d = compute_effect_size_cohens_d(group1, group2)
    n = len(group1) + len(group2)
    
    # correction factor for small samples
    j = 1 - (3 / (4 * n - 9))
    
    return d * j


def interpret_effect_size(d: float) -> str:
    """
    interpret cohen's d effect size.

    args:
        d: effect size value

    returns:
        interpretation string
    """
    d_abs = abs(d)
    
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    compute bootstrap confidence interval for a statistic.

    args:
        data: input data
        statistic: function to compute statistic
        n_bootstrap: number of bootstrap samples
        confidence_level: confidence level (e.g., 0.95)
        random_state: random seed

    returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(random_state)
    
    point_estimate = statistic(data)
    
    bootstrap_estimates = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_estimates.append(statistic(sample))
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    
    return point_estimate, lower, upper


def paired_comparison_test(
    values1: np.ndarray,
    values2: np.ndarray,
    test: str = 'auto',
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    perform paired comparison statistical test.

    args:
        values1: first set of paired values
        values2: second set of paired values
        test: 'ttest', 'wilcoxon', or 'auto'
        alternative: 'two-sided', 'less', or 'greater'

    returns:
        dict with statistic, p_value, effect_size
    """
    if len(values1) != len(values2):
        raise ValueError("paired values must have same length")
    
    differences = values1 - values2
    
    # check normality for auto test selection
    if test == 'auto':
        if len(differences) >= 20:
            _, normality_p = stats.shapiro(differences)
            test = 'ttest' if normality_p > 0.05 else 'wilcoxon'
        else:
            test = 'wilcoxon'  # safer for small samples
    
    if test == 'ttest':
        stat, p_value = ttest_rel(values1, values2, alternative=alternative)
    elif test == 'wilcoxon':
        stat, p_value = wilcoxon(values1, values2, alternative=alternative)
    else:
        raise ValueError(f"unknown test: {test}")
    
    # compute effect size (paired cohen's d)
    effect_size = np.mean(differences) / np.std(differences, ddof=1)
    
    return {
        'test': test,
        'statistic': stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_interpretation': interpret_effect_size(effect_size)
    }


def independent_comparison_test(
    group1: np.ndarray,
    group2: np.ndarray,
    test: str = 'auto',
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    perform independent samples comparison test.

    args:
        group1: first group values
        group2: second group values
        test: 'ttest', 'mannwhitney', or 'auto'
        alternative: 'two-sided', 'less', or 'greater'

    returns:
        dict with statistic, p_value, effect_size
    """
    # check normality for auto test selection
    if test == 'auto':
        if len(group1) >= 20 and len(group2) >= 20:
            _, p1 = stats.shapiro(group1)
            _, p2 = stats.shapiro(group2)
            test = 'ttest' if (p1 > 0.05 and p2 > 0.05) else 'mannwhitney'
        else:
            test = 'mannwhitney'
    
    if test == 'ttest':
        stat, p_value = ttest_ind(group1, group2, alternative=alternative)
    elif test == 'mannwhitney':
        stat, p_value = mannwhitneyu(group1, group2, alternative=alternative)
    else:
        raise ValueError(f"unknown test: {test}")
    
    effect_size = compute_effect_size_cohens_d(group1, group2)
    
    return {
        'test': test,
        'statistic': stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_interpretation': interpret_effect_size(effect_size)
    }


def compute_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'pearson'
) -> Dict[str, float]:
    """
    compute correlation with confidence interval.

    args:
        x: first variable
        y: second variable
        method: 'pearson' or 'spearman'

    returns:
        dict with r, p_value, ci_lower, ci_upper
    """
    if method == 'pearson':
        r, p_value = pearsonr(x, y)
    elif method == 'spearman':
        r, p_value = spearmanr(x, y)
    else:
        raise ValueError(f"unknown method: {method}")
    
    # fisher z-transform for ci
    n = len(x)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    
    ci_lower = np.tanh(z - 1.96 * se)
    ci_upper = np.tanh(z + 1.96 * se)
    
    return {
        'r': r,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'method': method
    }


def multiple_comparison_correction(
    p_values: List[float],
    method: str = 'bonferroni',
    alpha: float = 0.05
) -> Dict[str, Union[List[float], List[bool]]]:
    """
    apply multiple comparison correction.

    args:
        p_values: list of p-values
        method: 'bonferroni', 'holm', 'fdr_bh' (benjamini-hochberg)
        alpha: significance level

    returns:
        dict with adjusted_p_values, significant (bool list)
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    if method == 'bonferroni':
        adjusted = np.minimum(p_values * n, 1.0)
        
    elif method == 'holm':
        sorted_indices = np.argsort(p_values)
        adjusted = np.zeros(n)
        
        for i, idx in enumerate(sorted_indices):
            adjusted[idx] = min(p_values[idx] * (n - i), 1.0)
        
        # ensure monotonicity
        adjusted = np.maximum.accumulate(adjusted[sorted_indices])
        final_adjusted = np.zeros(n)
        for i, idx in enumerate(sorted_indices):
            final_adjusted[idx] = adjusted[i]
        adjusted = final_adjusted
        
    elif method == 'fdr_bh':
        sorted_indices = np.argsort(p_values)[::-1]
        adjusted = np.zeros(n)
        
        for i, idx in enumerate(sorted_indices):
            rank = n - i
            adjusted[idx] = min(p_values[idx] * n / rank, 1.0)
        
        # ensure monotonicity (in reverse)
        cummin = adjusted[sorted_indices[0]]
        for i, idx in enumerate(sorted_indices):
            cummin = min(cummin, adjusted[idx])
            adjusted[idx] = cummin
    else:
        raise ValueError(f"unknown method: {method}")
    
    significant = adjusted < alpha
    
    return {
        'original_p_values': p_values.tolist(),
        'adjusted_p_values': adjusted.tolist(),
        'significant': significant.tolist(),
        'method': method,
        'alpha': alpha
    }


def compare_layers_probing(
    layer_results: Dict[int, Dict[str, float]],
    baseline_layer: int = 0
) -> Dict[int, Dict[str, float]]:
    """
    statistically compare probing accuracy across layers.

    args:
        layer_results: dict mapping layer to cv results with 'scores' key
        baseline_layer: layer to compare against (usually first)

    returns:
        dict mapping layer to comparison stats
    """
    baseline_scores = np.array(layer_results[baseline_layer]['scores'])
    
    comparisons = {}
    
    for layer_idx, results in layer_results.items():
        if layer_idx == baseline_layer:
            continue
        
        layer_scores = np.array(results['scores'])
        
        # paired test since same cv folds
        comparison = paired_comparison_test(layer_scores, baseline_scores)
        comparison['layer'] = layer_idx
        comparison['mean_difference'] = np.mean(layer_scores) - np.mean(baseline_scores)
        
        comparisons[layer_idx] = comparison
    
    # apply multiple comparison correction
    p_values = [comparisons[l]['p_value'] for l in comparisons]
    if p_values:
        correction = multiple_comparison_correction(p_values, method='holm')
        
        for i, layer_idx in enumerate(comparisons):
            comparisons[layer_idx]['adjusted_p_value'] = correction['adjusted_p_values'][i]
            comparisons[layer_idx]['significant_corrected'] = correction['significant'][i]
    
    return comparisons


def analyze_patching_results(
    patching_results: Dict[int, Dict[str, float]],
    significance_threshold: float = 0.5
) -> Dict[str, Union[List[int], Dict]]:
    """
    analyze activation patching results statistically.

    args:
        patching_results: results from batch patching
        significance_threshold: recovery threshold for importance

    returns:
        analysis dict with important layers, statistics
    """
    layers = sorted(patching_results.keys())
    recoveries = [patching_results[l]['mean_recovery'] for l in layers]
    stds = [patching_results[l]['std_recovery'] for l in layers]
    
    # identify important layers
    important_layers = [l for l, r in zip(layers, recoveries) if r > significance_threshold]
    
    # compute summary statistics
    recoveries_arr = np.array(recoveries)
    
    analysis = {
        'important_layers': important_layers,
        'n_important': len(important_layers),
        'total_layers': len(layers),
        'importance_ratio': len(important_layers) / len(layers),
        'mean_recovery': float(np.mean(recoveries_arr)),
        'max_recovery': float(np.max(recoveries_arr)),
        'best_layer': int(layers[np.argmax(recoveries_arr)]),
        'recovery_std': float(np.std(recoveries_arr)),
        'recovery_range': float(np.max(recoveries_arr) - np.min(recoveries_arr))
    }
    
    # test if recovery is significantly above 0
    if len(recoveries) >= 3:
        t_stat, p_value = ttest_ind(recoveries, [0] * len(recoveries))
        analysis['above_zero_test'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value)
        }
    
    return analysis


def compute_clinical_alignment_statistics(
    probe_predictions: np.ndarray,
    clinical_values: np.ndarray,
    clinical_name: str
) -> Dict[str, float]:
    """
    compute detailed statistics for clinical feature alignment.

    args:
        probe_predictions: model predictions or probabilities
        clinical_values: ground truth clinical values
        clinical_name: name of clinical feature

    returns:
        comprehensive statistics dict
    """
    # remove nans
    valid_mask = ~np.isnan(clinical_values) & ~np.isnan(probe_predictions)
    preds = probe_predictions[valid_mask]
    clinical = clinical_values[valid_mask]
    
    if len(preds) < 10:
        warnings.warn(f"too few valid samples for {clinical_name}")
        return {'error': 'insufficient_samples', 'n_valid': len(preds)}
    
    # correlations
    pearson_result = compute_correlation(preds, clinical, 'pearson')
    spearman_result = compute_correlation(preds, clinical, 'spearman')
    
    # bootstrap ci for pearson r
    def corr_stat(data):
        x, y = data[:len(data)//2], data[len(data)//2:]
        r, _ = pearsonr(x, y)
        return r
    
    combined = np.concatenate([preds, clinical])
    _, r_ci_low, r_ci_high = bootstrap_confidence_interval(
        combined, statistic=corr_stat, n_bootstrap=1000
    )
    
    return {
        'clinical_feature': clinical_name,
        'n_samples': len(preds),
        'pearson_r': pearson_result['r'],
        'pearson_p': pearson_result['p_value'],
        'pearson_ci': (pearson_result['ci_lower'], pearson_result['ci_upper']),
        'spearman_r': spearman_result['r'],
        'spearman_p': spearman_result['p_value'],
        'prediction_mean': float(np.mean(preds)),
        'prediction_std': float(np.std(preds)),
        'clinical_mean': float(np.mean(clinical)),
        'clinical_std': float(np.std(clinical))
    }


def generate_results_summary(
    probing_results: Optional[Dict] = None,
    patching_results: Optional[Dict] = None,
    classification_metrics: Optional[Dict] = None
) -> Dict:
    """
    generate comprehensive results summary for paper.

    args:
        probing_results: layer-wise probing results
        patching_results: activation patching results
        classification_metrics: classifier performance metrics

    returns:
        formatted summary dict
    """
    summary = {
        'analysis_complete': True,
        'sections': {}
    }
    
    if probing_results:
        layers = sorted(probing_results.keys())
        accs = [probing_results[l]['mean'] for l in layers]
        best_layer = layers[np.argmax(accs)]
        
        summary['sections']['probing'] = {
            'best_layer': int(best_layer),
            'best_accuracy': float(max(accs)),
            'worst_accuracy': float(min(accs)),
            'improvement_over_baseline': float(max(accs) - accs[0]),
            'layer_comparisons': compare_layers_probing(probing_results)
        }
    
    if patching_results:
        summary['sections']['patching'] = analyze_patching_results(patching_results)
    
    if classification_metrics:
        summary['sections']['classification'] = {
            'accuracy': classification_metrics.get('accuracy'),
            'precision': classification_metrics.get('precision'),
            'recall': classification_metrics.get('recall'),
            'f1': classification_metrics.get('f1'),
            'auc': classification_metrics.get('auc')
        }
    
    return summary
