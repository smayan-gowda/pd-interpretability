"""
results aggregation and reporting for mechanistic interpretability.

provides comprehensive tools for aggregating results from probing,
patching, and cross-dataset experiments into publication-ready reports.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ExperimentMetadata:
    """metadata for an experiment run."""
    
    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset: str = ""
    model_checkpoint: str = ""
    random_seed: int = 42
    git_hash: str = ""
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProbingResult:
    """result from a single probing experiment."""
    
    layer: int
    mean_score: float
    std_score: float
    n_samples: int
    n_folds: int
    all_scores: List[float] = field(default_factory=list)
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PatchingResult:
    """result from activation patching experiment."""
    
    layer: int
    mean_recovery: float
    std_recovery: float
    n_pairs: int
    all_recoveries: List[float] = field(default_factory=list)
    head_results: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ClinicalProbingResult:
    """result from clinical feature probing."""
    
    feature_name: str
    layer: int
    r2_score: float
    std_score: float
    correlation: Optional[float] = None
    p_value: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ResultsAggregator:
    """
    aggregate and analyze results from multiple experiments.
    
    provides unified interface for collecting probing, patching,
    and cross-dataset results.
    """
    
    def __init__(self, experiment_name: str, output_dir: Optional[str] = None):
        """
        args:
            experiment_name: name for this experiment set
            output_dir: directory for saving results
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) if output_dir else Path('results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata = ExperimentMetadata(experiment_name=experiment_name)
        
        self.probing_results: Dict[int, ProbingResult] = {}
        self.patching_results: Dict[int, PatchingResult] = {}
        self.clinical_results: Dict[str, Dict[int, ClinicalProbingResult]] = {}
        self.cross_dataset_results: Dict = {}
        
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_probing_results(
        self,
        results: Dict[int, Dict[str, float]],
        compute_significance: bool = True
    ):
        """
        add layer-wise probing results.
        
        args:
            results: dict mapping layer to {mean, std, scores}
            compute_significance: whether to compute p-values
        """
        for layer, data in results.items():
            scores = data.get('scores', [])
            mean_score = data.get('mean', np.mean(scores) if scores else 0)
            std_score = data.get('std', np.std(scores) if scores else 0)
            
            p_value = None
            effect_size = None
            
            if compute_significance and scores:
                t_stat, p_value = stats.ttest_1samp(scores, 0.5)
                effect_size = (mean_score - 0.5) / std_score if std_score > 0 else 0
            
            self.probing_results[layer] = ProbingResult(
                layer=layer,
                mean_score=mean_score,
                std_score=std_score,
                n_samples=len(scores),
                n_folds=len(scores),
                all_scores=scores,
                p_value=p_value,
                effect_size=effect_size
            )
    
    def add_patching_results(
        self,
        results: Dict[int, Dict[str, float]]
    ):
        """
        add layer-wise patching results.
        
        args:
            results: dict mapping layer to {mean_recovery, std_recovery, ...}
        """
        for layer, data in results.items():
            if isinstance(layer, str) and not layer.isdigit():
                continue
            
            layer_idx = int(layer) if isinstance(layer, str) else layer
            
            self.patching_results[layer_idx] = PatchingResult(
                layer=layer_idx,
                mean_recovery=data.get('mean_recovery', 0),
                std_recovery=data.get('std_recovery', 0),
                n_pairs=data.get('n_pairs', 0),
                all_recoveries=data.get('recoveries', [])
            )
    
    def add_clinical_probing_results(
        self,
        results: Dict[str, Dict[int, Dict[str, float]]]
    ):
        """
        add clinical feature probing results.
        
        args:
            results: nested dict feature_name -> layer -> {mean, std}
        """
        for feature_name, layer_results in results.items():
            self.clinical_results[feature_name] = {}
            
            for layer, data in layer_results.items():
                self.clinical_results[feature_name][layer] = ClinicalProbingResult(
                    feature_name=feature_name,
                    layer=layer,
                    r2_score=data.get('mean', 0),
                    std_score=data.get('std', 0)
                )
    
    def add_cross_dataset_results(
        self,
        results: Dict
    ):
        """add cross-dataset evaluation results."""
        self.cross_dataset_results = results
    
    def get_best_probing_layer(self) -> Tuple[int, float]:
        """get layer with highest probing accuracy."""
        if not self.probing_results:
            raise ValueError("no probing results available")
        
        best_layer = max(
            self.probing_results.keys(),
            key=lambda x: self.probing_results[x].mean_score
        )
        return best_layer, self.probing_results[best_layer].mean_score
    
    def get_best_patching_layer(self) -> Tuple[int, float]:
        """get layer with highest patching recovery."""
        if not self.patching_results:
            raise ValueError("no patching results available")
        
        best_layer = max(
            self.patching_results.keys(),
            key=lambda x: self.patching_results[x].mean_recovery
        )
        return best_layer, self.patching_results[best_layer].mean_recovery
    
    def get_important_layers(
        self,
        probing_threshold: float = 0.6,
        patching_threshold: float = 0.1
    ) -> Dict[str, List[int]]:
        """
        identify important layers from both probing and patching.
        
        args:
            probing_threshold: minimum probing accuracy
            patching_threshold: minimum patching recovery
            
        returns:
            dict with 'probing', 'patching', 'both' layer lists
        """
        probing_important = [
            layer for layer, result in self.probing_results.items()
            if result.mean_score >= probing_threshold
        ]
        
        patching_important = [
            layer for layer, result in self.patching_results.items()
            if result.mean_recovery >= patching_threshold
        ]
        
        both = list(set(probing_important) & set(patching_important))
        
        return {
            'probing': sorted(probing_important),
            'patching': sorted(patching_important),
            'both': sorted(both)
        }
    
    def compute_probing_patching_correlation(self) -> Dict[str, float]:
        """
        compute correlation between probing accuracy and patching recovery.
        
        this tests whether layers that encode pd information also
        causally affect predictions.
        """
        common_layers = set(self.probing_results.keys()) & set(self.patching_results.keys())
        
        if len(common_layers) < 3:
            warnings.warn("not enough common layers for correlation")
            return {}
        
        layers = sorted(common_layers)
        probing_scores = [self.probing_results[l].mean_score for l in layers]
        patching_scores = [self.patching_results[l].mean_recovery for l in layers]
        
        spearman_r, spearman_p = stats.spearmanr(probing_scores, patching_scores)
        pearson_r, pearson_p = stats.pearsonr(probing_scores, patching_scores)
        
        return {
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'n_layers': len(layers)
        }
    
    def get_clinical_encoding_summary(self) -> pd.DataFrame:
        """
        get summary table of clinical feature encoding.
        
        returns:
            dataframe with feature, best_layer, r2 columns
        """
        rows = []
        
        for feature_name, layer_results in self.clinical_results.items():
            if not layer_results:
                continue
            
            best_layer = max(layer_results.keys(), key=lambda x: layer_results[x].r2_score)
            best_r2 = layer_results[best_layer].r2_score
            
            rows.append({
                'feature': feature_name,
                'best_layer': best_layer,
                'r2': best_r2,
                'std': layer_results[best_layer].std_score
            })
        
        return pd.DataFrame(rows)
    
    def generate_summary_report(self) -> str:
        """
        generate text summary of all results.
        
        returns:
            formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        lines.append(f"timestamp: {self.metadata.timestamp}")
        lines.append("=" * 70)
        
        # probing summary
        if self.probing_results:
            lines.append("\nPROBING RESULTS")
            lines.append("-" * 40)
            best_layer, best_acc = self.get_best_probing_layer()
            lines.append(f"best layer: {best_layer} (accuracy = {best_acc:.3f})")
            
            # layer-wise
            for layer in sorted(self.probing_results.keys()):
                r = self.probing_results[layer]
                sig = "*" if r.p_value and r.p_value < 0.05 else ""
                lines.append(f"  layer {layer:2d}: {r.mean_score:.3f} ± {r.std_score:.3f} {sig}")
        
        # patching summary
        if self.patching_results:
            lines.append("\nPATCHING RESULTS")
            lines.append("-" * 40)
            best_layer, best_rec = self.get_best_patching_layer()
            lines.append(f"best layer: {best_layer} (recovery = {best_rec:.3f})")
            
            for layer in sorted(self.patching_results.keys()):
                r = self.patching_results[layer]
                lines.append(f"  layer {layer:2d}: {r.mean_recovery:.3f} ± {r.std_recovery:.3f}")
        
        # probing-patching correlation
        if self.probing_results and self.patching_results:
            corr = self.compute_probing_patching_correlation()
            if corr:
                lines.append("\nPROBING-PATCHING CORRELATION")
                lines.append("-" * 40)
                lines.append(f"spearman r = {corr['spearman_r']:.3f} (p = {corr['spearman_p']:.4f})")
        
        # clinical feature summary
        if self.clinical_results:
            lines.append("\nCLINICAL FEATURE ENCODING")
            lines.append("-" * 40)
            df = self.get_clinical_encoding_summary()
            for _, row in df.iterrows():
                lines.append(f"  {row['feature']}: layer {row['best_layer']} (r² = {row['r2']:.3f})")
        
        # important layers
        if self.probing_results or self.patching_results:
            important = self.get_important_layers()
            lines.append("\nIMPORTANT LAYERS")
            lines.append("-" * 40)
            if important['both']:
                lines.append(f"  both probing & patching: {important['both']}")
            lines.append(f"  probing (acc > 0.6): {important['probing']}")
            lines.append(f"  patching (rec > 0.1): {important['patching']}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """convert all results to serializable dict."""
        return {
            'metadata': self.metadata.to_dict(),
            'probing': {
                str(k): v.to_dict() for k, v in self.probing_results.items()
            },
            'patching': {
                str(k): v.to_dict() for k, v in self.patching_results.items()
            },
            'clinical': {
                feat: {
                    str(layer): r.to_dict() for layer, r in layer_results.items()
                } for feat, layer_results in self.clinical_results.items()
            },
            'cross_dataset': self.cross_dataset_results
        }
    
    def save(self, filename: Optional[str] = None):
        """save all results to json file."""
        if filename is None:
            filename = f"{self.experiment_name}_{self._timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        return filepath
    
    def save_report(self, filename: Optional[str] = None):
        """save text report to file."""
        if filename is None:
            filename = f"{self.experiment_name}_{self._timestamp}_report.txt"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(self.generate_summary_report())
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'ResultsAggregator':
        """load aggregator from saved json file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        aggregator = cls(
            experiment_name=data['metadata']['experiment_name']
        )
        aggregator.metadata = ExperimentMetadata(**data['metadata'])
        
        # load probing results
        for layer_str, result_data in data.get('probing', {}).items():
            layer = int(layer_str)
            aggregator.probing_results[layer] = ProbingResult(**result_data)
        
        # load patching results
        for layer_str, result_data in data.get('patching', {}).items():
            layer = int(layer_str)
            aggregator.patching_results[layer] = PatchingResult(**result_data)
        
        # load clinical results
        for feat_name, layer_results in data.get('clinical', {}).items():
            aggregator.clinical_results[feat_name] = {}
            for layer_str, result_data in layer_results.items():
                layer = int(layer_str)
                aggregator.clinical_results[feat_name][layer] = ClinicalProbingResult(**result_data)
        
        aggregator.cross_dataset_results = data.get('cross_dataset', {})
        
        return aggregator


class HypothesisTester:
    """
    statistical testing for project hypotheses.
    
    provides formal statistical tests for the three main hypotheses.
    """
    
    def __init__(self, aggregator: ResultsAggregator):
        """
        args:
            aggregator: results aggregator with experiment data
        """
        self.aggregator = aggregator
    
    def test_hypothesis_1(
        self,
        phonatory_features: List[str] = None,
        prosodic_features: List[str] = None,
        early_layers: range = range(2, 5),
        middle_layers: range = range(5, 9)
    ) -> Dict[str, Any]:
        """
        test hypothesis 1: clinical features encoded in specific layers.
        
        hypothesis: phonatory features (jitter, shimmer) in early layers (2-4),
                   prosodic features (f0) in middle layers (5-8).
        
        args:
            phonatory_features: list of phonatory feature names
            prosodic_features: list of prosodic feature names
            early_layers: range of early layers
            middle_layers: range of middle layers
            
        returns:
            dict with test results
        """
        if phonatory_features is None:
            phonatory_features = ['jitter_local', 'jitter_rap', 'shimmer_local', 'shimmer_apq3']
        if prosodic_features is None:
            prosodic_features = ['f0_mean', 'f0_std', 'hnr']
        
        results = {
            'hypothesis': 'h1_clinical_encoding',
            'phonatory': {},
            'prosodic': {},
            'supported': False
        }
        
        clinical = self.aggregator.clinical_results
        
        # test phonatory features
        phonatory_peaks = []
        for feat in phonatory_features:
            if feat in clinical and clinical[feat]:
                best_layer = max(clinical[feat].keys(), key=lambda x: clinical[feat][x].r2_score)
                phonatory_peaks.append(best_layer)
                results['phonatory'][feat] = {
                    'best_layer': best_layer,
                    'r2': clinical[feat][best_layer].r2_score
                }
        
        # test prosodic features
        prosodic_peaks = []
        for feat in prosodic_features:
            if feat in clinical and clinical[feat]:
                best_layer = max(clinical[feat].keys(), key=lambda x: clinical[feat][x].r2_score)
                prosodic_peaks.append(best_layer)
                results['prosodic'][feat] = {
                    'best_layer': best_layer,
                    'r2': clinical[feat][best_layer].r2_score
                }
        
        # evaluate hypothesis
        phonatory_in_early = sum(1 for l in phonatory_peaks if l in early_layers) / max(len(phonatory_peaks), 1)
        prosodic_in_middle = sum(1 for l in prosodic_peaks if l in middle_layers) / max(len(prosodic_peaks), 1)
        
        results['phonatory_early_fraction'] = phonatory_in_early
        results['prosodic_middle_fraction'] = prosodic_in_middle
        results['phonatory_mean_layer'] = np.mean(phonatory_peaks) if phonatory_peaks else None
        results['prosodic_mean_layer'] = np.mean(prosodic_peaks) if prosodic_peaks else None
        
        results['supported'] = phonatory_in_early >= 0.5 and prosodic_in_middle >= 0.5
        
        return results
    
    def test_hypothesis_2(
        self,
        recovery_threshold: float = 0.1,
        probing_patching_correlation_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        test hypothesis 2: causal feature dependency.
        
        hypothesis: model predictions depend causally on representations
                   correlated with clinical biomarkers, not spurious features.
        
        args:
            recovery_threshold: minimum recovery to consider layer causal
            probing_patching_correlation_threshold: minimum correlation
            
        returns:
            dict with test results
        """
        results = {
            'hypothesis': 'h2_causal_dependency',
            'causal_layers': [],
            'probing_patching_correlation': None,
            'supported': False
        }
        
        # identify causal layers (high patching recovery)
        for layer, patching in self.aggregator.patching_results.items():
            if patching.mean_recovery >= recovery_threshold:
                results['causal_layers'].append(layer)
        
        # compute correlation between probing and patching
        corr = self.aggregator.compute_probing_patching_correlation()
        results['probing_patching_correlation'] = corr
        
        # hypothesis supported if correlation is positive and significant
        if corr:
            results['supported'] = (
                corr['spearman_r'] >= probing_patching_correlation_threshold and
                corr['spearman_p'] < 0.05 and
                len(results['causal_layers']) >= 3
            )
        
        return results
    
    def test_hypothesis_3(
        self,
        alignment_generalization_correlation_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        test hypothesis 3: generalization prediction.
        
        hypothesis: models with representations more aligned to clinical
                   biomarkers generalize better across datasets.
        
        args:
            alignment_generalization_correlation_threshold: minimum correlation
            
        returns:
            dict with test results
        """
        results = {
            'hypothesis': 'h3_generalization_prediction',
            'correlation': None,
            'supported': False,
            'notes': ''
        }
        
        cross_dataset = self.aggregator.cross_dataset_results
        
        if not cross_dataset:
            results['notes'] = 'no cross-dataset results available'
            return results
        
        # extract correlation if available
        if 'alignment_generalization_correlation' in cross_dataset:
            corr = cross_dataset['alignment_generalization_correlation']
            results['correlation'] = corr
            
            if isinstance(corr, dict):
                results['supported'] = (
                    corr.get('spearman_r', 0) >= alignment_generalization_correlation_threshold and
                    corr.get('p_value', 1) < 0.05
                )
            elif isinstance(corr, (int, float)):
                results['supported'] = corr >= alignment_generalization_correlation_threshold
        else:
            results['notes'] = 'correlation not computed in cross-dataset results'
        
        return results
    
    def run_all_hypothesis_tests(self) -> Dict[str, Dict]:
        """run all hypothesis tests and return combined results."""
        return {
            'hypothesis_1': self.test_hypothesis_1(),
            'hypothesis_2': self.test_hypothesis_2(),
            'hypothesis_3': self.test_hypothesis_3()
        }
    
    def generate_hypothesis_report(self) -> str:
        """generate formatted report of all hypothesis tests."""
        all_tests = self.run_all_hypothesis_tests()
        
        lines = []
        lines.append("=" * 70)
        lines.append("HYPOTHESIS TESTING REPORT")
        lines.append("=" * 70)
        
        # hypothesis 1
        h1 = all_tests['hypothesis_1']
        lines.append("\nHYPOTHESIS 1: Clinical Feature Encoding")
        lines.append("-" * 50)
        lines.append("claim: phonatory features (jitter, shimmer) encoded in early layers (2-4),")
        lines.append("       prosodic features (f0, hnr) encoded in middle layers (5-8)")
        lines.append("")
        if h1['phonatory_mean_layer']:
            lines.append(f"phonatory mean peak layer: {h1['phonatory_mean_layer']:.1f}")
        if h1['prosodic_mean_layer']:
            lines.append(f"prosodic mean peak layer: {h1['prosodic_mean_layer']:.1f}")
        lines.append(f"phonatory in early layers: {h1['phonatory_early_fraction']:.1%}")
        lines.append(f"prosodic in middle layers: {h1['prosodic_middle_fraction']:.1%}")
        lines.append(f"VERDICT: {'SUPPORTED' if h1['supported'] else 'NOT SUPPORTED'}")
        
        # hypothesis 2
        h2 = all_tests['hypothesis_2']
        lines.append("\nHYPOTHESIS 2: Causal Feature Dependency")
        lines.append("-" * 50)
        lines.append("claim: model predictions depend causally on representations")
        lines.append("       correlated with clinical biomarkers")
        lines.append("")
        lines.append(f"causal layers (recovery > 0.1): {h2['causal_layers']}")
        if h2['probing_patching_correlation']:
            corr = h2['probing_patching_correlation']
            lines.append(f"probing-patching correlation: r = {corr['spearman_r']:.3f} (p = {corr['spearman_p']:.4f})")
        lines.append(f"VERDICT: {'SUPPORTED' if h2['supported'] else 'NOT SUPPORTED'}")
        
        # hypothesis 3
        h3 = all_tests['hypothesis_3']
        lines.append("\nHYPOTHESIS 3: Generalization Prediction")
        lines.append("-" * 50)
        lines.append("claim: models with clinically-aligned representations")
        lines.append("       generalize better across datasets")
        lines.append("")
        if h3['correlation']:
            lines.append(f"alignment-generalization correlation: {h3['correlation']}")
        if h3['notes']:
            lines.append(f"notes: {h3['notes']}")
        lines.append(f"VERDICT: {'SUPPORTED' if h3['supported'] else 'NOT SUPPORTED'}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


def aggregate_multiple_experiments(
    experiment_dirs: List[str],
    output_path: str
) -> ResultsAggregator:
    """
    aggregate results from multiple experiment runs.
    
    args:
        experiment_dirs: list of paths to experiment result directories
        output_path: path for combined results
        
    returns:
        combined aggregator
    """
    combined = ResultsAggregator(
        experiment_name="combined_experiments",
        output_dir=output_path
    )
    
    all_probing = []
    all_patching = []
    
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        
        # find result files
        json_files = list(exp_path.glob("*.json"))
        
        for json_file in json_files:
            try:
                agg = ResultsAggregator.load(str(json_file))
                all_probing.append(agg.probing_results)
                all_patching.append(agg.patching_results)
            except Exception as e:
                warnings.warn(f"could not load {json_file}: {e}")
    
    # average results across experiments
    if all_probing:
        layers = set()
        for p in all_probing:
            layers.update(p.keys())
        
        for layer in layers:
            scores = [p[layer].mean_score for p in all_probing if layer in p]
            if scores:
                combined.probing_results[layer] = ProbingResult(
                    layer=layer,
                    mean_score=np.mean(scores),
                    std_score=np.std(scores),
                    n_samples=len(scores),
                    n_folds=len(scores)
                )
    
    return combined
