"""
cross-dataset generalization analysis module.

implements Phase 4 of the research plan:
- train dataset-specific models
- cross-dataset evaluation matrix
- clinical alignment score computation
- generalization-interpretability correlation
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
import json
import warnings

import torch
import torch.nn as nn
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm


@dataclass
class CrossDatasetResults:
    """container for cross-dataset evaluation results."""
    
    accuracy_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    f1_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    auc_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # generalization gaps
    generalization_gaps: Dict[str, float] = field(default_factory=dict)
    
    # clinical alignment
    clinical_alignment_scores: Dict[str, float] = field(default_factory=dict)
    
    # correlation analysis
    correlation_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Path):
        """save results to json."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "CrossDatasetResults":
        """load results from json."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class ClinicalAlignmentProfile:
    """clinical alignment profile for a model."""
    
    model_name: str = ""
    dataset_trained_on: str = ""
    
    # layerwise probing accuracy for each clinical feature
    layerwise_probing: Dict[str, Dict[int, float]] = field(default_factory=dict)
    
    # feature-wise alignment scores (average across layers)
    feature_scores: Dict[str, float] = field(default_factory=dict)
    
    # overall alignment score
    overall_alignment: float = 0.0
    
    # best layer for each feature
    best_layers: Dict[str, int] = field(default_factory=dict)
    
    def compute_overall_score(self):
        """compute overall clinical alignment score."""
        if self.feature_scores:
            self.overall_alignment = np.mean(list(self.feature_scores.values()))
        return self.overall_alignment
    
    def to_dict(self) -> Dict:
        """convert to dictionary."""
        return asdict(self)


class CrossDatasetEvaluator:
    """
    evaluate models across multiple datasets.
    
    implements the 3×3 (or N×N) cross-dataset evaluation matrix.
    """
    
    def __init__(
        self,
        datasets: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        args:
            datasets: dict mapping dataset name to pytorch dataset
            device: device to run on
        """
        self.datasets = datasets
        self.device = device
        self.dataset_names = list(datasets.keys())
    
    @torch.no_grad()
    def evaluate_model_on_dataset(
        self,
        model,
        dataset,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        evaluate a single model on a single dataset.
        
        args:
            model: classifier model
            dataset: pytorch dataset
            batch_size: batch size for evaluation
        
        returns:
            metrics dict
        """
        model = model.to(self.device)
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        # simple batching
        indices = list(range(len(dataset)))
        
        for start_idx in tqdm(range(0, len(indices), batch_size), desc="evaluating"):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            batch_inputs = []
            batch_labels = []
            
            for idx in batch_indices:
                sample = dataset[idx]
                batch_inputs.append(sample['input_values'])
                batch_labels.append(sample['label'])
            
            # pad to same length
            max_len = max(x.shape[0] for x in batch_inputs)
            padded_inputs = []
            
            for inp in batch_inputs:
                if inp.shape[0] < max_len:
                    padding = torch.zeros(max_len - inp.shape[0])
                    inp = torch.cat([inp, padding])
                padded_inputs.append(inp)
            
            input_batch = torch.stack(padded_inputs).to(self.device)
            
            try:
                outputs = model(input_batch)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_probs.extend(probs[:, 1].cpu().numpy().tolist())
                all_labels.extend(batch_labels)
            
            except Exception as e:
                warnings.warn(f"batch evaluation failed: {e}")
                continue
        
        if not all_preds:
            return {'accuracy': 0.0, 'f1': 0.0, 'auc': 0.5}
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='binary'),
        }
        
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_probs)
        except ValueError:
            metrics['auc'] = 0.5
        
        return metrics
    
    def build_evaluation_matrix(
        self,
        models: Dict[str, Any],
        batch_size: int = 8
    ) -> CrossDatasetResults:
        """
        build full cross-dataset evaluation matrix.
        
        args:
            models: dict mapping model name to model
                    (should match dataset names for proper analysis)
            batch_size: batch size for evaluation
        
        returns:
            CrossDatasetResults with full matrix
        """
        results = CrossDatasetResults()
        
        # initialize matrices
        for model_name in models.keys():
            results.accuracy_matrix[model_name] = {}
            results.f1_matrix[model_name] = {}
            results.auc_matrix[model_name] = {}
        
        # evaluate each model on each dataset
        for model_name, model in models.items():
            print(f"\nevaluating model: {model_name}")
            
            for dataset_name, dataset in self.datasets.items():
                print(f"  on dataset: {dataset_name}")
                
                metrics = self.evaluate_model_on_dataset(
                    model, dataset, batch_size
                )
                
                results.accuracy_matrix[model_name][dataset_name] = metrics['accuracy']
                results.f1_matrix[model_name][dataset_name] = metrics['f1']
                results.auc_matrix[model_name][dataset_name] = metrics['auc']
        
        # compute generalization gaps
        for model_name in models.keys():
            # find the dataset this model was trained on
            trained_dataset = model_name  # assumes model name matches dataset
            
            if trained_dataset in results.accuracy_matrix[model_name]:
                in_domain_acc = results.accuracy_matrix[model_name][trained_dataset]
                
                # compute average out-of-domain accuracy
                out_domain_accs = [
                    acc for ds_name, acc in results.accuracy_matrix[model_name].items()
                    if ds_name != trained_dataset
                ]
                
                if out_domain_accs:
                    avg_out_domain = np.mean(out_domain_accs)
                    results.generalization_gaps[model_name] = in_domain_acc - avg_out_domain
        
        return results


class ClinicalAlignmentAnalyzer:
    """
    analyze clinical alignment of model representations.
    
    computes how well each layer encodes clinical features.
    """
    
    def __init__(
        self,
        clinical_features: Dict[str, np.ndarray],
        sample_ids: List[str]
    ):
        """
        args:
            clinical_features: dict mapping feature name to values per sample
            sample_ids: list of sample identifiers
        """
        self.clinical_features = clinical_features
        self.sample_ids = sample_ids
        self.sample_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    
    def compute_alignment_profile(
        self,
        model_name: str,
        activations: np.ndarray,
        activation_sample_ids: List[str],
        n_layers: int = 12
    ) -> ClinicalAlignmentProfile:
        """
        compute clinical alignment profile for a model.
        
        args:
            model_name: name of the model
            activations: array of shape [n_samples, n_layers, hidden_size]
            activation_sample_ids: sample ids for activations
            n_layers: number of layers
        
        returns:
            ClinicalAlignmentProfile
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        
        profile = ClinicalAlignmentProfile(model_name=model_name)
        
        # map activation samples to clinical features
        valid_indices = []
        for i, sid in enumerate(activation_sample_ids):
            if sid in self.sample_to_idx:
                valid_indices.append(i)
        
        if not valid_indices:
            warnings.warn("no matching samples between activations and clinical features")
            return profile
        
        # probe each clinical feature at each layer
        for feature_name, feature_values in self.clinical_features.items():
            # binarize for probing
            median_val = np.nanmedian(feature_values)
            binary_labels = (feature_values > median_val).astype(int)
            
            layer_scores = {}
            
            for layer_idx in range(n_layers):
                # get activations for this layer
                layer_acts = activations[valid_indices, layer_idx, :]
                
                # get corresponding labels
                layer_labels = []
                valid_acts = []
                
                for i, act_idx in enumerate(valid_indices):
                    sid = activation_sample_ids[act_idx]
                    feat_idx = self.sample_to_idx[sid]
                    label = binary_labels[feat_idx]
                    
                    if not np.isnan(label):
                        layer_labels.append(int(label))
                        valid_acts.append(layer_acts[i])
                
                if len(valid_acts) < 10:
                    layer_scores[layer_idx] = 0.5
                    continue
                
                X = np.array(valid_acts)
                y = np.array(layer_labels)
                
                # scale and probe
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                try:
                    clf = LogisticRegression(max_iter=1000, random_state=42)
                    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
                    layer_scores[layer_idx] = float(np.mean(scores))
                except Exception as e:
                    layer_scores[layer_idx] = 0.5
            
            profile.layerwise_probing[feature_name] = layer_scores
            
            # compute feature-wise score (best layer performance)
            best_score = max(layer_scores.values())
            profile.feature_scores[feature_name] = best_score
            profile.best_layers[feature_name] = max(layer_scores, key=layer_scores.get)
        
        profile.compute_overall_score()
        
        return profile


class GeneralizationInterpretabilityAnalyzer:
    """
    analyze correlation between clinical alignment and generalization.
    
    tests the hypothesis: models with higher clinical alignment
    generalize better across datasets.
    """
    
    def __init__(self):
        """initialize analyzer."""
        self.alignment_profiles: Dict[str, ClinicalAlignmentProfile] = {}
        self.cross_dataset_results: Optional[CrossDatasetResults] = None
    
    def add_alignment_profile(self, profile: ClinicalAlignmentProfile):
        """add a clinical alignment profile."""
        self.alignment_profiles[profile.model_name] = profile
    
    def set_cross_dataset_results(self, results: CrossDatasetResults):
        """set cross-dataset evaluation results."""
        self.cross_dataset_results = results
    
    def compute_correlation(self) -> Dict[str, Any]:
        """
        compute correlation between clinical alignment and generalization.
        
        returns:
            correlation analysis results
        """
        if not self.alignment_profiles or not self.cross_dataset_results:
            raise ValueError("need both alignment profiles and cross-dataset results")
        
        alignment_scores = []
        generalization_gaps = []
        model_names = []
        
        for model_name, profile in self.alignment_profiles.items():
            if model_name in self.cross_dataset_results.generalization_gaps:
                alignment_scores.append(profile.overall_alignment)
                generalization_gaps.append(
                    self.cross_dataset_results.generalization_gaps[model_name]
                )
                model_names.append(model_name)
        
        if len(alignment_scores) < 3:
            return {
                'error': 'insufficient models for correlation analysis',
                'n_models': len(alignment_scores)
            }
        
        alignment_scores = np.array(alignment_scores)
        generalization_gaps = np.array(generalization_gaps)
        
        # spearman correlation (alignment vs generalization gap)
        # hypothesis: higher alignment -> smaller gap (negative correlation)
        spearman_corr, spearman_p = scipy_stats.spearmanr(
            alignment_scores, generalization_gaps
        )
        
        # pearson correlation
        pearson_corr, pearson_p = scipy_stats.pearsonr(
            alignment_scores, generalization_gaps
        )
        
        # alternative: alignment vs average cross-dataset accuracy
        avg_cross_accs = []
        for model_name in model_names:
            accs = list(self.cross_dataset_results.accuracy_matrix[model_name].values())
            avg_cross_accs.append(np.mean(accs))
        
        avg_cross_accs = np.array(avg_cross_accs)
        
        spearman_acc_corr, spearman_acc_p = scipy_stats.spearmanr(
            alignment_scores, avg_cross_accs
        )
        
        results = {
            'alignment_vs_gap': {
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p),
                'pearson_correlation': float(pearson_corr),
                'pearson_p_value': float(pearson_p),
                'interpretation': self._interpret_correlation(spearman_corr, spearman_p)
            },
            'alignment_vs_accuracy': {
                'spearman_correlation': float(spearman_acc_corr),
                'spearman_p_value': float(spearman_acc_p),
                'interpretation': self._interpret_correlation(spearman_acc_corr, spearman_acc_p)
            },
            'data': {
                'model_names': model_names,
                'alignment_scores': alignment_scores.tolist(),
                'generalization_gaps': generalization_gaps.tolist(),
                'avg_cross_dataset_accuracy': avg_cross_accs.tolist()
            },
            'n_models': len(model_names)
        }
        
        return results
    
    def _interpret_correlation(self, corr: float, p_value: float) -> str:
        """interpret correlation strength and significance."""
        if p_value > 0.05:
            sig = "not significant"
        elif p_value > 0.01:
            sig = "marginally significant"
        else:
            sig = "significant"
        
        if abs(corr) < 0.3:
            strength = "weak"
        elif abs(corr) < 0.6:
            strength = "moderate"
        else:
            strength = "strong"
        
        direction = "positive" if corr > 0 else "negative"
        
        return f"{strength} {direction} correlation ({sig})"
    
    def generate_report(self) -> Dict[str, Any]:
        """
        generate comprehensive analysis report.
        
        returns:
            full report dictionary
        """
        report = {
            'summary': {},
            'alignment_profiles': {},
            'cross_dataset_matrix': {},
            'correlation_analysis': {}
        }
        
        # alignment profiles
        for model_name, profile in self.alignment_profiles.items():
            report['alignment_profiles'][model_name] = profile.to_dict()
        
        # cross-dataset results
        if self.cross_dataset_results:
            report['cross_dataset_matrix'] = self.cross_dataset_results.to_dict()
        
        # correlation analysis
        try:
            report['correlation_analysis'] = self.compute_correlation()
        except Exception as e:
            report['correlation_analysis'] = {'error': str(e)}
        
        # summary
        if self.alignment_profiles:
            best_model = max(
                self.alignment_profiles.items(),
                key=lambda x: x[1].overall_alignment
            )
            report['summary']['best_aligned_model'] = best_model[0]
            report['summary']['best_alignment_score'] = best_model[1].overall_alignment
        
        if self.cross_dataset_results and self.cross_dataset_results.generalization_gaps:
            best_generalizer = min(
                self.cross_dataset_results.generalization_gaps.items(),
                key=lambda x: x[1]  # smallest gap = best generalization
            )
            report['summary']['best_generalizing_model'] = best_generalizer[0]
            report['summary']['smallest_generalization_gap'] = best_generalizer[1]
        
        return report


class DatasetSpecificTrainer:
    """
    train dataset-specific models for cross-dataset analysis.
    """
    
    def __init__(
        self,
        model_init_fn: Callable,
        training_args_fn: Callable,
        device: str = "cuda"
    ):
        """
        args:
            model_init_fn: function that returns a new model instance
            training_args_fn: function that returns training arguments
            device: device to train on
        """
        self.model_init_fn = model_init_fn
        self.training_args_fn = training_args_fn
        self.device = device
        self.trained_models: Dict[str, Any] = {}
    
    def train_on_dataset(
        self,
        dataset_name: str,
        train_dataset,
        eval_dataset,
        output_dir: Path
    ) -> Any:
        """
        train a model on a specific dataset.
        
        args:
            dataset_name: name of the dataset
            train_dataset: training dataset
            eval_dataset: evaluation dataset
            output_dir: directory to save model
        
        returns:
            trained model
        """
        from transformers import Trainer
        from ..models.classifier import DataCollatorWithPadding, compute_metrics
        
        output_dir = Path(output_dir) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model = self.model_init_fn()
        training_args = self.training_args_fn(output_dir=str(output_dir))
        
        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorWithPadding(),
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        
        self.trained_models[dataset_name] = model
        
        # save
        model.save(output_dir / "final_model")
        
        return model
    
    def train_all_datasets(
        self,
        datasets: Dict[str, Tuple[Any, Any]],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        train models on all datasets.
        
        args:
            datasets: dict mapping dataset name to (train, eval) tuple
            output_dir: base output directory
        
        returns:
            dict mapping dataset name to trained model
        """
        for dataset_name, (train_ds, eval_ds) in datasets.items():
            print(f"\ntraining on {dataset_name}...")
            
            self.train_on_dataset(
                dataset_name,
                train_ds,
                eval_ds,
                output_dir
            )
        
        return self.trained_models


def run_cross_dataset_analysis(
    datasets: Dict[str, Any],
    models: Dict[str, Any],
    clinical_features: Optional[Dict[str, np.ndarray]] = None,
    sample_ids: Optional[List[str]] = None,
    activations: Optional[Dict[str, np.ndarray]] = None,
    activation_sample_ids: Optional[Dict[str, List[str]]] = None,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    run complete cross-dataset generalization analysis.
    
    args:
        datasets: dict mapping dataset name to dataset
        models: dict mapping model name to model (trained on corresponding dataset)
        clinical_features: optional clinical feature values
        sample_ids: sample identifiers for clinical features
        activations: optional activations per model
        activation_sample_ids: sample ids for activations per model
        output_path: path to save results
    
    returns:
        comprehensive analysis results
    """
    # cross-dataset evaluation
    evaluator = CrossDatasetEvaluator(datasets)
    cross_results = evaluator.build_evaluation_matrix(models)
    
    # clinical alignment if features provided
    analyzer = GeneralizationInterpretabilityAnalyzer()
    analyzer.set_cross_dataset_results(cross_results)
    
    if clinical_features is not None and sample_ids is not None:
        alignment_analyzer = ClinicalAlignmentAnalyzer(clinical_features, sample_ids)
        
        for model_name in models.keys():
            if activations is not None and model_name in activations:
                model_acts = activations[model_name]
                model_sample_ids = activation_sample_ids.get(model_name, [])
                
                profile = alignment_analyzer.compute_alignment_profile(
                    model_name,
                    model_acts,
                    model_sample_ids
                )
                profile.dataset_trained_on = model_name
                analyzer.add_alignment_profile(profile)
    
    # generate report
    report = analyzer.generate_report()
    
    # save if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    return report
