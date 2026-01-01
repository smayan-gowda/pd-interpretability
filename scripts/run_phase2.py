#!/usr/bin/env python
"""
phase 2: probing classifier experiments.

this script:
1. extracts intermediate representations from wav2vec2
2. trains probing classifiers at each layer
3. probes for clinical features
4. validates with control tasks
5. generates layer-wise encoding heatmap

usage:
    python scripts/run_phase2.py --config configs/experiment_config.yaml
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.interpretability.extraction import Wav2Vec2ActivationExtractor
from src.models.probes import (
    LinearProbe,
    LayerwiseProber,
    MultiFeatureProber,
    ControlTaskProber,
    create_control_labels,
    permutation_test_probe
)
from src.features.clinical import get_pd_discriminative_features
from src.utils.visualization import (
    plot_layerwise_probing,
    plot_clinical_feature_heatmap
)
from src.utils.analysis import (
    compare_layers_probing,
    multiple_comparison_correction,
    generate_results_summary
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """load yaml configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_activations(
    processed_data: list,
    config: dict,
    output_dir: Path
) -> np.ndarray:
    """extract layer-wise activations from wav2vec2."""
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'facebook/wav2vec2-base-960h')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"using device: {device}")
    
    logger.info(f"loading model: {model_name}")
    extractor = Wav2Vec2ActivationExtractor(
        model_name=model_name,
        device=device
    )
    
    logger.info(f"extracting activations for {len(processed_data)} samples...")
    
    all_activations = []
    labels = []
    subject_ids = []
    
    for i, sample in enumerate(processed_data):
        if i % 100 == 0:
            logger.info(f"  processing sample {i}/{len(processed_data)}")
        
        audio = sample['audio']
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        
        # extract activations
        activations = extractor.extract_all_layers(audio)
        
        # mean pool over time
        pooled = []
        for layer_act in activations:
            pooled.append(layer_act.mean(dim=1).squeeze().cpu().numpy())
        
        all_activations.append(np.stack(pooled))
        labels.append(sample['label'])
        subject_ids.append(sample.get('subject_id', f'unknown_{i}'))
    
    activations_array = np.stack(all_activations)
    labels_array = np.array(labels)
    
    logger.info(f"activations shape: {activations_array.shape}")
    logger.info(f"  (samples, layers, hidden_size)")
    
    # save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'activations.npy', activations_array)
    np.save(output_dir / 'labels.npy', labels_array)
    np.save(output_dir / 'subject_ids.npy', np.array(subject_ids))
    
    return activations_array, labels_array, subject_ids


def run_probing_experiments(
    activations: np.ndarray,
    labels: np.ndarray,
    subject_ids: list,
    config: dict,
    output_dir: Path
) -> dict:
    """run layer-wise probing classifiers."""
    probing_config = config.get('probing', {})
    regularization = probing_config.get('regularization', 1.0)
    
    logger.info("running layer-wise probing for pd classification...")
    
    # convert subject ids to group indices for cv
    unique_subjects = list(set(subject_ids))
    subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
    groups = np.array([subject_to_idx[s] for s in subject_ids])
    
    # probe each layer
    prober = LayerwiseProber(
        task='classification',
        regularization=regularization
    )
    
    results = prober.probe_all_layers(
        activations, labels, groups=groups
    )
    
    logger.info("probing results:")
    for layer_idx in sorted(results.keys()):
        acc = results[layer_idx]['mean']
        std = results[layer_idx]['std']
        logger.info(f"  layer {layer_idx}: {acc:.3f} ± {std:.3f}")
    
    # find best layer
    best_layer = max(results, key=lambda x: results[x]['mean'])
    logger.info(f"\nbest layer: {best_layer} (acc={results[best_layer]['mean']:.3f})")
    
    # statistical comparison
    logger.info("\nstatistical comparison vs baseline (layer 0):")
    comparisons = compare_layers_probing(results, baseline_layer=0)
    
    for layer_idx, comp in comparisons.items():
        sig = "***" if comp.get('adjusted_p_value', 1) < 0.001 else \
              "**" if comp.get('adjusted_p_value', 1) < 0.01 else \
              "*" if comp.get('adjusted_p_value', 1) < 0.05 else "ns"
        logger.info(f"  layer {layer_idx}: p={comp['p_value']:.4f} "
                   f"(adj={comp.get('adjusted_p_value', comp['p_value']):.4f}) {sig}")
    
    # save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'probing_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # generate plot
    fig = plot_layerwise_probing(
        results,
        title="layer-wise pd classification probing accuracy",
        save_path=str(output_dir / 'probing_accuracy.png')
    )
    
    return results


def run_control_task_validation(
    activations: np.ndarray,
    labels: np.ndarray,
    metadata: list,
    config: dict,
    output_dir: Path
) -> dict:
    """validate probing with control tasks."""
    logger.info("\nrunning control task validation...")
    
    # create control labels
    control_labels = create_control_labels(metadata)
    
    # use best layer (or middle layer)
    n_layers = activations.shape[1]
    test_layer = n_layers // 2
    layer_acts = activations[:, test_layer, :]
    
    logger.info(f"testing layer {test_layer}")
    
    # fit probes with controls
    control_prober = ControlTaskProber()
    results = control_prober.fit_with_controls(
        layer_acts, labels, control_labels
    )
    
    logger.info("results:")
    for task, res in results.items():
        logger.info(f"  {task}: acc={res['mean']:.3f} ± {res['std']:.3f}")
    
    # validate quality
    quality = control_prober.validate_probe_quality()
    logger.info(f"\nquality validation:")
    for check, passed in quality.items():
        status = "✓" if passed else "✗"
        logger.info(f"  {status} {check}")
    
    # selectivity
    selectivity = control_prober.get_selectivity()
    logger.info(f"\nselectivity scores:")
    for control, score in selectivity.items():
        logger.info(f"  {control}: {score:.3f}")
    
    # save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validation_results = {
        'probing_results': results,
        'quality_checks': quality,
        'selectivity': selectivity
    }
    
    with open(output_dir / 'control_validation.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=float)
    
    return validation_results


def probe_clinical_features(
    activations: np.ndarray,
    clinical_features: dict,
    config: dict,
    output_dir: Path
) -> dict:
    """probe for clinical features across layers."""
    logger.info("\nprobing for clinical features...")
    
    feature_names = get_pd_discriminative_features()
    
    # filter to available features
    available = [f for f in feature_names if f in clinical_features]
    logger.info(f"probing {len(available)} clinical features")
    
    # create feature matrix
    n_samples = activations.shape[0]
    feature_matrix = np.zeros((n_samples, len(available)))
    
    for i, feat in enumerate(available):
        feature_matrix[:, i] = clinical_features[feat]
    
    # probe
    prober = MultiFeatureProber(
        feature_names=available,
        task='regression'
    )
    
    results = prober.probe_all_features(activations, feature_matrix)
    
    # generate heatmap
    encoding_matrix = prober.get_encoding_matrix()
    
    fig = plot_clinical_feature_heatmap(
        results,
        feature_names=available,
        title="clinical feature encoding across layers",
        save_path=str(output_dir / 'clinical_heatmap.png')
    )
    
    logger.info("clinical feature probing complete")
    logger.info(f"heatmap saved to {output_dir / 'clinical_heatmap.png'}")
    
    # save results
    with open(output_dir / 'clinical_probing.json', 'w') as f:
        # convert numpy to python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    return results


def run_permutation_tests(
    activations: np.ndarray,
    labels: np.ndarray,
    config: dict,
    output_dir: Path
) -> dict:
    """run permutation tests for statistical significance."""
    probing_config = config.get('probing', {})
    n_permutations = probing_config.get('n_permutations', 1000)
    
    logger.info(f"\nrunning permutation tests (n={n_permutations})...")
    
    results = {}
    n_layers = activations.shape[1]
    
    for layer_idx in range(n_layers):
        layer_acts = activations[:, layer_idx, :]
        
        probe = LinearProbe(task='classification')
        probe.fit(layer_acts, labels)
        
        observed, p_value = permutation_test_probe(
            probe, layer_acts, labels,
            n_permutations=n_permutations
        )
        
        results[layer_idx] = {
            'observed_accuracy': observed,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        sig = "*" if p_value < 0.05 else ""
        logger.info(f"  layer {layer_idx}: acc={observed:.3f}, p={p_value:.4f} {sig}")
    
    # save
    with open(output_dir / 'permutation_tests.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='phase 2: probing experiments')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml',
                        help='path to config file')
    parser.add_argument('--phase1-output', type=str, default='results/phase1',
                        help='phase 1 output directory')
    parser.add_argument('--output', type=str, default='results/phase2',
                        help='output directory')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='skip activation extraction (use cached)')
    parser.add_argument('--skip-clinical', action='store_true',
                        help='skip clinical feature probing')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output)
    phase1_dir = Path(args.phase1_output)
    
    logger.info("=" * 60)
    logger.info("phase 2: probing classifier experiments")
    logger.info("=" * 60)
    
    # load processed data from phase 1
    processed_path = phase1_dir / 'processed' / 'processed_segments.pt'
    if not processed_path.exists():
        logger.error(f"processed data not found: {processed_path}")
        logger.error("run phase 1 first!")
        return
    
    processed_data = torch.load(processed_path, weights_only=False)
    logger.info(f"loaded {len(processed_data)} processed segments")
    
    # step 1: extract activations
    activations_path = output_dir / 'activations' / 'activations.npy'
    
    if args.skip_extraction and activations_path.exists():
        logger.info("\n[step 1] loading cached activations...")
        activations = np.load(activations_path)
        labels = np.load(output_dir / 'activations' / 'labels.npy')
        subject_ids = np.load(output_dir / 'activations' / 'subject_ids.npy').tolist()
    else:
        logger.info("\n[step 1] extracting activations...")
        activations, labels, subject_ids = extract_activations(
            processed_data, config,
            output_dir / 'activations'
        )
    
    # step 2: run probing experiments
    logger.info("\n[step 2] running probing experiments...")
    probing_results = run_probing_experiments(
        activations, labels, subject_ids, config,
        output_dir / 'probing'
    )
    
    # step 3: control task validation
    logger.info("\n[step 3] control task validation...")
    metadata = [{'subject_id': s} for s in subject_ids]
    control_results = run_control_task_validation(
        activations, labels, metadata, config,
        output_dir / 'validation'
    )
    
    # step 4: clinical feature probing
    if not args.skip_clinical:
        clinical_path = phase1_dir / 'clinical' / 'clinical_features.csv'
        if clinical_path.exists():
            import pandas as pd
            clinical_df = pd.read_csv(clinical_path)
            clinical_features = clinical_df.to_dict('list')
            
            logger.info("\n[step 4] clinical feature probing...")
            clinical_results = probe_clinical_features(
                activations, clinical_features, config,
                output_dir / 'clinical'
            )
    
    # step 5: permutation tests
    logger.info("\n[step 5] running permutation tests...")
    perm_results = run_permutation_tests(
        activations, labels, config,
        output_dir / 'permutation'
    )
    
    # generate summary
    summary = generate_results_summary(
        probing_results=probing_results
    )
    
    with open(output_dir / 'phase2_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    
    logger.info("\n" + "=" * 60)
    logger.info("phase 2 complete!")
    logger.info(f"outputs saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
