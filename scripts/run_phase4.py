#!/usr/bin/env python
"""
phase 4: cross-dataset generalization analysis.

this script:
1. trains dataset-specific models
2. evaluates cross-dataset transfer
3. builds the N×N evaluation matrix
4. compares probing profiles across datasets
5. correlates interpretability with generalization

usage:
    python scripts/run_phase4.py --config configs/experiment_config.yaml
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import yaml

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.generalization import (
    CrossDatasetEvaluator,
    CrossDatasetResults,
    ProbingProfileComparator,
    GeneralizationInterpretabilityAnalyzer,
    run_cross_dataset_analysis
)
from src.data.datasets import MultiDatasetLoader, load_dataset_by_name
from src.utils.visualization import (
    plot_cross_dataset_matrix,
    plot_probing_profile_comparison
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


def load_datasets(config: dict) -> Dict[str, torch.utils.data.Dataset]:
    """load all datasets specified in config."""
    dataset_config = config.get('datasets', {})
    data_dir = Path(dataset_config.get('data_dir', 'data/raw'))
    
    datasets = {}
    
    # italian pvs
    if dataset_config.get('include_italian_pvs', True):
        try:
            datasets['italian_pvs'] = load_dataset_by_name(
                'italian_pvs',
                data_dir / 'italian_pvs'
            )
            logger.info(f"loaded italian_pvs: {len(datasets['italian_pvs'])} samples")
        except Exception as e:
            logger.warning(f"failed to load italian_pvs: {e}")
    
    # mdvr-kcl
    if dataset_config.get('include_mdvr_kcl', True):
        try:
            datasets['mdvr_kcl'] = load_dataset_by_name(
                'mdvr_kcl',
                data_dir / 'mdvr-kcl'
            )
            logger.info(f"loaded mdvr_kcl: {len(datasets['mdvr_kcl'])} samples")
        except Exception as e:
            logger.warning(f"failed to load mdvr_kcl: {e}")
    
    # arkansas
    if dataset_config.get('include_arkansas', True):
        try:
            datasets['arkansas'] = load_dataset_by_name(
                'arkansas',
                data_dir / 'arkansas (figshare)'
            )
            logger.info(f"loaded arkansas: {len(datasets['arkansas'])} samples")
        except Exception as e:
            logger.warning(f"failed to load arkansas: {e}")
    
    return datasets


def train_dataset_specific_models(
    datasets: Dict[str, torch.utils.data.Dataset],
    config: dict,
    output_dir: Path
) -> Dict[str, torch.nn.Module]:
    """
    train a separate model for each dataset.
    
    these models will be used for cross-dataset evaluation.
    """
    training_config = config.get('training', {})
    checkpoint_dir = output_dir / 'dataset_models'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    models = {}
    
    for dataset_name, dataset in datasets.items():
        logger.info(f"training model for {dataset_name}...")
        
        checkpoint_path = checkpoint_dir / f'{dataset_name}_model.pt'
        
        # check if already trained
        if checkpoint_path.exists() and not training_config.get('retrain', False):
            logger.info(f"loading existing model for {dataset_name}")
            models[dataset_name] = torch.load(checkpoint_path, weights_only=False)
            continue
        
        # import training function
        from src.models.classifier import train_wav2vec2_classifier
        
        model = train_wav2vec2_classifier(
            dataset=dataset,
            epochs=training_config.get('epochs', 10),
            batch_size=training_config.get('batch_size', 8),
            learning_rate=training_config.get('learning_rate', 1e-5),
            device=training_config.get('device', 'cpu')
        )
        
        torch.save(model, checkpoint_path)
        models[dataset_name] = model
        
        logger.info(f"saved model for {dataset_name}")
    
    return models


def build_evaluation_matrix(
    models: Dict[str, torch.nn.Module],
    datasets: Dict[str, torch.utils.data.Dataset],
    config: dict
) -> CrossDatasetResults:
    """
    build N×N cross-dataset evaluation matrix.
    
    rows = training dataset
    cols = evaluation dataset
    """
    eval_config = config.get('evaluation', {})
    device = eval_config.get('device', 'cpu')
    
    evaluator = CrossDatasetEvaluator(device=device)
    
    logger.info("building cross-dataset evaluation matrix...")
    
    results = evaluator.evaluate_matrix(
        models=models,
        datasets=datasets,
        metrics=['accuracy', 'auc', 'f1']
    )
    
    # log matrix
    dataset_names = list(datasets.keys())
    logger.info("\ncross-dataset evaluation matrix (accuracy):")
    logger.info("train \\ eval | " + " | ".join(f"{n:12}" for n in dataset_names))
    logger.info("-" * (15 + 15 * len(dataset_names)))
    
    for train_name in dataset_names:
        row = []
        for eval_name in dataset_names:
            acc = results.get_score(train_name, eval_name, 'accuracy')
            row.append(f"{acc:12.3f}" if acc else "    N/A     ")
        logger.info(f"{train_name:12} | " + " | ".join(row))
    
    return results


def compare_probing_profiles(
    models: Dict[str, torch.nn.Module],
    datasets: Dict[str, torch.utils.data.Dataset],
    activations_dir: Path,
    config: dict
) -> Dict[str, np.ndarray]:
    """
    compare probing profiles across datasets.
    
    determines if different datasets show consistent layer-wise patterns
    for encoding clinical features.
    """
    probing_config = config.get('probing', {})
    
    comparator = ProbingProfileComparator()
    
    logger.info("comparing probing profiles across datasets...")
    
    profiles = {}
    
    for dataset_name in datasets.keys():
        # load activations if available
        act_path = activations_dir / f'{dataset_name}_activations.npy'
        
        if act_path.exists():
            activations = np.load(act_path)
            
            # run layer-wise probing
            from src.models.probes import LayerwiseProber
            
            prober = LayerwiseProber(task='classification')
            
            # extract labels
            labels = np.array([
                sample.get('label', 0) 
                for sample in datasets[dataset_name]
            ])
            
            profile = prober.probe_all_layers(activations, labels)
            
            # convert to array
            n_layers = len(profile)
            profile_array = np.array([profile[i]['mean'] for i in range(n_layers)])
            profiles[dataset_name] = profile_array
            
            logger.info(f"{dataset_name} profile shape: {profile_array.shape}")
    
    # compute profile similarities
    if len(profiles) >= 2:
        similarity_matrix = comparator.compute_similarity_matrix(profiles)
        logger.info(f"\nprofile similarity matrix:\n{similarity_matrix}")
    
    return profiles


def analyze_generalization_interpretability(
    evaluation_matrix: CrossDatasetResults,
    probing_profiles: Dict[str, np.ndarray],
    patching_results: Optional[Dict] = None,
    output_dir: Path = None
) -> Dict:
    """
    analyze correlation between interpretability and generalization.
    
    key question: do models with clearer interpretability patterns
    generalize better across datasets?
    """
    analyzer = GeneralizationInterpretabilityAnalyzer()
    
    logger.info("analyzing generalization-interpretability correlation...")
    
    analysis = analyzer.analyze(
        evaluation_matrix=evaluation_matrix,
        probing_profiles=probing_profiles,
        patching_results=patching_results
    )
    
    # report key findings
    logger.info(f"\nkey findings:")
    logger.info(f"  - profile consistency score: {analysis.get('profile_consistency', 'N/A'):.3f}")
    logger.info(f"  - generalization score: {analysis.get('generalization_score', 'N/A'):.3f}")
    logger.info(f"  - correlation: {analysis.get('correlation', 'N/A'):.3f}")
    
    if output_dir:
        with open(output_dir / 'generalization_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
    
    return analysis


def generate_visualizations(
    evaluation_matrix: CrossDatasetResults,
    probing_profiles: Dict[str, np.ndarray],
    output_dir: Path
):
    """generate publication-quality visualizations."""
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # cross-dataset matrix heatmap
    plot_cross_dataset_matrix(
        evaluation_matrix,
        save_path=figures_dir / 'cross_dataset_matrix.pdf',
        metric='accuracy'
    )
    logger.info(f"saved cross-dataset matrix visualization")
    
    # probing profile comparison
    if probing_profiles:
        plot_probing_profile_comparison(
            probing_profiles,
            save_path=figures_dir / 'probing_profiles_comparison.pdf'
        )
        logger.info(f"saved probing profile comparison")


def main():
    """main execution."""
    parser = argparse.ArgumentParser(description='Phase 4: Cross-dataset generalization')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_config.yaml',
        help='path to config file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/phase4',
        help='output directory'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='skip model training, use existing checkpoints'
    )
    args = parser.parse_args()
    
    # load config
    config = load_config(args.config)
    
    # setup output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PHASE 4: CROSS-DATASET GENERALIZATION ANALYSIS")
    logger.info("=" * 60)
    
    # step 1: load datasets
    logger.info("\n[step 1] loading datasets...")
    datasets = load_datasets(config)
    
    if len(datasets) < 2:
        logger.error("need at least 2 datasets for cross-dataset analysis")
        return
    
    # step 2: train dataset-specific models
    if not args.skip_training:
        logger.info("\n[step 2] training dataset-specific models...")
        models = train_dataset_specific_models(datasets, config, output_dir)
    else:
        logger.info("\n[step 2] loading existing models...")
        models = {}
        for name in datasets.keys():
            path = output_dir / 'dataset_models' / f'{name}_model.pt'
            if path.exists():
                models[name] = torch.load(path, weights_only=False)
    
    # step 3: build evaluation matrix
    logger.info("\n[step 3] building cross-dataset evaluation matrix...")
    evaluation_matrix = build_evaluation_matrix(models, datasets, config)
    
    # save matrix
    evaluation_matrix.save(output_dir / 'evaluation_matrix.json')
    
    # step 4: compare probing profiles
    logger.info("\n[step 4] comparing probing profiles...")
    activations_dir = Path(config.get('activations', {}).get('dir', 'data/activations'))
    probing_profiles = compare_probing_profiles(models, datasets, activations_dir, config)
    
    # step 5: generalization-interpretability analysis
    logger.info("\n[step 5] analyzing generalization-interpretability correlation...")
    analysis = analyze_generalization_interpretability(
        evaluation_matrix,
        probing_profiles,
        output_dir=output_dir
    )
    
    # step 6: visualizations
    logger.info("\n[step 6] generating visualizations...")
    generate_visualizations(evaluation_matrix, probing_profiles, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"results saved to: {output_dir}")


if __name__ == '__main__':
    main()
