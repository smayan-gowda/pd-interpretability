#!/usr/bin/env python
"""
phase 1: dataset preparation and baseline establishment.

this script:
1. loads and preprocesses datasets
2. extracts clinical features
3. trains baseline wav2vec2 classifier
4. creates minimal pairs for patching

usage:
    python scripts/run_phase1.py --config configs/experiment_config.yaml
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

from src.data.datasets import (
    ItalianPVSDataset, 
    MDVRKCLDataset, 
    ArkansasDataset,
    CombinedPDDataset
)
from src.data.preprocessing import AudioPreprocessor, segment_audio
from src.features.clinical import (
    ClinicalFeatureExtractor,
    get_clinical_feature_names,
    batch_extract_features
)
from src.models.classifier import Wav2Vec2PDClassifier
from src.interpretability.patching import create_mfcc_matched_pairs


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """load yaml configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_datasets(config: dict) -> dict:
    """load all available datasets based on config."""
    data_config = config['data']
    datasets = {}
    
    # italian pvs
    italian_path = Path(data_config['italian_pvs_path'])
    if italian_path.exists():
        logger.info(f"loading italian pvs from {italian_path}")
        datasets['italian_pvs'] = ItalianPVSDataset(
            root_dir=italian_path,
            target_sr=data_config['target_sr']
        )
        logger.info(f"  loaded {len(datasets['italian_pvs'])} samples")
    
    # mdvr-kcl
    mdvr_path = Path(data_config['mdvr_kcl_path'])
    if mdvr_path.exists():
        logger.info(f"loading mdvr-kcl from {mdvr_path}")
        datasets['mdvr_kcl'] = MDVRKCLDataset(
            root_dir=mdvr_path,
            target_sr=data_config['target_sr']
        )
        logger.info(f"  loaded {len(datasets['mdvr_kcl'])} samples")
    
    # arkansas
    arkansas_path = Path(data_config['arkansas_path'])
    if arkansas_path.exists():
        logger.info(f"loading arkansas from {arkansas_path}")
        datasets['arkansas'] = ArkansasDataset(
            root_dir=arkansas_path,
            target_sr=data_config['target_sr']
        )
        logger.info(f"  loaded {len(datasets['arkansas'])} samples")
    
    return datasets


def preprocess_dataset(dataset, config: dict, output_dir: Path):
    """preprocess all samples with 3-second segmentation."""
    preproc_config = config['preprocessing']
    
    preprocessor = AudioPreprocessor(
        target_sr=config['data']['target_sr'],
        remove_silence=preproc_config.get('remove_silence', True),
        normalize=preproc_config.get('normalize', True),
        apply_vad=preproc_config.get('apply_vad', True)
    )
    
    processed_samples = []
    segment_duration = preproc_config.get('segment_duration', 3.0)
    overlap = preproc_config.get('overlap', 0.5)
    
    logger.info(f"preprocessing with {segment_duration}s segments, {overlap} overlap")
    
    for i in range(len(dataset)):
        sample = dataset[i]
        audio = sample['input_values']
        
        # preprocess
        processed = preprocessor(audio)
        
        if processed is None:
            continue
        
        # segment into 3-second clips
        segments = segment_audio(
            processed,
            sr=config['data']['target_sr'],
            segment_duration=segment_duration,
            overlap=overlap
        )
        
        for seg_idx, segment in enumerate(segments):
            processed_samples.append({
                'audio': segment,
                'label': sample['label'],
                'subject_id': sample.get('subject_id', f'unknown_{i}'),
                'task': sample.get('task', 'unknown'),
                'segment_index': seg_idx,
                'original_index': i
            })
    
    logger.info(f"created {len(processed_samples)} segments from {len(dataset)} samples")
    
    # save processed data
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(processed_samples, output_dir / 'processed_segments.pt')
    
    return processed_samples


def extract_clinical_features(dataset, config: dict, output_dir: Path):
    """extract clinical features for all samples."""
    clinical_config = config.get('clinical_features', {})
    
    extractor = ClinicalFeatureExtractor(
        sample_rate=config['data']['target_sr'],
        f0_min=clinical_config.get('f0_min', 75),
        f0_max=clinical_config.get('f0_max', 500)
    )
    
    features_list = []
    
    logger.info("extracting clinical features...")
    
    for i in range(min(len(dataset), 1000)):  # limit for demo
        sample = dataset[i]
        audio = sample['input_values'].numpy()
        
        features = extractor.extract_features(audio)
        features['label'] = sample['label']
        features['subject_id'] = sample.get('subject_id', f'unknown_{i}')
        
        features_list.append(features)
    
    # save features
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import pandas as pd
    features_df = pd.DataFrame(features_list)
    features_df.to_csv(output_dir / 'clinical_features.csv', index=False)
    
    logger.info(f"extracted features for {len(features_list)} samples")
    logger.info(f"features: {get_clinical_feature_names()}")
    
    return features_df


def create_minimal_pairs(dataset, config: dict, output_dir: Path):
    """create mfcc-matched minimal pairs for patching."""
    patching_config = config.get('patching', {})
    n_pairs = patching_config.get('n_pairs', 50)
    
    logger.info(f"creating {n_pairs} mfcc-matched minimal pairs...")
    
    pairs = create_mfcc_matched_pairs(
        dataset,
        n_pairs=n_pairs,
        same_task=True,
        n_mfcc=13
    )
    
    logger.info(f"created {len(pairs)} minimal pairs")
    
    # save pairs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pairs_data = []
    for hc_audio, pd_audio, label, distance in pairs:
        pairs_data.append({
            'hc_audio': hc_audio.numpy(),
            'pd_audio': pd_audio.numpy(),
            'label': label,
            'mfcc_distance': distance
        })
    
    torch.save(pairs_data, output_dir / 'minimal_pairs.pt')
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description='phase 1: data preparation')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml',
                        help='path to config file')
    parser.add_argument('--output', type=str, default='results/phase1',
                        help='output directory')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='skip preprocessing step')
    parser.add_argument('--skip-clinical', action='store_true',
                        help='skip clinical feature extraction')
    parser.add_argument('--skip-pairs', action='store_true',
                        help='skip minimal pair creation')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("phase 1: dataset preparation and baseline establishment")
    logger.info("=" * 60)
    
    # step 1: load datasets
    logger.info("\n[step 1] loading datasets...")
    datasets = load_datasets(config)
    
    if not datasets:
        logger.error("no datasets found! check paths in config.")
        return
    
    # combine datasets
    dataset_list = list(datasets.values())
    if len(dataset_list) > 1:
        combined = CombinedPDDataset(dataset_list)
    else:
        combined = dataset_list[0]
    
    logger.info(f"total samples: {len(combined)}")
    
    # step 2: preprocess
    if not args.skip_preprocessing:
        logger.info("\n[step 2] preprocessing and segmentation...")
        processed = preprocess_dataset(
            combined, config, 
            output_dir / 'processed'
        )
    
    # step 3: extract clinical features
    if not args.skip_clinical:
        logger.info("\n[step 3] extracting clinical features...")
        features_df = extract_clinical_features(
            combined, config,
            output_dir / 'clinical'
        )
    
    # step 4: create minimal pairs
    if not args.skip_pairs:
        logger.info("\n[step 4] creating minimal pairs...")
        pairs = create_minimal_pairs(
            combined, config,
            output_dir / 'pairs'
        )
    
    # save summary
    summary = {
        'datasets_loaded': list(datasets.keys()),
        'total_samples': len(combined),
        'config': config
    }
    
    with open(output_dir / 'phase1_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info("\n" + "=" * 60)
    logger.info("phase 1 complete!")
    logger.info(f"outputs saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
