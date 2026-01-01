#!/usr/bin/env python
"""
phase 3: activation patching experiments.

this script:
1. loads minimal pairs from phase 1
2. runs activation patching experiments
3. identifies causally important layers
4. computes causal contributions
5. generates patching visualizations

usage:
    python scripts/run_phase3.py --config configs/experiment_config.yaml
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

from src.interpretability.patching import (
    ActivationPatcher,
    AttentionHeadPatcher,
    compute_patching_importance,
    compute_causal_contribution
)
from src.utils.visualization import plot_patching_results
from src.utils.analysis import analyze_patching_results


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """load yaml configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_minimal_pairs(pairs_path: Path) -> list:
    """load minimal pairs created in phase 1."""
    pairs_data = torch.load(pairs_path, weights_only=False)
    
    pairs = []
    for p in pairs_data:
        pairs.append((
            torch.from_numpy(p['hc_audio']),
            torch.from_numpy(p['pd_audio']),
            p['label'],
            p['mfcc_distance']
        ))
    
    return pairs


def run_layerwise_patching(
    patcher: ActivationPatcher,
    pairs: list,
    config: dict,
    output_dir: Path
) -> dict:
    """run activation patching for each layer."""
    patching_config = config.get('patching', {})
    n_layers = patching_config.get('n_layers', 12)
    
    logger.info(f"running layer-wise activation patching on {len(pairs)} pairs...")
    
    results = {}
    
    for layer_idx in range(n_layers):
        logger.info(f"patching layer {layer_idx}...")
        
        recoveries = []
        
        for hc_audio, pd_audio, label, distance in pairs:
            try:
                recovery = patcher.patch_and_measure(
                    clean_input=hc_audio,
                    corrupted_input=pd_audio,
                    layer_idx=layer_idx
                )
                recoveries.append(recovery)
            except Exception as e:
                logger.warning(f"patching failed for pair: {e}")
                continue
        
        if recoveries:
            results[layer_idx] = {
                'mean_recovery': float(np.mean(recoveries)),
                'std_recovery': float(np.std(recoveries)),
                'median_recovery': float(np.median(recoveries)),
                'n_pairs': len(recoveries),
                'recoveries': recoveries
            }
            
            logger.info(f"  layer {layer_idx}: recovery={results[layer_idx]['mean_recovery']:.3f} "
                       f"± {results[layer_idx]['std_recovery']:.3f}")
    
    # save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # save without full recoveries list for json
    results_summary = {
        layer_idx: {k: v for k, v in data.items() if k != 'recoveries'}
        for layer_idx, data in results.items()
    }
    
    with open(output_dir / 'patching_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # save full results
    torch.save(results, output_dir / 'patching_results_full.pt')
    
    return results


def run_attention_head_patching(
    model,
    pairs: list,
    config: dict,
    output_dir: Path
) -> dict:
    """run attention head patching experiments."""
    patching_config = config.get('patching', {})
    n_layers = patching_config.get('n_layers', 12)
    n_heads = patching_config.get('n_heads', 12)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"running attention head patching ({n_layers} layers × {n_heads} heads)...")
    
    head_patcher = AttentionHeadPatcher(model, device=device)
    
    results = {}
    
    for layer_idx in range(n_layers):
        results[layer_idx] = {}
        
        for head_idx in range(n_heads):
            recoveries = []
            
            for hc_audio, pd_audio, label, distance in pairs[:20]:  # limit for speed
                try:
                    recovery = head_patcher.patch_head_and_measure(
                        clean_input=hc_audio,
                        corrupted_input=pd_audio,
                        layer_idx=layer_idx,
                        head_idx=head_idx
                    )
                    recoveries.append(recovery)
                except Exception as e:
                    continue
            
            if recoveries:
                results[layer_idx][head_idx] = {
                    'mean_recovery': float(np.mean(recoveries)),
                    'std_recovery': float(np.std(recoveries)),
                    'n_pairs': len(recoveries)
                }
        
        logger.info(f"  layer {layer_idx} complete")
    
    # save
    with open(output_dir / 'head_patching_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def identify_important_components(
    layer_results: dict,
    head_results: dict,
    config: dict
) -> dict:
    """identify causally important layers and heads."""
    patching_config = config.get('patching', {})
    threshold = patching_config.get('importance_threshold', 0.5)
    
    # important layers
    important_layers = compute_patching_importance(layer_results, threshold)
    
    # causal contributions
    layer_recoveries = {l: r['mean_recovery'] for l, r in layer_results.items()}
    contributions = compute_causal_contribution(layer_recoveries)
    
    # important heads
    important_heads = []
    for layer_idx, heads in head_results.items():
        for head_idx, data in heads.items():
            if data['mean_recovery'] > threshold:
                important_heads.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'recovery': data['mean_recovery']
                })
    
    # sort by importance
    important_heads.sort(key=lambda x: x['recovery'], reverse=True)
    
    return {
        'important_layers': important_layers,
        'layer_contributions': contributions,
        'important_heads': important_heads[:20],  # top 20
        'threshold': threshold
    }


def main():
    parser = argparse.ArgumentParser(description='phase 3: patching experiments')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml',
                        help='path to config file')
    parser.add_argument('--phase1-output', type=str, default='results/phase1',
                        help='phase 1 output directory')
    parser.add_argument('--output', type=str, default='results/phase3',
                        help='output directory')
    parser.add_argument('--skip-heads', action='store_true',
                        help='skip attention head patching (slow)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output)
    phase1_dir = Path(args.phase1_output)
    
    logger.info("=" * 60)
    logger.info("phase 3: activation patching experiments")
    logger.info("=" * 60)
    
    # load minimal pairs
    pairs_path = phase1_dir / 'pairs' / 'minimal_pairs.pt'
    if not pairs_path.exists():
        logger.error(f"minimal pairs not found: {pairs_path}")
        logger.error("run phase 1 first!")
        return
    
    pairs = load_minimal_pairs(pairs_path)
    logger.info(f"loaded {len(pairs)} minimal pairs")
    
    # log mfcc distances
    distances = [p[3] for p in pairs]
    logger.info(f"mfcc distances: min={min(distances):.2f}, "
               f"max={max(distances):.2f}, mean={np.mean(distances):.2f}")
    
    # setup patcher
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'facebook/wav2vec2-base-960h')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"loading model: {model_name}")
    
    from transformers import Wav2Vec2ForSequenceClassification
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    patcher = ActivationPatcher(model, device=device)
    
    # step 1: layer-wise patching
    logger.info("\n[step 1] layer-wise activation patching...")
    layer_results = run_layerwise_patching(
        patcher, pairs, config,
        output_dir / 'layers'
    )
    
    # step 2: attention head patching (optional)
    if not args.skip_heads:
        logger.info("\n[step 2] attention head patching...")
        head_results = run_attention_head_patching(
            model, pairs, config,
            output_dir / 'heads'
        )
    else:
        head_results = {}
    
    # step 3: identify important components
    logger.info("\n[step 3] identifying important components...")
    importance = identify_important_components(layer_results, head_results, config)
    
    logger.info(f"important layers: {importance['important_layers']}")
    logger.info(f"layer contributions: {importance['layer_contributions']}")
    
    if importance['important_heads']:
        logger.info(f"top 5 important heads:")
        for h in importance['important_heads'][:5]:
            logger.info(f"  layer {h['layer']}, head {h['head']}: {h['recovery']:.3f}")
    
    # save importance analysis
    with open(output_dir / 'importance_analysis.json', 'w') as f:
        json.dump(importance, f, indent=2)
    
    # step 4: generate visualizations
    logger.info("\n[step 4] generating visualizations...")
    
    fig = plot_patching_results(
        layer_results,
        title="activation patching: logit difference recovery",
        save_path=str(output_dir / 'patching_results.png')
    )
    
    # statistical analysis
    analysis = analyze_patching_results(layer_results)
    
    with open(output_dir / 'patching_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"\npatching analysis:")
    logger.info(f"  important layers: {analysis['important_layers']}")
    logger.info(f"  best layer: {analysis['best_layer']} "
               f"(recovery={analysis['max_recovery']:.3f})")
    logger.info(f"  importance ratio: {analysis['importance_ratio']:.2%}")
    
    # summary
    summary = {
        'n_pairs': len(pairs),
        'layer_results_summary': {
            l: {k: v for k, v in r.items() if k != 'recoveries'}
            for l, r in layer_results.items()
        },
        'importance': importance,
        'analysis': analysis
    }
    
    with open(output_dir / 'phase3_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("phase 3 complete!")
    logger.info(f"outputs saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
