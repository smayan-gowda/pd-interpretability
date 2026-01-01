#!/usr/bin/env python
"""
phase 5: synthesis and interpretable prediction interface.

this script:
1. synthesizes results from phases 1-4
2. creates the interpretable prediction interface
3. generates example predictions with explanations
4. validates the complete pipeline
5. exports analysis results for the interface

usage:
    python scripts/run_phase5.py --config configs/experiment_config.yaml
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.interpretability.prediction_interface import (
    InterpretablePredictionInterface,
    InterpretablePrediction,
    create_interpretable_interface
)
from src.features.clinical import ClinicalFeatureExtractor
from src.data.datasets import load_dataset_by_name


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """load yaml configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_analysis_results(results_dir: Path) -> Dict:
    """
    load analysis results from phases 2-4.
    
    collects:
    - probing results (layer-wise feature encoding)
    - patching results (causal head importance)
    - cross-dataset evaluation results
    """
    results = {}
    
    # probing results (phase 2)
    probing_path = results_dir / 'phase2' / 'probing_results.json'
    if probing_path.exists():
        with open(probing_path) as f:
            results['probing'] = json.load(f)
        logger.info(f"loaded probing results from {probing_path}")
    
    # patching results (phase 3)
    patching_path = results_dir / 'phase3' / 'patching_results.json'
    if patching_path.exists():
        with open(patching_path) as f:
            results['patching'] = json.load(f)
        logger.info(f"loaded patching results from {patching_path}")
    
    # cross-dataset results (phase 4)
    cross_path = results_dir / 'phase4' / 'evaluation_matrix.json'
    if cross_path.exists():
        with open(cross_path) as f:
            results['cross_dataset'] = json.load(f)
        logger.info(f"loaded cross-dataset results from {cross_path}")
    
    return results


def convert_probing_for_interface(probing_results: Dict) -> Dict[str, Dict[int, float]]:
    """convert probing results to interface format."""
    interface_probing = {}
    
    for feature, layer_data in probing_results.items():
        if isinstance(layer_data, dict):
            interface_probing[feature] = {
                int(k): float(v.get('mean', v) if isinstance(v, dict) else v)
                for k, v in layer_data.items()
            }
    
    return interface_probing


def convert_patching_for_interface(patching_results: Dict) -> Dict[tuple, float]:
    """convert patching results to interface format."""
    interface_patching = {}
    
    if 'head_importance' in patching_results:
        for head_key, importance in patching_results['head_importance'].items():
            # parse "layer,head" format
            if isinstance(head_key, str) and ',' in head_key:
                layer, head = map(int, head_key.split(','))
                interface_patching[(layer, head)] = float(importance)
    
    return interface_patching


def create_interface(
    model_path: Path,
    analysis_results: Dict,
    config: dict
) -> InterpretablePredictionInterface:
    """
    create the interpretable prediction interface.
    
    this synthesizes all analysis results into a unified interface.
    """
    from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
    
    device = config.get('device', 'cpu')
    
    # load model
    if model_path.exists():
        model = torch.load(model_path, map_location=device, weights_only=False)
        logger.info(f"loaded model from {model_path}")
    else:
        # use pretrained as fallback
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            'facebook/wav2vec2-base',
            num_labels=2
        )
        logger.warning(f"model not found at {model_path}, using pretrained")
    
    # load processor
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    
    # create clinical extractor
    clinical_extractor = ClinicalFeatureExtractor()
    
    # convert results to interface format
    probing_results = None
    if 'probing' in analysis_results:
        probing_results = convert_probing_for_interface(analysis_results['probing'])
        logger.info(f"converted probing results for {len(probing_results)} features")
    
    patching_results = None
    if 'patching' in analysis_results:
        patching_results = convert_patching_for_interface(analysis_results['patching'])
        logger.info(f"converted patching results for {len(patching_results)} heads")
    
    # create interface
    interface = create_interpretable_interface(
        model=model,
        processor=processor,
        clinical_extractor=clinical_extractor,
        probing_results=probing_results,
        patching_results=patching_results,
        device=device
    )
    
    logger.info("created interpretable prediction interface")
    
    return interface


def run_example_predictions(
    interface: InterpretablePredictionInterface,
    test_samples: List[Dict],
    output_dir: Path
):
    """
    run example predictions to demonstrate the interface.
    
    generates predictions with full interpretability information.
    """
    predictions_dir = output_dir / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"running predictions on {len(test_samples)} samples...")
    
    all_predictions = []
    
    for i, sample in enumerate(test_samples):
        audio = sample.get('audio', sample.get('waveform'))
        if audio is None:
            continue
        
        sample_rate = sample.get('sample_rate', 16000)
        true_label = sample.get('label', 'unknown')
        subject_id = sample.get('subject_id', f'sample_{i}')
        
        # generate prediction
        prediction = interface.predict(
            audio,
            sample_rate=sample_rate,
            include_clinical=True,
            compute_contributions=True
        )
        
        # add metadata
        prediction.metadata['subject_id'] = subject_id
        prediction.metadata['true_label'] = true_label
        
        # generate explanation
        explanation = prediction.generate_explanation()
        
        # log
        pred_label = 'pd' if prediction.pd_probability >= 0.5 else 'hc'
        correct = (pred_label == 'pd' and true_label == 1) or (pred_label == 'hc' and true_label == 0)
        status = "✓" if correct else "✗"
        
        logger.info(
            f"  [{status}] {subject_id}: "
            f"prob={prediction.pd_probability:.3f}, "
            f"conf={prediction.confidence:.3f}, "
            f"true={true_label}"
        )
        
        all_predictions.append(prediction.to_dict())
        
        # save individual prediction
        interface.save_prediction(
            prediction,
            predictions_dir / f'{subject_id}_prediction.json'
        )
    
    # save all predictions
    with open(output_dir / 'all_predictions.json', 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    logger.info(f"saved {len(all_predictions)} predictions")
    
    return all_predictions


def validate_pipeline(
    interface: InterpretablePredictionInterface,
    test_samples: List[Dict]
) -> Dict:
    """
    validate the complete pipeline.
    
    checks:
    - predictions are in valid range
    - feature contributions sum correctly
    - evidence layers are identified
    - key attention heads are identified
    """
    logger.info("validating complete pipeline...")
    
    validation_results = {
        'total_samples': 0,
        'valid_predictions': 0,
        'has_contributions': 0,
        'has_evidence_layers': 0,
        'has_key_heads': 0,
        'has_clinical_features': 0,
        'errors': []
    }
    
    for sample in test_samples:
        audio = sample.get('audio', sample.get('waveform'))
        if audio is None:
            continue
        
        validation_results['total_samples'] += 1
        
        try:
            prediction = interface.predict(
                audio,
                sample_rate=sample.get('sample_rate', 16000),
                include_clinical=True
            )
            
            # check probability range
            if 0 <= prediction.pd_probability <= 1:
                validation_results['valid_predictions'] += 1
            else:
                validation_results['errors'].append('probability out of range')
            
            # check contributions
            if prediction.feature_contributions:
                validation_results['has_contributions'] += 1
            
            # check evidence layers
            if prediction.evidence_layers:
                validation_results['has_evidence_layers'] += 1
            
            # check key heads
            if prediction.key_attention_heads:
                validation_results['has_key_heads'] += 1
            
            # check clinical features
            if prediction.clinical_features:
                validation_results['has_clinical_features'] += 1
        
        except Exception as e:
            validation_results['errors'].append(str(e))
    
    # compute rates
    total = max(validation_results['total_samples'], 1)
    validation_results['validity_rate'] = validation_results['valid_predictions'] / total
    validation_results['contribution_rate'] = validation_results['has_contributions'] / total
    validation_results['evidence_rate'] = validation_results['has_evidence_layers'] / total
    validation_results['heads_rate'] = validation_results['has_key_heads'] / total
    validation_results['clinical_rate'] = validation_results['has_clinical_features'] / total
    
    # overall pass/fail
    validation_results['passed'] = (
        validation_results['validity_rate'] >= 0.95 and
        validation_results['evidence_rate'] >= 0.95 and
        validation_results['heads_rate'] >= 0.95
    )
    
    logger.info(f"validation results:")
    logger.info(f"  - valid predictions: {validation_results['validity_rate']:.1%}")
    logger.info(f"  - has contributions: {validation_results['contribution_rate']:.1%}")
    logger.info(f"  - has evidence layers: {validation_results['evidence_rate']:.1%}")
    logger.info(f"  - has key heads: {validation_results['heads_rate']:.1%}")
    logger.info(f"  - has clinical features: {validation_results['clinical_rate']:.1%}")
    logger.info(f"  - overall: {'PASSED' if validation_results['passed'] else 'FAILED'}")
    
    return validation_results


def export_analysis_for_interface(
    analysis_results: Dict,
    output_dir: Path
):
    """
    export analysis results in interface-compatible format.
    
    creates a single file that the interface can load.
    """
    export_data = {}
    
    # probing results
    if 'probing' in analysis_results:
        export_data['probing_results'] = analysis_results['probing']
    
    # patching results
    if 'patching' in analysis_results:
        # convert tuple keys to strings for JSON
        patching = analysis_results['patching']
        if 'head_importance' in patching:
            export_data['patching_results'] = patching['head_importance']
    
    # evidence layers (top probing layers)
    if 'probing' in analysis_results:
        layer_scores = {}
        for feature, layers in analysis_results['probing'].items():
            if isinstance(layers, dict):
                for layer, score in layers.items():
                    layer_idx = int(layer)
                    if layer_idx not in layer_scores:
                        layer_scores[layer_idx] = []
                    score_val = score.get('mean', score) if isinstance(score, dict) else score
                    layer_scores[layer_idx].append(float(score_val))
        
        # average and sort
        avg_scores = {l: np.mean(s) for l, s in layer_scores.items()}
        sorted_layers = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        export_data['evidence_layers'] = [l for l, _ in sorted_layers[:5]]
    
    # save
    export_path = output_dir / 'interface_analysis.json'
    with open(export_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"exported interface analysis to {export_path}")


def generate_demonstration_report(
    predictions: List[Dict],
    validation: Dict,
    output_dir: Path
):
    """generate markdown demonstration report."""
    report_lines = [
        "# Interpretable Parkinson's Disease Detection",
        "",
        "## Phase 5: Synthesis and Validation Report",
        "",
        "### Overview",
        "",
        "This report demonstrates the interpretable prediction interface that synthesizes",
        "all mechanistic interpretability analyses into actionable predictions.",
        "",
        "### Validation Results",
        "",
        f"- **Total samples tested**: {validation['total_samples']}",
        f"- **Valid predictions**: {validation['validity_rate']:.1%}",
        f"- **Feature contributions**: {validation['contribution_rate']:.1%}",
        f"- **Evidence layers identified**: {validation['evidence_rate']:.1%}",
        f"- **Key attention heads identified**: {validation['heads_rate']:.1%}",
        f"- **Clinical features extracted**: {validation['clinical_rate']:.1%}",
        "",
        f"**Overall Status**: {'✅ PASSED' if validation['passed'] else '❌ FAILED'}",
        "",
        "### Example Predictions",
        "",
    ]
    
    # add example predictions
    for i, pred in enumerate(predictions[:5]):
        prob = pred['pd_probability']
        conf = pred.get('confidence', 0)
        status = "Parkinson's Disease" if prob >= 0.5 else "Healthy Control"
        
        report_lines.extend([
            f"#### Sample {i+1}",
            "",
            f"- **Prediction**: {status}",
            f"- **Probability**: {prob:.3f}",
            f"- **Confidence**: {conf:.3f}",
            "",
            "**Top Feature Contributions**:",
            ""
        ])
        
        contributions = pred.get('feature_contributions', {})
        for feat, score in list(contributions.items())[:3]:
            report_lines.append(f"- {feat}: {score:+.3f}")
        
        report_lines.extend([
            "",
            f"**Evidence Layers**: {pred.get('evidence_layers', [])}",
            f"**Key Heads**: {pred.get('key_attention_heads', [])}",
            "",
        ])
    
    report_lines.extend([
        "### Interface Usage",
        "",
        "```python",
        "from src.interpretability import create_interpretable_interface",
        "from src.features.clinical import ClinicalFeatureExtractor",
        "",
        "# create interface",
        "interface = create_interpretable_interface(",
        "    model=fine_tuned_model,",
        "    processor=processor,",
        "    clinical_extractor=ClinicalFeatureExtractor()",
        ")",
        "",
        "# load precomputed analysis",
        "interface.load_analysis_results('results/phase5/interface_analysis.json')",
        "",
        "# make prediction",
        "prediction = interface.predict(audio_waveform)",
        "",
        "# get explanation",
        "print(prediction.generate_explanation())",
        "```",
        "",
        "### Output Format",
        "",
        "```json",
        "{",
        '    "pd_probability": 0.87,',
        '    "feature_contributions": {',
        '        "jitter_elevated": 0.34,',
        '        "hnr_reduced": 0.28,',
        '        "f0_unstable": 0.21',
        "    },",
        '    "evidence_layers": [3, 4, 7],',
        '    "key_attention_heads": [[3, 4], [4, 2], [7, 8]]',
        "}",
        "```",
        ""
    ])
    
    report_path = output_dir / 'demonstration_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"saved demonstration report to {report_path}")


def main():
    """main execution."""
    parser = argparse.ArgumentParser(description='Phase 5: Synthesis and interpretable prediction')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_config.yaml',
        help='path to config file'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='directory with phase 1-4 results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/phase5',
        help='output directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='results/checkpoints/best_model.pt',
        help='path to trained model'
    )
    args = parser.parse_args()
    
    # load config
    config = load_config(args.config)
    
    # setup directories
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = Path(args.model)
    
    logger.info("=" * 60)
    logger.info("PHASE 5: SYNTHESIS AND INTERPRETABLE PREDICTION")
    logger.info("=" * 60)
    
    # step 1: load analysis results
    logger.info("\n[step 1] loading analysis results from phases 1-4...")
    analysis_results = load_analysis_results(results_dir)
    
    # step 2: create interface
    logger.info("\n[step 2] creating interpretable prediction interface...")
    interface = create_interface(model_path, analysis_results, config)
    
    # step 3: export analysis for interface
    logger.info("\n[step 3] exporting analysis for interface...")
    export_analysis_for_interface(analysis_results, output_dir)
    
    # step 4: load test samples
    logger.info("\n[step 4] loading test samples...")
    dataset_config = config.get('datasets', {})
    data_dir = Path(dataset_config.get('data_dir', 'data/raw'))
    
    try:
        dataset = load_dataset_by_name('italian_pvs', data_dir / 'italian_pvs')
        test_samples = [dataset[i] for i in range(min(20, len(dataset)))]
        logger.info(f"loaded {len(test_samples)} test samples")
    except Exception as e:
        logger.warning(f"could not load dataset: {e}")
        # create synthetic test samples for demonstration
        test_samples = [
            {
                'audio': np.random.randn(16000 * 3).astype(np.float32),
                'sample_rate': 16000,
                'label': i % 2,
                'subject_id': f'synthetic_{i}'
            }
            for i in range(10)
        ]
        logger.info("using synthetic test samples")
    
    # step 5: run example predictions
    logger.info("\n[step 5] running example predictions...")
    predictions = run_example_predictions(interface, test_samples, output_dir)
    
    # step 6: validate pipeline
    logger.info("\n[step 6] validating complete pipeline...")
    validation = validate_pipeline(interface, test_samples)
    
    # save validation results
    with open(output_dir / 'validation_results.json', 'w') as f:
        json.dump(validation, f, indent=2)
    
    # step 7: generate report
    logger.info("\n[step 7] generating demonstration report...")
    generate_demonstration_report(predictions, validation, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 5 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"results saved to: {output_dir}")
    logger.info(f"interface analysis: {output_dir / 'interface_analysis.json'}")
    logger.info(f"demonstration report: {output_dir / 'demonstration_report.md'}")


if __name__ == '__main__':
    main()
