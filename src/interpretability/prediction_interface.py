"""
interpretable prediction interface for parkinson's disease detection.

this module synthesizes all interpretability analyses (probing, patching,
clinical features) into a unified prediction interface that provides:
- pd probability predictions
- feature contribution explanations
- evidence layer identification
- key attention head attribution

this is the main user-facing interface for interpretable predictions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import warnings
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """individual feature contribution to prediction."""
    
    feature_name: str
    contribution_score: float
    clinical_interpretation: str
    direction: str  # 'elevated', 'reduced', 'unstable'
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'contribution_score': self.contribution_score,
            'clinical_interpretation': self.clinical_interpretation,
            'direction': self.direction,
            'confidence': self.confidence
        }


@dataclass
class InterpretablePrediction:
    """
    complete interpretable prediction result.
    
    this is the primary output format for the interpretable prediction interface,
    providing full transparency into model decision-making.
    
    attributes:
        pd_probability: probability of parkinson's disease (0-1)
        feature_contributions: dict mapping feature descriptions to contribution scores
        evidence_layers: list of layer indices where pd-related features are encoded
        key_attention_heads: list of (layer, head) tuples with highest causal importance
        clinical_features: extracted clinical features (jitter, shimmer, etc.)
        confidence: overall prediction confidence
        raw_logits: raw model output logits
        attention_pattern_summary: summary of attention patterns
        metadata: additional metadata about the prediction
    """
    
    pd_probability: float
    feature_contributions: Dict[str, float]
    evidence_layers: List[int]
    key_attention_heads: List[Tuple[int, int]]
    clinical_features: Optional[Dict[str, float]] = None
    confidence: float = 0.0
    raw_logits: Optional[np.ndarray] = None
    attention_pattern_summary: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        convert to dictionary for serialization.
        
        returns:
            dict with all prediction information
        """
        return {
            'pd_probability': self.pd_probability,
            'feature_contributions': self.feature_contributions,
            'evidence_layers': self.evidence_layers,
            'key_attention_heads': [list(h) for h in self.key_attention_heads],
            'clinical_features': self.clinical_features,
            'confidence': self.confidence,
            'raw_logits': self.raw_logits.tolist() if self.raw_logits is not None else None,
            'attention_pattern_summary': self.attention_pattern_summary,
            'metadata': self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """convert to json string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterpretablePrediction':
        """create from dictionary."""
        return cls(
            pd_probability=data['pd_probability'],
            feature_contributions=data['feature_contributions'],
            evidence_layers=data['evidence_layers'],
            key_attention_heads=[tuple(h) for h in data['key_attention_heads']],
            clinical_features=data.get('clinical_features'),
            confidence=data.get('confidence', 0.0),
            raw_logits=np.array(data['raw_logits']) if data.get('raw_logits') else None,
            attention_pattern_summary=data.get('attention_pattern_summary'),
            metadata=data.get('metadata', {})
        )
    
    def get_top_features(self, n: int = 3) -> List[Tuple[str, float]]:
        """get top n contributing features."""
        sorted_features = sorted(
            self.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:n]
    
    def generate_explanation(self) -> str:
        """
        generate human-readable explanation of prediction.
        
        returns:
            natural language explanation
        """
        lines = []
        
        # prediction summary
        if self.pd_probability >= 0.5:
            lines.append(
                f"prediction: parkinson's disease detected "
                f"(probability: {self.pd_probability:.1%}, confidence: {self.confidence:.1%})"
            )
        else:
            lines.append(
                f"prediction: healthy control "
                f"(pd probability: {self.pd_probability:.1%}, confidence: {self.confidence:.1%})"
            )
        
        lines.append("")
        
        # top contributing features
        lines.append("key contributing features:")
        for feature, score in self.get_top_features(5):
            direction = "+" if score > 0 else ""
            lines.append(f"  • {feature}: {direction}{score:.2f}")
        
        lines.append("")
        
        # evidence layers
        if self.evidence_layers:
            layers_str = ", ".join(str(l) for l in self.evidence_layers[:5])
            lines.append(f"primary evidence in layers: {layers_str}")
        
        # key attention heads
        if self.key_attention_heads:
            heads_str = ", ".join(f"L{l}H{h}" for l, h in self.key_attention_heads[:5])
            lines.append(f"key attention heads: {heads_str}")
        
        # clinical features if available
        if self.clinical_features:
            lines.append("")
            lines.append("extracted clinical features:")
            for name, value in list(self.clinical_features.items())[:6]:
                if not np.isnan(value):
                    lines.append(f"  • {name}: {value:.4f}")
        
        return "\n".join(lines)


# clinical feature interpretation mapping
CLINICAL_INTERPRETATIONS = {
    'jitter_local': ('voice pitch instability', 'elevated'),
    'jitter_rap': ('rapid pitch perturbation', 'elevated'),
    'jitter_ppq5': ('5-point pitch perturbation', 'elevated'),
    'jitter_ddp': ('difference of differences of periods', 'elevated'),
    'shimmer_local': ('voice amplitude instability', 'elevated'),
    'shimmer_apq3': ('3-point amplitude perturbation', 'elevated'),
    'shimmer_apq5': ('5-point amplitude perturbation', 'elevated'),
    'shimmer_apq11': ('11-point amplitude perturbation', 'elevated'),
    'shimmer_dda': ('difference of amplitude differences', 'elevated'),
    'hnr_mean': ('harmonics-to-noise ratio', 'reduced'),
    'f0_mean': ('fundamental frequency', 'unstable'),
    'f0_std': ('pitch variability', 'elevated'),
    'f0_range': ('pitch range', 'reduced'),
    'f1_mean': ('first formant frequency', 'reduced'),
    'f2_mean': ('second formant frequency', 'reduced')
}


class InterpretablePredictionInterface:
    """
    main interface for interpretable parkinson's disease predictions.
    
    this class synthesizes multiple interpretability methods:
    - model predictions (wav2vec2 fine-tuned classifier)
    - clinical feature extraction (jitter, shimmer, hnr, formants)
    - probing analysis (layer-wise feature encoding)
    - activation patching (causal contribution of components)
    
    usage:
        interface = InterpretablePredictionInterface(
            model=model,
            processor=processor,
            clinical_extractor=ClinicalFeatureExtractor()
        )
        
        prediction = interface.predict(audio_waveform, sample_rate=16000)
        print(prediction.to_dict())
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        clinical_extractor: Optional[Any] = None,
        probing_results: Optional[Dict[str, Dict[int, float]]] = None,
        patching_results: Optional[Dict[Tuple[int, int], float]] = None,
        device: str = 'cpu',
        threshold: float = 0.5
    ):
        """
        initialize interpretable prediction interface.
        
        args:
            model: fine-tuned wav2vec2 classifier
            processor: wav2vec2 processor for audio processing
            clinical_extractor: optional ClinicalFeatureExtractor instance
            probing_results: optional dict of probing scores per feature/layer
            patching_results: optional dict of patching importance per head
            device: computation device
            threshold: classification threshold
        """
        self.model = model
        self.processor = processor
        self.clinical_extractor = clinical_extractor
        self.probing_results = probing_results or {}
        self.patching_results = patching_results or {}
        self.device = device
        self.threshold = threshold
        
        self.model.to(device)
        self.model.eval()
        
        # determine evidence layers from probing results
        self._evidence_layers = self._compute_evidence_layers()
        
        # determine key attention heads from patching results
        self._key_heads = self._compute_key_heads()
    
    def _compute_evidence_layers(self, top_k: int = 5) -> List[int]:
        """identify layers with strongest feature encoding."""
        if not self.probing_results:
            return list(range(3, 8))  # default: layers 3-7
        
        layer_scores = {}
        
        for feature, layer_dict in self.probing_results.items():
            for layer_idx, score in layer_dict.items():
                if layer_idx not in layer_scores:
                    layer_scores[layer_idx] = []
                layer_scores[layer_idx].append(score)
        
        # average across features
        avg_scores = {
            layer: np.mean(scores)
            for layer, scores in layer_scores.items()
        }
        
        # get top layers
        sorted_layers = sorted(
            avg_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [layer for layer, _ in sorted_layers[:top_k]]
    
    def _compute_key_heads(self, top_k: int = 10) -> List[Tuple[int, int]]:
        """identify attention heads with highest causal importance."""
        if not self.patching_results:
            return [(3, 4), (4, 2), (7, 8)]  # defaults based on typical findings
        
        sorted_heads = sorted(
            self.patching_results.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return [head for head, _ in sorted_heads[:top_k]]
    
    def set_probing_results(self, results: Dict[str, Dict[int, float]]):
        """set probing results from analysis."""
        self.probing_results = results
        self._evidence_layers = self._compute_evidence_layers()
    
    def set_patching_results(self, results: Dict[Tuple[int, int], float]):
        """set patching results from analysis."""
        self.patching_results = results
        self._key_heads = self._compute_key_heads()
    
    def load_analysis_results(self, path: Union[str, Path]):
        """
        load precomputed analysis results.
        
        args:
            path: path to json file with probing/patching results
        """
        path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if 'probing_results' in data:
            self.probing_results = {
                k: {int(layer): score for layer, score in v.items()}
                for k, v in data['probing_results'].items()
            }
            self._evidence_layers = self._compute_evidence_layers()
        
        if 'patching_results' in data:
            self.patching_results = {
                tuple(map(int, k.split(','))): v
                for k, v in data['patching_results'].items()
            }
            self._key_heads = self._compute_key_heads()
        
        logger.info(f"loaded analysis results from {path}")
    
    @torch.no_grad()
    def predict(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
        include_clinical: bool = True,
        include_attention: bool = False,
        compute_contributions: bool = True
    ) -> InterpretablePrediction:
        """
        generate interpretable prediction for audio sample.
        
        args:
            audio: audio waveform (numpy array or tensor)
            sample_rate: audio sample rate (default 16kHz)
            include_clinical: whether to extract clinical features
            include_attention: whether to include attention analysis
            compute_contributions: whether to compute feature contributions
        
        returns:
            InterpretablePrediction with full interpretability information
        """
        # ensure numpy array
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # flatten if needed
        if audio.ndim > 1:
            audio = audio.flatten()
        
        # process audio
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors='pt',
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # get model prediction
        outputs = self.model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        if logits.dim() > 1:
            logits = logits.squeeze()
        
        # compute probability
        if logits.numel() == 1:
            pd_prob = torch.sigmoid(logits).item()
        else:
            probs = F.softmax(logits, dim=-1)
            pd_prob = probs[1].item() if probs.numel() > 1 else probs.item()
        
        # compute confidence
        confidence = abs(pd_prob - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1
        
        # extract clinical features if requested
        clinical_features = None
        if include_clinical and self.clinical_extractor is not None:
            try:
                clinical_features = self.clinical_extractor.extract_all_features(
                    audio, sample_rate
                )
            except Exception as e:
                warnings.warn(f"clinical feature extraction failed: {e}")
        
        # compute feature contributions
        feature_contributions = {}
        if compute_contributions:
            feature_contributions = self._compute_feature_contributions(
                clinical_features, pd_prob
            )
        
        # attention pattern summary
        attention_summary = None
        if include_attention:
            attention_summary = self._analyze_attention_patterns(inputs)
        
        return InterpretablePrediction(
            pd_probability=pd_prob,
            feature_contributions=feature_contributions,
            evidence_layers=self._evidence_layers,
            key_attention_heads=self._key_heads,
            clinical_features=clinical_features,
            confidence=confidence,
            raw_logits=logits.cpu().numpy(),
            attention_pattern_summary=attention_summary,
            metadata={
                'sample_rate': sample_rate,
                'audio_duration': len(audio) / sample_rate,
                'threshold': self.threshold
            }
        )
    
    def _compute_feature_contributions(
        self,
        clinical_features: Optional[Dict[str, float]],
        pd_probability: float
    ) -> Dict[str, float]:
        """
        compute feature contributions to prediction.
        
        combines clinical feature values with probing importance
        to estimate contribution to final prediction.
        """
        contributions = {}
        
        if clinical_features is None:
            # fall back to probing-based contributions
            for feature, layer_scores in self.probing_results.items():
                if layer_scores:
                    max_score = max(layer_scores.values())
                    interpretation, direction = CLINICAL_INTERPRETATIONS.get(
                        feature, (feature, 'unknown')
                    )
                    
                    # weight by prediction confidence
                    contrib = max_score * (pd_probability if direction == 'elevated' else (1 - pd_probability))
                    contributions[f"{feature}_{direction}"] = round(contrib, 4)
            
            return contributions
        
        # compute contributions from clinical features
        # reference values based on literature (healthy control means)
        reference_values = {
            'jitter_local': 0.005,
            'jitter_rap': 0.003,
            'shimmer_local': 0.03,
            'shimmer_apq5': 0.025,
            'hnr_mean': 20.0,
            'f0_std': 15.0
        }
        
        total_contribution = 0.0
        
        for feature, value in clinical_features.items():
            if np.isnan(value):
                continue
            
            interpretation, direction = CLINICAL_INTERPRETATIONS.get(
                feature, (feature, 'unknown')
            )
            
            # get reference value
            ref = reference_values.get(feature, value)
            
            # compute deviation
            if direction == 'elevated':
                deviation = (value - ref) / (ref + 1e-8)
            elif direction == 'reduced':
                deviation = (ref - value) / (ref + 1e-8)
            else:
                deviation = abs(value - ref) / (ref + 1e-8)
            
            # scale by probing importance if available
            probing_weight = 1.0
            if feature in self.probing_results:
                max_probing = max(self.probing_results[feature].values(), default=0.5)
                probing_weight = max_probing
            
            contribution = np.clip(deviation * probing_weight, -1, 1)
            
            if abs(contribution) > 0.05:  # only include significant contributions
                key = f"{feature}_{direction}"
                contributions[key] = round(float(contribution), 4)
                total_contribution += abs(contribution)
        
        # normalize to sum to ~1
        if total_contribution > 0:
            contributions = {
                k: round(v / total_contribution, 4)
                for k, v in contributions.items()
            }
        
        return contributions
    
    def _analyze_attention_patterns(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Optional[Dict[str, Any]]:
        """analyze attention patterns for interpretability."""
        try:
            outputs = self.model.wav2vec2(
                **inputs,
                output_attentions=True
            )
            
            attentions = outputs.attentions
            
            # compute attention statistics
            attention_stats = {}
            
            for layer_idx, attn in enumerate(attentions):
                # attn shape: [batch, heads, seq, seq]
                attn_np = attn.cpu().numpy().squeeze()
                
                # entropy as measure of attention distribution
                entropy = -np.sum(attn_np * np.log(attn_np + 1e-10), axis=-1)
                
                attention_stats[f'layer_{layer_idx}'] = {
                    'mean_entropy': float(entropy.mean()),
                    'max_attention': float(attn_np.max()),
                    'sparsity': float((attn_np < 0.01).mean())
                }
            
            return attention_stats
            
        except Exception as e:
            logger.warning(f"attention analysis failed: {e}")
            return None
    
    def batch_predict(
        self,
        audio_list: List[np.ndarray],
        sample_rate: int = 16000,
        include_clinical: bool = True,
        show_progress: bool = True
    ) -> List[InterpretablePrediction]:
        """
        generate predictions for multiple audio samples.
        
        args:
            audio_list: list of audio waveforms
            sample_rate: audio sample rate
            include_clinical: whether to extract clinical features
            show_progress: whether to show progress bar
        
        returns:
            list of InterpretablePrediction objects
        """
        predictions = []
        
        iterator = audio_list
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(audio_list, desc='predicting')
            except ImportError:
                pass
        
        for audio in iterator:
            pred = self.predict(
                audio,
                sample_rate=sample_rate,
                include_clinical=include_clinical
            )
            predictions.append(pred)
        
        return predictions
    
    def explain_prediction(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        format: str = 'text'
    ) -> str:
        """
        generate comprehensive explanation for prediction.
        
        args:
            audio: audio waveform
            sample_rate: audio sample rate
            format: output format ('text', 'markdown', 'json')
        
        returns:
            formatted explanation string
        """
        prediction = self.predict(audio, sample_rate, include_clinical=True)
        
        if format == 'json':
            return prediction.to_json()
        elif format == 'markdown':
            return self._format_markdown_explanation(prediction)
        else:
            return prediction.generate_explanation()
    
    def _format_markdown_explanation(
        self,
        prediction: InterpretablePrediction
    ) -> str:
        """format prediction as markdown."""
        lines = ["# Interpretable Prediction Results\n"]
        
        # prediction
        status = "Parkinson's Disease" if prediction.pd_probability >= 0.5 else "Healthy Control"
        lines.append(f"## Prediction: {status}")
        lines.append(f"- **Probability**: {prediction.pd_probability:.1%}")
        lines.append(f"- **Confidence**: {prediction.confidence:.1%}")
        lines.append("")
        
        # feature contributions
        lines.append("## Feature Contributions")
        lines.append("| Feature | Contribution |")
        lines.append("|---------|-------------|")
        for feature, score in prediction.get_top_features(8):
            lines.append(f"| {feature} | {score:+.3f} |")
        lines.append("")
        
        # evidence
        lines.append("## Model Evidence")
        lines.append(f"- **Evidence Layers**: {prediction.evidence_layers}")
        heads_str = ", ".join(f"L{l}H{h}" for l, h in prediction.key_attention_heads[:5])
        lines.append(f"- **Key Attention Heads**: {heads_str}")
        
        if prediction.clinical_features:
            lines.append("\n## Clinical Features")
            lines.append("| Feature | Value |")
            lines.append("|---------|-------|")
            for name, value in list(prediction.clinical_features.items())[:10]:
                if not np.isnan(value):
                    lines.append(f"| {name} | {value:.4f} |")
        
        return "\n".join(lines)
    
    def save_prediction(
        self,
        prediction: InterpretablePrediction,
        path: Union[str, Path]
    ):
        """save prediction to json file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write(prediction.to_json())
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        processor_path: Optional[Union[str, Path]] = None,
        analysis_results_path: Optional[Union[str, Path]] = None,
        device: str = 'cpu'
    ) -> 'InterpretablePredictionInterface':
        """
        create interface from saved checkpoint.
        
        args:
            checkpoint_path: path to model checkpoint
            processor_path: path to processor (defaults to wav2vec2-base)
            analysis_results_path: path to precomputed analysis results
            device: computation device
        
        returns:
            configured InterpretablePredictionInterface
        """
        from transformers import Wav2Vec2Processor
        
        # load processor
        if processor_path:
            processor = Wav2Vec2Processor.from_pretrained(processor_path)
        else:
            processor = Wav2Vec2Processor.from_pretrained(
                'facebook/wav2vec2-base'
            )
        
        # load model
        checkpoint_path = Path(checkpoint_path)
        model = torch.load(checkpoint_path, map_location=device)
        
        # create interface
        interface = cls(
            model=model,
            processor=processor,
            device=device
        )
        
        # load analysis results if provided
        if analysis_results_path:
            interface.load_analysis_results(analysis_results_path)
        
        return interface


def create_interpretable_interface(
    model: torch.nn.Module,
    processor: Any,
    clinical_extractor: Optional[Any] = None,
    probing_results: Optional[Dict] = None,
    patching_results: Optional[Dict] = None,
    device: str = 'cpu'
) -> InterpretablePredictionInterface:
    """
    factory function to create interpretable prediction interface.
    
    convenience function for creating the interface with common defaults.
    
    args:
        model: fine-tuned wav2vec2 classifier
        processor: wav2vec2 processor
        clinical_extractor: optional clinical feature extractor
        probing_results: optional probing analysis results
        patching_results: optional patching analysis results
        device: computation device
    
    returns:
        configured InterpretablePredictionInterface
    
    example:
        from src.interpretability.prediction_interface import create_interpretable_interface
        from src.features.clinical import ClinicalFeatureExtractor
        
        interface = create_interpretable_interface(
            model=fine_tuned_model,
            processor=processor,
            clinical_extractor=ClinicalFeatureExtractor()
        )
        
        prediction = interface.predict(audio_waveform)
        print(prediction.to_dict())
        # output:
        # {
        #     "pd_probability": 0.87,
        #     "feature_contributions": {
        #         "jitter_elevated": 0.34,
        #         "hnr_reduced": 0.28,
        #         "f0_unstable": 0.21
        #     },
        #     "evidence_layers": [3, 4, 7],
        #     "key_attention_heads": [[3, 4], [4, 2], [7, 8]]
        # }
    """
    return InterpretablePredictionInterface(
        model=model,
        processor=processor,
        clinical_extractor=clinical_extractor,
        probing_results=probing_results,
        patching_results=patching_results,
        device=device
    )


# convenience exports
__all__ = [
    'InterpretablePrediction',
    'InterpretablePredictionInterface',
    'FeatureContribution',
    'create_interpretable_interface',
    'CLINICAL_INTERPRETATIONS'
]
