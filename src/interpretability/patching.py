"""
activation patching utilities for causal analysis.

implements activation patching to determine which model components
causally affect predictions, including layer-level, head-level,
position-level patching, mean ablation, and clinical feature stratification.
"""

from typing import Dict, List, Optional, Tuple, Callable, Union, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import warnings
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification
from tqdm import tqdm


@dataclass
class PatchingResult:
    """container for patching experiment results."""
    
    layer_recoveries: Dict[int, float] = field(default_factory=dict)
    head_recoveries: Dict[Tuple[int, int], float] = field(default_factory=dict)
    position_recoveries: Dict[Tuple[int, int], float] = field(default_factory=dict)
    clean_logit_diff: float = 0.0
    corrupted_logit_diff: float = 0.0
    total_effect: float = 0.0
    
    def to_dict(self) -> Dict:
        """convert to json-serializable dict."""
        return {
            'layer_recoveries': self.layer_recoveries,
            'head_recoveries': {f"{k[0]}_{k[1]}": v for k, v in self.head_recoveries.items()},
            'position_recoveries': {f"{k[0]}_{k[1]}": v for k, v in self.position_recoveries.items()},
            'clean_logit_diff': self.clean_logit_diff,
            'corrupted_logit_diff': self.corrupted_logit_diff,
            'total_effect': self.total_effect
        }


@dataclass
class HeadImportanceRanking:
    """ranking of attention head importance from patching."""
    
    head_scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    head_rankings: List[Tuple[int, int, float]] = field(default_factory=list)
    important_heads: List[Tuple[int, int]] = field(default_factory=list)
    
    @classmethod
    def from_patching_results(
        cls,
        results: List[PatchingResult],
        top_k: int = 20,
        threshold: float = 0.1
    ) -> "HeadImportanceRanking":
        """create ranking from multiple patching results."""
        aggregated = defaultdict(list)
        
        for result in results:
            for (layer, head), recovery in result.head_recoveries.items():
                aggregated[(layer, head)].append(recovery)
        
        head_scores = {
            k: np.mean(v) for k, v in aggregated.items()
        }
        
        # create ranking
        rankings = sorted(
            [(layer, head, score) for (layer, head), score in head_scores.items()],
            key=lambda x: x[2],
            reverse=True
        )
        
        # identify important heads
        important = [
            (layer, head) for layer, head, score in rankings[:top_k]
            if score >= threshold
        ]
        
        return cls(
            head_scores=head_scores,
            head_rankings=rankings,
            important_heads=important
        )
    
    def to_dict(self) -> Dict:
        """convert to json-serializable dict."""
        return {
            'head_scores': {f"{k[0]}_{k[1]}": v for k, v in self.head_scores.items()},
            'head_rankings': [
                {'layer': l, 'head': h, 'score': s} for l, h, s in self.head_rankings
            ],
            'important_heads': [{'layer': l, 'head': h} for l, h in self.important_heads]
        }


def get_activation_patching_hook(
    source_activation: torch.Tensor,
    position: Optional[int] = None,
    positions: Optional[List[int]] = None
) -> Callable:
    """
    create TransformerLens-style hook for activation patching.
    
    args:
        source_activation: activation tensor to patch in [batch, seq, hidden]
        position: single position to patch (if None, patch all)
        positions: list of positions to patch (overrides position)
    
    returns:
        hook function compatible with register_forward_hook
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            out_tensor = output[0]
            is_tuple = True
        else:
            out_tensor = output
            is_tuple = False
        
        patched = out_tensor.clone()
        
        if positions is not None:
            for pos in positions:
                if pos < patched.shape[1]:
                    patched[:, pos, :] = source_activation[:, pos, :].to(patched.device)
        elif position is not None:
            if position < patched.shape[1]:
                patched[:, position, :] = source_activation[:, position, :].to(patched.device)
        else:
            # patch all positions
            patched = source_activation.to(patched.device)
        
        if is_tuple:
            return (patched,) + output[1:]
        return patched
    
    return hook


def get_mean_ablation_hook(
    mean_activation: torch.Tensor,
    position: Optional[int] = None
) -> Callable:
    """
    create hook for mean ablation (replace with dataset mean).
    
    args:
        mean_activation: mean activation across dataset [1, hidden] or [1, seq, hidden]
        position: position to ablate (if None, ablate all)
    
    returns:
        hook function
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            out_tensor = output[0]
            is_tuple = True
        else:
            out_tensor = output
            is_tuple = False
        
        ablated = out_tensor.clone()
        
        if position is not None:
            if mean_activation.dim() == 2:
                ablated[:, position, :] = mean_activation.to(ablated.device)
            else:
                ablated[:, position, :] = mean_activation[:, position, :].to(ablated.device)
        else:
            # ablate all positions with mean
            if mean_activation.dim() == 2:
                ablated = mean_activation.unsqueeze(1).expand_as(ablated).to(ablated.device)
            else:
                ablated = mean_activation.expand(ablated.shape[0], -1, -1).to(ablated.device)
        
        if is_tuple:
            return (ablated,) + output[1:]
        return ablated
    
    return hook


class ActivationPatcher:
    """
    perform activation patching experiments on wav2vec2.

    activation patching tests causality: if patching activations from a clean
    run into a corrupted run changes the prediction, those activations are
    causally important.
    
    supports:
    - layer-level patching
    - position-level patching  
    - attention head-level patching
    - mean ablation
    """

    def __init__(
        self,
        model: Wav2Vec2ForSequenceClassification,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        args:
            model: wav2vec2 classifier model
            device: device to run on
        """
        self.model = model
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

        self.num_layers = len(self.model.wav2vec2.encoder.layers)
        self.num_heads = self.model.wav2vec2.config.num_attention_heads
        self.hidden_size = self.model.wav2vec2.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.cached_activations = {}
        self.mean_activations = {}
        self.patch_hooks = []

    def _create_patch_hook(
        self,
        clean_activation: torch.Tensor
    ) -> Callable:
        """
        create hook that patches in clean activation.

        args:
            clean_activation: activation to patch in

        returns:
            hook function
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                patched = (clean_activation.to(output[0].device),) + output[1:]
            else:
                patched = clean_activation.to(output.device)
            return patched

        return hook

    @torch.no_grad()
    def get_clean_activations(
        self,
        input_values: torch.Tensor,
        include_attention: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        run clean forward pass and cache all layer activations.

        args:
            input_values: clean audio input
            include_attention: whether to cache attention outputs

        returns:
            (activations_dict, clean_logits, attention_dict or None)
        """
        activations = {}
        attention_outputs = {} if include_attention else None
        hooks = []

        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].clone()
                else:
                    activations[name] = output.clone()
            return hook
        
        def get_attention_output(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    attention_outputs[name] = output[0].clone()
                else:
                    attention_outputs[name] = output.clone()
            return hook

        for i, layer in enumerate(self.model.wav2vec2.encoder.layers):
            h = layer.register_forward_hook(get_activation(f'layer_{i}'))
            hooks.append(h)
            
            if include_attention:
                h_attn = layer.attention.register_forward_hook(
                    get_attention_output(f'attn_{i}')
                )
                hooks.append(h_attn)

        input_values = input_values.to(self.device)
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)

        outputs = self.model(input_values)

        for h in hooks:
            h.remove()

        return activations, outputs.logits.clone(), attention_outputs

    @torch.no_grad()
    def compute_mean_activations(
        self,
        dataset,
        max_samples: int = 500
    ) -> Dict[str, torch.Tensor]:
        """
        compute mean activations across dataset for ablation.
        
        args:
            dataset: pytorch dataset
            max_samples: max samples to use
        
        returns:
            dict mapping layer name to mean activation
        """
        layer_sums = {f'layer_{i}': None for i in range(self.num_layers)}
        counts = 0
        
        indices = list(range(min(len(dataset), max_samples)))
        
        for idx in tqdm(indices, desc="computing mean activations"):
            sample = dataset[idx]
            input_values = sample['input_values']
            
            activations, _, _ = self.get_clean_activations(input_values)
            
            for name, act in activations.items():
                # mean pool over sequence dimension
                act_mean = act.mean(dim=1)  # [1, hidden]
                
                if layer_sums[name] is None:
                    layer_sums[name] = act_mean.cpu()
                else:
                    layer_sums[name] += act_mean.cpu()
            
            counts += 1
        
        self.mean_activations = {
            name: (total / counts) for name, total in layer_sums.items()
        }
        
        return self.mean_activations

    @torch.no_grad()
    def patch_layer(
        self,
        corrupted_input: torch.Tensor,
        clean_activation: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        run forward pass with one layer patched.

        args:
            corrupted_input: corrupted audio input
            clean_activation: clean activation to patch in
            layer_idx: which layer to patch

        returns:
            patched logits
        """
        handle = self.model.wav2vec2.encoder.layers[layer_idx].register_forward_hook(
            self._create_patch_hook(clean_activation)
        )

        corrupted_input = corrupted_input.to(self.device)
        if corrupted_input.dim() == 1:
            corrupted_input = corrupted_input.unsqueeze(0)

        outputs = self.model(corrupted_input)

        handle.remove()

        return outputs.logits

    @torch.no_grad()
    def patch_position(
        self,
        corrupted_input: torch.Tensor,
        clean_activation: torch.Tensor,
        layer_idx: int,
        position: int
    ) -> torch.Tensor:
        """
        run forward pass with one position in one layer patched.
        
        args:
            corrupted_input: corrupted audio input
            clean_activation: clean activation to patch in
            layer_idx: which layer to patch
            position: which position to patch
        
        returns:
            patched logits
        """
        hook = get_activation_patching_hook(clean_activation, position=position)
        handle = self.model.wav2vec2.encoder.layers[layer_idx].register_forward_hook(hook)
        
        corrupted_input = corrupted_input.to(self.device)
        if corrupted_input.dim() == 1:
            corrupted_input = corrupted_input.unsqueeze(0)
        
        outputs = self.model(corrupted_input)
        handle.remove()
        
        return outputs.logits

    @torch.no_grad()
    def ablate_layer(
        self,
        input_values: torch.Tensor,
        layer_idx: int,
        mean_activation: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        run forward pass with layer ablated (replaced with mean).
        
        args:
            input_values: input audio
            layer_idx: which layer to ablate
            mean_activation: mean activation to use (uses cached if None)
        
        returns:
            ablated logits
        """
        if mean_activation is None:
            mean_activation = self.mean_activations.get(f'layer_{layer_idx}')
            if mean_activation is None:
                raise ValueError(f"no mean activation for layer {layer_idx}, call compute_mean_activations first")
        
        hook = get_mean_ablation_hook(mean_activation)
        handle = self.model.wav2vec2.encoder.layers[layer_idx].register_forward_hook(hook)
        
        input_values = input_values.to(self.device)
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)
        
        outputs = self.model(input_values)
        handle.remove()
        
        return outputs.logits

    @torch.no_grad()
    def patch_attention_head(
        self,
        corrupted_input: torch.Tensor,
        clean_attn_output: torch.Tensor,
        layer_idx: int,
        head_idx: int
    ) -> torch.Tensor:
        """
        patch a single attention head's output.
        
        args:
            corrupted_input: corrupted input audio
            clean_attn_output: clean attention output [batch, seq, hidden]
            layer_idx: which layer
            head_idx: which head (0 to num_heads-1)
        
        returns:
            patched logits
        """
        def patch_head_hook(module, input, output):
            if isinstance(output, tuple):
                attn_out = output[0].clone()
            else:
                attn_out = output.clone()
            
            # patch specific head dimensions
            start = head_idx * self.head_dim
            end = (head_idx + 1) * self.head_dim
            
            attn_out[:, :, start:end] = clean_attn_output[:, :, start:end].to(attn_out.device)
            
            if isinstance(output, tuple):
                return (attn_out,) + output[1:]
            return attn_out
        
        layer = self.model.wav2vec2.encoder.layers[layer_idx]
        handle = layer.attention.register_forward_hook(patch_head_hook)
        
        corrupted_input = corrupted_input.to(self.device)
        if corrupted_input.dim() == 1:
            corrupted_input = corrupted_input.unsqueeze(0)
        
        outputs = self.model(corrupted_input)
        handle.remove()
        
        return outputs.logits

    def run_layer_patching(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        clean_label: int
    ) -> PatchingResult:
        """
        patch each layer individually and measure effect.

        args:
            clean_input: clean audio (typically HC)
            corrupted_input: corrupted audio (typically PD)
            clean_label: true label for clean input (typically 0 for HC)

        returns:
            PatchingResult with layer recovery scores
        """
        clean_acts, clean_logits, _ = self.get_clean_activations(clean_input)

        corrupted_input_batch = corrupted_input.to(self.device)
        if corrupted_input_batch.dim() == 1:
            corrupted_input_batch = corrupted_input_batch.unsqueeze(0)

        corrupted_logits = self.model(corrupted_input_batch).logits

        # compute logit differences
        clean_logit_diff = (
            clean_logits[0, clean_label] - clean_logits[0, 1 - clean_label]
        ).item()

        corrupted_logit_diff = (
            corrupted_logits[0, clean_label] - corrupted_logits[0, 1 - clean_label]
        ).item()

        total_effect = clean_logit_diff - corrupted_logit_diff

        layer_recoveries = {}

        for layer_idx in range(self.num_layers):
            patched_logits = self.patch_layer(
                corrupted_input,
                clean_acts[f'layer_{layer_idx}'],
                layer_idx
            )

            patched_logit_diff = (
                patched_logits[0, clean_label] - patched_logits[0, 1 - clean_label]
            ).item()

            if abs(total_effect) > 1e-6:
                recovery = (patched_logit_diff - corrupted_logit_diff) / total_effect
            else:
                recovery = 0.0

            layer_recoveries[layer_idx] = recovery

        return PatchingResult(
            layer_recoveries=layer_recoveries,
            clean_logit_diff=clean_logit_diff,
            corrupted_logit_diff=corrupted_logit_diff,
            total_effect=total_effect
        )

    def run_head_patching(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        clean_label: int,
        target_layers: Optional[List[int]] = None
    ) -> PatchingResult:
        """
        patch each attention head individually.
        
        args:
            clean_input: clean audio
            corrupted_input: corrupted audio
            clean_label: clean label
            target_layers: layers to test (None = all layers)
        
        returns:
            PatchingResult with head recovery scores
        """
        clean_acts, clean_logits, clean_attn = self.get_clean_activations(
            clean_input, include_attention=True
        )
        
        corrupted_input_batch = corrupted_input.to(self.device)
        if corrupted_input_batch.dim() == 1:
            corrupted_input_batch = corrupted_input_batch.unsqueeze(0)
        
        corrupted_logits = self.model(corrupted_input_batch).logits
        
        clean_logit_diff = (
            clean_logits[0, clean_label] - clean_logits[0, 1 - clean_label]
        ).item()
        
        corrupted_logit_diff = (
            corrupted_logits[0, clean_label] - corrupted_logits[0, 1 - clean_label]
        ).item()
        
        total_effect = clean_logit_diff - corrupted_logit_diff
        
        if target_layers is None:
            target_layers = list(range(self.num_layers))
        
        head_recoveries = {}
        
        for layer_idx in target_layers:
            clean_attn_out = clean_attn.get(f'attn_{layer_idx}')
            if clean_attn_out is None:
                continue
            
            for head_idx in range(self.num_heads):
                patched_logits = self.patch_attention_head(
                    corrupted_input,
                    clean_attn_out,
                    layer_idx,
                    head_idx
                )
                
                patched_logit_diff = (
                    patched_logits[0, clean_label] - patched_logits[0, 1 - clean_label]
                ).item()
                
                if abs(total_effect) > 1e-6:
                    recovery = (patched_logit_diff - corrupted_logit_diff) / total_effect
                else:
                    recovery = 0.0
                
                head_recoveries[(layer_idx, head_idx)] = recovery
        
        return PatchingResult(
            head_recoveries=head_recoveries,
            clean_logit_diff=clean_logit_diff,
            corrupted_logit_diff=corrupted_logit_diff,
            total_effect=total_effect
        )

    def run_mean_ablation(
        self,
        input_values: torch.Tensor,
        true_label: int
    ) -> Dict[int, float]:
        """
        ablate each layer with mean activation and measure effect.
        
        args:
            input_values: input audio
            true_label: true label
        
        returns:
            dict mapping layer_idx to ablation effect (drop in correct logit)
        """
        if not self.mean_activations:
            raise ValueError("call compute_mean_activations first")
        
        input_batch = input_values.to(self.device)
        if input_batch.dim() == 1:
            input_batch = input_batch.unsqueeze(0)
        
        original_logits = self.model(input_batch).logits
        original_logit = original_logits[0, true_label].item()
        
        ablation_effects = {}
        
        for layer_idx in range(self.num_layers):
            ablated_logits = self.ablate_layer(input_values, layer_idx)
            ablated_logit = ablated_logits[0, true_label].item()
            
            # effect = how much the correct class logit dropped
            effect = original_logit - ablated_logit
            ablation_effects[layer_idx] = effect
        
        return ablation_effects

    def run_position_patching(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        clean_label: int,
        layer_idx: int,
        n_positions: int = 20
    ) -> Dict[int, float]:
        """
        patch individual positions within a layer.
        
        tests which sequence positions are most important.
        
        args:
            clean_input: clean audio
            corrupted_input: corrupted audio
            clean_label: clean label
            layer_idx: layer to patch positions in
            n_positions: number of positions to sample
        
        returns:
            dict mapping position index to recovery score
        """
        clean_acts, clean_logits, _ = self.get_clean_activations(clean_input)
        
        corrupted_input_batch = corrupted_input.to(self.device)
        if corrupted_input_batch.dim() == 1:
            corrupted_input_batch = corrupted_input_batch.unsqueeze(0)
        
        corrupted_logits = self.model(corrupted_input_batch).logits
        
        clean_logit_diff = (
            clean_logits[0, clean_label] - clean_logits[0, 1 - clean_label]
        ).item()
        
        corrupted_logit_diff = (
            corrupted_logits[0, clean_label] - corrupted_logits[0, 1 - clean_label]
        ).item()
        
        total_effect = clean_logit_diff - corrupted_logit_diff
        
        # get sequence length from cached activation
        clean_act = clean_acts[f'layer_{layer_idx}']
        seq_len = clean_act.shape[1]
        
        # sample positions evenly across sequence
        if n_positions >= seq_len:
            positions = list(range(seq_len))
        else:
            positions = np.linspace(0, seq_len - 1, n_positions, dtype=int).tolist()
        
        position_recoveries = {}
        
        for pos in positions:
            patched_logits = self.patch_position(
                corrupted_input, clean_act, layer_idx, pos
            )
            
            patched_logit_diff = (
                patched_logits[0, clean_label] - patched_logits[0, 1 - clean_label]
            ).item()
            
            if abs(total_effect) > 1e-6:
                recovery = (patched_logit_diff - corrupted_logit_diff) / total_effect
            else:
                recovery = 0.0
            
            position_recoveries[pos] = recovery
        
        return position_recoveries

    def run_batch_patching(
        self,
        clean_inputs: List[torch.Tensor],
        corrupted_inputs: List[torch.Tensor],
        clean_labels: List[int],
        include_heads: bool = False,
        target_layers: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        run patching on multiple (clean, corrupted) pairs.

        args:
            clean_inputs: list of clean audio tensors
            corrupted_inputs: list of corrupted audio tensors
            clean_labels: list of clean labels
            include_heads: whether to also do head-level patching
            target_layers: layers to test for head patching

        returns:
            comprehensive results dict
        """
        all_results = []

        for clean, corrupted, label in tqdm(
            zip(clean_inputs, corrupted_inputs, clean_labels),
            total=len(clean_inputs),
            desc="layer patching"
        ):
            try:
                result = self.run_layer_patching(clean, corrupted, label)
                all_results.append(result)
            except Exception as e:
                warnings.warn(f"patching failed: {e}")

        # aggregate layer results
        layer_summary = {}
        for layer_idx in range(self.num_layers):
            recoveries = [r.layer_recoveries.get(layer_idx, 0) for r in all_results]
            layer_summary[layer_idx] = {
                'mean_recovery': float(np.mean(recoveries)),
                'std_recovery': float(np.std(recoveries)),
                'median_recovery': float(np.median(recoveries)),
                'min_recovery': float(np.min(recoveries)),
                'max_recovery': float(np.max(recoveries)),
                'n_samples': len(recoveries)
            }

        results = {
            'layer_patching': layer_summary,
            'individual_results': [r.to_dict() for r in all_results]
        }

        # head-level patching if requested
        if include_heads:
            head_results = []
            
            # identify important layers from layer patching
            if target_layers is None:
                important = [
                    layer_idx for layer_idx, stats in layer_summary.items()
                    if stats['mean_recovery'] > 0.1
                ]
                target_layers = important if important else list(range(self.num_layers))
            
            for clean, corrupted, label in tqdm(
                zip(clean_inputs, corrupted_inputs, clean_labels),
                total=len(clean_inputs),
                desc="head patching"
            ):
                try:
                    result = self.run_head_patching(
                        clean, corrupted, label, target_layers
                    )
                    head_results.append(result)
                except Exception as e:
                    warnings.warn(f"head patching failed: {e}")
            
            # create head importance ranking
            head_ranking = HeadImportanceRanking.from_patching_results(head_results)
            results['head_patching'] = head_ranking.to_dict()

        return results

    def validate_with_ablation(
        self,
        dataset,
        patching_results: Dict,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """
        validate patching results with mean ablation.
        
        components that are important in patching should also show
        large effects under ablation.
        
        args:
            dataset: pytorch dataset
            patching_results: results from run_batch_patching
            max_samples: max samples for ablation
        
        returns:
            validation results including concordance
        """
        # compute mean activations if needed
        if not self.mean_activations:
            self.compute_mean_activations(dataset)
        
        # run ablation on samples
        all_ablation_effects = {i: [] for i in range(self.num_layers)}
        
        indices = list(range(min(len(dataset), max_samples)))
        
        for idx in tqdm(indices, desc="ablation validation"):
            sample = dataset[idx]
            input_values = sample['input_values']
            label = sample['label']
            
            try:
                effects = self.run_mean_ablation(input_values, label)
                for layer_idx, effect in effects.items():
                    all_ablation_effects[layer_idx].append(effect)
            except Exception as e:
                warnings.warn(f"ablation failed: {e}")
        
        # compute mean ablation effects
        ablation_summary = {
            layer_idx: {
                'mean_effect': float(np.mean(effects)),
                'std_effect': float(np.std(effects))
            }
            for layer_idx, effects in all_ablation_effects.items()
            if effects
        }
        
        # compute concordance with patching
        patching_ranks = []
        ablation_ranks = []
        
        layer_patching = patching_results.get('layer_patching', {})
        
        for layer_idx in range(self.num_layers):
            if layer_idx in layer_patching and layer_idx in ablation_summary:
                patching_ranks.append(layer_patching[layer_idx]['mean_recovery'])
                ablation_ranks.append(ablation_summary[layer_idx]['mean_effect'])
        
        # spearman correlation
        from scipy import stats as scipy_stats
        if len(patching_ranks) >= 3:
            correlation, p_value = scipy_stats.spearmanr(patching_ranks, ablation_ranks)
        else:
            correlation, p_value = 0.0, 1.0
        
        # detailed interpretation
        if np.isnan(correlation):
            interpretation = "undefined (insufficient variance)"
        elif correlation > 0.7 and p_value < 0.05:
            interpretation = "high concordance - patching and ablation strongly agree on layer importance"
        elif correlation > 0.4 and p_value < 0.1:
            interpretation = "moderate concordance - reasonable agreement between methods"
        elif correlation > 0.2:
            interpretation = "weak concordance - methods show some agreement"
        else:
            interpretation = "low concordance - methods disagree on component importance"
        
        return {
            'ablation_effects': ablation_summary,
            'concordance': {
                'spearman_correlation': float(correlation),
                'p_value': float(p_value),
                'interpretation': interpretation,
                'patching_ranks': patching_ranks,
                'ablation_ranks': ablation_ranks
            }
        }


class AttentionHeadPatcher:
    """
    patch individual attention heads to identify important heads.

    more fine-grained than layer-level patching.
    """

    def __init__(
        self,
        model: Wav2Vec2ForSequenceClassification,
        device: str = "cuda"
    ):
        """
        args:
            model: wav2vec2 classifier
            device: device to run on
        """
        self.model = model
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

        self.num_layers = len(self.model.wav2vec2.encoder.layers)
        self.num_heads = self.model.wav2vec2.config.num_attention_heads

    @torch.no_grad()
    def patch_attention_head(
        self,
        corrupted_input: torch.Tensor,
        clean_attention: torch.Tensor,
        layer_idx: int,
        head_idx: int
    ) -> torch.Tensor:
        """
        patch single attention head.

        args:
            corrupted_input: corrupted input
            clean_attention: clean attention weights
            layer_idx: which layer
            head_idx: which head

        returns:
            patched logits
        """
        def patch_head_hook(module, input, output):
            if isinstance(output, tuple):
                attn_output = output[0]
                head_dim = attn_output.shape[-1] // self.num_heads

                start_idx = head_idx * head_dim
                end_idx = (head_idx + 1) * head_dim

                attn_output[:, :, start_idx:end_idx] = clean_attention[:, :, start_idx:end_idx]

                return (attn_output,) + output[1:]

            return output

        layer = self.model.wav2vec2.encoder.layers[layer_idx]
        handle = layer.attention.register_forward_hook(patch_head_hook)

        corrupted_input = corrupted_input.to(self.device)
        if corrupted_input.dim() == 1:
            corrupted_input = corrupted_input.unsqueeze(0)

        outputs = self.model(corrupted_input)

        handle.remove()

        return outputs.logits


def create_minimal_pairs(
    dataset,
    n_pairs: int = 50,
    same_task: bool = True
) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    create (clean, corrupted) pairs for patching experiments.

    pairs clean (healthy) samples with corrupted (pd) samples.

    args:
        dataset: pytorch dataset
        n_pairs: number of pairs to create
        same_task: whether to match by task type

    returns:
        list of (clean_audio, corrupted_audio, clean_label) tuples
    """
    pd_samples = [i for i, s in enumerate(dataset.samples) if s['label'] == 1]
    hc_samples = [i for i, s in enumerate(dataset.samples) if s['label'] == 0]

    pairs = []
    used_hc = set()

    for pd_idx in pd_samples[:n_pairs]:
        pd_sample = dataset.samples[pd_idx]

        best_match = None
        best_score = float('inf')

        for hc_idx in hc_samples:
            if hc_idx in used_hc:
                continue

            hc_sample = dataset.samples[hc_idx]

            if same_task and pd_sample.get('task') == hc_sample.get('task'):
                score = 0
            else:
                score = 1

            if score < best_score:
                best_score = score
                best_match = hc_idx

        if best_match is not None:
            used_hc.add(best_match)

            hc_data = dataset[best_match]
            pd_data = dataset[pd_idx]

            pairs.append((
                hc_data['input_values'],
                pd_data['input_values'],
                0
            ))

    return pairs


def create_mfcc_matched_pairs(
    dataset,
    n_pairs: int = 50,
    same_task: bool = True,
    n_mfcc: int = 13,
    sr: int = 16000
) -> List[Tuple[torch.Tensor, torch.Tensor, int, float]]:
    """
    create (hc, pd) pairs matched by mfcc acoustic similarity.

    for each pd sample, finds the most acoustically similar hc sample
    using mfcc distance. this creates proper minimal pairs for
    activation patching experiments.

    args:
        dataset: pytorch dataset with 'input_values', 'label', 'task'
        n_pairs: number of pairs to create
        same_task: whether to require matching task types
        n_mfcc: number of mfcc coefficients
        sr: sampling rate

    returns:
        list of (hc_audio, pd_audio, hc_label, mfcc_distance) tuples
    """
    try:
        import librosa
    except ImportError:
        warnings.warn("librosa required for mfcc matching, falling back to random pairs")
        return create_minimal_pairs(dataset, n_pairs, same_task)

    pd_indices = [i for i, s in enumerate(dataset.samples) if s['label'] == 1]
    hc_indices = [i for i, s in enumerate(dataset.samples) if s['label'] == 0]

    if len(pd_indices) == 0 or len(hc_indices) == 0:
        warnings.warn("no pd or hc samples found")
        return []

    hc_mfccs = {}
    for hc_idx in tqdm(hc_indices, desc="computing hc mfccs"):
        try:
            sample = dataset[hc_idx]
            audio = sample['input_values'].numpy()
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            hc_mfccs[hc_idx] = {
                'mfcc_mean': mfcc.mean(axis=1),
                'mfcc_std': mfcc.std(axis=1),
                'task': sample.get('task', 'unknown')
            }
        except Exception as e:
            warnings.warn(f"failed to compute mfcc for sample {hc_idx}: {e}")
            continue

    pairs = []
    used_hc = set()

    for pd_idx in tqdm(pd_indices[:n_pairs * 2], desc="matching pairs"):
        try:
            pd_sample = dataset[pd_idx]
            pd_audio = pd_sample['input_values'].numpy()
            pd_task = pd_sample.get('task', 'unknown')

            pd_mfcc = librosa.feature.mfcc(y=pd_audio, sr=sr, n_mfcc=n_mfcc)
            pd_mfcc_mean = pd_mfcc.mean(axis=1)
            pd_mfcc_std = pd_mfcc.std(axis=1)

            best_match = None
            best_distance = float('inf')

            for hc_idx, hc_data in hc_mfccs.items():
                if hc_idx in used_hc:
                    continue

                if same_task and hc_data['task'] != pd_task:
                    continue

                mean_dist = np.linalg.norm(pd_mfcc_mean - hc_data['mfcc_mean'])
                std_dist = np.linalg.norm(pd_mfcc_std - hc_data['mfcc_std'])
                distance = mean_dist + 0.5 * std_dist

                if distance < best_distance:
                    best_distance = distance
                    best_match = hc_idx

            if best_match is not None:
                used_hc.add(best_match)

                hc_sample = dataset[best_match]

                pairs.append((
                    hc_sample['input_values'],
                    pd_sample['input_values'],
                    0,
                    best_distance
                ))

                if len(pairs) >= n_pairs:
                    break

        except Exception as e:
            warnings.warn(f"failed to match sample {pd_idx}: {e}")
            continue

    pairs.sort(key=lambda x: x[3])

    return pairs


def compute_mfcc_distance(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13
) -> float:
    """
    compute mfcc distance between two audio samples.

    args:
        audio1: first audio array
        audio2: second audio array
        sr: sampling rate
        n_mfcc: number of mfcc coefficients

    returns:
        euclidean distance between mfcc means
    """
    import librosa

    mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr, n_mfcc=n_mfcc)
    mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr, n_mfcc=n_mfcc)

    mean1 = mfcc1.mean(axis=1)
    mean2 = mfcc2.mean(axis=1)

    return float(np.linalg.norm(mean1 - mean2))


def compute_patching_importance(
    patching_results: Dict[int, Dict[str, float]],
    threshold: float = 0.5
) -> List[int]:
    """
    identify important layers from patching results.

    args:
        patching_results: results from run_batch_patching
        threshold: minimum recovery to be considered important

    returns:
        list of important layer indices
    """
    important_layers = []

    for layer_idx, metrics in patching_results.items():
        if metrics['mean_recovery'] > threshold:
            important_layers.append(layer_idx)

    return sorted(important_layers)


def compute_causal_contribution(
    layer_recoveries: Dict[int, float]
) -> Dict[int, float]:
    """
    compute normalized causal contribution for each layer.

    args:
        layer_recoveries: dict mapping layer to recovery score

    returns:
        dict mapping layer to normalized contribution
    """
    total_recovery = sum(max(0, r) for r in layer_recoveries.values())

    if total_recovery == 0:
        return {idx: 0.0 for idx in layer_recoveries.keys()}

    contributions = {
        idx: max(0, recovery) / total_recovery
        for idx, recovery in layer_recoveries.items()
    }

    return contributions


class ClinicalStratifiedPatcher:
    """
    path patching stratified by clinical feature values.
    
    tests hypothesis: if a head encodes a clinical feature (per probing),
    patching that head should preferentially affect predictions for samples
    with abnormal values of that feature.
    """
    
    def __init__(
        self,
        patcher: ActivationPatcher,
        clinical_features: Dict[str, np.ndarray],
        sample_ids: List[str]
    ):
        """
        args:
            patcher: ActivationPatcher instance
            clinical_features: dict mapping feature name to values per sample
            sample_ids: list of sample identifiers matching feature arrays
        """
        self.patcher = patcher
        self.clinical_features = clinical_features
        self.sample_ids = sample_ids
        self.sample_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    
    def stratify_by_feature(
        self,
        feature_name: str,
        n_strata: int = 3,
        method: str = "quantile"
    ) -> Dict[int, List[int]]:
        """
        stratify samples by clinical feature values.
        
        args:
            feature_name: name of clinical feature
            n_strata: number of strata (e.g., 3 for low/medium/high)
            method: 'quantile' or 'equal'
        
        returns:
            dict mapping stratum index to sample indices
        """
        if feature_name not in self.clinical_features:
            raise ValueError(f"unknown feature: {feature_name}")
        
        values = self.clinical_features[feature_name]
        
        if method == "quantile":
            quantiles = np.linspace(0, 1, n_strata + 1)
            thresholds = np.quantile(values[~np.isnan(values)], quantiles)
        else:
            valid_values = values[~np.isnan(values)]
            min_val, max_val = valid_values.min(), valid_values.max()
            thresholds = np.linspace(min_val, max_val, n_strata + 1)
        
        strata = {}
        for stratum in range(n_strata):
            low = thresholds[stratum]
            high = thresholds[stratum + 1]
            
            if stratum == n_strata - 1:
                mask = (values >= low) & (values <= high)
            else:
                mask = (values >= low) & (values < high)
            
            strata[stratum] = list(np.where(mask)[0])
        
        return strata
    
    def run_stratified_head_patching(
        self,
        dataset,
        feature_name: str,
        target_heads: List[Tuple[int, int]],
        n_pairs_per_stratum: int = 20
    ) -> Dict[str, Any]:
        """
        run head patching stratified by clinical feature.
        
        args:
            dataset: pytorch dataset
            feature_name: clinical feature to stratify by
            target_heads: list of (layer, head) tuples to test
            n_pairs_per_stratum: pairs per stratum
        
        returns:
            stratified patching results
        """
        strata = self.stratify_by_feature(feature_name, n_strata=3)
        stratum_names = {0: 'low', 1: 'medium', 2: 'high'}
        
        results = {}
        
        for stratum_idx, sample_indices in strata.items():
            stratum_name = stratum_names.get(stratum_idx, str(stratum_idx))
            
            if len(sample_indices) < 2:
                warnings.warn(f"stratum {stratum_name} has too few samples")
                continue
            
            # create pairs from this stratum
            pd_indices = [i for i in sample_indices if dataset.samples[i]['label'] == 1]
            hc_indices = [i for i in sample_indices if dataset.samples[i]['label'] == 0]
            
            if not pd_indices or not hc_indices:
                warnings.warn(f"stratum {stratum_name} lacks both classes")
                continue
            
            # create minimal pairs
            pairs = []
            for pd_idx in pd_indices[:n_pairs_per_stratum]:
                hc_idx = np.random.choice(hc_indices)
                
                hc_sample = dataset[hc_idx]
                pd_sample = dataset[pd_idx]
                
                pairs.append((
                    hc_sample['input_values'],
                    pd_sample['input_values'],
                    0
                ))
            
            if not pairs:
                continue
            
            # run head patching for this stratum
            head_effects = defaultdict(list)
            
            for clean, corrupted, label in tqdm(pairs, desc=f"patching {stratum_name}"):
                try:
                    # get clean activations with attention
                    clean_acts, clean_logits, clean_attn = self.patcher.get_clean_activations(
                        clean, include_attention=True
                    )
                    
                    corrupted_batch = corrupted.to(self.patcher.device)
                    if corrupted_batch.dim() == 1:
                        corrupted_batch = corrupted_batch.unsqueeze(0)
                    
                    corrupted_logits = self.patcher.model(corrupted_batch).logits
                    
                    clean_diff = (clean_logits[0, label] - clean_logits[0, 1-label]).item()
                    corrupted_diff = (corrupted_logits[0, label] - corrupted_logits[0, 1-label]).item()
                    total_effect = clean_diff - corrupted_diff
                    
                    for layer_idx, head_idx in target_heads:
                        clean_attn_out = clean_attn.get(f'attn_{layer_idx}')
                        if clean_attn_out is None:
                            continue
                        
                        patched_logits = self.patcher.patch_attention_head(
                            corrupted, clean_attn_out, layer_idx, head_idx
                        )
                        
                        patched_diff = (
                            patched_logits[0, label] - patched_logits[0, 1-label]
                        ).item()
                        
                        if abs(total_effect) > 1e-6:
                            recovery = (patched_diff - corrupted_diff) / total_effect
                        else:
                            recovery = 0.0
                        
                        head_effects[(layer_idx, head_idx)].append(recovery)
                
                except Exception as e:
                    warnings.warn(f"stratified patching failed: {e}")
            
            # summarize for this stratum
            stratum_results = {}
            for (layer, head), recoveries in head_effects.items():
                stratum_results[f"L{layer}H{head}"] = {
                    'mean_recovery': float(np.mean(recoveries)),
                    'std_recovery': float(np.std(recoveries)),
                    'n_samples': len(recoveries)
                }
            
            results[stratum_name] = {
                'n_pairs': len(pairs),
                'head_effects': stratum_results
            }
        
        # compute differential effects (high vs low stratum)
        differential = {}
        if 'high' in results and 'low' in results:
            for head_key in results['high']['head_effects'].keys():
                high_effect = results['high']['head_effects'].get(head_key, {}).get('mean_recovery', 0)
                low_effect = results['low']['head_effects'].get(head_key, {}).get('mean_recovery', 0)
                differential[head_key] = high_effect - low_effect
        
        results['differential_effects'] = differential
        results['feature_name'] = feature_name
        
        return results
    
    def validate_clinical_causality(
        self,
        dataset,
        head_probing_results: Dict[Tuple[int, int], Dict[str, float]],
        n_pairs: int = 50
    ) -> Dict[str, Any]:
        """
        test if heads that encode clinical features (per probing)
        also causally affect predictions in a feature-specific way.
        
        args:
            dataset: pytorch dataset
            head_probing_results: probing accuracy for each head on each feature
                {(layer, head): {'jitter': 0.8, 'shimmer': 0.7, ...}}
            n_pairs: number of pairs per feature
        
        returns:
            validation results
        """
        results = {}
        
        for (layer, head), feature_scores in head_probing_results.items():
            # find features this head encodes well
            top_features = sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            head_validation = {}
            
            for feature_name, probing_acc in top_features:
                if probing_acc < 0.6:  # skip if probing accuracy is low
                    continue
                
                if feature_name not in self.clinical_features:
                    continue
                
                # run stratified patching for this feature
                try:
                    stratified_results = self.run_stratified_head_patching(
                        dataset,
                        feature_name,
                        target_heads=[(layer, head)],
                        n_pairs_per_stratum=n_pairs // 3
                    )
                    
                    # check if patching effect is higher in high stratum
                    diff_effect = stratified_results.get('differential_effects', {}).get(
                        f"L{layer}H{head}", 0
                    )
                    
                    head_validation[feature_name] = {
                        'probing_accuracy': probing_acc,
                        'differential_patching_effect': diff_effect,
                        'causal_alignment': 'aligned' if diff_effect > 0.05 else 'not_aligned'
                    }
                
                except Exception as e:
                    warnings.warn(f"validation failed for {feature_name}: {e}")
            
            if head_validation:
                results[f"L{layer}H{head}"] = head_validation
        
        return results


class PathPatchingAnalyzer:
    """
    comprehensive path patching analysis combining probing and patching.
    
    implements the full analysis pipeline:
    1. identify important heads from layer/head patching
    2. probe heads for clinical feature encoding
    3. validate causality with stratified patching
    """
    
    def __init__(
        self,
        model: Wav2Vec2ForSequenceClassification,
        device: str = "cuda"
    ):
        """
        args:
            model: wav2vec2 classifier
            device: device to run on
        """
        self.model = model
        self.device = device
        self.patcher = ActivationPatcher(model, device)
    
    def run_full_analysis(
        self,
        dataset,
        clinical_features: Optional[Dict[str, np.ndarray]] = None,
        sample_ids: Optional[List[str]] = None,
        n_pairs: int = 100,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        run comprehensive patching analysis.
        
        args:
            dataset: pytorch dataset
            clinical_features: optional clinical feature values
            sample_ids: sample identifiers
            n_pairs: number of pairs for patching
            save_path: optional path to save results
        
        returns:
            comprehensive analysis results
        """
        results = {
            'metadata': {
                'n_pairs': n_pairs,
                'n_layers': self.patcher.num_layers,
                'n_heads': self.patcher.num_heads
            }
        }
        
        # create minimal pairs
        pairs = create_mfcc_matched_pairs(dataset, n_pairs=n_pairs)
        
        if not pairs:
            pairs = create_minimal_pairs(dataset, n_pairs=n_pairs)
        
        clean_inputs = [p[0] for p in pairs]
        corrupted_inputs = [p[1] for p in pairs]
        labels = [p[2] for p in pairs]
        
        # run layer and head patching
        patching_results = self.patcher.run_batch_patching(
            clean_inputs, corrupted_inputs, labels,
            include_heads=True
        )
        results['patching'] = patching_results
        
        # compute mean activations and validate with ablation
        self.patcher.compute_mean_activations(dataset)
        ablation_results = self.patcher.validate_with_ablation(
            dataset, patching_results
        )
        results['ablation_validation'] = ablation_results
        
        # clinical stratified analysis if features provided
        if clinical_features is not None and sample_ids is not None:
            stratified_patcher = ClinicalStratifiedPatcher(
                self.patcher, clinical_features, sample_ids
            )
            
            # get important heads
            important_heads = []
            if 'head_patching' in patching_results:
                for head_info in patching_results['head_patching'].get('important_heads', []):
                    important_heads.append((head_info['layer'], head_info['head']))
            
            if important_heads:
                stratified_results = {}
                for feature_name in clinical_features.keys():
                    try:
                        feature_results = stratified_patcher.run_stratified_head_patching(
                            dataset, feature_name, important_heads
                        )
                        stratified_results[feature_name] = feature_results
                    except Exception as e:
                        warnings.warn(f"stratified analysis failed for {feature_name}: {e}")
                
                results['clinical_stratified'] = stratified_results
        
        # save if path provided
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        return results
