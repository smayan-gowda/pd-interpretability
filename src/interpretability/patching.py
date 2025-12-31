"""
activation patching utilities for causal analysis.

implements activation patching to determine which model components
causally affect predictions.
"""

from typing import Dict, List, Optional, Tuple, Callable
import warnings

import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification
from tqdm import tqdm


class ActivationPatcher:
    """
    perform activation patching experiments on wav2vec2.

    activation patching tests causality: if patching activations from a clean
    run into a corrupted run changes the prediction, those activations are
    causally important.
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

        self.cached_activations = {}
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
        input_values: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        run clean forward pass and cache all layer activations.

        args:
            input_values: clean audio input

        returns:
            (activations_dict, clean_logits)
        """
        activations = {}
        hooks = []

        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].clone()
                else:
                    activations[name] = output.clone()
            return hook

        for i, layer in enumerate(self.model.wav2vec2.encoder.layers):
            h = layer.register_forward_hook(get_activation(f'layer_{i}'))
            hooks.append(h)

        input_values = input_values.to(self.device)
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)

        outputs = self.model(input_values)

        for h in hooks:
            h.remove()

        return activations, outputs.logits.clone()

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

    def run_layer_patching(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        clean_label: int
    ) -> Dict[int, float]:
        """
        patch each layer individually and measure effect.

        args:
            clean_input: clean audio
            corrupted_input: corrupted audio
            clean_label: true label for clean input

        returns:
            dict mapping layer_idx to logit difference recovered
        """
        clean_acts, clean_logits = self.get_clean_activations(clean_input)

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

        total_diff = clean_logit_diff - corrupted_logit_diff

        results = {}

        for layer_idx in range(self.num_layers):
            patched_logits = self.patch_layer(
                corrupted_input,
                clean_acts[f'layer_{layer_idx}'],
                layer_idx
            )

            patched_logit_diff = (
                patched_logits[0, clean_label] - patched_logits[0, 1 - clean_label]
            ).item()

            if abs(total_diff) > 1e-6:
                recovery = (patched_logit_diff - corrupted_logit_diff) / total_diff
            else:
                recovery = 0.0

            results[layer_idx] = recovery

        return results

    def run_batch_patching(
        self,
        clean_inputs: List[torch.Tensor],
        corrupted_inputs: List[torch.Tensor],
        clean_labels: List[int]
    ) -> Dict[int, Dict[str, float]]:
        """
        run patching on multiple (clean, corrupted) pairs.

        args:
            clean_inputs: list of clean audio tensors
            corrupted_inputs: list of corrupted audio tensors
            clean_labels: list of clean labels

        returns:
            dict mapping layer_idx to recovery statistics
        """
        layer_recoveries = {i: [] for i in range(self.num_layers)}

        for clean, corrupted, label in tqdm(
            zip(clean_inputs, corrupted_inputs, clean_labels),
            total=len(clean_inputs),
            desc="patching experiments"
        ):
            try:
                results = self.run_layer_patching(clean, corrupted, label)

                for layer_idx, recovery in results.items():
                    layer_recoveries[layer_idx].append(recovery)

            except Exception as e:
                warnings.warn(f"patching failed: {e}")

        summary = {}

        for layer_idx in range(self.num_layers):
            recoveries = layer_recoveries[layer_idx]

            if len(recoveries) > 0:
                summary[layer_idx] = {
                    'mean_recovery': np.mean(recoveries),
                    'std_recovery': np.std(recoveries),
                    'median_recovery': np.median(recoveries),
                    'min_recovery': np.min(recoveries),
                    'max_recovery': np.max(recoveries)
                }
            else:
                summary[layer_idx] = {
                    'mean_recovery': 0.0,
                    'std_recovery': 0.0,
                    'median_recovery': 0.0,
                    'min_recovery': 0.0,
                    'max_recovery': 0.0
                }

        return summary


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
