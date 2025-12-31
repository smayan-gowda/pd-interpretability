"""
activation extraction utilities for wav2vec2.

extracts intermediate layer representations for mechanistic interpretability
analysis including probing and activation patching.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2ForSequenceClassification
from tqdm import tqdm
import json


class Wav2Vec2ActivationExtractor:
    """
    extract activations from all transformer layers of wav2vec2.

    stores activations for later analysis with probing classifiers and
    activation patching experiments.
    """

    def __init__(
        self,
        model: Union[Wav2Vec2Model, Wav2Vec2ForSequenceClassification],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        args:
            model: wav2vec2 model or classifier
            device: device to run on
        """
        self.device = device

        if isinstance(model, Wav2Vec2ForSequenceClassification):
            self.model = model.wav2vec2
            self.classifier = model.classifier
            self.has_classifier = True
        else:
            self.model = model
            self.classifier = None
            self.has_classifier = False

        self.model = self.model.to(device)
        self.model.eval()

        self.num_layers = len(self.model.encoder.layers)
        self.hidden_size = self.model.config.hidden_size

        self.hooks = []
        self.activations = {}

    def _register_hooks(self):
        """register forward hooks to capture layer outputs."""
        self.activations = {}

        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
                else:
                    self.activations[name] = output.detach().cpu()
            return hook

        for i, layer in enumerate(self.model.encoder.layers):
            handle = layer.register_forward_hook(get_activation(f'layer_{i}'))
            self.hooks.append(handle)

        cnn_handle = self.model.feature_extractor.register_forward_hook(
            get_activation('cnn_features')
        )
        self.hooks.append(cnn_handle)

    def _remove_hooks(self):
        """remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def extract(
        self,
        input_values: torch.Tensor,
        pooling: str = 'mean',
        return_attention: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        extract activations from single audio sample.

        args:
            input_values: audio tensor [sequence_length] or [1, sequence_length]
            pooling: pooling strategy ('mean', 'max', 'cls', 'none')
            return_attention: whether to return attention weights

        returns:
            dict mapping layer names to activations
        """
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)

        self._register_hooks()

        input_values = input_values.to(self.device)

        outputs = self.model(
            input_values,
            output_attentions=return_attention,
            output_hidden_states=True
        )

        self._remove_hooks()

        result = {}

        for name, activation in self.activations.items():
            if pooling == 'mean':
                pooled = activation.mean(dim=1).squeeze(0).numpy()
            elif pooling == 'max':
                pooled = activation.max(dim=1).values.squeeze(0).numpy()
            elif pooling == 'cls':
                pooled = activation[:, 0, :].squeeze(0).numpy()
            elif pooling == 'none':
                pooled = activation.squeeze(0).numpy()
            else:
                raise ValueError(f"unknown pooling: {pooling}")

            result[name] = pooled

        if return_attention and outputs.attentions is not None:
            result['attentions'] = [
                att.cpu().numpy() for att in outputs.attentions
            ]

        return result

    def extract_batch(
        self,
        input_values_list: List[torch.Tensor],
        pooling: str = 'mean',
        batch_size: int = 8,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        extract activations from multiple samples.

        args:
            input_values_list: list of audio tensors
            pooling: pooling strategy
            batch_size: batch size for processing
            show_progress: whether to show progress bar

        returns:
            dict mapping layer names to stacked activations
        """
        all_activations = {f'layer_{i}': [] for i in range(self.num_layers)}
        all_activations['cnn_features'] = []

        iterator = range(0, len(input_values_list), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="extracting activations")

        for i in iterator:
            batch = input_values_list[i:i+batch_size]

            for audio in batch:
                acts = self.extract(audio, pooling=pooling)

                for key in all_activations.keys():
                    if key in acts:
                        all_activations[key].append(acts[key])

        result = {
            key: np.stack(vals, axis=0)
            for key, vals in all_activations.items()
        }

        return result

    def extract_to_memmap(
        self,
        input_values_list: List[torch.Tensor],
        output_path: Union[str, Path],
        pooling: str = 'mean',
        batch_size: int = 8
    ) -> np.memmap:
        """
        extract activations and save to memory-mapped file.

        efficient for large datasets that don't fit in memory.

        args:
            input_values_list: list of audio tensors
            output_path: path to save memmap file
            pooling: pooling strategy
            batch_size: batch size

        returns:
            memmap array [n_samples, n_layers, hidden_size]
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_samples = len(input_values_list)

        first_acts = self.extract(input_values_list[0], pooling=pooling)
        n_layers = len([k for k in first_acts.keys() if k.startswith('layer_')])
        hidden_size = first_acts['layer_0'].shape[-1]

        shape = (n_samples, n_layers, hidden_size)

        memmap_array = np.memmap(
            str(output_path),
            dtype='float32',
            mode='w+',
            shape=shape
        )

        for i in tqdm(range(0, n_samples, batch_size), desc="extracting to memmap"):
            batch = input_values_list[i:min(i+batch_size, n_samples)]

            for j, audio in enumerate(batch):
                try:
                    acts = self.extract(audio, pooling=pooling)

                    for layer_idx in range(n_layers):
                        memmap_array[i+j, layer_idx, :] = acts[f'layer_{layer_idx}']

                except Exception as e:
                    warnings.warn(f"failed to extract sample {i+j}: {e}")
                    memmap_array[i+j, :, :] = 0

        memmap_array.flush()

        metadata = {
            'shape': list(shape),
            'n_samples': n_samples,
            'n_layers': n_layers,
            'hidden_size': hidden_size,
            'pooling': pooling,
            'dtype': 'float32'
        }

        metadata_path = str(output_path).replace('.dat', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return memmap_array


def load_activations_memmap(
    memmap_path: Union[str, Path]
) -> Tuple[np.memmap, Dict]:
    """
    load memory-mapped activations file.

    args:
        memmap_path: path to memmap file

    returns:
        (memmap_array, metadata_dict)
    """
    memmap_path = Path(memmap_path)

    metadata_path = str(memmap_path).replace('.dat', '_metadata.json')

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    memmap_array = np.memmap(
        str(memmap_path),
        dtype=metadata.get('dtype', 'float32'),
        mode='r',
        shape=tuple(metadata['shape'])
    )

    return memmap_array, metadata


def extract_activations_from_dataset(
    model: Union[Wav2Vec2Model, Wav2Vec2ForSequenceClassification],
    dataset,
    output_path: Union[str, Path],
    pooling: str = 'mean',
    batch_size: int = 8,
    max_samples: Optional[int] = None
) -> Tuple[np.memmap, Dict]:
    """
    extract activations from entire dataset.

    args:
        model: wav2vec2 model
        dataset: pytorch dataset
        output_path: path to save activations
        pooling: pooling strategy
        batch_size: batch size for extraction
        max_samples: maximum samples to process

    returns:
        (memmap_array, metadata)
    """
    extractor = Wav2Vec2ActivationExtractor(model)

    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    input_values_list = []

    for i in tqdm(range(n_samples), desc="loading dataset"):
        sample = dataset[i]
        input_values_list.append(sample['input_values'])

    memmap_array = extractor.extract_to_memmap(
        input_values_list,
        output_path,
        pooling=pooling,
        batch_size=batch_size
    )

    metadata_path = str(output_path).replace('.dat', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return memmap_array, metadata


class AttentionExtractor:
    """
    extract attention weights from wav2vec2 for analysis.

    useful for visualizing which parts of audio the model attends to.
    """

    def __init__(
        self,
        model: Union[Wav2Vec2Model, Wav2Vec2ForSequenceClassification],
        device: str = "cuda"
    ):
        """
        args:
            model: wav2vec2 model
            device: device to run on
        """
        self.device = device

        if isinstance(model, Wav2Vec2ForSequenceClassification):
            self.model = model.wav2vec2
        else:
            self.model = model

        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def extract_attention(
        self,
        input_values: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        extract attention weights.

        args:
            input_values: audio tensor
            layer_idx: specific layer to extract (none = all layers)

        returns:
            attention weights [num_heads, seq_len, seq_len] or list of such
        """
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)

        input_values = input_values.to(self.device)

        outputs = self.model(
            input_values,
            output_attentions=True
        )

        attentions = [att.cpu().numpy() for att in outputs.attentions]

        if layer_idx is not None:
            return attentions[layer_idx].squeeze(0)
        else:
            return [att.squeeze(0) for att in attentions]
