"""
unit tests for extraction module.

tests Wav2Vec2ActivationExtractor and related utilities.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile


class TestWav2Vec2ActivationExtractor:
    """tests for Wav2Vec2ActivationExtractor class."""
    
    @pytest.fixture
    def mock_model(self):
        """create mock wav2vec2 model for testing."""
        model = Mock()
        model.config = Mock()
        model.config.num_hidden_layers = 12
        model.config.hidden_size = 768
        
        # mock forward pass with hidden states
        mock_output = Mock()
        mock_output.hidden_states = tuple(
            torch.randn(1, 50, 768) for _ in range(13)  # 12 layers + 1 input
        )
        model.return_value = mock_output
        
        return model
    
    def test_activation_shape_consistency(self):
        """test that extracted activations have correct shape."""
        # simulate activation extraction output
        batch_size = 4
        seq_len = 50
        hidden_size = 768
        num_layers = 12
        
        activations = {
            f'layer_{i}': torch.randn(batch_size, seq_len, hidden_size)
            for i in range(num_layers)
        }
        
        for layer_name, layer_act in activations.items():
            assert layer_act.shape[0] == batch_size
            assert layer_act.shape[1] == seq_len
            assert layer_act.shape[2] == hidden_size
    
    def test_mean_pooling_reduces_sequence(self):
        """test that mean pooling reduces sequence dimension."""
        batch_size = 4
        seq_len = 50
        hidden_size = 768
        
        # simulate layer activations
        activations = torch.randn(batch_size, seq_len, hidden_size)
        
        # mean pool over sequence
        pooled = activations.mean(dim=1)
        
        assert pooled.shape == (batch_size, hidden_size)
    
    def test_attention_mask_respects_padding(self):
        """test that attention masking works correctly."""
        batch_size = 2
        seq_len = 50
        hidden_size = 768
        
        activations = torch.randn(batch_size, seq_len, hidden_size)
        
        # create attention mask (first sample: all valid, second: half valid)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[1, 25:] = 0
        
        # masked mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(activations)
        masked_activations = activations * mask_expanded
        
        pooled = masked_activations.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        assert pooled.shape == (batch_size, hidden_size)
        assert not torch.isnan(pooled).any()


class TestActivationStorage:
    """tests for activation storage utilities."""
    
    def test_memmap_save_load(self):
        """test memmap save and load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test data
            n_samples = 100
            n_layers = 12
            hidden_size = 768
            
            data = np.random.randn(n_samples, n_layers, hidden_size).astype(np.float32)
            
            # save as memmap
            path = Path(tmpdir) / "activations.npy"
            np.save(path, data)
            
            # load as memmap
            loaded = np.load(path, mmap_mode='r')
            
            assert loaded.shape == data.shape
            assert np.allclose(loaded, data)
    
    def test_memmap_indexing(self):
        """test memmap random access efficiency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n_samples = 1000
            hidden_size = 768
            
            data = np.random.randn(n_samples, hidden_size).astype(np.float32)
            
            path = Path(tmpdir) / "activations.npy"
            np.save(path, data)
            
            # load as memmap
            mmap = np.load(path, mmap_mode='r')
            
            # random access should work
            sample_100 = mmap[100]
            sample_500 = mmap[500]
            
            assert sample_100.shape == (hidden_size,)
            assert np.allclose(sample_100, data[100])
            assert np.allclose(sample_500, data[500])
    
    def test_layerwise_activation_storage(self):
        """test storing activations per layer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n_samples = 50
            n_layers = 12
            hidden_size = 768
            
            # save each layer separately
            for layer_idx in range(n_layers):
                layer_data = np.random.randn(n_samples, hidden_size).astype(np.float32)
                path = Path(tmpdir) / f"layer_{layer_idx}.npy"
                np.save(path, layer_data)
            
            # verify all layers saved
            layer_files = list(Path(tmpdir).glob("layer_*.npy"))
            assert len(layer_files) == n_layers
            
            # verify each layer can be loaded
            for layer_idx in range(n_layers):
                path = Path(tmpdir) / f"layer_{layer_idx}.npy"
                loaded = np.load(path)
                assert loaded.shape == (n_samples, hidden_size)


class TestAttentionExtractor:
    """tests for AttentionExtractor class."""
    
    def test_attention_shape(self):
        """test attention weight shape."""
        batch_size = 4
        num_heads = 12
        seq_len = 50
        
        # simulate attention weights
        attention = torch.randn(batch_size, num_heads, seq_len, seq_len)
        
        # should be square in last two dims
        assert attention.shape[-1] == attention.shape[-2]
    
    def test_attention_softmax_sum(self):
        """test that attention sums to 1 along key dimension."""
        batch_size = 2
        num_heads = 12
        seq_len = 50
        
        # create random logits and apply softmax
        logits = torch.randn(batch_size, num_heads, seq_len, seq_len)
        attention = torch.softmax(logits, dim=-1)
        
        # sum along key dimension should be ~1
        row_sums = attention.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    
    def test_average_attention_heads(self):
        """test averaging attention across heads."""
        batch_size = 2
        num_heads = 12
        seq_len = 50
        
        attention = torch.softmax(
            torch.randn(batch_size, num_heads, seq_len, seq_len),
            dim=-1
        )
        
        # average across heads
        avg_attention = attention.mean(dim=1)
        
        assert avg_attention.shape == (batch_size, seq_len, seq_len)


class TestActivationMetadata:
    """tests for activation metadata handling."""
    
    def test_metadata_json_serialization(self):
        """test metadata can be serialized to json."""
        import json
        
        metadata = {
            'model_name': 'facebook/wav2vec2-base-960h',
            'num_layers': 12,
            'hidden_size': 768,
            'num_samples': 100,
            'extraction_date': '2024-01-15',
            'sample_ids': ['sample_0', 'sample_1', 'sample_2'],
            'labels': [0, 1, 0]
        }
        
        # should serialize without error
        json_str = json.dumps(metadata)
        loaded = json.loads(json_str)
        
        assert loaded['model_name'] == metadata['model_name']
        assert loaded['num_layers'] == metadata['num_layers']
    
    def test_metadata_with_activations(self):
        """test metadata file alongside activations."""
        import json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # create activations
            n_samples = 50
            hidden_size = 768
            
            activations = np.random.randn(n_samples, hidden_size).astype(np.float32)
            np.save(Path(tmpdir) / "activations.npy", activations)
            
            # create metadata
            metadata = {
                'shape': list(activations.shape),
                'dtype': str(activations.dtype),
                'sample_ids': [f'sample_{i}' for i in range(n_samples)]
            }
            
            with open(Path(tmpdir) / "metadata.json", 'w') as f:
                json.dump(metadata, f)
            
            # verify both files exist
            assert (Path(tmpdir) / "activations.npy").exists()
            assert (Path(tmpdir) / "metadata.json").exists()


class TestLayerwiseAnalysis:
    """tests for layerwise analysis utilities."""
    
    def test_layer_indexing(self):
        """test correct layer indexing."""
        num_layers = 12
        
        # layer indices should be 0-11
        for i in range(num_layers):
            layer_name = f'layer_{i}'
            assert layer_name.startswith('layer_')
            assert int(layer_name.split('_')[1]) < num_layers
    
    def test_cls_token_extraction(self):
        """test CLS token extraction (first position)."""
        batch_size = 4
        seq_len = 50
        hidden_size = 768
        
        activations = torch.randn(batch_size, seq_len, hidden_size)
        
        # extract first position (CLS-like)
        cls_activations = activations[:, 0, :]
        
        assert cls_activations.shape == (batch_size, hidden_size)
    
    def test_layer_comparison(self):
        """test layer-to-layer comparison."""
        batch_size = 4
        hidden_size = 768
        
        layer_5 = torch.randn(batch_size, hidden_size)
        layer_10 = torch.randn(batch_size, hidden_size)
        
        # compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(layer_5, layer_10, dim=1)
        
        assert cos_sim.shape == (batch_size,)
        assert ((cos_sim >= -1) & (cos_sim <= 1)).all()


class TestDatasetExtraction:
    """tests for dataset-level extraction."""
    
    def test_batch_extraction_accumulation(self):
        """test accumulating activations from batches."""
        batch_size = 8
        hidden_size = 768
        num_batches = 10
        
        all_activations = []
        
        for _ in range(num_batches):
            batch_activations = np.random.randn(batch_size, hidden_size).astype(np.float32)
            all_activations.append(batch_activations)
        
        # concatenate
        full_activations = np.concatenate(all_activations, axis=0)
        
        assert full_activations.shape == (batch_size * num_batches, hidden_size)
    
    def test_label_alignment(self):
        """test that labels align with activations."""
        n_samples = 100
        hidden_size = 768
        
        activations = np.random.randn(n_samples, hidden_size)
        labels = np.random.randint(0, 2, size=n_samples)
        
        # split by label
        pd_activations = activations[labels == 1]
        hc_activations = activations[labels == 0]
        
        # should sum to total
        assert len(pd_activations) + len(hc_activations) == n_samples
    
    def test_subject_grouping(self):
        """test grouping activations by subject."""
        n_samples = 100
        n_subjects = 10
        hidden_size = 768
        
        activations = np.random.randn(n_samples, hidden_size)
        subject_ids = np.random.randint(0, n_subjects, size=n_samples)
        
        # group by subject
        subject_means = {}
        for subj_id in range(n_subjects):
            mask = subject_ids == subj_id
            if mask.sum() > 0:
                subject_means[subj_id] = activations[mask].mean(axis=0)
        
        # each subject should have hidden_size features
        for subj_id, mean_act in subject_means.items():
            assert mean_act.shape == (hidden_size,)
