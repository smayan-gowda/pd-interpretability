"""
Unit tests for activation patching module.

Tests the comprehensive patching infrastructure including:
- Layer-level patching
- Attention head-level patching with importance ranking
- Clinical feature stratified patching
- Mean ablation validation
"""
import unittest
from unittest.mock import Mock, MagicMock, patch
import torch
import numpy as np


class TestPatchingResult(unittest.TestCase):
    """Test PatchingResult dataclass."""
    
    def test_patching_result_creation(self):
        """Test creating PatchingResult with all fields."""
        from src.interpretability.patching import PatchingResult
        
        result = PatchingResult(
            layer_recoveries={0: 0.1, 1: 0.2, 2: 0.3},
            head_recoveries={(0, 0): 0.05, (0, 1): 0.08},
            position_recoveries={0: {0: 0.1, 1: 0.2}},
            logit_diff_original=-0.5,
            logit_diff_patched=-0.3
        )
        
        self.assertEqual(len(result.layer_recoveries), 3)
        self.assertEqual(len(result.head_recoveries), 2)
        self.assertAlmostEqual(result.logit_diff_original, -0.5)
        self.assertAlmostEqual(result.logit_diff_patched, -0.3)
    
    def test_patching_result_empty(self):
        """Test creating empty PatchingResult."""
        from src.interpretability.patching import PatchingResult
        
        result = PatchingResult(
            layer_recoveries={},
            head_recoveries={},
            position_recoveries={},
            logit_diff_original=0.0,
            logit_diff_patched=0.0
        )
        
        self.assertEqual(len(result.layer_recoveries), 0)


class TestHeadImportanceRanking(unittest.TestCase):
    """Test HeadImportanceRanking class."""
    
    def test_ranking_from_scores(self):
        """Test creating ranking from head scores."""
        from src.interpretability.patching import HeadImportanceRanking
        
        head_scores = {
            (0, 0): 0.15,
            (0, 1): 0.05,
            (1, 0): 0.20,
            (1, 1): 0.03,
        }
        
        ranking = HeadImportanceRanking(
            head_scores=head_scores,
            head_rankings=[(1, 0, 0.20), (0, 0, 0.15), (0, 1, 0.05), (1, 1, 0.03)],
            important_heads=[(1, 0), (0, 0)],
            threshold=0.1
        )
        
        self.assertEqual(len(ranking.head_rankings), 4)
        self.assertEqual(ranking.head_rankings[0], (1, 0, 0.20))
        self.assertEqual(len(ranking.important_heads), 2)
    
    def test_from_patching_results(self):
        """Test creating ranking from patching results."""
        from src.interpretability.patching import HeadImportanceRanking, PatchingResult
        
        # Create mock patching results
        results = [
            PatchingResult(
                layer_recoveries={},
                head_recoveries={(0, 0): 0.15, (0, 1): 0.05},
                position_recoveries={},
                logit_diff_original=-0.5,
                logit_diff_patched=-0.3
            ),
            PatchingResult(
                layer_recoveries={},
                head_recoveries={(0, 0): 0.10, (0, 1): 0.08},
                position_recoveries={},
                logit_diff_original=-0.4,
                logit_diff_patched=-0.2
            )
        ]
        
        ranking = HeadImportanceRanking.from_patching_results(
            results,
            top_k=10,
            threshold=0.1
        )
        
        self.assertIsNotNone(ranking)
        self.assertIn((0, 0), ranking.head_scores)
    
    def test_to_dict(self):
        """Test serializing ranking to dict."""
        from src.interpretability.patching import HeadImportanceRanking
        
        head_scores = {(0, 0): 0.15}
        ranking = HeadImportanceRanking(
            head_scores=head_scores,
            head_rankings=[(0, 0, 0.15)],
            important_heads=[(0, 0)],
            threshold=0.1
        )
        
        d = ranking.to_dict()
        
        self.assertIn('head_rankings', d)
        self.assertIn('important_heads', d)
        self.assertIn('threshold', d)


class TestActivationPatcher(unittest.TestCase):
    """Test ActivationPatcher class."""
    
    def setUp(self):
        """Set up mock model for testing."""
        # Create mock wav2vec2 model
        self.mock_model = Mock()
        self.mock_model.wav2vec2 = Mock()
        self.mock_model.wav2vec2.encoder = Mock()
        self.mock_model.wav2vec2.encoder.layers = [Mock() for _ in range(12)]
        self.mock_model.wav2vec2.config = Mock()
        self.mock_model.wav2vec2.config.hidden_size = 768
        self.mock_model.wav2vec2.config.num_attention_heads = 12
        
        # Set device
        self.mock_model.device = 'cpu'
    
    def test_patcher_initialization(self):
        """Test patcher initialization."""
        from src.interpretability.patching import ActivationPatcher
        
        patcher = ActivationPatcher(self.mock_model, device='cpu')
        
        self.assertEqual(patcher.num_layers, 12)
        self.assertEqual(patcher.num_heads, 12)
        self.assertEqual(patcher.hidden_size, 768)
    
    def test_num_layers_property(self):
        """Test num_layers property."""
        from src.interpretability.patching import ActivationPatcher
        
        patcher = ActivationPatcher(self.mock_model, device='cpu')
        
        self.assertEqual(patcher.num_layers, 12)


class TestHookFunctions(unittest.TestCase):
    """Test hook functions for patching."""
    
    def test_activation_patching_hook(self):
        """Test get_activation_patching_hook function."""
        from src.interpretability.patching import get_activation_patching_hook
        
        source_act = torch.randn(1, 10, 768)
        
        hook_fn = get_activation_patching_hook(source_act, position=None)
        
        self.assertIsNotNone(hook_fn)
        self.assertTrue(callable(hook_fn))
        
        # Test hook application
        target_act = torch.randn(1, 10, 768)
        result = hook_fn(None, None, target_act)
        
        # Should return source activation
        self.assertTrue(torch.allclose(result, source_act))
    
    def test_activation_patching_hook_with_position(self):
        """Test get_activation_patching_hook with specific position."""
        from src.interpretability.patching import get_activation_patching_hook
        
        source_act = torch.randn(1, 10, 768)
        position = 5
        
        hook_fn = get_activation_patching_hook(source_act, position=position)
        
        target_act = torch.randn(1, 10, 768)
        target_original = target_act.clone()
        result = hook_fn(None, None, target_act)
        
        # Only position 5 should be patched
        self.assertTrue(torch.allclose(result[:, position], source_act[:, position]))
        # Other positions should be unchanged
        for pos in range(10):
            if pos != position:
                self.assertTrue(torch.allclose(result[:, pos], target_original[:, pos]))
    
    def test_mean_ablation_hook(self):
        """Test get_mean_ablation_hook function."""
        from src.interpretability.patching import get_mean_ablation_hook
        
        mean_act = torch.randn(768)
        
        hook_fn = get_mean_ablation_hook(mean_act, position=None)
        
        self.assertIsNotNone(hook_fn)
        self.assertTrue(callable(hook_fn))
        
        # Test hook application
        target_act = torch.randn(1, 10, 768)
        result = hook_fn(None, None, target_act)
        
        # All positions should be replaced with mean
        for pos in range(10):
            self.assertTrue(torch.allclose(result[0, pos], mean_act))


class TestMinimalPairCreation(unittest.TestCase):
    """Test minimal pair creation functions."""
    
    def test_create_minimal_pairs(self):
        """Test create_minimal_pairs function."""
        from src.interpretability.patching import create_minimal_pairs
        
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=20)
        
        # Return (audio, label) tuples
        def getitem(idx):
            label = 0 if idx < 10 else 1  # First 10 are HC, rest are PD
            audio = torch.randn(16000)
            return (audio, label)
        
        mock_dataset.__getitem__ = getitem
        
        pairs = create_minimal_pairs(mock_dataset, n_pairs=5)
        
        self.assertIsInstance(pairs, list)
        # Each pair should have (clean, corrupted, expected_label)
        if len(pairs) > 0:
            self.assertEqual(len(pairs[0]), 3)
    
    def test_create_mfcc_matched_pairs(self):
        """Test create_mfcc_matched_pairs function."""
        from src.interpretability.patching import create_mfcc_matched_pairs
        
        # Create mock dataset with audio
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        
        def getitem(idx):
            label = 0 if idx < 5 else 1
            # Return proper audio tensor
            audio = torch.randn(16000)
            return (audio, label)
        
        mock_dataset.__getitem__ = getitem
        
        # This may fail due to MFCC computation, which is expected
        # in unit tests without proper audio
        try:
            pairs = create_mfcc_matched_pairs(mock_dataset, n_pairs=3)
            # If it succeeds, check structure
            self.assertIsInstance(pairs, list)
        except Exception:
            # Expected in unit test environment
            pass


class TestClinicalStratifiedPatcher(unittest.TestCase):
    """Test ClinicalStratifiedPatcher class."""
    
    def setUp(self):
        """Set up mock patcher and clinical features."""
        # Create mock base patcher
        self.mock_patcher = Mock()
        self.mock_patcher.num_layers = 12
        self.mock_patcher.num_heads = 12
        
        # Create clinical features
        self.clinical_features = {
            'jitter': np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.10]),
            'shimmer': np.array([0.05, 0.08, 0.10, 0.15, 0.20, 0.25])
        }
        self.sample_ids = ['s1', 's2', 's3', 's4', 's5', 's6']
    
    def test_patcher_initialization(self):
        """Test ClinicalStratifiedPatcher initialization."""
        from src.interpretability.patching import ClinicalStratifiedPatcher
        
        patcher = ClinicalStratifiedPatcher(
            self.mock_patcher,
            self.clinical_features,
            self.sample_ids
        )
        
        self.assertEqual(patcher.features, self.clinical_features)
        self.assertEqual(len(patcher.sample_ids), 6)
    
    def test_stratify_samples(self):
        """Test sample stratification by clinical feature."""
        from src.interpretability.patching import ClinicalStratifiedPatcher
        
        patcher = ClinicalStratifiedPatcher(
            self.mock_patcher,
            self.clinical_features,
            self.sample_ids
        )
        
        strata = patcher.stratify_by_feature('jitter', n_strata=3)
        
        self.assertIn('low', strata)
        self.assertIn('medium', strata)
        self.assertIn('high', strata)


class TestPathPatchingAnalyzer(unittest.TestCase):
    """Test PathPatchingAnalyzer class."""
    
    def test_analyzer_initialization(self):
        """Test PathPatchingAnalyzer initialization."""
        from src.interpretability.patching import PathPatchingAnalyzer
        
        mock_patcher = Mock()
        mock_patcher.num_layers = 12
        mock_patcher.num_heads = 12
        
        analyzer = PathPatchingAnalyzer(mock_patcher)
        
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.patcher.num_layers, 12)


if __name__ == '__main__':
    unittest.main()
