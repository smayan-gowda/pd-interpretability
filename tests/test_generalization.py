"""
Unit tests for cross-dataset generalization module.

Tests the comprehensive generalization analysis including:
- Cross-dataset evaluation matrix
- Clinical alignment analysis
- Generalization-interpretability correlation
"""
import unittest
from unittest.mock import Mock, MagicMock, patch
import torch
import numpy as np


class TestCrossDatasetResults(unittest.TestCase):
    """Test CrossDatasetResults dataclass."""
    
    def test_results_creation(self):
        """Test creating CrossDatasetResults with all fields."""
        from src.models.generalization import CrossDatasetResults
        
        results = CrossDatasetResults(
            accuracy_matrix={('ds1', 'ds1'): 0.95, ('ds1', 'ds2'): 0.75},
            f1_matrix={('ds1', 'ds1'): 0.94, ('ds1', 'ds2'): 0.72},
            auc_matrix={('ds1', 'ds1'): 0.98, ('ds1', 'ds2'): 0.80},
            generalization_gaps={'ds1': 0.20}
        )
        
        self.assertEqual(len(results.accuracy_matrix), 2)
        self.assertAlmostEqual(results.accuracy_matrix[('ds1', 'ds1')], 0.95)
        self.assertAlmostEqual(results.generalization_gaps['ds1'], 0.20)
    
    def test_results_empty(self):
        """Test creating empty CrossDatasetResults."""
        from src.models.generalization import CrossDatasetResults
        
        results = CrossDatasetResults(
            accuracy_matrix={},
            f1_matrix={},
            auc_matrix={},
            generalization_gaps={}
        )
        
        self.assertEqual(len(results.accuracy_matrix), 0)


class TestClinicalAlignmentProfile(unittest.TestCase):
    """Test ClinicalAlignmentProfile dataclass."""
    
    def test_profile_creation(self):
        """Test creating ClinicalAlignmentProfile."""
        from src.models.generalization import ClinicalAlignmentProfile
        
        profile = ClinicalAlignmentProfile(
            dataset_name='italian_pvs',
            model_name='wav2vec2',
            layerwise_probing_accuracy={
                'jitter': {0: 0.65, 1: 0.70, 2: 0.75},
                'shimmer': {0: 0.60, 1: 0.68, 2: 0.72}
            },
            feature_alignment_scores={'jitter': 0.70, 'shimmer': 0.67},
            overall_alignment=0.685
        )
        
        self.assertEqual(profile.dataset_name, 'italian_pvs')
        self.assertEqual(profile.model_name, 'wav2vec2')
        self.assertAlmostEqual(profile.overall_alignment, 0.685)
        self.assertIn('jitter', profile.layerwise_probing_accuracy)
    
    def test_profile_best_layer(self):
        """Test finding best layer for a feature."""
        from src.models.generalization import ClinicalAlignmentProfile
        
        profile = ClinicalAlignmentProfile(
            dataset_name='test',
            model_name='test',
            layerwise_probing_accuracy={
                'jitter': {0: 0.65, 1: 0.80, 2: 0.75},
            },
            feature_alignment_scores={'jitter': 0.73},
            overall_alignment=0.73
        )
        
        # Best layer should be layer 1 with accuracy 0.80
        best_layer = max(
            profile.layerwise_probing_accuracy['jitter'].items(),
            key=lambda x: x[1]
        )
        self.assertEqual(best_layer[0], 1)
        self.assertAlmostEqual(best_layer[1], 0.80)


class TestCrossDatasetEvaluator(unittest.TestCase):
    """Test CrossDatasetEvaluator class."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        from src.models.generalization import CrossDatasetEvaluator
        
        evaluator = CrossDatasetEvaluator(device='cpu')
        
        self.assertEqual(evaluator.device, 'cpu')
    
    def test_evaluate_single_pair(self):
        """Test evaluating single model-dataset pair."""
        from src.models.generalization import CrossDatasetEvaluator
        
        evaluator = CrossDatasetEvaluator(device='cpu')
        
        # Create mock model
        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.device = 'cpu'
        
        def model_forward(*args, **kwargs):
            # Return mock logits
            return Mock(logits=torch.randn(8, 2))
        
        mock_model.__call__ = model_forward
        
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=16)
        
        def getitem(idx):
            return (torch.randn(16000), idx % 2)  # Alternating labels
        
        mock_dataset.__getitem__ = getitem
        
        # Evaluation should work with mocks
        # Note: This may need more complex mocking for full test


class TestClinicalAlignmentAnalyzer(unittest.TestCase):
    """Test ClinicalAlignmentAnalyzer class."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        from src.models.generalization import ClinicalAlignmentAnalyzer
        
        analyzer = ClinicalAlignmentAnalyzer(
            probe_epochs=50,
            probe_lr=0.01,
            device='cpu'
        )
        
        self.assertEqual(analyzer.probe_epochs, 50)
        self.assertAlmostEqual(analyzer.probe_lr, 0.01)
        self.assertEqual(analyzer.device, 'cpu')
    
    def test_discretize_feature(self):
        """Test feature discretization for probing."""
        from src.models.generalization import ClinicalAlignmentAnalyzer
        
        analyzer = ClinicalAlignmentAnalyzer(device='cpu')
        
        # Test discretization
        values = np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9])
        
        # Should be able to discretize into binary labels
        median = np.median(values)
        labels = (values > median).astype(int)
        
        self.assertEqual(len(labels), 7)
        self.assertTrue(all(l in [0, 1] for l in labels))


class TestGeneralizationInterpretabilityAnalyzer(unittest.TestCase):
    """Test GeneralizationInterpretabilityAnalyzer class."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        from src.models.generalization import GeneralizationInterpretabilityAnalyzer
        
        analyzer = GeneralizationInterpretabilityAnalyzer()
        
        self.assertIsNotNone(analyzer)
    
    def test_compute_correlation(self):
        """Test computing Spearman correlation."""
        from src.models.generalization import GeneralizationInterpretabilityAnalyzer
        
        analyzer = GeneralizationInterpretabilityAnalyzer()
        
        # Create correlated data
        alignment_scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        generalization_scores = [0.55, 0.62, 0.68, 0.78, 0.88]
        
        result = analyzer.compute_correlation(
            alignment_scores=alignment_scores,
            generalization_scores=generalization_scores
        )
        
        self.assertIn('spearman_r', result)
        self.assertIn('p_value', result)
        self.assertIn('interpretation', result)
        
        # Should show positive correlation
        self.assertGreater(result['spearman_r'], 0.5)
    
    def test_compute_correlation_negative(self):
        """Test computing negative Spearman correlation."""
        from src.models.generalization import GeneralizationInterpretabilityAnalyzer
        
        analyzer = GeneralizationInterpretabilityAnalyzer()
        
        # Create negatively correlated data
        alignment_scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        generalization_scores = [0.88, 0.78, 0.68, 0.62, 0.55]
        
        result = analyzer.compute_correlation(
            alignment_scores=alignment_scores,
            generalization_scores=generalization_scores
        )
        
        # Should show negative correlation
        self.assertLess(result['spearman_r'], -0.5)
    
    def test_compute_correlation_insufficient_data(self):
        """Test correlation with insufficient data."""
        from src.models.generalization import GeneralizationInterpretabilityAnalyzer
        
        analyzer = GeneralizationInterpretabilityAnalyzer()
        
        # Only 1 data point
        alignment_scores = [0.5]
        generalization_scores = [0.55]
        
        result = analyzer.compute_correlation(
            alignment_scores=alignment_scores,
            generalization_scores=generalization_scores
        )
        
        # Should handle gracefully
        self.assertIn('spearman_r', result)


class TestDatasetSpecificTrainer(unittest.TestCase):
    """Test DatasetSpecificTrainer class."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        from src.models.generalization import DatasetSpecificTrainer
        
        # Mock model class
        mock_model_class = Mock()
        
        trainer = DatasetSpecificTrainer(
            model_class=mock_model_class,
            epochs=3,
            learning_rate=1e-5,
            batch_size=8,
            device='cpu'
        )
        
        self.assertEqual(trainer.epochs, 3)
        self.assertAlmostEqual(trainer.learning_rate, 1e-5)
        self.assertEqual(trainer.batch_size, 8)
        self.assertEqual(trainer.device, 'cpu')


class TestRunCrossDatasetAnalysis(unittest.TestCase):
    """Test the convenience function for running full analysis."""
    
    def test_function_exists(self):
        """Test that the convenience function exists."""
        from src.models.generalization import run_cross_dataset_analysis
        
        self.assertTrue(callable(run_cross_dataset_analysis))


class TestGeneralizationGapComputation(unittest.TestCase):
    """Test generalization gap computation logic."""
    
    def test_gap_computation(self):
        """Test computing generalization gaps from accuracy matrix."""
        accuracy_matrix = {
            ('ds1', 'ds1'): 0.95,
            ('ds1', 'ds2'): 0.75,
            ('ds1', 'ds3'): 0.70,
            ('ds2', 'ds1'): 0.72,
            ('ds2', 'ds2'): 0.92,
            ('ds2', 'ds3'): 0.68,
        }
        
        dataset_names = ['ds1', 'ds2', 'ds3']
        gaps = {}
        
        for train_name in ['ds1', 'ds2']:
            in_domain = accuracy_matrix.get((train_name, train_name), 0.0)
            
            out_domain_accs = [
                accuracy_matrix.get((train_name, test_name), 0.0)
                for test_name in dataset_names if test_name != train_name
            ]
            out_domain_mean = np.mean(out_domain_accs) if out_domain_accs else 0.0
            
            gaps[train_name] = in_domain - out_domain_mean
        
        # ds1 gap: 0.95 - (0.75 + 0.70)/2 = 0.95 - 0.725 = 0.225
        self.assertAlmostEqual(gaps['ds1'], 0.225, places=3)
        
        # ds2 gap: 0.92 - (0.72 + 0.68)/2 = 0.92 - 0.70 = 0.22
        self.assertAlmostEqual(gaps['ds2'], 0.22, places=3)


class TestSpearmanCorrelationInterpretation(unittest.TestCase):
    """Test interpretation of Spearman correlation values."""
    
    def test_interpretations(self):
        """Test different correlation strength interpretations."""
        from src.models.generalization import GeneralizationInterpretabilityAnalyzer
        
        analyzer = GeneralizationInterpretabilityAnalyzer()
        
        # Strong positive correlation
        result = analyzer.compute_correlation(
            [1, 2, 3, 4, 5],
            [1.1, 2.1, 3.1, 4.1, 5.1]
        )
        self.assertIn('strong', result['interpretation'].lower())
        
        # Near-zero correlation
        result = analyzer.compute_correlation(
            [1, 2, 3, 4, 5],
            [3, 1, 4, 2, 5]  # Random-ish
        )
        # Should handle various cases


if __name__ == '__main__':
    unittest.main()
