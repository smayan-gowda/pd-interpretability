"""
unit tests for interpretable prediction interface.

tests the phase 5 synthesis component that provides interpretable predictions
with feature contributions, evidence layers, and key attention heads.
"""

import unittest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock, patch
import json
import tempfile
from pathlib import Path


class TestInterpretablePrediction(unittest.TestCase):
    """Test InterpretablePrediction dataclass."""
    
    def test_create_prediction(self):
        """Test creating prediction with all fields."""
        from src.interpretability.prediction_interface import InterpretablePrediction
        
        pred = InterpretablePrediction(
            pd_probability=0.87,
            feature_contributions={
                'jitter_elevated': 0.34,
                'hnr_reduced': 0.28,
                'f0_unstable': 0.21
            },
            evidence_layers=[3, 4, 7],
            key_attention_heads=[(3, 4), (4, 2), (7, 8)],
            clinical_features={'jitter_local': 0.015, 'hnr_mean': 15.2},
            confidence=0.74,
            raw_logits=np.array([0.5, 1.8])
        )
        
        self.assertAlmostEqual(pred.pd_probability, 0.87)
        self.assertEqual(len(pred.feature_contributions), 3)
        self.assertEqual(pred.evidence_layers, [3, 4, 7])
        self.assertEqual(len(pred.key_attention_heads), 3)
        self.assertIsNotNone(pred.clinical_features)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.interpretability.prediction_interface import InterpretablePrediction
        
        pred = InterpretablePrediction(
            pd_probability=0.75,
            feature_contributions={'jitter_elevated': 0.5},
            evidence_layers=[4, 5],
            key_attention_heads=[(4, 2)],
            confidence=0.5
        )
        
        d = pred.to_dict()
        
        self.assertIn('pd_probability', d)
        self.assertIn('feature_contributions', d)
        self.assertIn('evidence_layers', d)
        self.assertIn('key_attention_heads', d)
        self.assertEqual(d['pd_probability'], 0.75)
        self.assertEqual(d['key_attention_heads'], [[4, 2]])  # tuples become lists
    
    def test_to_json(self):
        """Test JSON serialization."""
        from src.interpretability.prediction_interface import InterpretablePrediction
        
        pred = InterpretablePrediction(
            pd_probability=0.65,
            feature_contributions={'shimmer_elevated': 0.3},
            evidence_layers=[5],
            key_attention_heads=[(5, 3)]
        )
        
        json_str = pred.to_json()
        
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed['pd_probability'], 0.65)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        from src.interpretability.prediction_interface import InterpretablePrediction
        
        data = {
            'pd_probability': 0.92,
            'feature_contributions': {'hnr_reduced': 0.4},
            'evidence_layers': [6, 7, 8],
            'key_attention_heads': [[6, 1], [7, 5]],
            'confidence': 0.84
        }
        
        pred = InterpretablePrediction.from_dict(data)
        
        self.assertAlmostEqual(pred.pd_probability, 0.92)
        self.assertEqual(pred.evidence_layers, [6, 7, 8])
        self.assertEqual(pred.key_attention_heads, [(6, 1), (7, 5)])
    
    def test_get_top_features(self):
        """Test getting top contributing features."""
        from src.interpretability.prediction_interface import InterpretablePrediction
        
        pred = InterpretablePrediction(
            pd_probability=0.8,
            feature_contributions={
                'jitter_elevated': 0.4,
                'hnr_reduced': 0.3,
                'shimmer_elevated': 0.2,
                'f0_unstable': 0.1
            },
            evidence_layers=[3],
            key_attention_heads=[(3, 1)]
        )
        
        top_3 = pred.get_top_features(3)
        
        self.assertEqual(len(top_3), 3)
        self.assertEqual(top_3[0][0], 'jitter_elevated')
        self.assertEqual(top_3[0][1], 0.4)
    
    def test_generate_explanation(self):
        """Test generating human-readable explanation."""
        from src.interpretability.prediction_interface import InterpretablePrediction
        
        pred = InterpretablePrediction(
            pd_probability=0.87,
            feature_contributions={
                'jitter_elevated': 0.34,
                'hnr_reduced': 0.28
            },
            evidence_layers=[3, 4, 7],
            key_attention_heads=[(3, 4), (4, 2)],
            confidence=0.74
        )
        
        explanation = pred.generate_explanation()
        
        self.assertIn("parkinson's disease", explanation.lower())
        self.assertIn("87", explanation)
        self.assertIn("jitter_elevated", explanation)
        self.assertIn("L3H4", explanation)


class TestFeatureContribution(unittest.TestCase):
    """Test FeatureContribution dataclass."""
    
    def test_create_contribution(self):
        """Test creating feature contribution."""
        from src.interpretability.prediction_interface import FeatureContribution
        
        contrib = FeatureContribution(
            feature_name='jitter_local',
            contribution_score=0.34,
            clinical_interpretation='voice pitch instability',
            direction='elevated',
            confidence=0.8
        )
        
        self.assertEqual(contrib.feature_name, 'jitter_local')
        self.assertAlmostEqual(contrib.contribution_score, 0.34)
        self.assertEqual(contrib.direction, 'elevated')
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.interpretability.prediction_interface import FeatureContribution
        
        contrib = FeatureContribution(
            feature_name='hnr_mean',
            contribution_score=0.28,
            clinical_interpretation='harmonics-to-noise ratio',
            direction='reduced',
            confidence=0.75
        )
        
        d = contrib.to_dict()
        
        self.assertIn('feature_name', d)
        self.assertIn('contribution_score', d)
        self.assertIn('direction', d)


class TestInterpretablePredictionInterface(unittest.TestCase):
    """Test InterpretablePredictionInterface class."""
    
    def setUp(self):
        """Set up mock model and processor."""
        # mock model
        self.mock_model = Mock()
        self.mock_model.eval = Mock()
        self.mock_model.to = Mock(return_value=self.mock_model)
        
        # mock output
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.2, 0.8]])
        self.mock_model.return_value = mock_output
        
        # mock processor
        self.mock_processor = Mock()
        self.mock_processor.return_value = {
            'input_values': torch.randn(1, 16000)
        }
    
    def test_init(self):
        """Test interface initialization."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        interface = InterpretablePredictionInterface(
            model=self.mock_model,
            processor=self.mock_processor,
            device='cpu'
        )
        
        self.assertIsNotNone(interface)
        self.mock_model.eval.assert_called_once()
        self.mock_model.to.assert_called_with('cpu')
    
    def test_init_with_probing_results(self):
        """Test initialization with probing results."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        probing_results = {
            'jitter_local': {3: 0.7, 4: 0.8, 5: 0.6},
            'shimmer_local': {4: 0.65, 5: 0.75, 6: 0.55}
        }
        
        interface = InterpretablePredictionInterface(
            model=self.mock_model,
            processor=self.mock_processor,
            probing_results=probing_results
        )
        
        # should compute evidence layers from probing
        self.assertTrue(len(interface._evidence_layers) > 0)
    
    def test_init_with_patching_results(self):
        """Test initialization with patching results."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        patching_results = {
            (3, 4): 0.15,
            (4, 2): 0.12,
            (7, 8): 0.10
        }
        
        interface = InterpretablePredictionInterface(
            model=self.mock_model,
            processor=self.mock_processor,
            patching_results=patching_results
        )
        
        # should compute key heads from patching
        self.assertTrue(len(interface._key_heads) > 0)
        self.assertIn((3, 4), interface._key_heads)
    
    def test_predict_basic(self):
        """Test basic prediction without clinical features."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        interface = InterpretablePredictionInterface(
            model=self.mock_model,
            processor=self.mock_processor
        )
        
        audio = np.random.randn(16000)
        
        prediction = interface.predict(
            audio,
            sample_rate=16000,
            include_clinical=False
        )
        
        self.assertIsNotNone(prediction)
        self.assertTrue(0 <= prediction.pd_probability <= 1)
        self.assertIsInstance(prediction.evidence_layers, list)
        self.assertIsInstance(prediction.key_attention_heads, list)
    
    def test_predict_with_tensor_input(self):
        """Test prediction with tensor input."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        interface = InterpretablePredictionInterface(
            model=self.mock_model,
            processor=self.mock_processor
        )
        
        audio = torch.randn(16000)
        
        prediction = interface.predict(
            audio,
            sample_rate=16000,
            include_clinical=False
        )
        
        self.assertIsNotNone(prediction)
    
    def test_predict_with_clinical_extractor(self):
        """Test prediction with clinical feature extraction."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        # mock clinical extractor
        mock_extractor = Mock()
        mock_extractor.extract_all_features = Mock(return_value={
            'jitter_local': 0.015,
            'shimmer_local': 0.045,
            'hnr_mean': 18.5
        })
        
        interface = InterpretablePredictionInterface(
            model=self.mock_model,
            processor=self.mock_processor,
            clinical_extractor=mock_extractor
        )
        
        audio = np.random.randn(16000)
        
        prediction = interface.predict(
            audio,
            sample_rate=16000,
            include_clinical=True
        )
        
        self.assertIsNotNone(prediction.clinical_features)
        self.assertIn('jitter_local', prediction.clinical_features)
    
    def test_set_probing_results(self):
        """Test setting probing results after init."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        interface = InterpretablePredictionInterface(
            model=self.mock_model,
            processor=self.mock_processor
        )
        
        new_results = {
            'jitter_local': {4: 0.85, 5: 0.80}
        }
        
        interface.set_probing_results(new_results)
        
        self.assertEqual(interface.probing_results, new_results)
    
    def test_set_patching_results(self):
        """Test setting patching results after init."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        interface = InterpretablePredictionInterface(
            model=self.mock_model,
            processor=self.mock_processor
        )
        
        new_results = {
            (5, 3): 0.20,
            (6, 7): 0.18
        }
        
        interface.set_patching_results(new_results)
        
        self.assertEqual(interface.patching_results, new_results)
        self.assertIn((5, 3), interface._key_heads)
    
    def test_batch_predict(self):
        """Test batch prediction."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        interface = InterpretablePredictionInterface(
            model=self.mock_model,
            processor=self.mock_processor
        )
        
        audio_list = [np.random.randn(16000) for _ in range(3)]
        
        predictions = interface.batch_predict(
            audio_list,
            include_clinical=False,
            show_progress=False
        )
        
        self.assertEqual(len(predictions), 3)
        for pred in predictions:
            self.assertTrue(0 <= pred.pd_probability <= 1)
    
    def test_explain_prediction_text(self):
        """Test text explanation generation."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        interface = InterpretablePredictionInterface(
            model=self.mock_model,
            processor=self.mock_processor
        )
        
        audio = np.random.randn(16000)
        
        explanation = interface.explain_prediction(
            audio,
            format='text'
        )
        
        self.assertIsInstance(explanation, str)
        self.assertTrue(len(explanation) > 0)
    
    def test_explain_prediction_json(self):
        """Test JSON explanation generation."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        interface = InterpretablePredictionInterface(
            model=self.mock_model,
            processor=self.mock_processor
        )
        
        audio = np.random.randn(16000)
        
        explanation = interface.explain_prediction(
            audio,
            format='json'
        )
        
        # should be valid JSON
        parsed = json.loads(explanation)
        self.assertIn('pd_probability', parsed)
    
    def test_explain_prediction_markdown(self):
        """Test markdown explanation generation."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        interface = InterpretablePredictionInterface(
            model=self.mock_model,
            processor=self.mock_processor
        )
        
        audio = np.random.randn(16000)
        
        explanation = interface.explain_prediction(
            audio,
            format='markdown'
        )
        
        self.assertIn('#', explanation)  # markdown headers
        self.assertIn('Prediction', explanation)
    
    def test_save_and_load_prediction(self):
        """Test saving prediction to file."""
        from src.interpretability.prediction_interface import (
            InterpretablePredictionInterface,
            InterpretablePrediction
        )
        
        interface = InterpretablePredictionInterface(
            model=self.mock_model,
            processor=self.mock_processor
        )
        
        audio = np.random.randn(16000)
        prediction = interface.predict(audio, include_clinical=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'prediction.json'
            interface.save_prediction(prediction, path)
            
            self.assertTrue(path.exists())
            
            # load and verify
            with open(path) as f:
                loaded = json.load(f)
            
            self.assertIn('pd_probability', loaded)
    
    def test_load_analysis_results(self):
        """Test loading precomputed analysis results."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        interface = InterpretablePredictionInterface(
            model=self.mock_model,
            processor=self.mock_processor
        )
        
        analysis_data = {
            'probing_results': {
                'jitter_local': {'3': 0.7, '4': 0.8},
                'shimmer_local': {'4': 0.65, '5': 0.7}
            },
            'patching_results': {
                '3,4': 0.15,
                '5,2': 0.12
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'analysis.json'
            with open(path, 'w') as f:
                json.dump(analysis_data, f)
            
            interface.load_analysis_results(path)
        
        self.assertIn('jitter_local', interface.probing_results)
        self.assertIn((3, 4), interface.patching_results)


class TestCreateInterpretableInterface(unittest.TestCase):
    """Test factory function."""
    
    def test_factory_function(self):
        """Test create_interpretable_interface function."""
        from src.interpretability.prediction_interface import create_interpretable_interface
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)
        
        mock_processor = Mock()
        
        interface = create_interpretable_interface(
            model=mock_model,
            processor=mock_processor
        )
        
        self.assertIsNotNone(interface)


class TestClinicalInterpretations(unittest.TestCase):
    """Test clinical feature interpretations."""
    
    def test_interpretations_exist(self):
        """Test that clinical interpretations are defined."""
        from src.interpretability.prediction_interface import CLINICAL_INTERPRETATIONS
        
        self.assertIn('jitter_local', CLINICAL_INTERPRETATIONS)
        self.assertIn('shimmer_local', CLINICAL_INTERPRETATIONS)
        self.assertIn('hnr_mean', CLINICAL_INTERPRETATIONS)
    
    def test_interpretation_format(self):
        """Test interpretation tuple format."""
        from src.interpretability.prediction_interface import CLINICAL_INTERPRETATIONS
        
        for feature, (interpretation, direction) in CLINICAL_INTERPRETATIONS.items():
            self.assertIsInstance(interpretation, str)
            self.assertIn(direction, ['elevated', 'reduced', 'unstable'])


class TestComputeFeatureContributions(unittest.TestCase):
    """Test feature contribution computation."""
    
    def test_compute_with_clinical_features(self):
        """Test contribution computation with clinical features."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_processor = Mock()
        
        interface = InterpretablePredictionInterface(
            model=mock_model,
            processor=mock_processor
        )
        
        clinical_features = {
            'jitter_local': 0.02,  # elevated (ref ~0.005)
            'hnr_mean': 12.0  # reduced (ref ~20.0)
        }
        
        contributions = interface._compute_feature_contributions(
            clinical_features,
            pd_probability=0.8
        )
        
        self.assertIsInstance(contributions, dict)
    
    def test_compute_without_clinical_features(self):
        """Test contribution computation from probing only."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_processor = Mock()
        
        probing_results = {
            'jitter_local': {3: 0.7, 4: 0.8},
            'shimmer_local': {4: 0.65}
        }
        
        interface = InterpretablePredictionInterface(
            model=mock_model,
            processor=mock_processor,
            probing_results=probing_results
        )
        
        contributions = interface._compute_feature_contributions(
            None,  # no clinical features
            pd_probability=0.85
        )
        
        self.assertIsInstance(contributions, dict)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_audio(self):
        """Test handling of empty audio."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)
        
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.5, 0.5]])
        mock_model.return_value = mock_output
        
        mock_processor = Mock()
        mock_processor.return_value = {'input_values': torch.zeros(1, 100)}
        
        interface = InterpretablePredictionInterface(
            model=mock_model,
            processor=mock_processor
        )
        
        audio = np.zeros(100)
        
        # should not raise
        prediction = interface.predict(audio, include_clinical=False)
        self.assertIsNotNone(prediction)
    
    def test_multidimensional_audio(self):
        """Test handling of multi-dimensional audio."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)
        
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.3, 0.7]])
        mock_model.return_value = mock_output
        
        mock_processor = Mock()
        mock_processor.return_value = {'input_values': torch.randn(1, 16000)}
        
        interface = InterpretablePredictionInterface(
            model=mock_model,
            processor=mock_processor
        )
        
        # 2D audio (should be flattened)
        audio = np.random.randn(1, 16000)
        
        prediction = interface.predict(audio, include_clinical=False)
        self.assertIsNotNone(prediction)
    
    def test_clinical_extraction_failure(self):
        """Test graceful handling of clinical extraction failure."""
        from src.interpretability.prediction_interface import InterpretablePredictionInterface
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)
        
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.2, 0.8]])
        mock_model.return_value = mock_output
        
        mock_processor = Mock()
        mock_processor.return_value = {'input_values': torch.randn(1, 16000)}
        
        # extractor that raises
        mock_extractor = Mock()
        mock_extractor.extract_all_features = Mock(side_effect=ValueError("extraction failed"))
        
        interface = InterpretablePredictionInterface(
            model=mock_model,
            processor=mock_processor,
            clinical_extractor=mock_extractor
        )
        
        audio = np.random.randn(16000)
        
        # should not raise, just warn
        with self.assertWarns(Warning):
            prediction = interface.predict(audio, include_clinical=True)
        
        self.assertIsNone(prediction.clinical_features)


if __name__ == '__main__':
    unittest.main()
