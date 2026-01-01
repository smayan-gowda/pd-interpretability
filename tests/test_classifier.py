"""
unit tests for classifier module.

tests Wav2Vec2PDClassifier, training utilities, and evaluation functions.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json


class TestWav2Vec2PDClassifier:
    """tests for Wav2Vec2PDClassifier class."""
    
    @pytest.fixture
    def mock_model(self):
        """create mock wav2vec2 model."""
        model = Mock()
        model.config = Mock()
        model.config.hidden_size = 768
        model.wav2vec2 = Mock()
        model.wav2vec2.feature_extractor = Mock()
        return model
    
    def test_compute_metrics_perfect_predictions(self):
        """test compute_metrics with perfect predictions."""
        from src.models.classifier import compute_metrics
        
        # perfect predictions
        eval_pred = (
            np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]),
            np.array([1, 0, 1, 0])
        )
        
        metrics = compute_metrics(eval_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['auc'] == 1.0
        assert metrics['sensitivity'] == 1.0
        assert metrics['specificity'] == 1.0
    
    def test_compute_metrics_random_predictions(self):
        """test compute_metrics with random predictions."""
        from src.models.classifier import compute_metrics
        
        # 50% accuracy
        eval_pred = (
            np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1]]),
            np.array([1, 0, 0, 1])
        )
        
        metrics = compute_metrics(eval_pred)
        
        assert metrics['accuracy'] == 0.5
        assert 'f1' in metrics
        assert 'sensitivity' in metrics
        assert 'specificity' in metrics
    
    def test_compute_metrics_all_required_keys(self):
        """test that compute_metrics returns all required keys."""
        from src.models.classifier import compute_metrics
        
        eval_pred = (
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([0, 1])
        )
        
        metrics = compute_metrics(eval_pred)
        
        required_keys = ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']
        for key in required_keys:
            assert key in metrics
    
    def test_create_training_args_defaults(self):
        """test create_training_args with defaults."""
        from src.models.classifier import create_training_args
        
        with tempfile.TemporaryDirectory() as tmpdir:
            args = create_training_args(output_dir=tmpdir)
            
            assert args.output_dir == tmpdir
            assert args.num_train_epochs == 20
            assert args.per_device_train_batch_size == 8
            assert args.learning_rate == 1e-4
            assert args.fp16 == True
    
    def test_create_training_args_custom(self):
        """test create_training_args with custom values."""
        from src.models.classifier import create_training_args
        
        with tempfile.TemporaryDirectory() as tmpdir:
            args = create_training_args(
                output_dir=tmpdir,
                num_epochs=10,
                batch_size=4,
                learning_rate=5e-5,
                warmup_ratio=0.2
            )
            
            assert args.num_train_epochs == 10
            assert args.per_device_train_batch_size == 4
            assert args.learning_rate == 5e-5
            assert args.warmup_ratio == 0.2


class TestDataCollatorWithPadding:
    """tests for DataCollatorWithPadding class."""
    
    def test_collate_single_sample(self):
        """test collation of single sample."""
        from src.models.classifier import DataCollatorWithPadding
        
        collator = DataCollatorWithPadding()
        
        batch = [{
            'input_values': torch.randn(16000),
            'labels': torch.tensor(0)
        }]
        
        result = collator(batch)
        
        assert 'input_values' in result
        assert 'labels' in result
        assert result['input_values'].shape[0] == 1
        assert result['labels'].shape[0] == 1
    
    def test_collate_multiple_samples_same_length(self):
        """test collation of multiple samples with same length."""
        from src.models.classifier import DataCollatorWithPadding
        
        collator = DataCollatorWithPadding()
        
        batch = [
            {'input_values': torch.randn(16000), 'labels': torch.tensor(0)},
            {'input_values': torch.randn(16000), 'labels': torch.tensor(1)},
        ]
        
        result = collator(batch)
        
        assert result['input_values'].shape == (2, 16000)
        assert result['labels'].shape == (2,)
    
    def test_collate_multiple_samples_different_lengths(self):
        """test collation pads shorter samples."""
        from src.models.classifier import DataCollatorWithPadding
        
        collator = DataCollatorWithPadding()
        
        batch = [
            {'input_values': torch.randn(16000), 'labels': torch.tensor(0)},
            {'input_values': torch.randn(8000), 'labels': torch.tensor(1)},
        ]
        
        result = collator(batch)
        
        # should be padded to longest
        assert result['input_values'].shape == (2, 16000)
        assert result['labels'].shape == (2,)
    
    def test_attention_mask_creation(self):
        """test attention mask is created correctly."""
        from src.models.classifier import DataCollatorWithPadding
        
        collator = DataCollatorWithPadding()
        
        batch = [
            {'input_values': torch.randn(16000), 'labels': torch.tensor(0)},
            {'input_values': torch.randn(8000), 'labels': torch.tensor(1)},
        ]
        
        result = collator(batch)
        
        assert 'attention_mask' in result
        
        # first sample should have all 1s
        assert result['attention_mask'][0].sum() == 16000
        
        # second sample should have some 0s (padding)
        assert result['attention_mask'][1].sum() == 8000


class TestCrossValidation:
    """tests for cross-validation module."""
    
    def test_cvresults_aggregate_empty(self):
        """test CVResults with empty results."""
        from src.models.cross_validation import CVResults
        
        results = CVResults.aggregate([])
        
        assert results.fold_metrics == []
        assert results.mean_metrics == {}
        assert results.std_metrics == {}
    
    def test_cvresults_aggregate_single_fold(self):
        """test CVResults aggregation with single fold."""
        from src.models.cross_validation import CVResults
        
        fold_result = CVResults(
            fold_metrics=[{'accuracy': 0.85, 'f1': 0.82}],
            mean_metrics={'accuracy': 0.85, 'f1': 0.82},
            std_metrics={'accuracy': 0.0, 'f1': 0.0},
            n_folds=1
        )
        
        results = CVResults.aggregate([fold_result])
        
        assert results.n_folds == 1
        assert results.mean_metrics['accuracy'] == 0.85
    
    def test_cvresults_aggregate_multiple_folds(self):
        """test CVResults aggregation with multiple folds."""
        from src.models.cross_validation import CVResults
        
        # simulate 3-fold results
        fold_results = []
        for acc, f1 in [(0.80, 0.78), (0.85, 0.83), (0.90, 0.88)]:
            fold_results.append(CVResults(
                fold_metrics=[{'accuracy': acc, 'f1': f1}],
                mean_metrics={'accuracy': acc, 'f1': f1},
                std_metrics={'accuracy': 0.0, 'f1': 0.0},
                n_folds=1
            ))
        
        results = CVResults.aggregate(fold_results)
        
        assert results.n_folds == 3
        assert abs(results.mean_metrics['accuracy'] - 0.85) < 0.01
        assert results.std_metrics['accuracy'] > 0
    
    def test_cvresults_to_dict(self):
        """test CVResults serialization."""
        from src.models.cross_validation import CVResults
        
        results = CVResults(
            fold_metrics=[{'accuracy': 0.85}],
            mean_metrics={'accuracy': 0.85},
            std_metrics={'accuracy': 0.05},
            n_folds=5
        )
        
        d = results.to_dict()
        
        assert 'fold_metrics' in d
        assert 'mean_metrics' in d
        assert 'std_metrics' in d
        assert 'n_folds' in d
    
    def test_cvresults_save_load(self):
        """test CVResults save and load."""
        from src.models.cross_validation import CVResults
        
        results = CVResults(
            fold_metrics=[{'accuracy': 0.85, 'f1': 0.82}],
            mean_metrics={'accuracy': 0.85, 'f1': 0.82},
            std_metrics={'accuracy': 0.05, 'f1': 0.04},
            n_folds=5
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cv_results.json"
            results.save(path)
            
            loaded = CVResults.load(path)
            
            assert loaded.n_folds == results.n_folds
            assert loaded.mean_metrics == results.mean_metrics


class TestExperimentTracking:
    """tests for experiment tracking utilities."""
    
    def test_experiment_config_defaults(self):
        """test ExperimentConfig default values."""
        from src.utils.experiment import ExperimentConfig
        
        config = ExperimentConfig()
        
        assert config.model_name == "facebook/wav2vec2-base-960h"
        assert config.num_labels == 2
        assert config.freeze_feature_extractor == True
        assert config.num_epochs == 20
        assert config.random_seed == 42
    
    def test_experiment_config_custom(self):
        """test ExperimentConfig with custom values."""
        from src.utils.experiment import ExperimentConfig
        
        config = ExperimentConfig(
            experiment_name="test_experiment",
            num_epochs=10,
            learning_rate=5e-5
        )
        
        assert config.experiment_name == "test_experiment"
        assert config.num_epochs == 10
        assert config.learning_rate == 5e-5
    
    def test_experiment_config_save_load(self):
        """test ExperimentConfig serialization."""
        from src.utils.experiment import ExperimentConfig
        
        config = ExperimentConfig(
            experiment_name="test_experiment",
            num_epochs=15
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(path)
            
            loaded = ExperimentConfig.load(path)
            
            assert loaded.experiment_name == config.experiment_name
            assert loaded.num_epochs == config.num_epochs
    
    def test_metrics_logger_log(self):
        """test MetricsLogger logging."""
        from src.utils.experiment import MetricsLogger
        
        logger = MetricsLogger()
        
        logger.log({'loss': 0.5, 'accuracy': 0.8}, epoch=1, phase='train')
        logger.log({'loss': 0.4, 'accuracy': 0.85}, epoch=2, phase='train')
        
        assert len(logger.history['train']) == 2
        assert logger.history['train'][0]['loss'] == 0.5
        assert logger.history['train'][1]['accuracy'] == 0.85
    
    def test_metrics_logger_get_best(self):
        """test MetricsLogger get_best."""
        from src.utils.experiment import MetricsLogger
        
        logger = MetricsLogger()
        
        logger.log({'accuracy': 0.8}, epoch=1, phase='val')
        logger.log({'accuracy': 0.9}, epoch=2, phase='val')
        logger.log({'accuracy': 0.85}, epoch=3, phase='val')
        
        best = logger.get_best('accuracy', phase='val', mode='max')
        
        assert best['accuracy'] == 0.9
        assert best['epoch'] == 2
    
    def test_metrics_logger_summary(self):
        """test MetricsLogger summary generation."""
        from src.utils.experiment import MetricsLogger
        
        logger = MetricsLogger()
        
        for i, acc in enumerate([0.7, 0.8, 0.85, 0.9]):
            logger.log({'accuracy': acc}, epoch=i+1, phase='train')
        
        summary = logger.summary()
        
        assert 'train' in summary
        assert 'accuracy' in summary['train']
        assert summary['train']['accuracy']['final'] == 0.9
        assert summary['train']['accuracy']['best'] == 0.9
    
    def test_experiment_tracker_lifecycle(self):
        """test ExperimentTracker full lifecycle."""
        from src.utils.experiment import ExperimentTracker, ExperimentConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(experiment_name="test")
            tracker = ExperimentTracker(
                experiment_name="test",
                output_dir=tmpdir,
                config=config
            )
            
            tracker.start()
            
            for epoch in range(3):
                tracker.log_metrics({'loss': 0.5 - epoch*0.1}, epoch=epoch+1, phase='train')
                tracker.log_metrics({'accuracy': 0.7 + epoch*0.1}, epoch=epoch+1, phase='val')
            
            tracker.finish(status='completed')
            
            # verify files created
            assert (tracker.output_dir / "config.json").exists()
            assert (tracker.output_dir / "metadata.json").exists()
            assert (tracker.output_dir / "metrics.json").exists()
            assert (tracker.output_dir / "summary.json").exists()
    
    def test_create_experiment_id(self):
        """test unique experiment id creation."""
        from src.utils.experiment import ExperimentConfig, create_experiment_id
        
        config1 = ExperimentConfig(experiment_name="exp1", num_epochs=10)
        config2 = ExperimentConfig(experiment_name="exp2", num_epochs=10)
        config3 = ExperimentConfig(experiment_name="exp1", num_epochs=10)
        
        id1 = create_experiment_id(config1)
        id2 = create_experiment_id(config2)
        id3 = create_experiment_id(config3)
        
        # different configs should have different ids
        assert id1 != id2
        
        # same config should have same id
        assert id1 == id3


class TestIntegration:
    """integration tests for classifier and cross-validation."""
    
    def test_training_args_are_valid(self):
        """test that training args can be created without error."""
        from src.models.classifier import create_training_args
        
        with tempfile.TemporaryDirectory() as tmpdir:
            args = create_training_args(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=2
            )
            
            # verify key training arguments
            assert args.do_train == True
            assert args.do_eval == True
            assert args.evaluation_strategy == "epoch"
            assert args.save_strategy == "epoch"
            assert args.load_best_model_at_end == True
