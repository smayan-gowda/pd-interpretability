"""
Tests for results aggregation module.

Tests the ResultsAggregator and HypothesisTester classes.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

from src.utils.results import (
    ExperimentMetadata,
    ProbingResult,
    PatchingResult,
    ClinicalProbingResult,
    ResultsAggregator,
    HypothesisTester,
    aggregate_multiple_experiments
)


class TestDataclasses:
    """Test result dataclasses."""
    
    def test_experiment_metadata(self):
        """Test ExperimentMetadata creation."""
        meta = ExperimentMetadata(
            experiment_name='test',
            model_name='wav2vec2',
            num_layers=12
        )
        
        assert meta.experiment_name == 'test'
        assert meta.model_name == 'wav2vec2'
        assert meta.num_layers == 12
        assert meta.timestamp is not None
    
    def test_probing_result(self):
        """Test ProbingResult creation and validation."""
        result = ProbingResult(
            layer=5,
            mean_score=0.85,
            std_score=0.03,
            n_folds=5
        )
        
        assert result.layer == 5
        assert result.mean_score == 0.85
        assert result.std_score == 0.03
        assert result.n_folds == 5
    
    def test_patching_result(self):
        """Test PatchingResult creation."""
        result = PatchingResult(
            layer=7,
            mean_recovery=0.45,
            std_recovery=0.08,
            n_pairs=100
        )
        
        assert result.layer == 7
        assert result.mean_recovery == 0.45
        assert result.n_pairs == 100
    
    def test_clinical_probing_result(self):
        """Test ClinicalProbingResult creation."""
        result = ClinicalProbingResult(
            feature_name='jitter',
            layer=3,
            r2_score=0.42,
            std_score=0.05
        )
        
        assert result.feature_name == 'jitter'
        assert result.layer == 3
        assert result.r2_score == 0.42


class TestResultsAggregator:
    """Test ResultsAggregator class."""
    
    @pytest.fixture
    def aggregator(self, tmp_path):
        """Create aggregator with temp directory."""
        return ResultsAggregator(
            experiment_name='test_experiment',
            output_dir=str(tmp_path)
        )
    
    @pytest.fixture
    def sample_probing_results(self):
        """Sample probing results."""
        return {
            1: {'mean': 0.65, 'std': 0.05, 'fold_scores': [0.63, 0.67, 0.64, 0.66, 0.65]},
            2: {'mean': 0.72, 'std': 0.04, 'fold_scores': [0.70, 0.74, 0.71, 0.73, 0.72]},
            3: {'mean': 0.78, 'std': 0.03, 'fold_scores': [0.76, 0.80, 0.77, 0.79, 0.78]},
            4: {'mean': 0.82, 'std': 0.03, 'fold_scores': [0.80, 0.84, 0.81, 0.83, 0.82]},
            5: {'mean': 0.85, 'std': 0.02, 'fold_scores': [0.84, 0.86, 0.84, 0.86, 0.85]},
            6: {'mean': 0.83, 'std': 0.03, 'fold_scores': [0.81, 0.85, 0.82, 0.84, 0.83]},
        }
    
    @pytest.fixture
    def sample_patching_results(self):
        """Sample patching results."""
        return {
            1: {'mean_recovery': 0.10, 'std_recovery': 0.02, 'n_pairs': 50},
            2: {'mean_recovery': 0.18, 'std_recovery': 0.03, 'n_pairs': 50},
            3: {'mean_recovery': 0.25, 'std_recovery': 0.04, 'n_pairs': 50},
            4: {'mean_recovery': 0.38, 'std_recovery': 0.05, 'n_pairs': 50},
            5: {'mean_recovery': 0.45, 'std_recovery': 0.05, 'n_pairs': 50},
            6: {'mean_recovery': 0.35, 'std_recovery': 0.04, 'n_pairs': 50},
        }
    
    def test_add_probing_results(self, aggregator, sample_probing_results):
        """Test adding probing results."""
        aggregator.add_probing_results(sample_probing_results)
        
        assert len(aggregator.probing_results) == 6
        assert 5 in aggregator.probing_results
        assert aggregator.probing_results[5].mean_score == 0.85
    
    def test_add_patching_results(self, aggregator, sample_patching_results):
        """Test adding patching results."""
        aggregator.add_patching_results(sample_patching_results)
        
        assert len(aggregator.patching_results) == 6
        assert 5 in aggregator.patching_results
        assert aggregator.patching_results[5].mean_recovery == 0.45
    
    def test_get_best_probing_layer(self, aggregator, sample_probing_results):
        """Test finding best probing layer."""
        aggregator.add_probing_results(sample_probing_results)
        
        best_layer, best_score = aggregator.get_best_probing_layer()
        
        assert best_layer == 5
        assert best_score == 0.85
    
    def test_get_best_patching_layer(self, aggregator, sample_patching_results):
        """Test finding best patching layer."""
        aggregator.add_patching_results(sample_patching_results)
        
        best_layer, best_recovery = aggregator.get_best_patching_layer()
        
        assert best_layer == 5
        assert best_recovery == 0.45
    
    def test_get_important_layers(self, aggregator, sample_probing_results, sample_patching_results):
        """Test finding important layers."""
        aggregator.add_probing_results(sample_probing_results)
        aggregator.add_patching_results(sample_patching_results)
        
        important = aggregator.get_important_layers(probing_threshold=0.80, patching_threshold=0.30)
        
        assert 'probing' in important
        assert 'patching' in important
        assert 'both' in important
        assert 5 in important['both']  # layer 5 should be important for both
    
    def test_compute_probing_patching_correlation(self, aggregator, sample_probing_results, sample_patching_results):
        """Test computing correlation between probing and patching."""
        aggregator.add_probing_results(sample_probing_results)
        aggregator.add_patching_results(sample_patching_results)
        
        corr = aggregator.compute_probing_patching_correlation()
        
        assert corr is not None
        assert 'spearman_r' in corr
        assert 'spearman_p' in corr
        assert 'n_layers' in corr
        assert corr['n_layers'] == 6
        # should be positively correlated
        assert corr['spearman_r'] > 0
    
    def test_save_and_load(self, aggregator, sample_probing_results, tmp_path):
        """Test saving and loading results."""
        aggregator.add_probing_results(sample_probing_results)
        
        # save
        save_path = aggregator.save('test_results.json')
        assert Path(save_path).exists()
        
        # load and verify
        with open(save_path, 'r') as f:
            loaded = json.load(f)
        
        assert 'metadata' in loaded
        assert 'probing' in loaded
        assert len(loaded['probing']) == 6
    
    def test_generate_summary_report(self, aggregator, sample_probing_results, sample_patching_results):
        """Test summary report generation."""
        aggregator.add_probing_results(sample_probing_results)
        aggregator.add_patching_results(sample_patching_results)
        
        report = aggregator.generate_summary_report()
        
        assert 'EXPERIMENT SUMMARY' in report
        assert 'probing' in report.lower()
        assert 'patching' in report.lower()


class TestHypothesisTester:
    """Test HypothesisTester class."""
    
    @pytest.fixture
    def aggregator_with_data(self, tmp_path):
        """Create aggregator with sample data."""
        agg = ResultsAggregator(
            experiment_name='test',
            output_dir=str(tmp_path)
        )
        
        # probing results
        probing = {i: {'mean': 0.6 + 0.03*i, 'std': 0.02} for i in range(1, 13)}
        agg.add_probing_results(probing)
        
        # patching results
        patching = {i: {'mean_recovery': 0.05 + 0.04*i, 'std_recovery': 0.02} for i in range(1, 13)}
        agg.add_patching_results(patching)
        
        # clinical probing
        clinical = {
            'jitter': {i: {'r2': 0.5 if i <= 4 else 0.2, 'std': 0.05} for i in range(1, 13)},
            'shimmer': {i: {'r2': 0.45 if i <= 4 else 0.15, 'std': 0.05} for i in range(1, 13)},
            'hnr': {i: {'r2': 0.3 if 5 <= i <= 8 else 0.1, 'std': 0.05} for i in range(1, 13)},
            'f0_mean': {i: {'r2': 0.35 if 5 <= i <= 8 else 0.1, 'std': 0.05} for i in range(1, 13)},
        }
        agg.add_clinical_probing_results(clinical)
        
        return agg
    
    def test_hypothesis_1_clinical_encoding(self, aggregator_with_data):
        """Test hypothesis 1 (clinical feature encoding)."""
        tester = HypothesisTester(aggregator_with_data)
        
        h1 = tester.test_hypothesis_1()
        
        assert 'supported' in h1
        assert 'phonatory' in h1 or 'prosodic' in h1
    
    def test_hypothesis_2_causal_dependency(self, aggregator_with_data):
        """Test hypothesis 2 (causal dependency)."""
        tester = HypothesisTester(aggregator_with_data)
        
        h2 = tester.test_hypothesis_2()
        
        assert 'supported' in h2
        assert 'causal_layers' in h2
        assert 'probing_patching_correlation' in h2
    
    def test_hypothesis_3_generalization(self, aggregator_with_data):
        """Test hypothesis 3 (cross-dataset generalization)."""
        # add cross-dataset results
        cross = {
            'dataset_a': {'dataset_a': 0.90, 'dataset_b': 0.75},
            'dataset_b': {'dataset_a': 0.72, 'dataset_b': 0.88}
        }
        aggregator_with_data.add_cross_dataset_results(cross)
        
        tester = HypothesisTester(aggregator_with_data)
        
        h3 = tester.test_hypothesis_3()
        
        assert 'supported' in h3
        assert 'cross_dataset_mean' in h3 or 'no_data' in h3
    
    def test_run_all_hypothesis_tests(self, aggregator_with_data):
        """Test running all hypothesis tests."""
        tester = HypothesisTester(aggregator_with_data)
        
        results = tester.run_all_hypothesis_tests()
        
        assert 'hypothesis_1' in results
        assert 'hypothesis_2' in results
        assert 'hypothesis_3' in results
    
    def test_generate_hypothesis_report(self, aggregator_with_data):
        """Test hypothesis report generation."""
        tester = HypothesisTester(aggregator_with_data)
        tester.run_all_hypothesis_tests()
        
        report = tester.generate_hypothesis_report()
        
        assert 'HYPOTHESIS' in report
        assert 'H1' in report or 'hypothesis 1' in report.lower()


class TestAggregateMultipleExperiments:
    """Test multi-experiment aggregation."""
    
    def test_aggregate_two_experiments(self, tmp_path):
        """Test aggregating results from multiple experiments."""
        # create two aggregators
        agg1 = ResultsAggregator('exp1', str(tmp_path / 'exp1'))
        agg2 = ResultsAggregator('exp2', str(tmp_path / 'exp2'))
        
        (tmp_path / 'exp1').mkdir()
        (tmp_path / 'exp2').mkdir()
        
        # add data
        agg1.add_probing_results({1: {'mean': 0.80, 'std': 0.02}})
        agg2.add_probing_results({1: {'mean': 0.82, 'std': 0.03}})
        
        # save
        path1 = agg1.save('results.json')
        path2 = agg2.save('results.json')
        
        # aggregate
        combined = aggregate_multiple_experiments([path1, path2])
        
        assert 'experiments' in combined
        assert len(combined['experiments']) == 2


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_aggregator(self, tmp_path):
        """Test aggregator with no data."""
        agg = ResultsAggregator('empty', str(tmp_path))
        
        report = agg.generate_summary_report()
        assert report is not None
        
        # should handle missing data gracefully
        best = agg.get_best_probing_layer()
        assert best == (None, None)
    
    def test_partial_data(self, tmp_path):
        """Test with only some data types."""
        agg = ResultsAggregator('partial', str(tmp_path))
        
        # only probing, no patching
        agg.add_probing_results({1: {'mean': 0.80, 'std': 0.02}})
        
        # correlation should return None
        corr = agg.compute_probing_patching_correlation()
        assert corr is None
    
    def test_single_layer(self, tmp_path):
        """Test with single layer."""
        agg = ResultsAggregator('single', str(tmp_path))
        
        agg.add_probing_results({5: {'mean': 0.85, 'std': 0.03}})
        
        best_layer, best_score = agg.get_best_probing_layer()
        assert best_layer == 5
        assert best_score == 0.85


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
