"""
Tests for publication figure generation module.

Tests the FigureGenerator class and helper functions.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.figures import (
    set_publication_style,
    FigureGenerator,
    PALETTES,
    create_poster_figure
)


class TestPublicationStyle:
    """Test publication style settings."""
    
    def test_set_publication_style_default(self):
        """Test default publication style."""
        set_publication_style()
        
        # check that style was applied
        assert plt.rcParams['figure.dpi'] >= 100
        assert plt.rcParams['savefig.dpi'] >= 300
    
    def test_set_publication_style_poster(self):
        """Test poster style with larger fonts."""
        set_publication_style(context='poster')
        
        # fonts should be larger for poster
        assert plt.rcParams['font.size'] >= 12
    
    def test_set_publication_style_paper(self):
        """Test paper style."""
        set_publication_style(context='paper')
        
        # should not raise
        assert True


class TestPalettes:
    """Test color palettes."""
    
    def test_palettes_exist(self):
        """Test that required palettes exist."""
        assert 'categorical' in PALETTES
        assert 'sequential' in PALETTES
        assert 'diverging' in PALETTES
    
    def test_palettes_have_colors(self):
        """Test that palettes have enough colors."""
        assert len(PALETTES['categorical']) >= 5
        assert len(PALETTES['sequential']) >= 5
        assert len(PALETTES['diverging']) >= 5
    
    def test_palettes_are_valid_colors(self):
        """Test that palette values are valid color strings."""
        for name, colors in PALETTES.items():
            for color in colors:
                # should be able to convert to RGBA
                rgba = matplotlib.colors.to_rgba(color)
                assert len(rgba) == 4


class TestFigureGenerator:
    """Test FigureGenerator class."""
    
    @pytest.fixture
    def fig_gen(self, tmp_path):
        """Create figure generator with temp output directory."""
        return FigureGenerator(output_dir=str(tmp_path))
    
    @pytest.fixture
    def sample_probing_results(self):
        """Sample probing results for testing."""
        return {
            i: {'mean': 0.6 + 0.02*i, 'std': 0.03}
            for i in range(1, 13)
        }
    
    @pytest.fixture
    def sample_patching_results(self):
        """Sample patching results for testing."""
        return {
            i: {'mean_recovery': 0.05 + 0.03*i, 'std_recovery': 0.02}
            for i in range(1, 13)
        }
    
    @pytest.fixture
    def sample_clinical_results(self):
        """Sample clinical probing results."""
        return {
            'jitter': {i: {'mean': 0.3 + 0.01*i, 'std': 0.02} for i in range(1, 13)},
            'shimmer': {i: {'mean': 0.28 + 0.01*i, 'std': 0.02} for i in range(1, 13)},
            'hnr': {i: {'mean': 0.25 + 0.015*i, 'std': 0.02} for i in range(1, 13)},
            'f0_mean': {i: {'mean': 0.2 + 0.02*i, 'std': 0.03} for i in range(1, 13)},
        }
    
    @pytest.fixture
    def sample_hypothesis_results(self):
        """Sample hypothesis test results."""
        return {
            'hypothesis_1': {
                'supported': True,
                'phonatory_early_fraction': 0.75,
                'prosodic_middle_fraction': 0.80
            },
            'hypothesis_2': {
                'supported': True,
                'causal_layers': [5, 6, 7, 8],
                'probing_patching_correlation': {
                    'spearman_r': 0.85,
                    'spearman_p': 0.0003
                }
            },
            'hypothesis_3': {
                'supported': True,
                'cross_dataset_mean': 0.76,
                'generalization_gap': 0.10
            }
        }
    
    def test_figure_1_overview(self, fig_gen, sample_probing_results, sample_patching_results):
        """Test overview figure generation."""
        fig = fig_gen.figure_1_overview(
            sample_probing_results,
            sample_patching_results,
            model_accuracy=0.85
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # check figure was saved
        expected_path = Path(fig_gen.output_dir) / 'figure_1_overview.pdf'
        assert expected_path.exists()
        
        plt.close(fig)
    
    def test_figure_2_clinical_encoding(self, fig_gen, sample_clinical_results):
        """Test clinical encoding heatmap."""
        fig = fig_gen.figure_2_clinical_encoding(sample_clinical_results)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        expected_path = Path(fig_gen.output_dir) / 'figure_2_clinical_encoding.pdf'
        assert expected_path.exists()
        
        plt.close(fig)
    
    def test_figure_3_hypothesis_summary(self, fig_gen, sample_hypothesis_results):
        """Test hypothesis summary figure."""
        fig = fig_gen.figure_3_hypothesis_summary(sample_hypothesis_results)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        expected_path = Path(fig_gen.output_dir) / 'figure_3_hypothesis_summary.pdf'
        assert expected_path.exists()
        
        plt.close(fig)
    
    def test_figure_4_cross_dataset(self, fig_gen):
        """Test cross-dataset matrix figure."""
        cross_results = {
            'Italian': {'Italian': 0.90, 'mDVR-KCL': 0.75, 'UCI': 0.70},
            'mDVR-KCL': {'Italian': 0.72, 'mDVR-KCL': 0.88, 'UCI': 0.68},
            'UCI': {'Italian': 0.65, 'mDVR-KCL': 0.67, 'UCI': 0.85}
        }
        
        fig = fig_gen.figure_4_cross_dataset(cross_results)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        expected_path = Path(fig_gen.output_dir) / 'figure_4_cross_dataset.pdf'
        assert expected_path.exists()
        
        plt.close(fig)
    
    def test_figure_5_attention_analysis(self, fig_gen):
        """Test attention analysis figure."""
        # create sample attention data
        attention_data = {
            'head_importance': np.random.rand(12, 12),  # 12 layers x 12 heads
            'layer_labels': [f'L{i}' for i in range(1, 13)],
            'head_labels': [f'H{i}' for i in range(1, 13)]
        }
        
        fig = fig_gen.figure_5_attention_analysis(attention_data)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        expected_path = Path(fig_gen.output_dir) / 'figure_5_attention_analysis.pdf'
        assert expected_path.exists()
        
        plt.close(fig)
    
    def test_generate_all_figures(self, fig_gen, sample_probing_results, 
                                   sample_patching_results, sample_clinical_results,
                                   sample_hypothesis_results):
        """Test batch figure generation."""
        cross_results = {
            'A': {'A': 0.90, 'B': 0.75},
            'B': {'A': 0.72, 'B': 0.88}
        }
        
        saved = fig_gen.generate_all_figures(
            probing_results=sample_probing_results,
            patching_results=sample_patching_results,
            clinical_results=sample_clinical_results,
            hypothesis_results=sample_hypothesis_results,
            cross_dataset_results=cross_results
        )
        
        assert len(saved) >= 4  # at least 4 figures should be generated
        
        for path in saved:
            assert Path(path).exists()
        
        plt.close('all')


class TestCreatePosterFigure:
    """Test poster figure creation."""
    
    def test_create_poster_figure(self, tmp_path):
        """Test creating a poster-sized figure."""
        fig = create_poster_figure(width=48, height=36)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # check size (in inches)
        size = fig.get_size_inches()
        assert size[0] == 48
        assert size[1] == 36
        
        plt.close(fig)


class TestFigureQuality:
    """Test figure quality standards."""
    
    @pytest.fixture
    def fig_gen(self, tmp_path):
        """Create figure generator."""
        return FigureGenerator(output_dir=str(tmp_path), dpi=300)
    
    def test_high_dpi(self, fig_gen):
        """Test that figures are saved at high DPI."""
        assert fig_gen.dpi >= 300
    
    def test_vector_format(self, fig_gen):
        """Test that PDF (vector) format is used."""
        probing = {i: {'mean': 0.5 + 0.02*i, 'std': 0.02} for i in range(1, 5)}
        patching = {i: {'mean_recovery': 0.1 + 0.05*i, 'std_recovery': 0.02} for i in range(1, 5)}
        
        fig = fig_gen.figure_1_overview(probing, patching)
        
        pdf_path = Path(fig_gen.output_dir) / 'figure_1_overview.pdf'
        assert pdf_path.exists()
        
        # PDF should be larger than a few KB (actual content, not empty)
        assert pdf_path.stat().st_size > 1000
        
        plt.close(fig)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def fig_gen(self, tmp_path):
        """Create figure generator."""
        return FigureGenerator(output_dir=str(tmp_path))
    
    def test_empty_probing_results(self, fig_gen):
        """Test with empty probing results."""
        fig = fig_gen.figure_1_overview({}, {})
        
        # should handle gracefully
        assert fig is not None
        plt.close(fig)
    
    def test_single_layer_results(self, fig_gen):
        """Test with single layer."""
        probing = {5: {'mean': 0.85, 'std': 0.03}}
        patching = {5: {'mean_recovery': 0.45, 'std_recovery': 0.05}}
        
        fig = fig_gen.figure_1_overview(probing, patching)
        
        assert fig is not None
        plt.close(fig)
    
    def test_missing_std(self, fig_gen):
        """Test with missing std values."""
        probing = {i: {'mean': 0.5 + 0.02*i} for i in range(1, 5)}  # no std
        patching = {i: {'mean_recovery': 0.1 + 0.05*i} for i in range(1, 5)}  # no std
        
        fig = fig_gen.figure_1_overview(probing, patching)
        
        assert fig is not None
        plt.close(fig)
    
    def test_clinical_single_feature(self, fig_gen):
        """Test clinical heatmap with single feature."""
        clinical = {
            'jitter': {i: {'mean': 0.3 + 0.01*i, 'std': 0.02} for i in range(1, 5)}
        }
        
        fig = fig_gen.figure_2_clinical_encoding(clinical)
        
        assert fig is not None
        plt.close(fig)
    
    def test_hypothesis_partial_results(self, fig_gen):
        """Test hypothesis figure with partial results."""
        hypothesis = {
            'hypothesis_1': {'supported': True},
            'hypothesis_2': {'supported': False},
            # hypothesis_3 missing
        }
        
        fig = fig_gen.figure_3_hypothesis_summary(hypothesis)
        
        assert fig is not None
        plt.close(fig)


class TestColorblindFriendly:
    """Test colorblind-friendly design."""
    
    def test_categorical_palette_distinct(self):
        """Test that categorical colors are perceptually distinct."""
        colors = PALETTES['categorical']
        
        # convert to RGB
        rgb_colors = [matplotlib.colors.to_rgb(c) for c in colors]
        
        # check that colors are sufficiently different
        # using simple Euclidean distance in RGB space
        for i, c1 in enumerate(rgb_colors):
            for j, c2 in enumerate(rgb_colors):
                if i < j:
                    dist = np.sqrt(sum((a - b)**2 for a, b in zip(c1, c2)))
                    # colors should be at least somewhat different
                    assert dist > 0.1, f"Colors {i} and {j} are too similar"
    
    def test_sequential_palette_monotonic(self):
        """Test that sequential palette has monotonic luminance."""
        colors = PALETTES['sequential']
        
        # convert to RGB and compute luminance
        luminances = []
        for c in colors:
            rgb = matplotlib.colors.to_rgb(c)
            # approximate luminance
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            luminances.append(lum)
        
        # should be roughly monotonic (allowing small variations)
        diffs = np.diff(luminances)
        # most differences should be positive or small negative
        assert sum(d > -0.1 for d in diffs) >= len(diffs) * 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
