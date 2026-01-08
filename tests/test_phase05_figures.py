import pytest
from pathlib import Path


def test_phase05_vector_figures_exist():
    out = Path('results/probing')
    expected = [
        'fig_p5_01_layerwise_probing_regen.pdf',
        'fig_p5_02_clinical_feature_heatmap_regen.pdf',
        'fig_p5_03_selectivity_regen.pdf',
    ]
    for f in expected:
        p = out / f
        assert p.exists(), f"Missing figure: {p}"
        assert p.stat().st_size > 1000, f"Figure {p} seems too small"
