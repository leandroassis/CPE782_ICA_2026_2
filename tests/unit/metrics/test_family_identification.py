"""Testes unitarios para ica.metrics.family_identification (skill ica-evaluation, Secao 3)."""

import numpy as np

from ica.metrics.family_identification import FamilyIdentificationMetric, identify_family


def test_identifies_gaussian_source(rng):
    """Uma fonte genuinamente gaussiana deve ser identificada como tal (ou proxima, por KS)."""
    y = rng.normal(size=20000)
    result = identify_family(y)
    assert result.best.name == "gaussiana"


def test_identifies_uniform_source(rng):
    """Uma fonte uniforme deve ser identificada como tal."""
    y = rng.uniform(-1, 1, size=20000)
    result = identify_family(y)
    assert result.best.name == "uniforme"


def test_identifies_laplace_source(rng):
    """Uma fonte Laplaciana deve ser identificada como tal."""
    y = rng.laplace(size=20000)
    result = identify_family(y)
    assert result.best.name == "laplaciana"


def test_identifies_exponential_source(rng):
    """Uma fonte exponencial (assimetrica) deve ser identificada corretamente."""
    y = rng.exponential(size=20000)
    result = identify_family(y)
    assert result.best.name == "exponencial"


def test_identifies_exponential_source_even_with_flipped_sign(rng):
    """A identificacao deve ser robusta ao sinal indeterminado da ICA."""
    y = -rng.exponential(size=20000)
    result = identify_family(y)
    assert result.best.name == "exponencial"


def test_ranking_has_all_six_candidates_sorted_by_ks(rng):
    """O ranking deve ter as 6 candidatas, ordenadas por KS crescente."""
    y = rng.laplace(size=5000)
    result = identify_family(y)
    assert len(result.ranking) == 6
    ks_values = [fit.ks_statistic for fit in result.ranking]
    assert ks_values == sorted(ks_values)
    assert result.best is result.ranking[0]
    assert result.runner_up is result.ranking[1]


class _FakeModel:
    def __init__(self, sources):
        self.sources_ = sources


def test_family_identification_metric_returns_one_result_per_component(rng):
    """FamilyIdentificationMetric.compute deve devolver um resultado por componente."""
    sources = np.vstack([rng.normal(size=5000), rng.laplace(size=5000)])
    model = _FakeModel(sources)
    results = FamilyIdentificationMetric().compute(model)
    assert len(results) == 2
