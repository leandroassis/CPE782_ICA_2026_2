"""Testes unitarios para ica.metrics.family_identification (skill ica-evaluation, Secao 3)."""

import numpy as np
import pytest
from scipy import stats

from ica.metrics.family_identification import (
    FamilyIdentificationMetric,
    family_pdf,
    identify_family,
)


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


def test_family_pdf_matches_analytic_gaussian_density_on_original_scale(rng):
    """Para uma fonte gaussiana com media/escala arbitrarias, family_pdf deve bater com norm.pdf."""
    y = rng.normal(loc=5.0, scale=2.0, size=50000)
    result = identify_family(y)
    assert result.best.name == "gaussiana"

    x = np.linspace(y.min(), y.max(), 50)
    density = family_pdf(result, x)
    expected = stats.norm.pdf(x, loc=y.mean(), scale=y.std())
    assert np.allclose(density, expected, atol=0.01)


def test_family_pdf_integrates_to_approximately_one():
    """A densidade reconstruida em escala original deve integrar (numericamente) a ~1."""
    rng = np.random.default_rng(0)
    y = rng.laplace(loc=-3.0, scale=1.5, size=20000)
    result = identify_family(y)

    x = np.linspace(y.min(), y.max(), 5000)
    density = family_pdf(result, x)
    integral = np.trapezoid(density, x)
    assert integral == pytest.approx(1.0, abs=0.02)


def test_family_pdf_defaults_to_best_but_accepts_explicit_fit(rng):
    """family_pdf deve aceitar uma candidata explicita (ex.: runner_up) em vez de best."""
    y = rng.laplace(size=20000)
    result = identify_family(y)
    x = np.array([0.0, 1.0, -1.0])

    default_density = family_pdf(result, x)
    explicit_density = family_pdf(result, x, fit=result.best)
    runner_up_density = family_pdf(result, x, fit=result.runner_up)

    assert np.array_equal(default_density, explicit_density)
    assert not np.array_equal(default_density, runner_up_density)
