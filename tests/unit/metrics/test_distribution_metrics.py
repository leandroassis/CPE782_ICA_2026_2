"""Testes unitarios para ica.metrics.distribution_metrics (skill ica-evaluation, Secao 4)."""

import numpy as np

from ica.metrics.distribution_metrics import (
    DistributionFitMetric,
    distribution_fit_quality,
    distribution_metrics_battery,
)


def test_distribution_fit_quality_is_good_for_samples_from_the_same_distribution(rng):
    """KS deve ser baixo (alto p-valor) para duas amostras da mesma distribuicao."""
    a = rng.laplace(size=5000)
    b = rng.laplace(size=5000)
    result = distribution_fit_quality(a, b)
    assert result.ks_statistic < 0.05
    assert result.ks_pvalue > 0.05


def test_distribution_fit_quality_is_poor_for_different_distributions(rng):
    """KS deve ser alto (baixo p-valor) para amostras de distribuicoes bem diferentes."""
    a = rng.laplace(size=5000)
    b = rng.uniform(-1, 1, size=5000)
    result = distribution_fit_quality(a, b)
    assert result.ks_statistic > 0.1
    assert result.ks_pvalue < 0.01


def test_distribution_metrics_battery_matches_and_orders_by_true_index(rng, make_sources):
    """A bateria deve casar (hungaro) e devolver na ordem das fontes verdadeiras."""
    S = make_sources(["laplace", "uniform"], 5000, rng)
    permuted = np.stack([S[1], S[0]])
    results = distribution_metrics_battery(S, permuted)
    assert len(results) == 2
    for result in results:
        assert result.ks_statistic < 0.1


def test_distribution_metrics_battery_corrects_sign_ambiguity_before_scoring(rng):
    """Regressao: uma fonte assimetrica com sinal invertido deve pontuar bem, nao mal.

    Para uma familia assimetrica (aqui, exponencial deslocada para media
    zero), a versao espelhada (``-x``) tem KS alto contra a original --
    exatamente o que a ambiguidade de sinal da ICA pode produzir. A bateria
    deve realinhar o sinal (via ``MatchResult.signs``) antes do teste.
    """
    true_source = rng.exponential(size=5000)
    true_source -= true_source.mean()
    sign_flipped_estimate = -true_source.copy()

    results = distribution_metrics_battery(
        np.vstack([true_source, rng.laplace(size=5000)]),
        np.vstack([sign_flipped_estimate, rng.laplace(size=5000)]),
    )

    assert results[0].ks_statistic < 0.05


class _FakeModel:
    def __init__(self, sources_true, sources, domain):
        self.sources_true_ = sources_true
        self.sources_ = sources
        self.domain_ = domain


def test_distribution_fit_metric_returns_none_without_ground_truth():
    """DistributionFitMetric.compute deve devolver None sem sources_true_."""
    model = _FakeModel(None, np.zeros((2, 100)), "distribution")
    assert DistributionFitMetric().compute(model) is None


def test_distribution_fit_metric_returns_none_outside_distribution_domain():
    """DistributionFitMetric.compute deve devolver None fora do dominio distribuicao."""
    model = _FakeModel(np.zeros((2, 100)), np.zeros((2, 100)), "image")
    assert DistributionFitMetric().compute(model) is None
