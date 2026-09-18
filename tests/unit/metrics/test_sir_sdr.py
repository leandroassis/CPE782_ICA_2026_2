"""Testes unitarios para ica.metrics.sir_sdr (skill ica-evaluation, Secao 4)."""

import numpy as np

from ica.metrics.sir_sdr import SIRSDRMetric, si_sdr, sir_sdr, sir_sdr_battery


def test_si_sdr_is_infinite_for_identical_signals(rng):
    """SI-SDR deve ser infinito quando o estimado e o alvo sao identicos."""
    target = rng.normal(size=2000)
    assert si_sdr(target, target) == float("inf")


def test_si_sdr_is_invariant_to_scale_of_the_estimate(rng):
    """SI-SDR deve ser o mesmo se o estimado for reescalado por uma constante."""
    target = rng.normal(size=2000)
    noise = rng.normal(size=2000) * 0.1
    estimate = target + noise
    assert abs(si_sdr(estimate, target) - si_sdr(3.0 * estimate, target)) < 1e-6


def test_si_sdr_decreases_with_more_noise(rng):
    """Mais ruido aditivo deve reduzir o SI-SDR."""
    target = rng.normal(size=5000)
    low_noise = target + rng.normal(size=5000) * 0.01
    high_noise = target + rng.normal(size=5000) * 1.0
    assert si_sdr(low_noise, target) > si_sdr(high_noise, target)


def test_sir_sdr_recovers_high_values_for_a_well_matched_clean_source(rng):
    """Para uma fonte bem casada e limpa, SIR/SDR/SI-SDR devem ser altos."""
    s1 = rng.laplace(size=5000)
    s2 = rng.uniform(-1, 1, size=5000)
    all_true = np.vstack([s1, s2])
    result = sir_sdr(estimate=s1, target_index=0, all_true_sources=all_true)
    assert result.sir_db > 40
    assert result.sdr_db > 40
    assert result.si_sdr_db > 40


def test_sir_sdr_battery_matches_sources_and_orders_by_true_index(rng, make_sources):
    """sir_sdr_battery deve casar e devolver os resultados na ordem das fontes verdadeiras."""
    S = make_sources(["laplace", "uniform"], 5000, rng)
    permuted = np.stack([S[1], S[0]])  # ordem trocada
    results = sir_sdr_battery(S, permuted)
    assert len(results) == 2
    assert results[0].si_sdr_db > 40
    assert results[1].si_sdr_db > 40


class _FakeModel:
    def __init__(self, sources_true, sources):
        self.sources_true_ = sources_true
        self.sources_ = sources


def test_sir_sdr_metric_returns_none_without_ground_truth():
    """SIRSDRMetric.compute deve devolver None sem sources_true_."""
    model = _FakeModel(None, np.zeros((2, 10)))
    assert SIRSDRMetric().compute(model) is None
