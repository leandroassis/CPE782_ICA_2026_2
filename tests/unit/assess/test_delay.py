"""Testes unitarios para ica.assess.delay (skill bss-assessment, Secao 4)."""

import numpy as np

from ica.assess.delay import cross_correlation_lag, delay_battery, gcc_phat


def _make_shifted_pair(rng, n_samples=4000, true_shift=37):
    base = rng.normal(size=n_samples + true_shift)
    xi = base[: n_samples]
    xj = base[true_shift : true_shift + n_samples]
    return xi, xj


def test_cross_correlation_lag_detects_zero_lag_for_instantaneous_mixture(rng):
    """Numa mistura instantanea (sem atraso), o pico deve ficar em lag 0."""
    s1 = rng.laplace(size=5000)
    s2 = rng.uniform(-1, 1, size=5000)
    A = np.array([[1.0, 0.5], [0.3, 1.0]])
    X = A @ np.vstack([s1, s2])
    lag = cross_correlation_lag(X[0], X[1])
    assert lag == 0


def test_gcc_phat_recovers_known_shift(rng):
    """GCC-PHAT deve recuperar o deslocamento verdadeiro entre dois canais."""
    xi, xj = _make_shifted_pair(rng, true_shift=37)
    estimate = gcc_phat(xi, xj)
    assert estimate.lag_samples == -37 or estimate.lag_samples == 37


def test_gcc_phat_converts_to_seconds_when_sample_rate_given(rng):
    """Com sample_rate informado, lag_seconds deve ser lag_samples / sample_rate."""
    xi, xj = _make_shifted_pair(rng, true_shift=10)
    estimate = gcc_phat(xi, xj, sample_rate=1000.0)
    assert estimate.lag_seconds == estimate.lag_samples / 1000.0


def test_delay_battery_has_zero_diagonal(rng):
    """A matriz de lags deve ter diagonal zero (um canal contra ele mesmo)."""
    X = rng.normal(size=(3, 2000))
    lags = delay_battery(X)
    assert np.all(np.diag(lags) == 0)
    assert lags.shape == (3, 3)
