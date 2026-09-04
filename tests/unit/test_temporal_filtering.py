"""Testes unitarios para TemporalFiltering (livro-texto, Secao 13.1, paginas 263-267)."""

import numpy as np
import pytest

from ica.preprocessing.temporal_filtering import TemporalFiltering


def test_invalid_kind_raises():
    """Kind fora de {'low', 'high', 'band'} deve levantar ValueError."""
    with pytest.raises(ValueError):
        TemporalFiltering(kind="invalid")


def test_invalid_window_raises():
    """Window menor que 1 deve levantar ValueError."""
    with pytest.raises(ValueError):
        TemporalFiltering(kind="low", window=0)


def test_estimation_only_is_true():
    """TemporalFiltering deve ser marcado estimation_only (livro-texto, Secao 13.1, p.264)."""
    assert TemporalFiltering().estimation_only is True


def test_fit_is_a_no_op_and_returns_self(rng):
    """Fit nao ajusta nenhum parametro (filtro deterministico); deve so retornar self."""
    step = TemporalFiltering()
    X = rng.normal(size=(2, 50))
    assert step.fit(X) is step


def test_transform_preserves_shape(rng):
    """Transform deve manter a shape de entrada para os tres tipos de filtro."""
    X = rng.normal(size=(3, 100))
    for kind in ("low", "high", "band"):
        Y = TemporalFiltering(kind=kind, window=5).transform(X)
        assert Y.shape == X.shape


def test_low_pass_reduces_variance_of_noisy_signal(rng):
    """Passa-baixa (media movel) deve reduzir a variancia de ruido branco (Secao 13.1.2)."""
    X = rng.normal(size=(2, 5000))
    filtered = TemporalFiltering(kind="low", window=9).transform(X)
    assert filtered.var(axis=1).mean() < X.var(axis=1).mean()


def test_low_pass_matches_moving_average_convolution(rng):
    """O passa-baixa deve ser exatamente a media movel de `window` amostras por linha."""
    X = rng.normal(size=(2, 40))
    window = 4
    filtered = TemporalFiltering(kind="low", window=window).transform(X)
    kernel = np.ones(window) / window
    expected = np.array([np.convolve(row, kernel, mode="same") for row in X])
    assert np.allclose(filtered, expected)


def test_high_pass_matches_first_order_differencing(rng):
    """O passa-alta deve ser exatamente a diferenciacao de 1a ordem (Eq. 13.4)."""
    X = rng.normal(size=(2, 40))
    filtered = TemporalFiltering(kind="high").transform(X)
    expected = np.diff(X, axis=1, prepend=X[:, :1])
    assert np.array_equal(filtered, expected)


def test_high_pass_removes_constant_trend(rng):
    """Passa-alta deve remover uma tendencia constante, deixando so a flutuacao (Secao 13.1.3)."""
    trend = 100.0
    fluctuation = rng.normal(scale=0.1, size=(1, 200))
    X = trend + fluctuation
    filtered = TemporalFiltering(kind="high").transform(X)
    assert abs(filtered.mean()) < 1.0


def test_band_applies_low_then_high_pass(rng):
    """kind='band' deve ser equivalente a aplicar passa-baixa seguido de passa-alta."""
    X = rng.normal(size=(2, 60))
    window = 3
    band = TemporalFiltering(kind="band", window=window).transform(X)

    low = TemporalFiltering(kind="low", window=window).transform(X)
    expected = TemporalFiltering(kind="high").transform(low)

    assert np.array_equal(band, expected)


def test_inverse_transform_raises_not_implemented(rng):
    """inverse_transform deve levantar NotImplementedError: o filtro nao e reversivel."""
    step = TemporalFiltering()
    Y = rng.normal(size=(2, 50))
    with pytest.raises(NotImplementedError):
        step.inverse_transform(Y)
