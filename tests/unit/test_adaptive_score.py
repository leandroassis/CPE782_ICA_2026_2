"""Testes unitarios para AdaptiveScore (ICA_BACKGROUND.md, Secao 3.4)."""

import numpy as np

from ica.nonlinearities.adaptive import AdaptiveScore
from ica.nonlinearities.base import NonlinearityTemplate


class _SpyNonlinearity(NonlinearityTemplate):
    """Duble de teste que registra as chamadas recebidas e devolve um valor constante."""

    def __init__(self, marker: float) -> None:
        self.marker = marker
        self.score_calls: list[np.ndarray] = []
        self.derivative_calls: list[np.ndarray] = []

    def score(self, y: np.ndarray) -> np.ndarray:
        self.score_calls.append(y)
        return np.full_like(y, self.marker)

    def derivative(self, y: np.ndarray) -> np.ndarray:
        self.derivative_calls.append(y)
        return np.full_like(y, self.marker)


def test_laplace_source_is_classified_as_supergaussian(rng, make_sources):
    """Uma fonte Laplaciana (cauda pesada) deve ter gamma_i < 0 e usar g_+.

    Direcao confirmada empiricamente em ICA_BACKGROUND.md, Secao 3.4:
    gamma_i e o negativo da estatistica de chaveamento do Extended
    Infomax (Lee, Girolami & Sejnowski, 1999), que e positiva para fontes
    supergaussianas.
    """
    y = make_sources(["laplace"], 20_000, rng)
    adaptive = AdaptiveScore()
    adaptive.score(y)
    assert adaptive.gamma_[0] < 0
    assert adaptive.is_super_gaussian_[0]


def test_uniform_source_is_classified_as_subgaussian(rng, make_sources):
    """Uma fonte Uniforme (achatada) deve ter gamma_i > 0 e usar g_-."""
    y = make_sources(["uniform"], 20_000, rng)
    adaptive = AdaptiveScore()
    adaptive.score(y)
    assert adaptive.gamma_[0] > 0
    assert not adaptive.is_super_gaussian_[0]


def test_mixed_sources_switch_independently_per_component(rng, make_sources):
    """Componentes diferentes devem poder ser classificadas diferentemente na mesma chamada."""
    y = make_sources(["laplace", "uniform"], 20_000, rng)
    adaptive = AdaptiveScore()
    adaptive.score(y)
    assert adaptive.is_super_gaussian_[0]
    assert not adaptive.is_super_gaussian_[1]


def test_score_delegates_to_injected_nonlinearities():
    """O metodo score deve delegar, por componente, ao duble injetado correspondente."""
    super_spy = _SpyNonlinearity(marker=100.0)
    sub_spy = _SpyNonlinearity(marker=-100.0)
    adaptive = AdaptiveScore(super_gaussian=super_spy, sub_gaussian=sub_spy)

    rng = np.random.default_rng(0)
    laplace_row = rng.laplace(size=5000)
    laplace_row = (laplace_row - laplace_row.mean()) / laplace_row.std()
    uniform_row = rng.uniform(-1, 1, size=5000)
    uniform_row = (uniform_row - uniform_row.mean()) / uniform_row.std()
    y = np.vstack([laplace_row, uniform_row])

    result = adaptive.score(y)

    assert len(super_spy.score_calls) == 1
    assert len(sub_spy.score_calls) == 1
    assert np.all(result[0] == 100.0)
    assert np.all(result[1] == -100.0)


def test_derivative_delegates_to_injected_nonlinearities():
    """O metodo derivative deve delegar, por componente, ao duble injetado correspondente."""
    super_spy = _SpyNonlinearity(marker=7.0)
    sub_spy = _SpyNonlinearity(marker=-7.0)
    adaptive = AdaptiveScore(super_gaussian=super_spy, sub_gaussian=sub_spy)

    rng = np.random.default_rng(0)
    laplace_row = rng.laplace(size=5000)
    laplace_row = (laplace_row - laplace_row.mean()) / laplace_row.std()
    uniform_row = rng.uniform(-1, 1, size=5000)
    uniform_row = (uniform_row - uniform_row.mean()) / uniform_row.std()
    y = np.vstack([laplace_row, uniform_row])

    result = adaptive.derivative(y)

    assert len(super_spy.derivative_calls) == 1
    assert len(sub_spy.derivative_calls) == 1
    assert np.all(result[0] == 7.0)
    assert np.all(result[1] == -7.0)
