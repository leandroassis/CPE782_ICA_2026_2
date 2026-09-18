"""Testes unitarios para ICAAlgorithm (Template Method base).

Ver ``.claude/skills/ica-ml/SKILL.md``, Secoes 6-7.
"""

import numpy as np

from ica.algorithms.base import ICAAlgorithm
from ica.algorithms.natural_gradient import NaturalGradientICA
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.nonlinearities.subgaussian import SubGaussianScore
from ica.nonlinearities.supergaussian import SuperGaussianScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.whitening import Whitening


class _ConstantStepAlgorithm(ICAAlgorithm):
    """Duble de teste cujo passo de atualizacao move B por um incremento fixo."""

    def __init__(self, step_size: float, **kwargs) -> None:
        super().__init__(nonlinearity=SuperGaussianScore(), **kwargs)
        self._step_size = step_size

    def _update_step(self, B: np.ndarray, X: np.ndarray) -> np.ndarray:
        return B + self._step_size

    def _convergence_residual(self, B_old: np.ndarray, B_new: np.ndarray, X: np.ndarray) -> float:
        return float(np.linalg.norm(B_new - B_old) / np.linalg.norm(B_old))


def test_stops_at_max_iterations_when_never_converging():
    """Sem nunca atingir a tolerancia, o loop deve parar exatamente em max_iterations."""
    algorithm = _ConstantStepAlgorithm(step_size=1.0, max_iterations=7, tolerance=1e-12)
    algorithm.fit(np.eye(3))
    assert algorithm.converged_ is False
    assert algorithm.n_iterations_ == 7
    assert len(algorithm.history_) == 7


def test_stops_early_when_tolerance_is_hit():
    """Quando a variacao de B cai abaixo da tolerancia, o loop deve parar antes do limite."""
    algorithm = _ConstantStepAlgorithm(step_size=0.0, max_iterations=50, tolerance=1e-9)
    algorithm.fit(np.eye(3))
    assert algorithm.converged_ is True
    assert algorithm.n_iterations_ == 1


def test_default_initialization_is_identity():
    """Sem random_state, B deve comecar como a matriz identidade."""
    algorithm = _ConstantStepAlgorithm(step_size=0.0, max_iterations=1)
    algorithm.fit(np.eye(4))
    assert np.allclose(algorithm.unmixing_matrix_, np.eye(4))


def test_random_state_produces_orthogonal_initialization():
    """Com random_state, a inicializacao de B deve ser ortogonal."""
    algorithm = _ConstantStepAlgorithm(step_size=0.0, max_iterations=1, random_state=0)
    algorithm.fit(np.eye(3))
    B0 = algorithm.unmixing_matrix_
    assert np.allclose(B0 @ B0.T, np.eye(3), atol=1e-8)


def test_elapsed_time_is_recorded_and_non_negative():
    """elapsed_time_ deve ser preenchido e nao-negativo apos fit."""
    algorithm = _ConstantStepAlgorithm(step_size=0.0, max_iterations=1)
    algorithm.fit(np.eye(3))
    assert algorithm.elapsed_time_ is not None
    assert algorithm.elapsed_time_ >= 0.0


def test_wrong_nonlinearity_violates_stability_and_degrades_separation(
    rng, make_sources, make_mixing_matrix, best_match_correlation
):
    """Forcar g_- sobre fontes supergaussianas viola a condicao de estabilidade.

    Convencao do livro (skill ica-ml, Secao 3; Teor. 9.1): o estimador de
    ML e localmente consistente quando ``E{s g(s) - g'(s)} > 0``. Usar a
    nao-linearidade do ramo errado inverte esse sinal: a solucao correta
    deixa de ser um ponto fixo estavel, e a separacao se degrada.

    Nao se testa divergencia numerica (NaN) aqui: o modo de falha previsto
    pela teoria e perda de consistencia, nao estouro.
    """
    S = make_sources(["laplace", "laplace"], 3000, rng)
    A = make_mixing_matrix(rng, 2)
    X = A @ S
    X_whitened = Whitening().fit_transform(Centering().fit_transform(X))

    def stability(nonlinearity, y):
        return float(np.mean(y * nonlinearity.score(y) - nonlinearity.derivative(y)))

    assert stability(SuperGaussianScore(), S) > 0
    assert stability(SubGaussianScore(), S) < 0

    wrong_algorithm = NaturalGradientICA(
        nonlinearity=SubGaussianScore(), learning_rate=0.001, max_iterations=500
    )
    wrong_algorithm.fit(X_whitened)
    recovered_wrong = wrong_algorithm.unmixing_matrix_ @ X_whitened

    correct_algorithm = NaturalGradientICA(
        nonlinearity=AdaptiveScore(), learning_rate=0.001, max_iterations=500
    )
    correct_algorithm.fit(X_whitened)
    recovered_correct = correct_algorithm.unmixing_matrix_ @ X_whitened

    assert np.all(np.isfinite(recovered_correct))
    assert best_match_correlation(S, recovered_correct) > 0.8
    assert best_match_correlation(S, recovered_correct) > best_match_correlation(S, recovered_wrong)
