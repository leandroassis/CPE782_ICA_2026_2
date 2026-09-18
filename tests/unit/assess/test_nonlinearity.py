"""Testes unitarios para ica.assess.nonlinearity (skill bss-assessment, Secao 3)."""

import numpy as np

from ica.assess.nonlinearity import (
    distance_correlation,
    joint_diagonalization_defect,
    residual_dependence,
)
from ica.preprocessing.centering import Centering
from ica.preprocessing.whitening import Whitening


def test_residual_dependence_is_low_for_independent_components(rng, make_sources):
    """Componentes ja independentes devem ter dependencia residual baixa."""
    Y = make_sources(["laplace", "uniform", "gaussian"], 20000, rng)
    result = residual_dependence(Y)
    assert result.max_residual_dependence < 0.1
    assert not result.is_likely_nonlinear


def test_residual_dependence_is_high_for_nonlinearly_coupled_signals(rng):
    """Sinais com dependencia nao-linear explicita (y2 = y1^2) devem disparar o flag."""
    base = rng.laplace(size=20000)
    coupled = base**2 - np.mean(base**2)
    Y = np.vstack([base, coupled, rng.normal(size=20000)])
    result = residual_dependence(Y)
    assert result.is_likely_nonlinear


def test_distance_correlation_is_near_zero_for_independent_signals(rng):
    """DCorr deve ser baixo para sinais independentes."""
    x = rng.normal(size=3000)
    y = rng.normal(size=3000)
    assert distance_correlation(x, y) < 0.1


def test_distance_correlation_detects_nonlinear_dependence(rng):
    """DCorr deve detectar dependencia nao-linear (y = x^2) que a correlacao linear perderia."""
    x = rng.uniform(-1, 1, size=3000)
    y = x**2
    assert distance_correlation(x, y) > 0.3
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.1


def test_joint_diagonalization_defect_is_low_for_genuinely_linear_mixture(rng, make_sources):
    """Para uma mistura linear verdadeira, o defeito de diagonalizacao conjunta deve ser baixo."""
    S = make_sources(["laplace", "uniform", "laplace"], 20000, rng)
    A = rng.normal(size=(3, 3))
    X = A @ S
    whitened = Whitening().fit_transform(Centering().fit_transform(X))
    assert joint_diagonalization_defect(whitened) < 0.02


def test_joint_diagonalization_defect_is_higher_for_nonlinear_mixing(rng, make_sources):
    """Para uma mistura com termos cruzados nao-lineares, o defeito deve ser maior."""
    S = make_sources(["laplace", "uniform", "laplace"], 20000, rng)
    nonlinear_mix = np.vstack(
        [
            S[0] + 0.5 * S[1] * S[2],
            S[1] - 0.3 * S[0] * S[2],
            S[2] + 0.2 * S[0] * S[1],
        ]
    )
    linear_mix = rng.normal(size=(3, 3)) @ S

    whitened_nonlinear = Whitening().fit_transform(Centering().fit_transform(nonlinear_mix))
    whitened_linear = Whitening().fit_transform(Centering().fit_transform(linear_mix))

    assert joint_diagonalization_defect(whitened_nonlinear) > joint_diagonalization_defect(
        whitened_linear
    )
