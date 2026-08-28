"""Testes unitarios para SuperGaussianScore (ICA_BACKGROUND.md, Secao 3.4)."""

import numpy as np
from scipy.integrate import quad

from ica.nonlinearities.supergaussian import SuperGaussianScore


def test_score_matches_closed_form():
    """score(y) deve ser exatamente 2*tanh(y)."""
    y = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    result = SuperGaussianScore().score(y)
    assert np.allclose(result, 2.0 * np.tanh(y))


def test_derivative_matches_closed_form():
    """derivative(y) deve ser exatamente 2*(1 - tanh(y)**2)."""
    y = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    result = SuperGaussianScore().derivative(y)
    assert np.allclose(result, 2.0 * (1.0 - np.tanh(y) ** 2))


def test_score_is_zero_at_origin():
    """g_+(0) deve ser 0, pois tanh(0) = 0."""
    result = SuperGaussianScore().score(np.zeros((2, 3)))
    assert np.allclose(result, 0.0)


def test_derivative_is_positive_everywhere():
    """g_+'(y) = 2(1 - tanh^2(y)) e sempre >= 0, pois |tanh| <= 1."""
    y = np.linspace(-10, 10, 201).reshape(1, -1)
    result = SuperGaussianScore().derivative(y)
    assert np.all(result >= 0.0)


def test_log_density_matches_closed_form():
    """log_density(y) deve ser exatamente -log(2) - 2*log(cosh(y))."""
    y = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    result = SuperGaussianScore().log_density(y)
    expected = -np.log(2.0) - 2.0 * np.log(np.cosh(y))
    assert np.allclose(result, expected)


def test_log_density_is_negative_derivative_of_score():
    """score(y) deve ser exatamente -d/dy log_density(y) (definicao de score function)."""
    nonlinearity = SuperGaussianScore()
    eps = 1e-6
    for s in [-2.0, -0.5, 0.3, 1.7]:
        numeric_derivative = (
            nonlinearity.log_density(np.array([[s + eps]]))[0, 0]
            - nonlinearity.log_density(np.array([[s - eps]]))[0, 0]
        ) / (2 * eps)
        assert abs(-numeric_derivative - nonlinearity.score(np.array([[s]]))[0, 0]) < 1e-8


def test_log_density_is_a_properly_normalized_density():
    """exp(log_density) deve integrar a 1 sobre toda a reta (e uma densidade valida)."""
    nonlinearity = SuperGaussianScore()

    def density(s: float) -> float:
        return np.exp(nonlinearity.log_density(np.array([[s]]))[0, 0])

    integral, _ = quad(density, -50, 50)
    assert abs(integral - 1.0) < 1e-6
