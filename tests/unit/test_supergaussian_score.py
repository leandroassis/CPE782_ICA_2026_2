"""Testes unitarios para SuperGaussianScore (ICA_BACKGROUND.md, Secao 3.4)."""

import numpy as np

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
