"""Testes unitarios para SubGaussianScore (ICA_BACKGROUND.md, Secao 3.4)."""

import numpy as np

from ica.nonlinearities.subgaussian import SubGaussianScore


def test_score_matches_closed_form():
    """score(y) deve ser exatamente tanh(y) - y."""
    y = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    result = SubGaussianScore().score(y)
    assert np.allclose(result, np.tanh(y) - y)


def test_derivative_matches_closed_form():
    """derivative(y) deve ser exatamente -tanh(y)**2."""
    y = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    result = SubGaussianScore().derivative(y)
    assert np.allclose(result, -np.tanh(y) ** 2)


def test_score_is_zero_at_origin():
    """g_-(0) deve ser 0, pois tanh(0) - 0 = 0."""
    result = SubGaussianScore().score(np.zeros((2, 3)))
    assert np.allclose(result, 0.0)


def test_derivative_is_non_positive_everywhere():
    """g_-'(y) = -tanh^2(y) e sempre <= 0."""
    y = np.linspace(-10, 10, 201).reshape(1, -1)
    result = SubGaussianScore().derivative(y)
    assert np.all(result <= 0.0)
