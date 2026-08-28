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


def test_log_density_matches_closed_form():
    """log_density(y) deve ser exatamente y^2/2 - log(cosh(y))."""
    y = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    result = SubGaussianScore().log_density(y)
    expected = (y**2) / 2.0 - np.log(np.cosh(y))
    assert np.allclose(result, expected)


def test_log_density_is_negative_derivative_of_score():
    """score(y) deve ser exatamente -d/dy log_density(y) (definicao de score function).

    Vale mesmo esta "densidade suposta" nao sendo normalizavel (ver
    test_log_density_is_not_a_valid_probability_density) -- e a
    definicao que importa para a log-verossimilhanca que o algoritmo de
    fato ascende (ICA_BACKGROUND.md, Secao 3.2-3.3).
    """
    nonlinearity = SubGaussianScore()
    eps = 1e-6
    for s in [-2.0, -0.5, 0.3, 1.7]:
        numeric_derivative = (
            nonlinearity.log_density(np.array([[s + eps]]))[0, 0]
            - nonlinearity.log_density(np.array([[s - eps]]))[0, 0]
        ) / (2 * eps)
        assert abs(-numeric_derivative - nonlinearity.score(np.array([[s]]))[0, 0]) < 1e-8


def test_log_density_is_not_a_valid_probability_density():
    """log_density diverge para +infinito nas caudas (y^2/2 domina log(cosh(y))).

    Documenta explicitamente a diferenca em relacao a
    SuperGaussianScore.log_density (que e a densidade logistica,
    propriamente normalizada): esta funcao nao integra a 1 -- e um
    substituto de trabalho consistente com score, nao uma densidade
    valida (ver ICA_BACKGROUND.md, Secao 3.3).
    """
    nonlinearity = SubGaussianScore()
    small = nonlinearity.log_density(np.array([[0.0]]))[0, 0]
    large = nonlinearity.log_density(np.array([[50.0]]))[0, 0]
    assert large > small + 1000
