"""Testes unitarios para SubGaussianScore (skill ica-ml, Secoes 3-4)."""

import numpy as np

from ica.nonlinearities.subgaussian import SubGaussianScore
from ica.nonlinearities.supergaussian import SuperGaussianScore


def test_score_matches_closed_form():
    """score(y) deve ser exatamente tanh(y) - y (livro, sem sinal na frente de g)."""
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
    """log_density(y) deve ser exatamente log(cosh(y)) - y^2/2."""
    y = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    result = SubGaussianScore().log_density(y)
    expected = np.log(np.cosh(y)) - (y**2) / 2.0
    assert np.allclose(result, expected)


def test_log_density_is_the_antiderivative_of_score():
    """score(y) deve ser exatamente d/dy log_density(y) (convencao do livro, sem sinal)."""
    nonlinearity = SubGaussianScore()
    eps = 1e-6
    for s in [-2.0, -0.5, 0.3, 1.7]:
        numeric_derivative = (
            nonlinearity.log_density(np.array([[s + eps]]))[0, 0]
            - nonlinearity.log_density(np.array([[s - eps]]))[0, 0]
        ) / (2 * eps)
        assert abs(numeric_derivative - nonlinearity.score(np.array([[s]]))[0, 0]) < 1e-8


def test_log_density_decays_in_the_tails():
    """log_density deve decair nas caudas: -y^2/2 domina log(cosh(y)) ~ |y|.

    Garante que a densidade suposta subgaussiana e integravel.
    """
    nonlinearity = SubGaussianScore()
    center = nonlinearity.log_density(np.array([[0.0]]))[0, 0]
    tail = nonlinearity.log_density(np.array([[50.0]]))[0, 0]
    assert tail < center - 1000


def test_sign_convention_matches_supergaussian():
    """g_- e g_+ devem seguir a MESMA convencao ``g = (log p)'`` (livro, Cap. 9).

    Uma inconsistencia de sinal entre as duas nao-linearidades faria o
    algoritmo ascender a verossimilhanca nas componentes de um ramo e
    descende-la nas do outro -- invisivel no FastICA (invariante ao
    sinal de g), mas fatal para os metodos de gradiente.
    """
    y = np.array([[-2.0, -0.5, 0.3, 1.7]])
    eps = 1e-6
    for nonlinearity in [SubGaussianScore(), SuperGaussianScore()]:
        numeric_derivative = (
            nonlinearity.log_density(y + eps) - nonlinearity.log_density(y - eps)
        ) / (2 * eps)
        assert np.allclose(numeric_derivative, nonlinearity.score(y), atol=1e-7)
