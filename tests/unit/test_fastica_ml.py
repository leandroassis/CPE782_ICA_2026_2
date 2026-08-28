"""Testes unitarios para FastICAML (ICA_BACKGROUND.md, Secao 4.3)."""

import numpy as np

from ica.algorithms.fastica_ml import FastICAML
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.nonlinearities.supergaussian import SuperGaussianScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.whitening import Whitening


def test_update_step_matches_closed_form():
    """_update_step deve seguir o ponto fixo em bloco + ortogonalizacao simetrica.

    Valores de referencia (alpha_i, beta_i e o resultado final apos
    ortogonalizar) calculados independentemente a partir da formula
    (ICA_BACKGROUND.md, Secao 4.3), nao lidos de volta do proprio metodo.
    """
    B = np.array([[1.0, 0.2], [-0.1, 1.0]])
    X = np.array([[1.0, -2.0, 0.5, 3.0], [0.5, 1.0, -1.5, 2.0]])
    algorithm = FastICAML(nonlinearity=SuperGaussianScore(), max_iterations=1)

    result = algorithm._update_step(B, X)

    expected = np.array(
        [[0.4885439546, 0.129250794], [-0.2907870831, 0.7729396137]]
    )
    assert np.allclose(result, expected, atol=1e-8)


def test_update_step_orthogonalizes_output_against_covariance():
    """Apos cada passo, B Cx B^T deve ser (numericamente) a identidade."""
    B = np.array([[1.0, 0.2], [-0.1, 1.0]])
    X = np.array([[1.0, -2.0, 0.5, 3.0], [0.5, 1.0, -1.5, 2.0]])
    algorithm = FastICAML(nonlinearity=SuperGaussianScore(), max_iterations=1)

    B_new = algorithm._update_step(B, X)
    covariance = (X @ X.T) / X.shape[1]

    assert np.allclose(B_new @ covariance @ B_new.T, np.eye(2), atol=1e-8)


def test_recovers_two_laplace_sources_quickly(
    rng, make_sources, make_mixing_matrix, best_match_correlation
):
    """FastICAML deve recuperar as fontes com correlacao muito alta em poucas iteracoes."""
    S = make_sources(["laplace", "laplace"], 3000, rng)
    A = make_mixing_matrix(rng, 2)
    X = A @ S
    X_whitened = Whitening().fit_transform(Centering().fit_transform(X))

    algorithm = FastICAML(nonlinearity=AdaptiveScore(), max_iterations=100)
    algorithm.fit(X_whitened)
    recovered = algorithm.unmixing_matrix_ @ X_whitened

    assert algorithm.converged_
    assert algorithm.n_iterations_ < 100
    assert best_match_correlation(S, recovered) > 0.99


def test_log_likelihood_is_non_decreasing(rng, make_sources, make_mixing_matrix):
    """A log-verossimilhanca media (ICA_BACKGROUND.md, Secao 3.2) deve crescer a cada iteracao."""
    S = make_sources(["laplace", "laplace"], 3000, rng)
    A = make_mixing_matrix(rng, 2)
    X = A @ S
    X_whitened = Whitening().fit_transform(Centering().fit_transform(X))

    algorithm = FastICAML(nonlinearity=AdaptiveScore(), max_iterations=100)
    algorithm.fit(X_whitened)

    log_likelihood = algorithm.log_likelihood_history_
    assert len(log_likelihood) == algorithm.n_iterations_
    assert log_likelihood[-1] >= log_likelihood[0]
    assert np.mean(np.diff(log_likelihood) >= -1e-6) > 0.95


def test_learning_rate_is_ignored():
    """FastICAML e livre de taxa de aprendizado -- o parametro e aceito mas nao usado.

    Documenta explicitamente a excecao registrada em ICA_BACKGROUND.md,
    Secao 4.3/4.4 ("Livre de taxa mu"): duas instancias com
    ``learning_rate`` diferentes devem produzir o mesmo resultado.
    """
    X = np.array([[1.0, -2.0, 0.5, 3.0, -0.2], [0.5, 1.0, -1.5, 2.0, 0.7]])
    nonlinearity = SuperGaussianScore()
    algorithm_a = FastICAML(nonlinearity=nonlinearity, learning_rate=0.001, max_iterations=1)
    algorithm_b = FastICAML(nonlinearity=nonlinearity, learning_rate=50.0, max_iterations=1)
    result_a = algorithm_a._update_step(np.eye(2), X)
    result_b = algorithm_b._update_step(np.eye(2), X)
    assert np.allclose(result_a, result_b)


def test_fit_populates_attribute_contract(rng):
    """Apos fit(), todos os atributos publicos documentados devem estar preenchidos."""
    X = rng.normal(size=(2, 200))
    algorithm = FastICAML(nonlinearity=SuperGaussianScore(), max_iterations=5)
    algorithm.fit(X)

    assert algorithm.unmixing_matrix_.shape == (2, 2)
    assert isinstance(algorithm.converged_, bool)
    assert len(algorithm.history_) == algorithm.n_iterations_
    assert len(algorithm.log_likelihood_history_) == algorithm.n_iterations_
    assert algorithm.elapsed_time_ is not None and algorithm.elapsed_time_ >= 0.0
