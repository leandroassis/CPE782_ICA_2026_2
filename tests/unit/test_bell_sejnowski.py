"""Testes unitarios para BellSejnowskiICA (ICA_BACKGROUND.md, Secao 4.1)."""

import numpy as np

from ica.algorithms.bell_sejnowski import BellSejnowskiICA
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.nonlinearities.supergaussian import SuperGaussianScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.whitening import Whitening


def test_update_step_matches_closed_form():
    """_update_step deve ser exatamente B + lr*((B^T)^-1 - g(Y)X^T/T), com Y=BX.

    Valores de referencia calculados independentemente a partir da formula
    (ICA_BACKGROUND.md, Secao 4.1), nao lidos de volta do proprio metodo.
    """
    B = np.array([[1.0, 0.2], [-0.1, 1.0]])
    X = np.array([[1.0, -2.0, 0.5, 3.0], [0.5, 1.0, -1.5, 2.0]])
    algorithm = BellSejnowskiICA(
        nonlinearity=SuperGaussianScore(), learning_rate=0.1, max_iterations=1
    )

    result = algorithm._update_step(B, X)

    expected = np.array(
        [[0.8087330411, 0.1521574029], [-0.1727065539, 0.8847829425]]
    )
    assert np.allclose(result, expected, atol=1e-9)


def test_runs_on_non_whitened_data_exercising_matrix_inversion(
    rng, make_sources, make_mixing_matrix
):
    """Diferente do Gradiente Natural, Bell-Sejnowski deve funcionar mesmo sem branqueamento.

    Exercita de fato o termo ``(B^T)^-1`` sobre dados apenas centralizados
    (nao branqueados), garantindo que a inversao de matriz nao produz
    valores nao-finitos.
    """
    S = make_sources(["laplace", "laplace"], 3000, rng)
    A = make_mixing_matrix(rng, 2)
    X = A @ S
    X_centered = Centering().fit_transform(X)

    algorithm = BellSejnowskiICA(
        nonlinearity=AdaptiveScore(), learning_rate=0.01, max_iterations=500
    )
    algorithm.fit(X_centered)

    assert np.all(np.isfinite(algorithm.unmixing_matrix_))
    recovered = algorithm.unmixing_matrix_ @ X_centered
    assert np.all(np.isfinite(recovered))


def test_recovers_two_laplace_sources_when_whitened(
    rng, make_sources, make_mixing_matrix, best_match_correlation
):
    """Com dados branqueados, Bell-Sejnowski deve recuperar as fontes com alta correlacao."""
    S = make_sources(["laplace", "laplace"], 3000, rng)
    A = make_mixing_matrix(rng, 2)
    X = A @ S
    X_whitened = Whitening().fit_transform(Centering().fit_transform(X))

    algorithm = BellSejnowskiICA(
        nonlinearity=AdaptiveScore(), learning_rate=0.01, max_iterations=500
    )
    algorithm.fit(X_whitened)
    recovered = algorithm.unmixing_matrix_ @ X_whitened

    assert best_match_correlation(S, recovered) > 0.95


def test_fit_populates_attribute_contract(rng):
    """Apos fit(), todos os atributos publicos documentados devem estar preenchidos."""
    X = rng.normal(size=(2, 200))
    algorithm = BellSejnowskiICA(nonlinearity=SuperGaussianScore(), max_iterations=5)
    algorithm.fit(X)

    assert algorithm.unmixing_matrix_.shape == (2, 2)
    assert isinstance(algorithm.converged_, bool)
    assert len(algorithm.history_) == algorithm.n_iterations_
    assert algorithm.elapsed_time_ is not None and algorithm.elapsed_time_ >= 0.0
