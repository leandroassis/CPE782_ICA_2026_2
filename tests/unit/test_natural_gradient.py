"""Testes unitarios para NaturalGradientICA (ICA_BACKGROUND.md, Secao 4.2)."""

import numpy as np

from ica.algorithms.natural_gradient import NaturalGradientICA
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.nonlinearities.supergaussian import SuperGaussianScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.whitening import Whitening


def test_update_step_matches_closed_form():
    """_update_step deve ser exatamente B + lr*(I - g(Y)Y^T/T)B, com Y=BX.

    Valores de referencia calculados independentemente a partir da formula
    (ICA_BACKGROUND.md, Secao 4.2), nao lidos de volta do proprio metodo.
    """
    B = np.array([[1.0, 0.2], [-0.1, 1.0]])
    X = np.array([[1.0, -2.0, 0.5, 3.0], [0.5, 1.0, -1.5, 2.0]])
    algorithm = NaturalGradientICA(
        nonlinearity=SuperGaussianScore(), learning_rate=0.1, max_iterations=1
    )

    result = algorithm._update_step(B, X)

    expected = np.array(
        [[0.8020361118, 0.1311170031], [-0.1849553252, 0.8729036048]]
    )
    assert np.allclose(result, expected, atol=1e-9)


def test_recovers_two_laplace_sources(
    rng, make_sources, make_mixing_matrix, best_match_correlation
):
    """fit() deve recuperar 2 fontes Laplacianas com alta correlacao (a menos de ambiguidades)."""
    S = make_sources(["laplace", "laplace"], 3000, rng)
    A = make_mixing_matrix(rng, 2)
    X = A @ S
    X_whitened = Whitening().fit_transform(Centering().fit_transform(X))

    algorithm = NaturalGradientICA(nonlinearity=AdaptiveScore(), max_iterations=500)
    algorithm.fit(X_whitened)
    recovered = algorithm.unmixing_matrix_ @ X_whitened

    assert np.all(np.isfinite(recovered))
    assert best_match_correlation(S, recovered) > 0.95


def test_fit_populates_attribute_contract(rng):
    """Apos fit(), todos os atributos publicos documentados devem estar preenchidos."""
    X = rng.normal(size=(2, 200))
    algorithm = NaturalGradientICA(nonlinearity=SuperGaussianScore(), max_iterations=5)
    algorithm.fit(X)

    assert algorithm.unmixing_matrix_.shape == (2, 2)
    assert isinstance(algorithm.converged_, bool)
    assert algorithm.n_iterations_ == 5 or algorithm.converged_
    assert len(algorithm.history_) == algorithm.n_iterations_
    assert algorithm.elapsed_time_ is not None and algorithm.elapsed_time_ >= 0.0


def test_default_learning_rate_is_conservative():
    """O default de learning_rate deve ser mais conservador que o da classe base.

    Verificado empiricamente: com g_- (subgaussiana, nao-linearidade
    correta para fontes achatadas) o termo multiplicativo
    ``[I - g(y)y^T]B`` diverge numericamente para taxas de aprendizado
    maiores (>= 0.002) quando ha fontes subgaussianas -- ver
    ICA_BACKGROUND.md, Secao 4.2.
    """
    default_algorithm = NaturalGradientICA(nonlinearity=SuperGaussianScore())
    assert default_algorithm.learning_rate <= 0.001
