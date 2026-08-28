"""Testes unitarios para Whitening (ICA_BACKGROUND.md, Secao 2.2)."""

import numpy as np

from ica.preprocessing.centering import Centering
from ica.preprocessing.whitening import Whitening


def test_transform_produces_identity_covariance(rng):
    """A covariancia amostral dos dados branqueados deve ser exatamente a identidade.

    Esta e uma identidade algebrica exata (V vem da EVD da propria
    covariancia amostral de X), nao uma aproximacao estatistica -- vale
    para qualquer X, independentemente da distribuicao ou do tamanho
    amostral.
    """
    X = rng.normal(size=(3, 500))
    Z = Whitening().fit_transform(X)
    covariance = (Z @ Z.T) / Z.shape[1]
    assert np.allclose(covariance, np.eye(3), atol=1e-8)


def test_whitened_mixing_matrix_is_approximately_orthogonal(rng, make_sources, make_mixing_matrix):
    """V @ A deve ficar proxima de ortogonal apos o branqueamento (ICA_BACKGROUND.md, Secao 2.2).

    So e exatamente ortogonal se a covariancia amostral das fontes for
    exatamente a identidade, o que so vale no limite T -> infinito; por
    isso a amostra e grande e a tolerancia reflete o ruido estatistico
    residual esperado (~1/sqrt(T)).
    """
    S = make_sources(["laplace", "uniform", "gaussian"], 20_000, rng)
    A = make_mixing_matrix(rng, 3)
    X = A @ S
    whitening = Whitening().fit(Centering().fit_transform(X))
    effective_mixing = whitening.whitening_matrix_ @ A
    should_be_identity = effective_mixing @ effective_mixing.T
    assert np.allclose(should_be_identity, np.eye(3), atol=0.05)


def test_inverse_transform_round_trips(rng):
    """inverse_transform(transform(X)) deve recuperar X (a menos de erro numerico)."""
    X = rng.normal(size=(4, 1000))
    X_centered = X - X.mean(axis=1, keepdims=True)
    step = Whitening()
    whitened = step.fit_transform(X_centered)
    reconstructed = step.inverse_transform(whitened)
    assert np.allclose(reconstructed, X_centered, atol=1e-8)


def test_whitening_matrix_shape(rng):
    """whitening_matrix_ deve ter shape (n_misturas, n_misturas)."""
    X = rng.normal(size=(6, 300))
    step = Whitening().fit(X)
    assert step.whitening_matrix_.shape == (6, 6)
