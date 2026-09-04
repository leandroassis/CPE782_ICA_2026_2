"""Testes unitarios para PCA (livro-texto, Secao 13.2, paginas 267-269)."""

import numpy as np
import pytest

from ica.preprocessing.pca import PCA


def test_invalid_float_n_components_raises():
    """n_components como float fora de (0, 1] deve levantar ValueError."""
    with pytest.raises(ValueError):
        PCA(n_components=1.5)
    with pytest.raises(ValueError):
        PCA(n_components=0.0)


def test_invalid_int_n_components_raises():
    """n_components como int menor que 1 deve levantar ValueError."""
    with pytest.raises(ValueError):
        PCA(n_components=0)


def test_fit_raises_when_int_n_components_exceeds_n_mixtures(rng):
    """n_components (int) maior que o numero de misturas deve levantar ValueError."""
    X = rng.normal(size=(3, 100))
    with pytest.raises(ValueError):
        PCA(n_components=5).fit(X)


def test_transform_shape_matches_n_components(rng):
    """Transform deve reduzir X de (n_misturas, T) para (n_components, T)."""
    X = rng.normal(size=(5, 200))
    step = PCA(n_components=2).fit(X)
    Z = step.transform(X)
    assert Z.shape == (2, 200)


def test_components_are_orthonormal(rng):
    """Os autovetores retidos devem ser ortonormais (EVD de matriz simetrica)."""
    X = rng.normal(size=(4, 300))
    step = PCA(n_components=3).fit(X)
    gram = step.components_.T @ step.components_
    assert np.allclose(gram, np.eye(3), atol=1e-8)


def test_retains_highest_variance_directions(rng):
    """PCA deve reter as direcoes de maior variancia, nao quaisquer duas.

    Constroi dados com variancia muito maior nos dois primeiros eixos
    (apos rotacao) e verifica que a razao de variancia explicada retida
    e proxima de 1 -- ou seja, as direcoes de baixa variancia (ruido)
    foram descartadas.
    """
    n = 50_000
    high_variance = rng.normal(scale=10.0, size=(2, n))
    low_variance = rng.normal(scale=0.01, size=(2, n))
    X = np.vstack([high_variance, low_variance])
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    X = rotation @ X

    step = PCA(n_components=2).fit(X)

    assert step.explained_variance_ratio_.sum() > 0.999


def test_float_n_components_selects_minimum_components_for_variance_threshold(rng):
    """n_components float deve escolher o menor k que atinge o limiar de variancia."""
    n = 50_000
    high_variance = rng.normal(scale=10.0, size=(1, n))
    low_variance = rng.normal(scale=0.01, size=(3, n))
    X = np.vstack([high_variance, low_variance])

    step = PCA(n_components=0.9).fit(X)

    assert step.n_components_ == 1


def test_float_n_components_equal_to_one_retains_all_mixtures(rng):
    """n_components=1.0 deve reter todas as misturas (100% da variancia)."""
    X = rng.normal(size=(4, 500))
    step = PCA(n_components=1.0).fit(X)
    assert step.n_components_ == 4


def test_explained_variance_ratio_sums_to_at_most_one(rng):
    """A soma de explained_variance_ratio_ nao deve exceder 1 (fracao da variancia total)."""
    X = rng.normal(size=(4, 500))
    step = PCA(n_components=2).fit(X)
    assert step.explained_variance_ratio_.sum() <= 1.0 + 1e-8


def test_inverse_transform_round_trips_when_no_reduction(rng):
    """Sem reducao de dimensao (n_components == n_misturas), a reconstrucao deve ser exata."""
    X = rng.normal(size=(3, 400))
    step = PCA(n_components=3)
    Z = step.fit_transform(X)
    reconstructed = step.inverse_transform(Z)
    assert np.allclose(reconstructed, X, atol=1e-8)


def test_inverse_transform_is_lossy_approximation_when_reducing(rng):
    """Com reducao de dimensao, a reconstrucao deve ser uma aproximacao, nao exata."""
    n = 5_000
    high_variance = rng.normal(scale=10.0, size=(2, n))
    low_variance = rng.normal(scale=1.0, size=(2, n))
    X = np.vstack([high_variance, low_variance])

    step = PCA(n_components=2).fit(X)
    reconstructed = step.inverse_transform(step.transform(X))

    assert not np.allclose(reconstructed, X, atol=0.5)
    residual = np.linalg.norm(X - reconstructed) / np.linalg.norm(X)
    assert residual < 0.5


def test_linear_matrix_is_none_before_fit():
    """linear_matrix_ deve ser None antes de fit (components_ ainda nao definido)."""
    assert PCA(n_components=2).linear_matrix_ is None


def test_linear_matrix_is_components_transpose_after_fit(rng):
    """linear_matrix_ deve ser exatamente components_.T apos fit."""
    X = rng.normal(size=(4, 200))
    step = PCA(n_components=2).fit(X)
    assert np.array_equal(step.linear_matrix_, step.components_.T)


def test_estimation_only_is_false():
    """PCA e um passo de canal (nao de tempo): estimation_only deve ser False."""
    assert PCA(n_components=2).estimation_only is False
