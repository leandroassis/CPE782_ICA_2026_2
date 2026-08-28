"""Testes unitarios para Centering (ICA_BACKGROUND.md, Secao 2.1)."""

import numpy as np

from ica.preprocessing.centering import Centering


def test_transform_produces_zero_mean(rng):
    """Apos transform, cada mistura (linha) deve ter media amostral proxima de zero."""
    X = rng.normal(loc=5.0, scale=2.0, size=(3, 500))
    centered = Centering().fit_transform(X)
    assert np.allclose(centered.mean(axis=1), 0.0, atol=1e-10)


def test_mean_attribute_matches_row_means(rng):
    """mean_ deve conter exatamente a media amostral por linha usada em fit."""
    X = rng.normal(size=(4, 200))
    step = Centering().fit(X)
    assert np.allclose(step.mean_, X.mean(axis=1))


def test_inverse_transform_round_trips(rng):
    """inverse_transform(transform(X)) deve recuperar X exatamente."""
    X = rng.normal(loc=-3.0, scale=1.5, size=(3, 300))
    step = Centering()
    centered = step.fit_transform(X)
    reconstructed = step.inverse_transform(centered)
    assert np.allclose(reconstructed, X)


def test_mean_shape_matches_number_of_mixtures(rng):
    """mean_ deve ter shape (n_misturas,)."""
    X = rng.normal(size=(5, 100))
    step = Centering().fit(X)
    assert step.mean_.shape == (5,)
