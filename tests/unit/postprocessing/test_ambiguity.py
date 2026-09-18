"""Testes unitarios para ica.postprocessing.ambiguity (skill ica-evaluation, Secao 1)."""

import numpy as np
from scipy.stats import kurtosis, skew

from ica.postprocessing.ambiguity import (
    fix_scale,
    fix_sign,
    resolve_ambiguities,
    stable_permutation_order,
)


def test_fix_scale_normalizes_to_unit_variance_for_distribution_domain():
    """Para dominio 'distribution', fix_scale so normaliza a variancia unitaria."""
    rng = np.random.default_rng(0)
    Y = rng.normal(loc=5.0, scale=3.0, size=(2, 5000))
    result = fix_scale(Y, "distribution")
    assert np.allclose(result.std(axis=1), 1.0, atol=1e-8)


def test_fix_scale_rescales_audio_to_unit_peak():
    """Para dominio 'audio', fix_scale deve trazer o pico absoluto para <= 1."""
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(2, 2000)) * 50.0
    result = fix_scale(Y, "audio")
    assert np.allclose(np.max(np.abs(result), axis=1), 1.0, atol=1e-8)


def test_fix_scale_rescales_image_to_zero_one_range():
    """Para dominio 'image', fix_scale deve trazer cada componente para [0, 1]."""
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(2, 2000)) * 10.0
    result = fix_scale(Y, "image")
    assert np.allclose(result.min(axis=1), 0.0, atol=1e-8)
    assert np.allclose(result.max(axis=1), 1.0, atol=1e-8)


def test_fix_sign_makes_asymmetric_component_positively_skewed():
    """Para uma componente claramente assimetrica, fix_sign deve deixar a assimetria positiva."""
    rng = np.random.default_rng(0)
    exponential = rng.exponential(size=(1, 5000))
    negatively_skewed = -(exponential - exponential.mean())

    result = fix_sign(negatively_skewed)

    assert skew(result, axis=1)[0] > 0


def test_fix_sign_leaves_symmetric_component_untouched():
    """Para uma componente simetrica (skew ~ 0), fix_sign nao deve alterar os valores."""
    rng = np.random.default_rng(0)
    laplace = rng.laplace(size=(1, 5000))
    result = fix_sign(laplace)
    assert np.allclose(result, laplace)


def test_stable_permutation_order_sorts_by_decreasing_excess_kurtosis_magnitude():
    """A ordem deve colocar a componente mais nao-gaussiana (maior |curtose|) primeiro."""
    rng = np.random.default_rng(0)
    gaussian_like = rng.normal(size=(1, 20000))
    laplace_like = rng.laplace(size=(1, 20000))
    Y = np.vstack([gaussian_like, laplace_like])

    order = stable_permutation_order(Y)

    assert order[0] == 1  # laplaciana (curtose alta) vem primeiro


def test_stable_permutation_order_is_deterministic():
    """A mesma entrada deve sempre produzir a mesma ordem (reproduzibilidade)."""
    rng = np.random.default_rng(0)
    Y = rng.laplace(size=(4, 3000))
    assert list(stable_permutation_order(Y)) == list(stable_permutation_order(Y))


def test_resolve_ambiguities_preserves_recoverability_up_to_reordering(rng, make_sources):
    """resolve_ambiguities nao deve destruir a informacao das fontes -- so normaliza/reordena."""
    S = make_sources(["laplace", "uniform"], 5000, rng)
    scrambled = np.stack([-3.0 * S[1], 5.0 * S[0]])

    resolved = resolve_ambiguities(scrambled, domain="distribution")

    assert resolved.shape == scrambled.shape
    # cada linha resolvida deve estar fortemente correlacionada com alguma fonte original
    correlations = np.abs(np.corrcoef(np.vstack([S, resolved]))[:2, 2:])
    assert np.all(correlations.max(axis=0) > 0.99)


def test_resolve_ambiguities_kurtosis_matches_expected_after_scaling():
    """A curtose excedente deve ser preservada pela normalizacao de escala (invariante)."""
    rng = np.random.default_rng(1)
    laplace = rng.laplace(size=(1, 20000))
    before = kurtosis(laplace, axis=1, fisher=True)[0]

    resolved = resolve_ambiguities(laplace, domain="distribution")

    after = kurtosis(resolved, axis=1, fisher=True)[0]
    assert abs(before - after) < 0.05
