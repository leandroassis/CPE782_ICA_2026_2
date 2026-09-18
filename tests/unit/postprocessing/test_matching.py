"""Testes unitarios para ica.postprocessing.matching (skill ica-evaluation, Secao 2)."""

import numpy as np

from ica.postprocessing.matching import best_match_correlation, hungarian_match


def test_hungarian_match_recovers_permuted_and_sign_flipped_sources(rng, make_sources):
    """hungarian_match deve casar corretamente mesmo com permutacao e sinal trocados."""
    S = make_sources(["laplace", "uniform", "gaussian"], 5000, rng)
    permuted_and_flipped = np.stack([-S[2], S[0], -S[1]])

    match = hungarian_match(S, permuted_and_flipped)

    assert list(match.reference_indices) == [0, 1, 2]
    assert list(match.matched_indices) == [1, 2, 0]
    assert match.mean_correlation > 0.99


def test_hungarian_match_signs_flag_anti_correlated_pairs(rng, make_sources):
    """Signs deve ser -1 exatamente para os pares casados que vieram invertidos."""
    S = make_sources(["laplace", "uniform", "gaussian"], 5000, rng)
    permuted_and_flipped = np.stack([-S[2], S[0], -S[1]])

    match = hungarian_match(S, permuted_and_flipped)

    # candidatos: 0=-S[2], 1=S[0], 2=-S[1] -- so a referencia 0 (S[0]) casa
    # com um candidato NAO invertido (candidate 1); as outras duas invertem.
    signs_by_reference = dict(zip(match.reference_indices.tolist(), match.signs.tolist()))
    assert signs_by_reference[0] == 1.0  # S[0] <-> S[0] (candidate 1)
    assert signs_by_reference[1] == -1.0  # S[1] <-> -S[1] (candidate 2)
    assert signs_by_reference[2] == -1.0  # S[2] <-> -S[2] (candidate 0)


def test_best_match_correlation_is_near_one_for_identical_sources(rng, make_sources):
    """best_match_correlation deve ser ~1.0 quando as fontes sao identicas."""
    S = make_sources(["laplace", "uniform"], 3000, rng)
    assert best_match_correlation(S, S) > 0.999


def test_best_match_correlation_is_low_for_independent_noise(rng):
    """best_match_correlation deve ser baixo para sinais independentes (sem relacao)."""
    a = rng.normal(size=(2, 5000))
    b = rng.normal(size=(2, 5000))
    assert best_match_correlation(a, b) < 0.2
