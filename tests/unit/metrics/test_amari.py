"""Testes unitarios para ica.metrics.amari (skill ica-evaluation, Secao 4)."""

import numpy as np

from ica.metrics.amari import AmariIndex, amari_index


def test_amari_index_is_zero_for_perfect_separation():
    """G = permutacao x escala -> Amari = 0."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.linalg.inv(A)  # separacao perfeita: G = I
    assert amari_index(B, A) < 1e-10


def test_amari_index_is_invariant_to_permutation_and_scale():
    """Amari deve ser o mesmo se B for reescalada/permutada por linha."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.linalg.inv(A)
    B_scaled_and_permuted = np.array([5.0 * B[1], -2.0 * B[0]])
    assert abs(amari_index(B, A) - amari_index(B_scaled_and_permuted, A)) < 1e-10


def test_amari_index_grows_with_leakage_between_sources():
    """Uma separacao pior (mais vazamento) deve ter Amari maior."""
    A = np.eye(2)
    good_B = np.eye(2)
    bad_B = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert amari_index(bad_B, A) > amari_index(good_B, A)


class _FakeModel:
    def __init__(self, full_unmixing_matrix, mixing_matrix_true):
        self.full_unmixing_matrix_ = full_unmixing_matrix
        self.mixing_matrix_true_ = mixing_matrix_true


def test_amari_metric_returns_none_without_ground_truth():
    """AmariIndex.compute deve devolver None sem mixing_matrix_true_."""
    model = _FakeModel(np.eye(2), None)
    assert AmariIndex().compute(model) is None


def test_amari_metric_computes_index_with_ground_truth():
    """AmariIndex.compute deve delegar a amari_index quando ha gabarito."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    model = _FakeModel(np.linalg.inv(A), A)
    assert AmariIndex().compute(model) < 1e-10
