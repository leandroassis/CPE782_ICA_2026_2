"""Testes unitarios para ica.harness.selection (skill ica-evaluation, Secao 5)."""

import pytest

from ica.harness.selection import log_likelihood_per_sample, select_best


class _FakeModel:
    def __init__(self, log_likelihood_history):
        self.log_likelihood_history_ = log_likelihood_history


class _FakeCell:
    def __init__(self, log_likelihood_final, amari=None):
        self.model = _FakeModel([0.0, log_likelihood_final])
        self.metrics = {"amari_index": amari}


def test_log_likelihood_per_sample_reads_last_value():
    """Deve ler o ultimo valor da trajetoria de log-verossimilhanca."""
    cell = _FakeCell(log_likelihood_final=-1.23)
    assert log_likelihood_per_sample(cell) == -1.23


def test_select_best_picks_highest_log_likelihood():
    """Sem empate, vence a celula de maior log-L/amostra."""
    worse = _FakeCell(log_likelihood_final=-5.0)
    better = _FakeCell(log_likelihood_final=-1.0)
    assert select_best([worse, better]) is better


def test_select_best_breaks_tie_by_lower_amari():
    """Em empate de log-L, vence a celula de menor indice de Amari."""
    tied_high_amari = _FakeCell(log_likelihood_final=-2.0, amari=0.5)
    tied_low_amari = _FakeCell(log_likelihood_final=-2.0, amari=0.1)
    assert select_best([tied_high_amari, tied_low_amari]) is tied_low_amari


def test_select_best_treats_missing_amari_as_worst_tiebreak():
    """Uma celula sem gabarito (amari=None) so perde o desempate, nunca o criterio primario."""
    no_ground_truth = _FakeCell(log_likelihood_final=-2.0, amari=None)
    with_ground_truth = _FakeCell(log_likelihood_final=-2.0, amari=0.9)
    assert select_best([no_ground_truth, with_ground_truth]) is with_ground_truth


def test_select_best_raises_on_empty_list():
    """Uma lista vazia de celulas deve levantar ValueError."""
    with pytest.raises(ValueError):
        select_best([])
