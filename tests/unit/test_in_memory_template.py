"""Testes unitarios para InMemorySignalMatrixTemplate."""

import numpy as np

from ica.data.in_memory_template import InMemorySignalMatrixTemplate
from ica.interfaces import SignalMatrix


def test_load_returns_the_wrapped_signal_matrix():
    """load() deve devolver exatamente o SignalMatrix recebido no construtor."""
    signal_matrix = SignalMatrix(data=np.eye(2), domain="distribution", meta={"k": 1})
    template = InMemorySignalMatrixTemplate(signal_matrix)
    assert template.load() is signal_matrix


def test_n_mixtures_reflects_signal_matrix_rows():
    """n_mixtures deve refletir o numero de linhas do SignalMatrix."""
    signal_matrix = SignalMatrix(data=np.zeros((3, 10)), domain="distribution", meta={})
    template = InMemorySignalMatrixTemplate(signal_matrix)
    assert template.n_mixtures == 3


def test_load_ground_truth_returns_none_by_default():
    """Sem gabarito informado, load_ground_truth deve devolver (None, None)."""
    signal_matrix = SignalMatrix(data=np.zeros((2, 10)), domain="distribution", meta={})
    template = InMemorySignalMatrixTemplate(signal_matrix)
    assert template.load_ground_truth("qualquer/coisa") == (None, None)


def test_load_ground_truth_returns_injected_ground_truth():
    """Com gabarito informado no construtor, load_ground_truth deve devolve-lo."""
    signal_matrix = SignalMatrix(data=np.zeros((2, 10)), domain="distribution", meta={})
    A = np.eye(2)
    S = np.zeros((2, 10))
    template = InMemorySignalMatrixTemplate(signal_matrix, ground_truth=(A, S))
    mixing_matrix_true, sources_true = template.load_ground_truth("ignored")
    assert mixing_matrix_true is A
    assert sources_true is S


def test_discover_runs_is_always_empty():
    """discover_runs nao e aplicavel a este template (nao ha runs em disco)."""
    assert InMemorySignalMatrixTemplate.discover_runs("qualquer/coisa") == []
