"""Indice de Amari -- qualidade de separacao contra o gabarito.

Skill ``ica-evaluation``, Secao 4, e ``references/metrics-formulas.md``,
Secao 1. Metrica primaria de qualidade de separacao (com gabarito) e o
desempate de "melhor separacao" em ``ica.harness.selection``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ica.metrics.base import Metric

if TYPE_CHECKING:
    from ica.model import ICAModel


def amari_index(B: np.ndarray, A: np.ndarray) -> float:
    """Calcula o indice de Amari da matriz global ``G = B @ A``.

    ``Amari = 0`` para separacao perfeita (``G`` = permutacao x escala);
    cresce com o vazamento entre fontes. Invariante a permutacao e escala
    de ``B`` -- por isso pode ser calculado sobre o ``B`` cru ou sobre
    qualquer reescala/reordenacao dele (ex.: apos
    :func:`~ica.postprocessing.ambiguity.resolve_ambiguities`), com o mesmo
    resultado.

    Parameters
    ----------
    B : np.ndarray
        Matriz de separacao estimada (tipicamente
        ``ICAModel.full_unmixing_matrix_``, no espaco das misturas
        originais), shape ``(n, n)``.
    A : np.ndarray
        Matriz de mistura verdadeira (gabarito), shape ``(n, n)``.

    Returns
    -------
    float
        Indice de Amari, ``>= 0``.
    """
    G = B @ A
    n = G.shape[0]
    abs_G = np.abs(G)
    row_term = np.sum(np.sum(abs_G, axis=1) / np.max(abs_G, axis=1) - 1.0)
    col_term = np.sum(np.sum(abs_G, axis=0) / np.max(abs_G, axis=0) - 1.0)
    return float((row_term + col_term) / (2.0 * n * (n - 1)))


class AmariIndex(Metric):
    """Indice de Amari de um ``ICAModel`` ajustado, contra o gabarito (quando disponivel).

    Auto-desabilita (``None``) quando ``model.mixing_matrix_true_`` nao foi
    populado (sem ``groundtruth_root`` ou sem gabarito para o run).
    """

    name = "amari_index"

    def compute(self, model: ICAModel) -> float | None:
        """Calcula :func:`amari_index` de ``model``.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado.

        Returns
        -------
        float or None
            O indice de Amari, ou ``None`` sem gabarito disponivel.
        """
        if model.mixing_matrix_true_ is None:
            return None
        return amari_index(model.full_unmixing_matrix_, model.mixing_matrix_true_)
