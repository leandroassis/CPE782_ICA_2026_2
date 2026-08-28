"""Metrica: tempo de execucao do algoritmo de ICA.

Ver context/TASK_DESCRIPTION.md ("tempo de execucao").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ica.metrics.base import Metric

if TYPE_CHECKING:
    from ica.model import ICAModel


class ExecutionTime(Metric):
    """Tempo de execucao do algoritmo de ICA, em segundos."""

    name = "execution_time_seconds"

    def compute(self, model: ICAModel) -> float:
        """Le ``model.elapsed_time_``.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado.

        Returns
        -------
        float
            Tempo de execucao, em segundos.
        """
        return float(model.elapsed_time_)
