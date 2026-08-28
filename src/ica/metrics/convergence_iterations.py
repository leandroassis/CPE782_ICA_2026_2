"""Metrica: numero de iteracoes ate a convergencia (ou o limite maximo).

Ver context/TASK_DESCRIPTION.md ("quantidade de iteracoes necessarias para convergir").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ica.metrics.base import Metric

if TYPE_CHECKING:
    from ica.model import ICAModel


class ConvergenceIterations(Metric):
    """Numero de iteracoes que o algoritmo executou ate convergir (ou parar)."""

    name = "convergence_iterations"

    def compute(self, model: ICAModel) -> float:
        """Le ``model.n_iterations_``.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado.

        Returns
        -------
        float
            Numero de iteracoes executadas.
        """
        return float(model.n_iterations_)
