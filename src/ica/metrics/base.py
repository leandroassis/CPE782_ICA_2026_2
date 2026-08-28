"""Interface base para metricas de avaliacao de um ICAModel ajustado.

Ver context/DEVELOPMENT_GUIDELINES.md, Secao 2.6.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ica.model import ICAModel


class Metric(ABC):
    """Uma metrica calculada sobre um ICAModel ja ajustado (apos ``fit()``).

    Attributes
    ----------
    name : str
        Nome usado para indexar o resultado em ``ICAModel.evaluate()``.
    """

    name: str

    @abstractmethod
    def compute(self, model: ICAModel) -> float | np.ndarray:
        """Calcula a metrica a partir do estado de um ICAModel ajustado.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado (``model.fit()`` ja foi chamado).

        Returns
        -------
        float or np.ndarray
            Valor da metrica.
        """
