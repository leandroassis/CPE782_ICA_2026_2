"""Metrica: nao-gaussianidade (curtose de Fisher) de cada fonte recuperada.

Ver context/ICA_BACKGROUND.md, Secao 1.2; context/TASK_DESCRIPTION.md
("metricas de nao gaussianidade").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import kurtosis

from ica.metrics.base import Metric

if TYPE_CHECKING:
    from ica.model import ICAModel


class NonGaussianityScore(Metric):
    """Curtose de Fisher (excesso de curtose) de cada fonte recuperada.

    Valores positivos indicam fontes supergaussianas, negativos indicam
    subgaussianas e valores proximos de zero indicam gaussianidade
    residual -- um sinal de separacao malsucedida, ja que fontes
    gaussianas sao inseparaveis (ICA_BACKGROUND.md, Secao 1.2).
    """

    name = "non_gaussianity_kurtosis"

    def compute(self, model: ICAModel) -> np.ndarray:
        """Calcula a curtose de Fisher de cada linha de ``model.sources_``.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado.

        Returns
        -------
        np.ndarray
            Vetor de curtoses, shape ``(n_componentes,)``.
        """
        return kurtosis(model.sources_, axis=1, fisher=True)
