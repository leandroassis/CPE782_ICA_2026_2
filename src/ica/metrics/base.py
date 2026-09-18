"""Interface base para metricas de avaliacao de um ICAModel ajustado."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

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
    def compute(self, model: ICAModel) -> Any:
        """Calcula a metrica a partir do estado de um ICAModel ajustado.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado (``model.fit()`` ja foi chamado).

        Returns
        -------
        Any
            Valor da metrica -- tipicamente ``float`` ou ``np.ndarray``,
            mas metricas de validacao (Amari, SIR/SDR, PSNR/SSIM, KS/AD,
            identificacao de familia) podem devolver ``None`` (sem
            gabarito disponivel) ou uma lista de resultados estruturados
            (um por fonte).
        """
