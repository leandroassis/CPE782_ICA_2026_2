"""Interface base para visualizacoes qualitativas/quantitativas de um ICAModel ajustado.

Ver context/DEVELOPMENT_GUIDELINES.md, Secao 2.6; context/TASK_DESCRIPTION.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

if TYPE_CHECKING:
    from ica.model import ICAModel


class Visualizer(ABC):
    """Gera e salva figura(s) de diagnostico a partir de um ICAModel ajustado."""

    @abstractmethod
    def plot(self, model: ICAModel, output_dir: Path) -> list[Path]:
        """Renderiza e salva figura(s) sobre o resultado de um ICAModel.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado (``model.fit()`` ja foi chamado).
        output_dir : pathlib.Path
            Diretorio onde as figuras devem ser salvas (criado se
            necessario).

        Returns
        -------
        list of pathlib.Path
            Caminhos dos arquivos efetivamente escritos.
        """
