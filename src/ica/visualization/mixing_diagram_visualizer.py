"""Diagrama de mistura: scatter pairwise das misturas e das fontes recuperadas.

Ver context/TASK_DESCRIPTION.md ("diagramas de mistura"). Este diagnostico
nao depende da matriz de mistura A verdadeira -- a passagem de um
diagrama correlacionado (misturas, ex.: paralelogramo/losango, ver
ICA_BACKGROUND.md Secao 1) para eixos alinhados (fontes independentes) e
visivel diretamente nos dados observados e recuperados. Por isso e
produzido para todo run, inclusive os reais, que nao tem gabarito.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from ica.visualization.base import Visualizer

if TYPE_CHECKING:
    from ica.model import ICAModel


class MixingDiagramVisualizer(Visualizer):
    """Plota, para pares de componentes, o scatter das misturas e das fontes recuperadas.

    Parameters
    ----------
    max_pairs : int, default=3
        Numero maximo de pares de componentes a exibir (evita graficos
        gigantes quando ha muitas misturas).
    """

    def __init__(self, max_pairs: int = 3) -> None:
        self.max_pairs = max_pairs

    def plot(self, model: ICAModel, output_dir: Path) -> list[Path]:
        """Salva um grid PNG com scatter de misturas (linha 1) e fontes recuperadas (linha 2).

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado.
        output_dir : pathlib.Path
            Diretorio de saida.

        Returns
        -------
        list of pathlib.Path
            Caminho do PNG gerado.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n = model.mixtures_.shape[0]
        pairs = list(combinations(range(n), 2))[: self.max_pairs]
        n_pairs = max(len(pairs), 1)

        fig, axes = plt.subplots(2, n_pairs, figsize=(3.5 * n_pairs, 7), squeeze=False)
        for column, (i, j) in enumerate(pairs):
            axes[0, column].scatter(model.mixtures_[i], model.mixtures_[j], s=2, alpha=0.4)
            axes[0, column].set_title(f"misturas {i + 1} x {j + 1}")
            axes[1, column].scatter(
                model.sources_[i], model.sources_[j], s=2, alpha=0.4, color="darkorange"
            )
            axes[1, column].set_title(f"fontes recuperadas {i + 1} x {j + 1}")
        fig.tight_layout()

        path = output_dir / "diagrama_de_mistura.png"
        fig.savefig(path)
        plt.close(fig)
        return [path]
