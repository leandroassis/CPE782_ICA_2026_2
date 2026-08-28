"""Visualizacao qualitativa de distribuicoes: histogramas com referencia gaussiana.

Ver context/TASK_DESCRIPTION.md ("forma dos histogramas reconstruidos").
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from ica.visualization.base import Visualizer

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ica.model import ICAModel


class HistogramVisualizer(Visualizer):
    """Plota histogramas das misturas e das fontes recuperadas, com gaussiana de referencia."""

    def plot(self, model: ICAModel, output_dir: Path) -> list[Path]:
        """Salva um grid PNG com os histogramas de ``model.mixtures_`` e ``model.sources_``.

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

        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
        for i in range(n):
            self._plot_with_gaussian_reference(axes[0, i], model.mixtures_[i], f"mistura {i + 1}")
            self._plot_with_gaussian_reference(
                axes[1, i], model.sources_[i], f"fonte recuperada {i + 1}"
            )
        fig.tight_layout()

        path = output_dir / "histogramas_misturas_vs_fontes.png"
        fig.savefig(path)
        plt.close(fig)
        return [path]

    @staticmethod
    def _plot_with_gaussian_reference(ax: Axes, values: np.ndarray, title: str) -> None:
        """Desenha o histograma normalizado de ``values`` com a gaussiana equivalente sobreposta.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Eixo onde desenhar.
        values : np.ndarray
            Amostras 1D a histografar.
        title : str
            Titulo do subplot.
        """
        ax.hist(values, bins=50, density=True, alpha=0.7)
        x = np.linspace(values.min(), values.max(), 200)
        ax.plot(x, norm.pdf(x, loc=values.mean(), scale=values.std()), linestyle="--")
        ax.set_title(title)
