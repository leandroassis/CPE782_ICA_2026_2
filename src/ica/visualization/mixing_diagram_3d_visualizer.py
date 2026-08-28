"""Visualizacao 3D: nuvem de pontos das misturas e das fontes recuperadas.

Ver context/TASK_DESCRIPTION.md ("diagramas de mistura"). Generaliza para
n=3 os diagramas 2D de ICA_BACKGROUND.md, Secao 1 (Figs. 7.5-7.9: as
misturas formam uma nuvem correlacionada -- paralelepipedo/losango --
enquanto as fontes recuperadas, sendo independentes, formam uma nuvem
alinhada aos eixos). So produz saida quando ha exatamente 3
misturas/fontes; nos demais casos (2, 4+ ou 9 componentes) uma nuvem 3D
unica nao capturaria a relacao completa entre todos os pares, que ja e
coberta por :class:`~ica.visualization.mixing_diagram_visualizer.MixingDiagramVisualizer`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra a projecao "3d")

from ica.visualization.base import Visualizer

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ica.model import ICAModel


class MixingDiagram3DVisualizer(Visualizer):
    """Plota, quando ha exatamente 3 componentes, a nuvem 3D de misturas e fontes recuperadas.

    Parameters
    ----------
    max_points : int, default=5000
        Numero maximo de pontos plotados por nuvem; se houver mais
        amostras, uma subamostra aleatoria (semente fixa) e usada para
        manter o grafico legivel e leve.
    """

    def __init__(self, max_points: int = 5000) -> None:
        self.max_points = max_points

    def plot(self, model: ICAModel, output_dir: Path) -> list[Path]:
        """Salva um PNG com a nuvem 3D das misturas e das fontes recuperadas.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado.
        output_dir : pathlib.Path
            Diretorio de saida.

        Returns
        -------
        list of pathlib.Path
            Caminho do PNG gerado, ou lista vazia se o modelo nao tiver
            exatamente 3 misturas/fontes (esta visualizacao nao se
            aplica nesse caso).
        """
        if model.mixtures_.shape[0] != 3:
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        indices = self._sample_indices(model.mixtures_.shape[1])

        fig = plt.figure(figsize=(13, 6))
        ax_mixtures = fig.add_subplot(1, 2, 1, projection="3d")
        ax_sources = fig.add_subplot(1, 2, 2, projection="3d")

        self._scatter(ax_mixtures, model.mixtures_, indices, "misturas", "mistura")
        self._scatter(ax_sources, model.sources_, indices, "fontes recuperadas", "fonte")

        fig.subplots_adjust(left=0.03, right=0.97, wspace=0.35)
        path = output_dir / "nuvem_3d_misturas_vs_fontes.png"
        fig.savefig(path)
        plt.close(fig)
        return [path]

    def _sample_indices(self, n_samples: int) -> np.ndarray:
        """Escolhe ate ``max_points`` indices (sem reposicao) para plotar.

        Parameters
        ----------
        n_samples : int
            Numero total de amostras disponiveis.

        Returns
        -------
        np.ndarray
            Indices a plotar, shape ``(min(n_samples, max_points),)``.
        """
        if n_samples <= self.max_points:
            return np.arange(n_samples)
        rng = np.random.default_rng(0)
        return rng.choice(n_samples, size=self.max_points, replace=False)

    @staticmethod
    def _scatter(
        ax: Axes, data: np.ndarray, indices: np.ndarray, title: str, axis_label: str
    ) -> None:
        """Desenha o scatter 3D de ``data[:, indices]`` no eixo informado.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Eixo 3D (``projection="3d"``) onde desenhar.
        data : np.ndarray
            Matriz com exatamente 3 linhas, shape ``(3, n_amostras)``.
        indices : np.ndarray
            Indices de amostras a plotar.
        title : str
            Titulo do subplot.
        axis_label : str
            Prefixo usado nos rotulos dos eixos (ex.: ``"mistura"``).
        """
        ax.scatter(data[0, indices], data[1, indices], data[2, indices], s=2, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel(f"{axis_label} 1")
        ax.set_ylabel(f"{axis_label} 2")
        ax.set_zlabel(f"{axis_label} 3", labelpad=8)
