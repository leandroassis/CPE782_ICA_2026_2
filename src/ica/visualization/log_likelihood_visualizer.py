"""Visualizacao de convergencia: log-verossimilhanca media a cada iteracao.

Ver context/ICA_BACKGROUND.md, Secao 3.2; context/TASK_DESCRIPTION.md
("quantidade de iteracoes necessarias para convergir").
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from ica.visualization.base import Visualizer

if TYPE_CHECKING:
    from ica.model import ICAModel


class LogLikelihoodVisualizer(Visualizer):
    """Plota a log-verossimilhanca media -- ``(1/T) log L(B)`` -- ao longo das iteracoes.

    Le ``model.log_likelihood_history_``, calculado a cada passo por
    :meth:`ICAAlgorithm._log_likelihood
    <ica.algorithms.base.ICAAlgorithm._log_likelihood>` (ICA_BACKGROUND.md,
    Secao 3.2). Uma curva crescente/estabilizando indica que o algoritmo
    esta de fato ascendendo o objetivo; quedas ou oscilacoes fortes
    indicam instabilidade numerica (ver, por exemplo, a nota sobre
    ``NaturalGradientICA`` e nao-linearidades nao-limitadas em
    ICA_BACKGROUND.md, Secao 4.2).
    """

    def plot(self, model: ICAModel, output_dir: Path) -> list[Path]:
        """Salva um PNG com a curva de log-verossimilhanca por iteracao.

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

        history = model.log_likelihood_history_
        iterations = range(1, len(history) + 1)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(iterations, history, marker=".", markersize=3, linewidth=1)
        ax.set_xlabel("iteracao")
        ax.set_ylabel("log-verossimilhanca media  (1/T) log L(B)")
        ax.set_title("Convergencia da log-verossimilhanca")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        path = output_dir / "log_verossimilhanca.png"
        fig.savefig(path)
        plt.close(fig)
        return [path]
