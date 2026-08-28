"""Visualizacao qualitativa de imagens: grade de misturas vs. fontes recuperadas.

Ver context/TASK_DESCRIPTION.md ("clareza das imagens recuperadas").
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from ica.visualization.base import Visualizer

if TYPE_CHECKING:
    from ica.data.image_template import ImageTemplate
    from ica.model import ICAModel


class ImageVisualizer(Visualizer):
    """Plota lado a lado as imagens de mistura originais e as fontes recuperadas.

    Parameters
    ----------
    data : ImageTemplate
        Template da amostra de imagem, usado para reformatar vetores
        planos em imagens 2D via ``reconstruct`` (Protocol
        ``Reconstructable``, ver ``ica.interfaces``).
    """

    def __init__(self, data: ImageTemplate) -> None:
        self._data = data

    def plot(self, model: ICAModel, output_dir: Path) -> list[Path]:
        """Salva um grid PNG com as misturas (linha 1) e as fontes recuperadas (linha 2).

        Quando a amostra e RGB (``data.is_rgb_``), tambem salva um
        composto colorido best-effort (ver :meth:`_save_rgb_composites`).

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado.
        output_dir : pathlib.Path
            Diretorio de saida.

        Returns
        -------
        list of pathlib.Path
            Caminhos dos arquivos gerados.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        n = model.mixtures_.shape[0]

        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
        for i in range(n):
            axes[0, i].imshow(self._data.reconstruct(model.mixtures_[i]), cmap="gray")
            axes[0, i].set_title(f"mistura {i + 1}")
            axes[0, i].axis("off")
            axes[1, i].imshow(self._data.reconstruct(model.sources_[i]), cmap="gray")
            axes[1, i].set_title(f"fonte recuperada {i + 1}")
            axes[1, i].axis("off")
        fig.tight_layout()

        grid_path = output_dir / "imagens_misturas_vs_fontes.png"
        fig.savefig(grid_path)
        plt.close(fig)
        written = [grid_path]

        if getattr(self._data, "is_rgb_", False) and n >= 9:
            written.append(self._save_rgb_composites(model, output_dir))

        return written

    def _save_rgb_composites(self, model: ICAModel, output_dir: Path) -> Path:
        """Salva um painel comparando o composto RGB (best-effort) antes/depois.

        Assume o agrupamento consecutivo por imagem (colunas 1-3, 4-6,
        7-9) descrito em ``ImageTemplate.reconstruct_rgb_triplet`` -- uma
        convencao cosmetica, nao verificavel a partir dos dados de
        mistura. O grid grayscale salvo por :meth:`plot` continua sendo a
        verificacao qualitativa autoritativa.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado.
        output_dir : pathlib.Path
            Diretorio de saida.

        Returns
        -------
        pathlib.Path
            Caminho do PNG gerado.
        """
        fig, axes = plt.subplots(2, 3, figsize=(9, 6), squeeze=False)
        for group in range(3):
            rows = slice(group * 3, group * 3 + 3)
            mixture_rgb = self._data.reconstruct_rgb_triplet(list(model.mixtures_[rows]))
            source_rgb = self._data.reconstruct_rgb_triplet(list(model.sources_[rows]))
            axes[0, group].imshow(mixture_rgb)
            axes[0, group].set_title(f"mistura RGB {group + 1}")
            axes[0, group].axis("off")
            axes[1, group].imshow(source_rgb)
            axes[1, group].set_title(f"fonte RGB {group + 1} (best-effort)")
            axes[1, group].axis("off")
        fig.tight_layout()

        path = output_dir / "imagens_rgb_composto.png"
        fig.savefig(path)
        plt.close(fig)
        return path
