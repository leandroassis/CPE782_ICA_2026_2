"""Interfaces (Protocols) pequenas e focadas, seguindo o Interface Segregation Principle.

Ver ``context/DEVELOPMENT_GUIDELINES.md``, Secao 3 (Interface Segregation): em
vez de uma unica interface inchada, cada capacidade opcional que uma
implementacao de :class:`~ica.data.base.DataTemplate` pode oferecer e
expressa como um Protocol separado, verificavel em tempo de execucao via
``isinstance``.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Reconstructable(Protocol):
    """Amostras cuja saida vetorial pode ser reconstruida em uma forma visualizavel.

    Implementada por :class:`~ica.data.image_template.ImageTemplate`, que
    sabe reformatar um vetor de pixels serializados de volta em uma imagem
    2D (ou RGB).
    """

    def reconstruct(self, source_vector: np.ndarray) -> np.ndarray:
        """Reformata um vetor de fonte recuperada em uma forma visualizavel.

        Parameters
        ----------
        source_vector : np.ndarray
            Vetor 1D de forma ``(n_amostras,)``, tipicamente uma linha de
            ``ICAModel.sources_``.

        Returns
        -------
        np.ndarray
            Array reformatado, pronto para ser exibido (ex.: ``(H, W)``).
        """
        ...


@runtime_checkable
class Exportable(Protocol):
    """Amostras cuja saida vetorial pode ser exportada de volta a um arquivo de mídia.

    Implementada por :class:`~ica.data.audio_template.AudioTemplate`, que
    sabe converter um vetor de fonte recuperada de volta em um arquivo
    ``.wav``.
    """

    def export(self, signal: np.ndarray, output_path: Path) -> None:
        """Grava um sinal recuperado em disco no formato nativo da amostra.

        Parameters
        ----------
        signal : np.ndarray
            Vetor 1D de forma ``(n_amostras,)`` a ser exportado.
        output_path : pathlib.Path
            Caminho do arquivo de saida.
        """
        ...
