"""Adapter de ``DataTemplate`` para um ``SignalMatrix`` ja construido em memoria.

Usado por ``ica.harness.grid`` para os modos de condicionamento B/C de
imagem (a matriz ja vem pronta de
:meth:`~ica.data.image_template.ImageTemplate.mode_b_plane_matrices`/
:meth:`~ica.data.image_template.ImageTemplate.mode_c_matrix`, nao de um
arquivo) e por ``ica.harness.synthetic`` para injetar fontes sinteticas no
mesmo fluxo (``io/`` -> ``ICAModel``) usado pelos dados reais.
"""

from pathlib import Path

import numpy as np

from ica.data.base import DataTemplate
from ica.interfaces import SignalMatrix


class InMemorySignalMatrixTemplate(DataTemplate):
    """Envolve um :class:`~ica.interfaces.SignalMatrix` ja pronto como um ``DataTemplate``.

    Parameters
    ----------
    signal_matrix : SignalMatrix
        Misturas ja construidas (ex.: um modo de condicionamento de
        imagem, ou uma mistura sintetica).
    ground_truth : tuple of (np.ndarray or None, np.ndarray or None), optional
        ``(mixing_matrix_true, sources_true)`` a devolver em
        :meth:`load_ground_truth`, quando disponivel (ex.: gerador
        sintetico). Por padrao, ``(None, None)``.
    run : str, default="in_memory"
        Rotulo do run, so para logging/depuracao.
    """

    def __init__(
        self,
        signal_matrix: SignalMatrix,
        ground_truth: tuple[np.ndarray | None, np.ndarray | None] | None = None,
        run: str = "in_memory",
    ) -> None:
        super().__init__(run=run, data_root=Path("."))
        self._signal_matrix = signal_matrix
        self._ground_truth = ground_truth or (None, None)

    def load(self) -> SignalMatrix:
        """Devolve o :class:`SignalMatrix` recebido no construtor.

        Returns
        -------
        SignalMatrix
            O mesmo objeto passado a este template.
        """
        return self._signal_matrix

    @property
    def n_mixtures(self) -> int:
        """Numero de linhas do ``SignalMatrix`` envolvido.

        Returns
        -------
        int
            ``signal_matrix.n_signals``.
        """
        return self._signal_matrix.n_signals

    def load_ground_truth(
        self, groundtruth_root: Path
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Devolve o gabarito recebido no construtor, ignorando ``groundtruth_root``.

        Parameters
        ----------
        groundtruth_root : pathlib.Path
            Ignorado -- o gabarito, quando existe, ja foi passado ao
            construtor.

        Returns
        -------
        tuple of (np.ndarray or None, np.ndarray or None)
            O gabarito recebido no construtor.
        """
        return self._ground_truth

    @classmethod
    def discover_runs(cls, data_root: Path) -> list[str]:
        """Nao aplicavel -- este template nao descobre runs em disco.

        Returns
        -------
        list of str
            Sempre vazia.
        """
        return []
