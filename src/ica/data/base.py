"""Interface base para carregadores de amostras do trabalho (DataTemplate).

Bloco ``io/`` de ``.claude/PIPELINE_MAP.md``: adapters dominio <-> ``SignalMatrix``.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ica.interfaces import SignalMatrix


class DataTemplate(ABC):
    """Carrega uma amostra (imagens, distribuicoes ou audio) como um :class:`SignalMatrix`.

    Cada implementacao concreta so conhece o formato de arquivo especifico
    da sua amostra; o restante do pipeline (pre-processamento, algoritmos de
    ICA) so depende desta interface comum. Run disponiveis, numero de
    misturas e (quando aplicavel) tamanhos amostrais nunca sao hardcoded --
    sao descobertos a partir do sistema de arquivos.

    Layout de dados (chave ja liberada): misturas em
    ``<data_root>/<run>/...`` (``data_root`` tipicamente ``data/mix/<tipo>``)
    espelhado pelo gabarito em ``<groundtruth_root>/<run>/...``
    (``groundtruth_root`` tipicamente ``data/groundtruth/<tipo>``) --
    :meth:`load_ground_truth`. Nem todo run tem gabarito completo (ex.:
    alguns runs de distribuicao nao tem a matriz `A` verdadeira) --
    :meth:`load_ground_truth` e sempre tolerante a arquivos ausentes,
    devolvendo ``None`` no lugar do que faltar.

    Parameters
    ----------
    run : str
        Identificador do run a carregar (ex.: ``"run1"``).
    data_root : pathlib.Path
        Diretorio raiz onde as misturas estao organizadas (tipicamente
        ``data/mix/<tipo>/``).
    """

    def __init__(self, run: str, data_root: Path) -> None:
        self.run = run
        self.data_root = Path(data_root)

    @abstractmethod
    def load(self) -> SignalMatrix:
        """Carrega a amostra do disco como um :class:`SignalMatrix` de misturas.

        Returns
        -------
        SignalMatrix
            Misturas, ``data`` shape ``(n_misturas, n_amostras)``, dtype
            ``float64``, com os metadados de domínio necessarios para
            de-matrixizar de volta a forma nativa.
        """

    @property
    @abstractmethod
    def n_mixtures(self) -> int:
        """Numero de misturas (fontes) desta amostra, descoberto do disco.

        Returns
        -------
        int
            Numero de colunas/arquivos de mistura disponiveis para este run.
        """

    @classmethod
    @abstractmethod
    def discover_runs(cls, data_root: Path) -> list[str]:
        """Lista os runs disponiveis em ``data_root`` para este tipo de amostra.

        Parameters
        ----------
        data_root : pathlib.Path
            Diretorio raiz onde as amostras estao organizadas.

        Returns
        -------
        list of str
            Identificadores de run disponiveis, ordenados.
        """

    def load_ground_truth(
        self, groundtruth_root: Path
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Carrega a matriz de mistura verdadeira e as fontes verdadeiras deste run, se existirem.

        Implementacao padrao: nenhum gabarito (``(None, None)``). Cada
        subclasse concreta sobrescreve com a descoberta de arquivos do seu
        dominio -- ver :meth:`~ica.data.image_template.ImageTemplate.load_ground_truth`,
        :meth:`~ica.data.audio_template.AudioTemplate.load_ground_truth`,
        :meth:`~ica.data.distribution_template.DistributionTemplate.load_ground_truth`.

        Parameters
        ----------
        groundtruth_root : pathlib.Path
            Diretorio raiz do gabarito para este tipo de amostra
            (tipicamente ``data/groundtruth/<tipo>``).

        Returns
        -------
        tuple of (np.ndarray or None, np.ndarray or None)
            ``(mixing_matrix_true, sources_true)``. Qualquer um dos dois
            pode ser ``None`` quando o arquivo correspondente nao existe
            para este run -- nunca levanta excecao por ausencia de gabarito.
        """
        return None, None

    @staticmethod
    def _discover_runs_with_matching_file(
        data_root: Path, predicate: Callable[[Path], bool]
    ) -> list[str]:
        """Lista subdiretorios de ``data_root`` com ao menos um arquivo casando ``predicate``.

        Helper compartilhado pelas implementacoes concretas de
        :meth:`discover_runs`, que diferem apenas no criterio (nome de
        arquivo esperado) usado para reconhecer um run valido.

        Parameters
        ----------
        data_root : pathlib.Path
            Diretorio raiz a inspecionar.
        predicate : callable
            Funcao ``(pathlib.Path) -> bool`` aplicada a cada arquivo
            dentro de cada subdiretorio.

        Returns
        -------
        list of str
            Nomes dos subdiretorios que contem ao menos um arquivo
            casando com ``predicate``, ordenados.
        """
        data_root = Path(data_root)
        if not data_root.exists():
            return []
        runs = [
            entry.name
            for entry in data_root.iterdir()
            if entry.is_dir() and any(predicate(f) for f in entry.iterdir())
        ]
        return sorted(runs)
