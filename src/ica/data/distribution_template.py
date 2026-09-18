"""Carregador de amostras de distribuicoes (DataTemplate)."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ica.data.base import DataTemplate
from ica.interfaces import SignalMatrix

_FILENAME_PATTERN = re.compile(r"^mix_(\d+)_stats\.csv$")


class DistributionTemplate(DataTemplate):
    """Carrega misturas de variaveis aleatorias, com um CSV por tamanho amostral.

    Cada linha do CSV e uma amostra i.i.d.; cada coluna (``misturaN``) e
    uma mistura observada. :meth:`load` transpoe para a convencao
    ``(n_misturas, n_amostras)`` usada em todo o pacote.

    Parameters
    ----------
    run : str
        Identificador do run (ex.: ``"run1"``).
    data_root : pathlib.Path
        Diretorio raiz das amostras de distribuicao (``data/mix/dist``).
    sample_size : int
        Tamanho amostral a carregar (deve corresponder a um arquivo
        ``mix_{sample_size}_stats.csv`` existente no run).
    """

    def __init__(self, run: str, data_root: Path, sample_size: int) -> None:
        super().__init__(run=run, data_root=data_root)
        self.sample_size = sample_size

    def _csv_path(self) -> Path:
        return self.data_root / self.run / f"mix_{self.sample_size}_stats.csv"

    def load(self) -> SignalMatrix:
        """Le o CSV do tamanho amostral configurado e devolve X transposto.

        Returns
        -------
        SignalMatrix
            ``data`` shape ``(n_misturas, sample_size)``, ``domain="distribution"``,
            ``meta={"sample_size": ...}``.

        Raises
        ------
        FileNotFoundError
            Se nao existir um CSV para ``sample_size`` neste run.
        """
        csv_path = self._csv_path()
        if not csv_path.exists():
            available = self.discover_sample_sizes(self.data_root, self.run)
            raise FileNotFoundError(
                f"Nao ha amostra de tamanho {self.sample_size} em {self.run}. "
                f"Tamanhos disponiveis: {available}."
            )
        samples_by_mixture = pd.read_csv(csv_path).to_numpy(dtype=np.float64)
        return SignalMatrix(
            data=samples_by_mixture.T,
            domain="distribution",
            meta={"sample_size": self.sample_size},
        )

    @property
    def n_mixtures(self) -> int:
        """Numero de colunas ``misturaN`` no CSV deste run/tamanho amostral.

        Returns
        -------
        int
            Numero de misturas.
        """
        header = pd.read_csv(self._csv_path(), nrows=0)
        return len(header.columns)

    def load_ground_truth(
        self, groundtruth_root: Path
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Carrega ``A`` verdadeira e as fontes verdadeiras deste run/tamanho amostral.

        Descobre os arquivos por glob, pois o nome da matriz de mistura
        varia por run (``mix_matrix_run{k}.csv``) e nem todo run a tem
        (ex.: ``dist/run6`` so tem fontes verdadeiras, sem ``A``).

        Parameters
        ----------
        groundtruth_root : pathlib.Path
            Diretorio raiz do gabarito de distribuicoes (``data/groundtruth/dist``).

        Returns
        -------
        tuple of (np.ndarray or None, np.ndarray or None)
            ``(mixing_matrix_true, sources_true)``, shapes ``(n, n)`` e
            ``(n, sample_size)``. ``None`` no lugar do que faltar.
        """
        run_dir = Path(groundtruth_root) / self.run
        if not run_dir.exists():
            return None, None

        mixing_matrix_true = None
        mix_matrix_paths = sorted(run_dir.glob("mix_matrix*.csv"))
        if mix_matrix_paths:
            mixing_matrix_true = pd.read_csv(mix_matrix_paths[0]).to_numpy(dtype=np.float64)

        sources_true = None
        sources_path = run_dir / f"sources_{self.sample_size}_stats.csv"
        if sources_path.exists():
            sources_true = pd.read_csv(sources_path).to_numpy(dtype=np.float64).T

        return mixing_matrix_true, sources_true

    @classmethod
    def discover_runs(cls, data_root: Path) -> list[str]:
        """Lista os runs de distribuicao disponiveis (com ao menos um CSV de mistura).

        Parameters
        ----------
        data_root : pathlib.Path
            Diretorio raiz das amostras de distribuicao (``data/mix/dist``).

        Returns
        -------
        list of str
            Identificadores de run disponiveis, ordenados.
        """
        return cls._discover_runs_with_matching_file(
            data_root, lambda f: bool(_FILENAME_PATTERN.match(f.name))
        )

    @staticmethod
    def discover_sample_sizes(data_root: Path, run: str) -> list[int]:
        """Lista os tamanhos amostrais disponiveis para um run especifico.

        Parameters
        ----------
        data_root : pathlib.Path
            Diretorio raiz das amostras de distribuicao (``data/mix/dist``).
        run : str
            Identificador do run.

        Returns
        -------
        list of int
            Tamanhos amostrais disponiveis, em ordem crescente.
        """
        run_dir = Path(data_root) / run
        if not run_dir.exists():
            return []
        sizes = []
        for entry in run_dir.iterdir():
            match = _FILENAME_PATTERN.match(entry.name)
            if match:
                sizes.append(int(match.group(1)))
        return sorted(sizes)
