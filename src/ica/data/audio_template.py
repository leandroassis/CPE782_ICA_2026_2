"""Carregador de amostras de audio (DataTemplate)."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from ica.data.base import DataTemplate
from ica.interfaces import SignalMatrix

_FILENAME_PATTERN = re.compile(r"^mixture_(\d+)\.wav$")
_SOURCE_FILENAME_PATTERN = re.compile(r"^source_(\d+)\.wav$")


class AudioTemplate(DataTemplate):
    """Carrega misturas de audio (um arquivo ``.wav`` por microfone/mistura).

    Normaliza as amostras PCM inteiras para ponto flutuante em
    ``[-1, 1]`` em :meth:`load`, e reverte essa normalizacao em
    :meth:`export` (Protocol ``Exportable``, ver ``ica.interfaces``) para
    gravar fontes recuperadas.

    Parameters
    ----------
    run : str
        Identificador do run (ex.: ``"run1"``).
    data_root : pathlib.Path
        Diretorio raiz das amostras de audio (``data/mix/audio``).

    Attributes
    ----------
    sample_rate_ : int or None
        Taxa de amostragem (Hz), definida apos :meth:`load`.
    """

    def __init__(self, run: str, data_root: Path) -> None:
        super().__init__(run=run, data_root=data_root)
        self.sample_rate_: int | None = None

    def _run_dir(self) -> Path:
        return self.data_root / self.run

    def _mixture_paths(self) -> list[Path]:
        candidates = [
            entry for entry in self._run_dir().iterdir() if _FILENAME_PATTERN.match(entry.name)
        ]

        def _mixture_index(path: Path) -> int:
            return int(_FILENAME_PATTERN.match(path.name).group(1))

        return sorted(candidates, key=_mixture_index)

    def load(self) -> SignalMatrix:
        """Le todos os arquivos ``mixture_*.wav`` do run e devolve X normalizado.

        Returns
        -------
        SignalMatrix
            ``data`` shape ``(n_misturas, n_amostras)``, valores em
            ``[-1, 1]``, ``domain="audio"``, ``meta={"sample_rate": ...}``.

        Raises
        ------
        ValueError
            Se as misturas do run tiverem taxas de amostragem ou
            numeros de amostras diferentes entre si.
        """
        signals = []
        sample_rates = []
        for path in self._mixture_paths():
            sample_rate, raw_signal = wavfile.read(path)
            sample_rates.append(sample_rate)
            signals.append(self._normalize(raw_signal))

        if len(set(sample_rates)) > 1:
            raise ValueError(f"Taxas de amostragem inconsistentes entre misturas: {sample_rates}.")
        if len({signal.shape[0] for signal in signals}) > 1:
            raise ValueError("As misturas deste run tem numeros de amostras diferentes.")

        self.sample_rate_ = sample_rates[0]
        return SignalMatrix(
            data=np.vstack(signals),
            domain="audio",
            meta={"sample_rate": self.sample_rate_},
        )

    @staticmethod
    def _normalize(raw_signal: np.ndarray) -> np.ndarray:
        """Converte PCM inteiro para ``float64`` em ``[-1, 1]``.

        Parameters
        ----------
        raw_signal : np.ndarray
            Sinal PCM inteiro lido de um arquivo ``.wav``.

        Returns
        -------
        np.ndarray
            Sinal normalizado, ``float64``.
        """
        info = np.iinfo(raw_signal.dtype)
        return raw_signal.astype(np.float64) / max(abs(int(info.min)), int(info.max))

    def export(self, signal: np.ndarray, output_path: Path) -> None:
        """Grava um sinal recuperado como PCM 16-bit no disco.

        Como a ICA recupera as fontes a menos de uma ambiguidade de
        escala, o sinal e normalizado pelo seu proprio pico antes de
        converter para inteiro, evitando estouro (*clipping*) ou saida
        silenciosa.

        Parameters
        ----------
        signal : np.ndarray
            Vetor 1D de forma ``(n_amostras,)``, tipicamente uma linha de
            ``ICAModel.sources_``.
        output_path : pathlib.Path
            Caminho do arquivo ``.wav`` de saida.

        Raises
        ------
        RuntimeError
            Se chamado antes de :meth:`load` (``sample_rate_`` ainda nao
            definido).
        """
        if self.sample_rate_ is None:
            raise RuntimeError("sample_rate_ ainda nao definido; chame load() antes de export().")
        peak = np.max(np.abs(signal))
        normalized = signal / peak if peak > 0 else signal
        pcm = (normalized * np.iinfo(np.int16).max).astype(np.int16)
        wavfile.write(output_path, self.sample_rate_, pcm)

    @property
    def n_mixtures(self) -> int:
        """Numero de arquivos ``mixture_*.wav`` neste run.

        Returns
        -------
        int
            Numero de misturas.
        """
        return len(self._mixture_paths())

    def load_ground_truth(
        self, groundtruth_root: Path
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Carrega ``A`` verdadeira e as fontes verdadeiras (``source_*.wav``) deste run.

        Parameters
        ----------
        groundtruth_root : pathlib.Path
            Diretorio raiz do gabarito de audio (``data/groundtruth/audio``).

        Returns
        -------
        tuple of (np.ndarray or None, np.ndarray or None)
            ``(mixing_matrix_true, sources_true)``. ``sources_true`` tem
            shape ``(n_fontes, n_amostras)``, normalizado como em
            :meth:`load`. ``None`` no lugar do que faltar.
        """
        run_dir = Path(groundtruth_root) / self.run
        if not run_dir.exists():
            return None, None

        mixing_matrix_true = None
        mix_matrix_paths = sorted(run_dir.glob("mix_matrix*.csv"))
        if mix_matrix_paths:
            mixing_matrix_true = pd.read_csv(mix_matrix_paths[0]).to_numpy(dtype=np.float64)

        source_paths = sorted(
            (p for p in run_dir.glob("source_*.wav") if _SOURCE_FILENAME_PATTERN.match(p.name)),
            key=lambda p: int(_SOURCE_FILENAME_PATTERN.match(p.name).group(1)),
        )
        sources_true = None
        if source_paths:
            sources_true = np.vstack(
                [self._normalize(wavfile.read(p)[1]) for p in source_paths]
            )

        return mixing_matrix_true, sources_true

    @classmethod
    def discover_runs(cls, data_root: Path) -> list[str]:
        """Lista os runs de audio disponiveis (com ao menos um arquivo de mistura).

        Parameters
        ----------
        data_root : pathlib.Path
            Diretorio raiz das amostras de audio (``data/mix/audio``).

        Returns
        -------
        list of str
            Identificadores de run disponiveis, ordenados.
        """
        return cls._discover_runs_with_matching_file(
            data_root, lambda f: bool(_FILENAME_PATTERN.match(f.name))
        )
