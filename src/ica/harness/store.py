"""Cache retomavel por celula da grade.

Bloco ``harness/`` de ``.claude/PIPELINE_MAP.md``: "store retomavel com
seeds". Cada celula (sample, run, algoritmo, modo, ...) e serializada em
disco por uma chave estavel; uma segunda chamada com os mesmos parametros
le do cache em vez de recomputar -- permite interromper e retomar uma
grade grande sem perder trabalho ja feito.
"""

import hashlib
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ica.harness.grid import CellResult


def cell_key(sample: str, run: str, algorithm: str, mode: str, **extra: Any) -> str:
    """Chave estavel (hash curto) para uma celula, a partir de seus parametros.

    Parameters
    ----------
    sample, run, algorithm, mode : str
        Identificacao da celula (ver :class:`~ica.harness.grid.CellResult`).
    **extra
        Parametros adicionais que afetam o resultado (ex.: ``sample_size``,
        ``max_iterations``, ``seed``), incluidos na chave para que
        celulas com parametros diferentes nunca colidam no cache.

    Returns
    -------
    str
        Hash SHA-1 truncado (16 caracteres hexadecimais).
    """
    raw = f"{sample}|{run}|{algorithm}|{mode}|{sorted(extra.items())}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class ResumableStore:
    """Cache em disco (pickle) de :class:`~ica.harness.grid.CellResult`, por celula.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Diretorio onde os arquivos ``<hash>.pkl`` sao guardados (criado se
        necessario).
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def has(self, sample: str, run: str, algorithm: str, mode: str, **extra: Any) -> bool:
        """Verifica se ja existe um resultado em cache para esta celula.

        Returns
        -------
        bool
            ``True`` se o arquivo de cache existe.
        """
        return self._path_for_key(cell_key(sample, run, algorithm, mode, **extra)).exists()

    def load(self, sample: str, run: str, algorithm: str, mode: str, **extra: Any) -> CellResult:
        """Le o resultado em cache desta celula.

        Returns
        -------
        CellResult
            O resultado previamente salvo.

        Raises
        ------
        FileNotFoundError
            Se nao houver cache para esta celula (ver :meth:`has`).
        """
        path = self._path_for_key(cell_key(sample, run, algorithm, mode, **extra))
        with path.open("rb") as handle:
            return pickle.load(handle)

    def save(
        self, sample: str, run: str, algorithm: str, mode: str, result: CellResult, **extra: Any
    ) -> Path:
        """Salva ``result`` no cache desta celula.

        Returns
        -------
        pathlib.Path
            Caminho do arquivo escrito.
        """
        path = self._path_for_key(cell_key(sample, run, algorithm, mode, **extra))
        with path.open("wb") as handle:
            pickle.dump(result, handle)
        return path

    def get_or_compute(
        self,
        sample: str,
        run: str,
        algorithm: str,
        mode: str,
        compute: Callable[[], CellResult],
        **extra: Any,
    ) -> CellResult:
        """Le do cache se existir; senao computa via ``compute()`` e salva.

        Parameters
        ----------
        sample, run, algorithm, mode : str
            Identificacao da celula.
        compute : callable
            Funcao sem argumentos que roda a celula do zero (tipicamente
            ``lambda: run_cell(sample, run, algorithm, mode, ...)``).
        **extra
            Ver :func:`cell_key`.

        Returns
        -------
        CellResult
            Resultado (do cache ou recem-computado).
        """
        if self.has(sample, run, algorithm, mode, **extra):
            return self.load(sample, run, algorithm, mode, **extra)
        result = compute()
        self.save(sample, run, algorithm, mode, result, **extra)
        return result
