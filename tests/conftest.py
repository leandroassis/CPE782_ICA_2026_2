"""Fixtures compartilhadas: geracao de fontes sinteticas e avaliacao de recuperacao.

Ver context/DEVELOPMENT_GUIDELINES.md, Secao 5: os testes usam fontes
sinteticas com fundo de verdade conhecido (nunca os dados reais de
``data/``, que nao tem matriz de mistura ``A`` disponivel), e comparam
recuperacao por correlacao maxima -- nao por igualdade direta -- para
respeitar as ambiguidades de escala/sinal/permutacao (ICA_BACKGROUND.md,
Secao 1.3).
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from ica.data.base import DataTemplate


@pytest.fixture
def rng() -> np.random.Generator:
    """Gerador de numeros aleatorios com semente fixa, para testes reprodutiveis."""
    return np.random.default_rng(42)


class _ArrayDataTemplate(DataTemplate):
    """DataTemplate de teste que envolve uma matriz ja em memoria (sem tocar em disco)."""

    def __init__(self, X: np.ndarray) -> None:
        super().__init__(run="synthetic", data_root=Path("."))
        self._X = X

    def load(self) -> np.ndarray:
        return self._X

    @property
    def n_mixtures(self) -> int:
        return self._X.shape[0]

    @classmethod
    def discover_runs(cls, data_root: Path) -> list[str]:
        return ["synthetic"]


@pytest.fixture
def array_data_template():
    """Fabrica de um DataTemplate de teste que apenas devolve uma matriz X ja pronta.

    Returns
    -------
    callable
        Funcao ``(X) -> DataTemplate`` para uso em testes de integracao
        que exercitam :class:`~ica.model.ICAModel` de ponta a ponta sem
        depender de arquivos em disco.
    """
    return _ArrayDataTemplate


@pytest.fixture
def make_sources():
    """Fabrica de matrizes de fontes independentes sinteticas.

    Returns
    -------
    callable
        Funcao ``(kinds, n_samples, rng) -> np.ndarray`` que gera uma
        fonte i.i.d. por entrada de ``kinds`` (``"laplace"``,
        ``"uniform"`` ou ``"gaussian"``), padronizada para media zero e
        variancia unitaria, empilhadas em uma matriz ``(len(kinds),
        n_samples)``.
    """

    def _make_sources(
        kinds: list[str], n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        rows = []
        for kind in kinds:
            if kind == "laplace":
                raw = rng.laplace(loc=0.0, scale=1.0, size=n_samples)
            elif kind == "uniform":
                raw = rng.uniform(low=-1.0, high=1.0, size=n_samples)
            elif kind == "gaussian":
                raw = rng.normal(loc=0.0, scale=1.0, size=n_samples)
            else:
                raise ValueError(f"Tipo de fonte desconhecido: {kind!r}")
            rows.append((raw - raw.mean()) / raw.std())
        return np.vstack(rows)

    return _make_sources


@pytest.fixture
def make_mixing_matrix():
    """Fabrica de matrizes de mistura ``A`` aleatorias e bem-condicionadas.

    Returns
    -------
    callable
        Funcao ``(rng, n, max_condition_number=10.0) -> np.ndarray`` que
        sorteia uma matriz ``n x n`` ate encontrar uma com numero de
        condicao abaixo do limite informado (evita misturas quase
        singulares, que inviabilizariam a separacao mesmo com o
        algoritmo correto).
    """

    def _make_mixing_matrix(
        rng: np.random.Generator, n: int, max_condition_number: float = 10.0
    ) -> np.ndarray:
        for _ in range(100):
            candidate = rng.normal(size=(n, n))
            if np.linalg.cond(candidate) < max_condition_number:
                return candidate
        raise RuntimeError("Nao foi possivel gerar uma matriz de mistura bem-condicionada.")

    return _make_mixing_matrix


@pytest.fixture
def best_match_correlation():
    """Fabrica da metrica de recuperacao invariante a escala/sinal/permutacao.

    Returns
    -------
    callable
        Funcao ``(S_true, S_hat) -> float`` que casa cada fonte
        verdadeira com a fonte recuperada mais correlacionada (em modulo,
        via algoritmo hungaro sobre a matriz de correlacoes absolutas) e
        retorna a media das correlacoes casadas. Um valor proximo de 1.0
        indica recuperacao bem-sucedida, independentemente de reordenacao,
        troca de sinal ou reescala das componentes.
    """

    def _best_match_correlation(S_true: np.ndarray, S_hat: np.ndarray) -> float:
        n = S_true.shape[0]
        absolute_correlation = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                absolute_correlation[i, j] = abs(np.corrcoef(S_true[i], S_hat[j])[0, 1])
        row_ind, col_ind = linear_sum_assignment(-absolute_correlation)
        return float(absolute_correlation[row_ind, col_ind].mean())

    return _best_match_correlation
