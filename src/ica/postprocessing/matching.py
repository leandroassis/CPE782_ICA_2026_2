"""Casamento humgaro entre componentes recuperadas e fontes/planos de referencia.

Skill ``ica-evaluation``, Secao 2 e ``references/metrics-formulas.md``, Secao
2: resolve a permutacao (e o sinal, via valor absoluto) so para pontuar --
nunca para reordenar a figura-vitrine (ver
:mod:`ica.postprocessing.ambiguity`). Tambem reaproveitado por
:mod:`ica.postprocessing.regroup` para casar planos RGB por correlacao
espacial: o mesmo problema de atribuicao, so que entre dois conjuntos de
componentes recuperadas em vez de recuperadas-vs-verdadeiras.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchResult:
    """Resultado do casamento hungaro entre duas coleções de sinais.

    Attributes
    ----------
    reference_indices : np.ndarray
        Indices (em ``reference``) de cada par casado.
    matched_indices : np.ndarray
        Indices (em ``candidates``) casados, na mesma ordem de
        ``reference_indices`` -- ``matched_indices[k]`` e o candidato
        casado a ``reference_indices[k]``.
    correlations : np.ndarray
        Correlacao absoluta de cada par casado, mesma ordem.
    mean_correlation : float
        Media de ``correlations`` -- proxima de 1.0 indica bom casamento.
    """

    reference_indices: np.ndarray
    matched_indices: np.ndarray
    correlations: np.ndarray
    mean_correlation: float


def hungarian_match(reference: np.ndarray, candidates: np.ndarray) -> MatchResult:
    """Casa cada sinal de ``reference`` ao sinal mais correlacionado (em modulo) de ``candidates``.

    Implementa ``references/metrics-formulas.md``, Secao 2: custo
    ``C_ij = 1 - |corr(reference_i, candidates_j)|``, resolvido por
    atribuicao otima (``scipy.optimize.linear_sum_assignment``). Resolve
    permutacao e sinal (via ``|.|``) apenas para o calculo -- nao reordena
    nenhuma das duas entradas.

    Parameters
    ----------
    reference : np.ndarray
        Sinais de referencia, shape ``(n_referencia, n_amostras)`` --
        tipicamente as fontes verdadeiras, ou os componentes de um plano
        RGB usado como ancora.
    candidates : np.ndarray
        Sinais candidatos, shape ``(n_candidatos, n_amostras)``.

    Returns
    -------
    MatchResult
        Casamento otimo, com ``len(reference_indices) ==
        min(n_referencia, n_candidatos)``.
    """
    n_reference = reference.shape[0]
    n_candidates = candidates.shape[0]
    correlation = np.zeros((n_reference, n_candidates))
    for i in range(n_reference):
        for j in range(n_candidates):
            correlation[i, j] = abs(np.corrcoef(reference[i], candidates[j])[0, 1])

    reference_indices, matched_indices = linear_sum_assignment(-correlation)
    matched_correlations = correlation[reference_indices, matched_indices]
    return MatchResult(
        reference_indices=reference_indices,
        matched_indices=matched_indices,
        correlations=matched_correlations,
        mean_correlation=float(matched_correlations.mean()),
    )


def best_match_correlation(reference: np.ndarray, candidates: np.ndarray) -> float:
    """Atalho: media das correlacoes do melhor casamento hungaro entre ``reference``/``candidates``.

    Parameters
    ----------
    reference : np.ndarray
        Sinais de referencia, shape ``(n_referencia, n_amostras)``.
    candidates : np.ndarray
        Sinais candidatos, shape ``(n_candidatos, n_amostras)``.

    Returns
    -------
    float
        :attr:`MatchResult.mean_correlation`.
    """
    return hungarian_match(reference, candidates).mean_correlation
