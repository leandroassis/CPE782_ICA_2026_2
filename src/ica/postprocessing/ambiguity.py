"""Resolucao das 3 ambiguidades da ICA: escala, sinal e permutacao.

Skill ``ica-evaluation``, Secao 1. Aplicadas **nesta ordem** sobre as
componentes recuperadas ``Y`` (linhas = componentes, colunas = amostras).
Resolve a ordem **cega** (por nao-gaussianidade decrescente) -- a ordem
"correta" via casamento hungaro (:mod:`ica.postprocessing.matching`) e usada
**apenas** para pontuar em ``ica.metrics``, nunca para reordenar aqui: o que
sai daqui e exatamente o que a figura-vitrine mostra.
"""

import numpy as np
from scipy.stats import kurtosis, skew

from ica.interfaces import Domain

_DEFAULT_SKEW_THRESHOLD = 0.1


def fix_scale(Y: np.ndarray, domain: Domain) -> np.ndarray:
    """Normaliza cada componente a variancia unitaria e rescala pelo dominio de saida.

    A variancia e livre (convencao ``E{s_i^2}=1`` da ICA), entao a faixa
    final e escolha de reconstrucao (skill ica-evaluation, Secao 1):
    imagem -> ``[0, 1]``, audio -> ``[-1, 1]``, distribuicao -> mantida
    padronizada (para a identificacao de familia pos-separacao).

    Parameters
    ----------
    Y : np.ndarray
        Componentes recuperadas, shape ``(n_componentes, n_amostras)``.
    domain : {"image", "distribution", "audio"}
        Dominio de saida.

    Returns
    -------
    np.ndarray
        Componentes reescaladas, mesma shape de ``Y``.
    """
    std = Y.std(axis=1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    unit_variance = Y / std

    if domain == "audio":
        peak = np.max(np.abs(unit_variance), axis=1, keepdims=True)
        peak = np.where(peak > 0, peak, 1.0)
        return unit_variance / peak

    if domain == "image":
        channel_min = unit_variance.min(axis=1, keepdims=True)
        channel_max = unit_variance.max(axis=1, keepdims=True)
        span = np.where(channel_max > channel_min, channel_max - channel_min, 1.0)
        return (unit_variance - channel_min) / span

    return unit_variance


def fix_sign(Y: np.ndarray, skew_threshold: float = _DEFAULT_SKEW_THRESHOLD) -> np.ndarray:
    """Fixa o sinal de cada componente pela assimetria (skill ica-evaluation, Secao 1).

    Componentes **assimetricas** (``|skew| > skew_threshold``) tem o sinal
    escolhido para que a assimetria fique positiva. Componentes
    **simetricas** (Laplaciana, Uniforme, Gaussiana...) tem sinal
    indeterminado por natureza -- mantidas como estao (convencao
    documentada, irrelevante para a percepcao em imagem/audio).

    Parameters
    ----------
    Y : np.ndarray
        Componentes recuperadas, shape ``(n_componentes, n_amostras)``.
    skew_threshold : float, default=0.1
        Limiar de assimetria abaixo do qual a componente e tratada como
        simetrica (sinal nao alterado).

    Returns
    -------
    np.ndarray
        Componentes com sinal fixado, mesma shape de ``Y``.
    """
    skewness = skew(Y, axis=1)
    sign = np.where(np.abs(skewness) > skew_threshold, np.sign(skewness), 1.0)
    sign = np.where(sign == 0, 1.0, sign)
    return Y * sign[:, np.newaxis]


def stable_permutation_order(Y: np.ndarray) -> np.ndarray:
    """Ordem cega, estavel e deterministica por nao-gaussianidade decrescente.

    Usa a curtose excedente em modulo (``|gamma_2|``) como proxy de
    nao-gaussianidade -- mesmo indicador do chaveamento super/sub da ICA-ML
    (skill ica-ml, Secao 5) e da impressao HOS pre-BSS (skill
    bss-assessment, Secao 2). Rotulo arbitrario, porem reproduzivel (skill
    ica-evaluation, Secao 1).

    Parameters
    ----------
    Y : np.ndarray
        Componentes recuperadas, shape ``(n_componentes, n_amostras)``.

    Returns
    -------
    np.ndarray
        Indices que ordenam ``Y`` por ``|curtose excedente|`` decrescente.
    """
    excess_kurtosis = kurtosis(Y, axis=1, fisher=True)
    return np.argsort(-np.abs(excess_kurtosis))


def resolve_ambiguities(Y: np.ndarray, domain: Domain) -> np.ndarray:
    """Aplica escala -> sinal -> ordem cega estavel, nessa ordem (skill ica-evaluation, Secao 1).

    E o que popula ``ICAModel.sources_`` -- portanto tambem o que a
    figura-vitrine mostra. O casamento hungaro contra o gabarito
    (:mod:`ica.postprocessing.matching`) roda a parte, so dentro de
    ``ica.metrics``, e nunca realimenta esta funcao.

    Parameters
    ----------
    Y : np.ndarray
        Componentes recuperadas cruas (``B @ x``), shape
        ``(n_componentes, n_amostras)``.
    domain : {"image", "distribution", "audio"}
        Dominio de saida, usado por :func:`fix_scale`.

    Returns
    -------
    np.ndarray
        Componentes com escala, sinal e ordem resolvidos.
    """
    Y = fix_scale(Y, domain)
    Y = fix_sign(Y)
    order = stable_permutation_order(Y)
    return Y[order]
