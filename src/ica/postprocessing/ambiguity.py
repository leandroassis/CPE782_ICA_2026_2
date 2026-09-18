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
    """Aplica sinal -> escala -> ordem cega estavel, nessa ordem.

    ``fix_sign`` roda **antes** de ``fix_scale``, e nao depois (apesar da
    ordem "escala, sinal, ordem" da skill ica-evaluation, Secao 1, que trata
    o sinal como independente da escala escolhida): a decisao de
    :func:`fix_sign` e por assimetria (``skew``), invariante a qualquer
    transformacao afim ``a*Y+b`` com ``a>0`` -- portanto identica antes ou
    depois de ``fix_scale``. A ordem importa, porem, para o *dominio de
    saida*: o rescale de imagem em :func:`fix_scale` estica cada componente
    para ``[0, 1]`` via min-max; multiplicar esse resultado por ``-1``
    (se aplicado depois) joga a componente para ``[-1, 0]``, fora da faixa
    que a figura-vitrine e as metricas perceptuais (PSNR/SSIM, ``MAX=1``)
    assumem -- exibida como preto solido pelo ``imshow(vmin=0, vmax=1)``.
    Aplicando o sinal antes, o min-max de :func:`fix_scale` sempre recalcula
    seu proprio min/max sobre a componente ja com sinal resolvido, entao a
    saida cai sempre dentro da faixa do dominio, nunca fora dela.

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
        Componentes com sinal, escala e ordem resolvidos.
    """
    Y = fix_sign(Y)
    Y = fix_scale(Y, domain)
    order = stable_permutation_order(Y)
    return Y[order]
