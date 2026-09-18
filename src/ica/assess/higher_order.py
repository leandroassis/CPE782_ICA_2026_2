"""Impressao de estatisticas de ordem superior (HOS) pre-BSS.

Skill ``bss-assessment``, Secao 2, com apoio de ``references/higher-order-theory.md``.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis, skew

# E{G_i(nu)} para nu ~ N(0,1), constantes padrao (Hyvarinen), usadas pela
# aproximacao robusta da negentropia por funcoes de contraste.
_E_LOG_COSH_STANDARD_NORMAL = 0.37457
_E_NEG_GAUSSIAN_KERNEL_STANDARD_NORMAL = -1.0 / np.sqrt(2.0)
# Pesos k_i padrao (Hyvarinen & Oja, 2000) para G1(u)=log cosh(u), G2(u)=-exp(-u^2/2).
_CONTRAST_WEIGHT_LOG_COSH = 36.0 / (8.0 * np.sqrt(3.0) - 9.0)
_CONTRAST_WEIGHT_GAUSSIAN_KERNEL = 1.0 / (2.0 - 6.0 / np.pi)


@dataclass
class HigherOrderStatistics:
    """Impressao HOS de um unico canal.

    Attributes
    ----------
    skewness : float
        Assimetria ``gamma_1``.
    excess_kurtosis : float
        Curtose excedente ``gamma_2``; ``>0`` supergaussiana, ``<0``
        subgaussiana -- mesmo indicador do chaveamento da ICA-ML (skill
        ica-ml, Secao 5).
    negentropy_cumulant : float
        Aproximacao classica da negentropia (sensivel a outliers pela
        curtose).
    negentropy_contrast : float
        Aproximacao robusta por funcoes de contraste (``log cosh`` e
        gaussiana negativa).
    """

    skewness: float
    excess_kurtosis: float
    negentropy_cumulant: float
    negentropy_contrast: float


def negentropy_cumulant_approximation(y: np.ndarray) -> float:
    """``J(y) ~ (1/12) E{y^3}^2 + (1/48) gamma_2^2`` (aproximacao classica por cumulantes).

    Parameters
    ----------
    y : np.ndarray
        Canal padronizado ou nao, shape ``(n_amostras,)``.

    Returns
    -------
    float
        Negentropia aproximada, ``>= 0`` no limite assintotico (pode ser
        levemente negativa por ruido amostral em ``T`` pequeno).
    """
    standardized = (y - y.mean()) / y.std()
    third_moment = np.mean(standardized**3)
    excess_kurt = kurtosis(standardized, fisher=True)
    return float((third_moment**2) / 12.0 + (excess_kurt**2) / 48.0)


def negentropy_contrast_approximation(y: np.ndarray) -> float:
    """``J(y) ~ k1 (E{G1(y)}-E{G1(nu)})^2 + k2 (E{G2(y)}-E{G2(nu)})^2`` (aproximacao robusta).

    ``G1(u) = log cosh(u)``, ``G2(u) = -exp(-u^2/2)``, ``nu`` gaussiana
    padrao -- a mesma familia de nao-quadraticas usada pelo FastICA por
    negentropia (Hyvarinen & Oja, 2000), mais estavel a outliers que a
    aproximacao por cumulantes.

    Parameters
    ----------
    y : np.ndarray
        Canal, shape ``(n_amostras,)``.

    Returns
    -------
    float
        Negentropia aproximada, ``>= 0``.
    """
    standardized = (y - y.mean()) / y.std()
    g1 = np.mean(np.log(np.cosh(standardized)))
    g2 = np.mean(-np.exp(-0.5 * standardized**2))
    term1 = _CONTRAST_WEIGHT_LOG_COSH * (g1 - _E_LOG_COSH_STANDARD_NORMAL) ** 2
    term2 = _CONTRAST_WEIGHT_GAUSSIAN_KERNEL * (g2 - _E_NEG_GAUSSIAN_KERNEL_STANDARD_NORMAL) ** 2
    return float(term1 + term2)


def higher_order_statistics(y: np.ndarray) -> HigherOrderStatistics:
    """Calcula a impressao HOS completa de um canal.

    Parameters
    ----------
    y : np.ndarray
        Canal, shape ``(n_amostras,)``.

    Returns
    -------
    HigherOrderStatistics
        Assimetria, curtose excedente e as duas aproximacoes de negentropia.
    """
    return HigherOrderStatistics(
        skewness=float(skew(y)),
        excess_kurtosis=float(kurtosis(y, fisher=True)),
        negentropy_cumulant=negentropy_cumulant_approximation(y),
        negentropy_contrast=negentropy_contrast_approximation(y),
    )


def higher_order_battery(X: np.ndarray) -> list[HigherOrderStatistics]:
    """Roda :func:`higher_order_statistics` por canal (linha) de ``X``.

    Parameters
    ----------
    X : np.ndarray
        Misturas ou componentes, shape ``(n_canais, n_amostras)``.

    Returns
    -------
    list of HigherOrderStatistics
        Um resultado por canal, na ordem das linhas de ``X``.
    """
    return [higher_order_statistics(row) for row in X]
