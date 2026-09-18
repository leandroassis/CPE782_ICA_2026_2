"""Identificacao da familia de distribuicao de uma componente recuperada.

Skill ``ica-evaluation``, Secao 3, e ``references/metrics-formulas.md``,
Secao 7. Feita **sobre a componente recuperada**, nunca sobre a mistura
(TLC -- misturas gaussianizam, skill bss-assessment). 6 candidatas:
gaussiana, uniforme, Laplaciana, exponencial, Rayleigh, qui-quadrado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from ica.metrics.base import Metric

if TYPE_CHECKING:
    from ica.model import ICAModel

# Familias simetricas: sign-invariantes, ajustadas direto sobre a
# componente padronizada. Assimetricas: exigem o sinal correto (fixado
# pela assimetria) antes do ajuste, pois tem suporte de um lado so.
_SYMMETRIC_FAMILIES = {
    "gaussiana": stats.norm,
    "uniforme": stats.uniform,
    "laplaciana": stats.laplace,
}
_ASYMMETRIC_FAMILIES = {
    "exponencial": stats.expon,
    "rayleigh": stats.rayleigh,
    "qui-quadrado": stats.chi2,
}
_ALL_FAMILIES = {**_SYMMETRIC_FAMILIES, **_ASYMMETRIC_FAMILIES}


@dataclass
class FamilyFitResult:
    """Ajuste de uma familia candidata a uma componente."""

    name: str
    ks_statistic: float
    parameters: tuple


@dataclass
class FamilyIdentificationResult:
    """Resultado completo da identificacao de familia de uma componente.

    Attributes
    ----------
    best, runner_up : FamilyFitResult
        Melhor e segunda-melhor familia por bondade de ajuste (menor KS).
    ranking : list of FamilyFitResult
        Todas as 6 candidatas, ordenadas por KS crescente.
    mean, std : float
        Media/desvio-padrao de ``y`` usados para padronizar antes do
        ajuste -- necessarios para reconstruir a densidade em escala
        original (:func:`family_pdf`).
    sign : float
        Sinal (``+-1``) aplicado antes de ajustar as familias assimetricas
        (fixado pela assimetria de ``y`` padronizado).
    """

    best: FamilyFitResult
    runner_up: FamilyFitResult
    ranking: list[FamilyFitResult]
    mean: float
    std: float
    sign: float


def identify_family(y: np.ndarray) -> FamilyIdentificationResult:
    """Identifica a familia de distribuicao mais provavel de uma componente recuperada.

    Procedimento (skill ica-evaluation, Secao 3): padroniza a componente;
    para as 3 familias assimetricas, fixa o sinal pela assimetria (elas
    exigem suporte positivo); ajusta as 6 candidatas por MLE
    (``scipy.stats.<dist>.fit``); ranqueia por Kolmogorov-Smirnov.

    Parameters
    ----------
    y : np.ndarray
        Componente recuperada, shape ``(n_amostras,)``.

    Returns
    -------
    FamilyIdentificationResult
        Melhor, segunda-melhor e o ranking completo.
    """
    mean, std = float(y.mean()), float(y.std())
    standardized = (y - mean) / std
    sign = 1.0 if stats.skew(standardized) >= 0 else -1.0
    sign_fixed = sign * standardized

    fits = []
    for name, distribution in _SYMMETRIC_FAMILIES.items():
        fits.append(_fit_family(name, distribution, standardized))
    for name, distribution in _ASYMMETRIC_FAMILIES.items():
        fits.append(_fit_family(name, distribution, sign_fixed))

    ranking = sorted(fits, key=lambda fit: fit.ks_statistic)
    return FamilyIdentificationResult(
        best=ranking[0], runner_up=ranking[1], ranking=ranking, mean=mean, std=std, sign=sign
    )


def family_pdf(
    result: FamilyIdentificationResult, x: np.ndarray, fit: FamilyFitResult | None = None
) -> np.ndarray:
    """Densidade do ajuste (``fit`` ou ``result.best``) avaliada em ``x``, na escala original.

    As familias simetricas foram ajustadas sobre ``(y - mean) / std``; as
    assimetricas sobre ``sign * (y - mean) / std`` (exigem suporte
    positivo). Muda de variavel para devolver a densidade na escala de
    ``y``: ``f_Y(x) = f_Z(z) / std``, com ``z`` a mesma transformacao usada
    no ajuste (``|sign| = 1`` nao afeta o jacobiano).

    Parameters
    ----------
    result : FamilyIdentificationResult
        Resultado de :func:`identify_family` sobre o sinal ``y`` de interesse.
    x : np.ndarray
        Pontos, na escala original de ``y``, onde avaliar a densidade.
    fit : FamilyFitResult, optional
        Por padrao, ``result.best``; passe ``result.runner_up`` (ou
        qualquer entrada de ``result.ranking``) para plotar outra candidata.

    Returns
    -------
    np.ndarray
        Densidade avaliada em ``x``, mesma shape.
    """
    fit = fit or result.best
    distribution = _ALL_FAMILIES[fit.name]
    z = (x - result.mean) / result.std
    if fit.name in _ASYMMETRIC_FAMILIES:
        z = result.sign * z
    return distribution.pdf(z, *fit.parameters) / result.std


def _fit_family(name: str, distribution, data: np.ndarray) -> FamilyFitResult:
    """Ajusta ``distribution`` a ``data`` por MLE e mede a bondade por KS."""
    try:
        parameters = distribution.fit(data)
        ks_statistic, _ = stats.kstest(data, distribution.cdf, args=parameters)
    except (ValueError, RuntimeError):
        parameters = ()
        ks_statistic = float("inf")
    return FamilyFitResult(name=name, ks_statistic=float(ks_statistic), parameters=parameters)


class FamilyIdentificationMetric(Metric):
    """Identificacao de familia por componente recuperada (sempre disponivel, sem gabarito)."""

    name = "family_identification_per_source"

    def compute(self, model: ICAModel) -> list[FamilyIdentificationResult]:
        """Calcula :func:`identify_family` para cada componente de ``model.sources_``.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado.

        Returns
        -------
        list of FamilyIdentificationResult
            Um resultado por componente, na ordem de ``model.sources_``.
        """
        return [identify_family(row) for row in model.sources_]
