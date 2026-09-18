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
    """

    best: FamilyFitResult
    runner_up: FamilyFitResult
    ranking: list[FamilyFitResult]


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
    standardized = (y - y.mean()) / y.std()
    sign = 1.0 if stats.skew(standardized) >= 0 else -1.0
    sign_fixed = sign * standardized

    fits = []
    for name, distribution in _SYMMETRIC_FAMILIES.items():
        fits.append(_fit_family(name, distribution, standardized))
    for name, distribution in _ASYMMETRIC_FAMILIES.items():
        fits.append(_fit_family(name, distribution, sign_fixed))

    ranking = sorted(fits, key=lambda fit: fit.ks_statistic)
    return FamilyIdentificationResult(best=ranking[0], runner_up=ranking[1], ranking=ranking)


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
