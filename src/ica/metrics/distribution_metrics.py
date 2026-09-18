"""KS/Anderson-Darling -- bondade de ajuste para distribuicao, por fonte casada.

Skill ``ica-evaluation``, Secao 4, e ``references/metrics-formulas.md``,
Secao 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import anderson_ksamp, ks_2samp

from ica.metrics.base import Metric
from ica.postprocessing.matching import hungarian_match

if TYPE_CHECKING:
    from ica.model import ICAModel


@dataclass
class DistributionFitQuality:
    """Bondade de ajuste de uma componente recuperada contra sua fonte verdadeira casada."""

    ks_statistic: float
    ks_pvalue: float
    anderson_darling_statistic: float


def distribution_fit_quality(estimated: np.ndarray, true: np.ndarray) -> DistributionFitQuality:
    """KS e AD de duas amostras (``references/metrics-formulas.md``, Secao 5).

    ``D = sup_x |F_n(x) - F(x)|`` (KS); AD pondera as caudas, preferivel
    quando a discriminacao depende delas. Menor = melhor ajuste.

    Parameters
    ----------
    estimated : np.ndarray
        Componente recuperada, shape ``(n_amostras,)``.
    true : np.ndarray
        Fonte verdadeira casada (pode ter tamanho amostral diferente).

    Returns
    -------
    DistributionFitQuality
        Estatisticas KS (com p-valor) e AD de 2 amostras.
    """
    ks_result = ks_2samp(estimated, true)
    # So o statistic e consumido; variant="midrank" fixa o metodo que o
    # SciPy >= 1.17 passou a exigir explicitamente, sem mudar o resultado.
    ad_result = anderson_ksamp([estimated, true], variant="midrank")
    return DistributionFitQuality(
        ks_statistic=float(ks_result.statistic),
        ks_pvalue=float(ks_result.pvalue),
        anderson_darling_statistic=float(ad_result.statistic),
    )


def distribution_metrics_battery(
    sources_true: np.ndarray, sources_estimated: np.ndarray
) -> list[DistributionFitQuality]:
    """Casa (hungaro) e calcula :func:`distribution_fit_quality` por par.

    Parameters
    ----------
    sources_true : np.ndarray
        Fontes verdadeiras, shape ``(n_fontes, n_amostras_verdadeiras)``.
    sources_estimated : np.ndarray
        Componentes recuperadas, shape ``(n_componentes, n_amostras)``.

    Returns
    -------
    list of DistributionFitQuality
        Um resultado por fonte verdadeira casada, na ordem de ``sources_true``.
    """
    match = hungarian_match(sources_true, sources_estimated)
    order = np.argsort(match.reference_indices)
    return [
        distribution_fit_quality(
            estimated=sources_estimated[match.matched_indices[i]],
            true=sources_true[match.reference_indices[i]],
        )
        for i in order
    ]


class DistributionFitMetric(Metric):
    """KS/AD por fonte casada, quando o gabarito esta disponivel (dominio distribuicao)."""

    name = "distribution_fit_per_source"

    def compute(self, model: ICAModel) -> list[DistributionFitQuality] | None:
        """Calcula :func:`distribution_metrics_battery` sobre ``model``.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado sobre uma amostra de distribuicao.

        Returns
        -------
        list of DistributionFitQuality or None
            Um resultado por fonte, ou ``None`` sem gabarito/fora do dominio.
        """
        if model.sources_true_ is None or model.domain_ != "distribution":
            return None
        return distribution_metrics_battery(model.sources_true_, model.sources_)
