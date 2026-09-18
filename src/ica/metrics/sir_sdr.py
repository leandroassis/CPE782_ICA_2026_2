"""SIR/SDR classicos e SI-SDR -- razoes de energia por fonte casada.

Skill ``ica-evaluation``, Secao 4, e ``references/metrics-formulas.md``,
Secao 3. O casamento (:mod:`ica.postprocessing.matching`) so serve para
alinhar cada componente recuperada a sua fonte verdadeira -- nunca para
reordenar ``ICAModel.sources_``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ica.metrics.base import Metric
from ica.postprocessing.matching import hungarian_match

if TYPE_CHECKING:
    from ica.model import ICAModel


@dataclass
class SourceSeparationRatios:
    """SIR, SDR e SI-SDR (em dB) de uma componente recuperada contra sua fonte casada."""

    sir_db: float
    sdr_db: float
    si_sdr_db: float


def si_sdr(estimate: np.ndarray, target: np.ndarray) -> float:
    """SDR invariante a escala (Le Roux et al., 2019).

    ``s_target = (<estimate,target>/||target||^2) * target``,
    ``e_noise = estimate - s_target``,
    ``SI-SDR = 10 log10(||s_target||^2 / ||e_noise||^2)``.

    Parameters
    ----------
    estimate : np.ndarray
        Componente recuperada, shape ``(n_amostras,)``.
    target : np.ndarray
        Fonte verdadeira casada, mesma shape.

    Returns
    -------
    float
        SI-SDR em dB.
    """
    scale = np.dot(estimate, target) / np.dot(target, target)
    s_target = scale * target
    e_noise = estimate - s_target
    noise_energy = np.sum(e_noise**2)
    if noise_energy <= 0:
        return float("inf")
    return float(10.0 * np.log10(np.sum(s_target**2) / noise_energy))


def sir_sdr(
    estimate: np.ndarray,
    target_index: int,
    all_true_sources: np.ndarray,
) -> SourceSeparationRatios:
    """SIR/SDR classicos (BSS_EVAL, Vincent et al., 2006) + SI-SDR.

    Decompoe ``estimate`` pela projecao de minimos quadrados no subespaco
    gerado por **todas** as fontes verdadeiras: ``s_target`` (projecao na
    fonte casada), ``e_interference`` (vazamento das demais fontes dentro
    dessa projecao) e ``e_artifacts`` (residuo fora do subespaco, ex. ruido
    de estimacao).

    Parameters
    ----------
    estimate : np.ndarray
        Componente recuperada, shape ``(n_amostras,)``.
    target_index : int
        Indice (em ``all_true_sources``) da fonte verdadeira casada a
        ``estimate``.
    all_true_sources : np.ndarray
        Todas as fontes verdadeiras do run, shape ``(n_fontes, n_amostras)``.

    Returns
    -------
    SourceSeparationRatios
        SIR, SDR e SI-SDR em dB. Maior = melhor.
    """
    coefficients, *_ = np.linalg.lstsq(all_true_sources.T, estimate, rcond=None)
    projection = coefficients @ all_true_sources
    s_target = coefficients[target_index] * all_true_sources[target_index]
    e_interference = projection - s_target
    e_artifacts = estimate - projection

    target_energy = np.sum(s_target**2)
    interference_energy = np.sum(e_interference**2)
    total_error_energy = np.sum((e_interference + e_artifacts) ** 2)

    sir_db = (
        float(10.0 * np.log10(target_energy / interference_energy))
        if interference_energy > 0
        else float("inf")
    )
    sdr_db = (
        float(10.0 * np.log10(target_energy / total_error_energy))
        if total_error_energy > 0
        else float("inf")
    )
    return SourceSeparationRatios(
        sir_db=sir_db,
        sdr_db=sdr_db,
        si_sdr_db=si_sdr(estimate, all_true_sources[target_index]),
    )


def sir_sdr_battery(
    sources_true: np.ndarray, sources_estimated: np.ndarray
) -> list[SourceSeparationRatios]:
    """Casa ``sources_estimated`` a ``sources_true`` (hungaro) e calcula SIR/SDR/SI-SDR por par.

    Parameters
    ----------
    sources_true : np.ndarray
        Fontes verdadeiras, shape ``(n_fontes, n_amostras)``.
    sources_estimated : np.ndarray
        Componentes recuperadas, shape ``(n_componentes, n_amostras)``.

    Returns
    -------
    list of SourceSeparationRatios
        Um resultado por fonte verdadeira casada, na ordem de
        ``sources_true``.
    """
    match = hungarian_match(sources_true, sources_estimated)
    order = np.argsort(match.reference_indices)
    return [
        sir_sdr(
            estimate=sources_estimated[match.matched_indices[i]],
            target_index=match.reference_indices[i],
            all_true_sources=sources_true,
        )
        for i in order
    ]


class SIRSDRMetric(Metric):
    """SIR, SDR e SI-SDR por fonte casada, quando o gabarito esta disponivel."""

    name = "sir_sdr_per_source"

    def compute(self, model: ICAModel) -> list[SourceSeparationRatios] | None:
        """Calcula :func:`sir_sdr_battery` sobre ``model``.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado.

        Returns
        -------
        list of SourceSeparationRatios or None
            Um resultado por fonte verdadeira, ou ``None`` sem gabarito.
        """
        if model.sources_true_ is None:
            return None
        return sir_sdr_battery(model.sources_true_, model.sources_)
