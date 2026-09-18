"""Metricas de avaliacao de um ICAModel ajustado.

Bloco ``evaluate/`` de ``.claude/PIPELINE_MAP.md`` (parte escopada a um
unico modelo): metricas sempre disponiveis
(``ConvergenceIterations``/``ExecutionTime``/``LogLikelihood``/``NonGaussianityScore``)
e metricas de validacao contra o gabarito (``AmariIndex``, ``SIRSDRMetric``,
``PSNRMetric``/``SSIMMetric``, ``DistributionFitMetric``), que se
auto-desabilitam (``None``) sem ``ICAModel.mixing_matrix_true_``/
``sources_true_``. A selecao da "melhor separacao" e a figura-vitrine, que
precisam de varios modelos de um run, ficam em ``ica.harness``.
"""

from ica.metrics.amari import AmariIndex, amari_index
from ica.metrics.base import Metric
from ica.metrics.convergence_iterations import ConvergenceIterations
from ica.metrics.distribution_metrics import (
    DistributionFitMetric,
    DistributionFitQuality,
    distribution_fit_quality,
    distribution_metrics_battery,
)
from ica.metrics.execution_time import ExecutionTime
from ica.metrics.family_identification import (
    FamilyFitResult,
    FamilyIdentificationMetric,
    FamilyIdentificationResult,
    family_pdf,
    identify_family,
)
from ica.metrics.image_metrics import (
    PSNRMetric,
    SSIMMetric,
    image_metrics_battery,
    min_max_normalize,
    psnr,
    ssim,
)
from ica.metrics.log_likelihood import LogLikelihood
from ica.metrics.non_gaussianity import NonGaussianityScore
from ica.metrics.sir_sdr import (
    SIRSDRMetric,
    SourceSeparationRatios,
    si_sdr,
    sir_sdr,
    sir_sdr_battery,
)

__all__ = [
    "Metric",
    "ConvergenceIterations",
    "ExecutionTime",
    "NonGaussianityScore",
    "LogLikelihood",
    "amari_index",
    "AmariIndex",
    "si_sdr",
    "sir_sdr",
    "sir_sdr_battery",
    "SourceSeparationRatios",
    "SIRSDRMetric",
    "psnr",
    "ssim",
    "min_max_normalize",
    "image_metrics_battery",
    "PSNRMetric",
    "SSIMMetric",
    "distribution_fit_quality",
    "distribution_metrics_battery",
    "DistributionFitQuality",
    "DistributionFitMetric",
    "identify_family",
    "family_pdf",
    "FamilyFitResult",
    "FamilyIdentificationResult",
    "FamilyIdentificationMetric",
]
