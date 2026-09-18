"""Bloco ``assess/`` de ``.claude/PIPELINE_MAP.md`` -- bateria estatistica pre-BSS.

Skill ``bss-assessment``: gaussianidade/viabilidade (:mod:`ica.assess.gaussianity`),
impressao de ordem superior (:mod:`ica.assess.higher_order`), diagnostico de
nao-linearidade (:mod:`ica.assess.nonlinearity`) e de atraso
(:mod:`ica.assess.delay`), agregados por :func:`ica.assess.report.assess_run`.
"""

from ica.assess.delay import DelayEstimate, cross_correlation_lag, delay_battery, gcc_phat
from ica.assess.gaussianity import GaussianityTestResult, gaussianity_battery, gaussianity_tests
from ica.assess.higher_order import (
    HigherOrderStatistics,
    higher_order_battery,
    higher_order_statistics,
)
from ica.assess.nonlinearity import (
    ResidualDependenceResult,
    distance_correlation,
    joint_diagonalization_defect,
    residual_dependence,
)
from ica.assess.report import AssessmentReport, ChannelAssessment, assess_run

__all__ = [
    "gaussianity_tests",
    "gaussianity_battery",
    "GaussianityTestResult",
    "higher_order_statistics",
    "higher_order_battery",
    "HigherOrderStatistics",
    "residual_dependence",
    "distance_correlation",
    "joint_diagonalization_defect",
    "ResidualDependenceResult",
    "cross_correlation_lag",
    "gcc_phat",
    "delay_battery",
    "DelayEstimate",
    "assess_run",
    "AssessmentReport",
    "ChannelAssessment",
]
