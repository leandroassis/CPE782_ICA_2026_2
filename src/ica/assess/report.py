"""Agrega a bateria de ``assess/`` num relatorio unico por run.

Skill ``bss-assessment``, "Saidas do bloco assess/": por run/canal, tabela
de gaussianidade + decisao de viabilidade, impressao HOS, flag de
nao-linearidade (pos-ICA) e flag de atraso (so audio).
"""

from dataclasses import dataclass

import numpy as np

from ica.assess.delay import delay_battery
from ica.assess.gaussianity import GaussianityTestResult, gaussianity_battery
from ica.assess.higher_order import HigherOrderStatistics, higher_order_battery
from ica.assess.nonlinearity import (
    ResidualDependenceResult,
    joint_diagonalization_defect,
    residual_dependence,
)
from ica.interfaces import Domain
from ica.preprocessing.centering import Centering
from ica.preprocessing.whitening import Whitening

# Limiares por tamanho de efeito (skill bss-assessment, Secao 5) -- nao por
# p-valor cru, que rejeita a normalidade por desvios triviais em T grande.
_GAUSSIAN_LIKE_KURTOSIS_THRESHOLD = 0.3
_GAUSSIAN_LIKE_NEGENTROPY_THRESHOLD = 0.02


@dataclass
class ChannelAssessment:
    """Avaliacao pre-BSS de um unico canal de mistura.

    Attributes
    ----------
    gaussianity : GaussianityTestResult
        Resultado dos 3 testes de gaussianidade.
    higher_order : HigherOrderStatistics
        Impressao HOS (assimetria, curtose excedente, negentropia).
    is_gaussian_like : bool
        Decisao por tamanho de efeito (``|curtose excedente|`` e
        negentropia robusta abaixo dos limiares), nao por p-valor.
    """

    gaussianity: GaussianityTestResult
    higher_order: HigherOrderStatistics
    is_gaussian_like: bool


@dataclass
class AssessmentReport:
    """Relatorio agregado da bateria ``assess/`` para um run.

    Attributes
    ----------
    domain : {"image", "distribution", "audio"}
        Dominio da amostra avaliada.
    channels : list of ChannelAssessment
        Uma avaliacao por canal de mistura.
    n_gaussian_like_channels : int
        Quantos canais sao estatisticamente indistinguiveis de gaussianos.
    viability_flag : {"viavel", "risco"}
        Sinal de viabilidade da separacao (via TLC) -- **nao** uma
        conclusao sobre quantas fontes sao gaussianas.
    nonlinearity : ResidualDependenceResult
        Dependencia residual pos-ICA (skill bss-assessment, Secao 3).
    joint_diagonalization_defect : float
        Pre-screen corroborante (pre-separacao, sobre dados branqueados);
        ver ressalva em :mod:`ica.assess.nonlinearity` sobre fontes
        proximas de gaussianas.
    delay_lags : np.ndarray or None
        Matriz de lags GCC-PHAT entre canais; ``None`` para dominios sem
        eixo temporal real (imagem, distribuicao).
    """

    domain: Domain
    channels: list[ChannelAssessment]
    n_gaussian_like_channels: int
    viability_flag: str
    nonlinearity: ResidualDependenceResult
    joint_diagonalization_defect: float
    delay_lags: np.ndarray | None


def _is_gaussian_like(statistics: HigherOrderStatistics) -> bool:
    """Decide gaussianidade por tamanho de efeito (curtose e negentropia), nao p-valor."""
    return (
        abs(statistics.excess_kurtosis) < _GAUSSIAN_LIKE_KURTOSIS_THRESHOLD
        and statistics.negentropy_contrast < _GAUSSIAN_LIKE_NEGENTROPY_THRESHOLD
    )


def assess_run(
    mixtures: np.ndarray,
    sources: np.ndarray,
    domain: Domain,
    sample_rate: float | None = None,
) -> AssessmentReport:
    """Roda a bateria completa de ``assess/`` sobre um run.

    Parameters
    ----------
    mixtures : np.ndarray
        Misturas cruas, shape ``(n_canais, n_amostras)`` -- usadas para
        gaussianidade, HOS, atraso e o pre-screen de diagonalizacao
        conjunta (branqueadas internamente para esse ultimo).
    sources : np.ndarray
        Componentes recuperadas pela melhor separacao linear obtida
        (``ICAModel.sources_``) -- usadas para o diagnostico de
        nao-linearidade (skill bss-assessment, Secao 3: nunca medido sobre
        dados so branqueados).
    domain : {"image", "distribution", "audio"}
        Dominio da amostra.
    sample_rate : float, optional
        Taxa de amostragem (Hz), para o diagnostico de atraso em audio.

    Returns
    -------
    AssessmentReport
        Relatorio agregado.
    """
    gaussianity_results = gaussianity_battery(mixtures)
    hos_results = higher_order_battery(mixtures)
    channels = [
        ChannelAssessment(
            gaussianity=gaussianity,
            higher_order=hos,
            is_gaussian_like=_is_gaussian_like(hos),
        )
        for gaussianity, hos in zip(gaussianity_results, hos_results, strict=True)
    ]
    n_gaussian_like = sum(channel.is_gaussian_like for channel in channels)
    viability_flag = "risco" if n_gaussian_like == len(channels) else "viavel"

    nonlinearity_diagnostic = residual_dependence(sources)

    whitened = Whitening().fit_transform(Centering().fit_transform(mixtures))
    joint_diag_defect = joint_diagonalization_defect(whitened)

    delay_lags = delay_battery(mixtures, sample_rate=sample_rate) if domain == "audio" else None

    return AssessmentReport(
        domain=domain,
        channels=channels,
        n_gaussian_like_channels=n_gaussian_like,
        viability_flag=viability_flag,
        nonlinearity=nonlinearity_diagnostic,
        joint_diagonalization_defect=joint_diag_defect,
        delay_lags=delay_lags,
    )
