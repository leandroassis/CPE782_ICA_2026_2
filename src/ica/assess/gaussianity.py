"""Bateria de gaussianidade / viabilidade pre-BSS.

Skill ``bss-assessment``, Secao 1: por canal de mistura, quao longe da
normalidade esta a distribuicao. Um sinal de **viabilidade** da separacao
(via TLC -- muitos canais fortemente nao-gaussianos sugerem estrutura de
ordem superior a explorar), nunca uma conclusao sobre a origem das fontes
(isso e medido pos-separacao, ``ica.metrics.family_identification``).
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class GaussianityTestResult:
    """Resultado dos 3 testes de gaussianidade sobre um unico canal.

    Attributes
    ----------
    dagostino_pearson_statistic, dagostino_pearson_pvalue : float
        Teste K² (primario), combina assimetria e curtose transformadas.
    jarque_bera_statistic, jarque_bera_pvalue : float
        Teste JB (checagem cruzada), barato mas otimista em T pequeno.
    anderson_darling_statistic : float
        Teste AD (sensivel a caudas, baseado na EDF).
    """

    dagostino_pearson_statistic: float
    dagostino_pearson_pvalue: float
    jarque_bera_statistic: float
    jarque_bera_pvalue: float
    anderson_darling_statistic: float


def gaussianity_tests(y: np.ndarray) -> GaussianityTestResult:
    """Roda D'Agostino-Pearson K², Jarque-Bera e Anderson-Darling sobre um canal 1D.

    Parameters
    ----------
    y : np.ndarray
        Um canal (mistura ou componente), shape ``(n_amostras,)``.

    Returns
    -------
    GaussianityTestResult
        As 3 estatisticas (e p-valores, quando aplicavel).
    """
    k2_statistic, k2_pvalue = stats.normaltest(y)
    jb_statistic, jb_pvalue = stats.jarque_bera(y)
    # So o statistic e consumido (nao pvalue/critical_value); method="interpolate"
    # so fixa o metodo de calculo do p-valor que nao usamos aqui, evitando o
    # FutureWarning do SciPy >= 1.17 sem mudar o comportamento.
    anderson_result = stats.anderson(y, dist="norm", method="interpolate")
    return GaussianityTestResult(
        dagostino_pearson_statistic=float(k2_statistic),
        dagostino_pearson_pvalue=float(k2_pvalue),
        jarque_bera_statistic=float(jb_statistic),
        jarque_bera_pvalue=float(jb_pvalue),
        anderson_darling_statistic=float(anderson_result.statistic),
    )


def gaussianity_battery(X: np.ndarray) -> list[GaussianityTestResult]:
    """Roda :func:`gaussianity_tests` por canal (linha) de ``X``.

    Parameters
    ----------
    X : np.ndarray
        Misturas, shape ``(n_canais, n_amostras)``.

    Returns
    -------
    list of GaussianityTestResult
        Um resultado por canal, na ordem das linhas de ``X``.
    """
    return [gaussianity_tests(row) for row in X]
