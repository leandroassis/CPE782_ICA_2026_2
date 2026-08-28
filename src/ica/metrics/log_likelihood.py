"""Metrica: log-verossimilhanca media atingida ao final do ajuste.

Ver context/ICA_BACKGROUND.md, Secao 3.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ica.metrics.base import Metric

if TYPE_CHECKING:
    from ica.model import ICAModel


class LogLikelihood(Metric):
    """Log-verossimilhanca media -- ``(1/T) log L(B)`` -- na ultima iteracao do ajuste.

    Le o ultimo valor de ``model.log_likelihood_history_``, a trajetoria
    completa calculada iteracao a iteracao por
    :meth:`ICAAlgorithm._log_likelihood
    <ica.algorithms.base.ICAAlgorithm._log_likelihood>` durante ``fit()``.

    Essa trajetoria precisa ser construida dentro do proprio algoritmo:
    o calculo depende da matriz de separacao *intermediaria* a cada
    passo, disponivel apenas enquanto o loop de otimizacao roda -- uma
    ``Metric`` so e avaliada depois que ``ICAModel.fit()`` termina, sobre
    o modelo ja ajustado, entao nao poderia calcular os valores
    intermediarios ela mesma. Esta classe segue o mesmo padrao de
    :class:`~ica.metrics.convergence_iterations.ConvergenceIterations` e
    :class:`~ica.metrics.execution_time.ExecutionTime`: expoe, como uma
    metrica nomeada e reaproveitavel no pipeline de ``evaluate()``, um
    valor ja calculado alhures -- sem duplicar a formula.
    """

    name = "log_likelihood"

    def compute(self, model: ICAModel) -> float:
        """Le o ultimo valor de ``model.log_likelihood_history_``.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado.

        Returns
        -------
        float
            Log-verossimilhanca media na ultima iteracao executada.
        """
        return float(model.log_likelihood_history_[-1])
