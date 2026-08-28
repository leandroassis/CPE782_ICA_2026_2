"""Fachada de alto nivel que orquestra o pipeline completo de ICA.

Ver context/DEVELOPMENT_GUIDELINES.md, Secao 2.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ica.algorithms.base import ICAAlgorithm
from ica.data.base import DataTemplate
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.whitening import Whitening

if TYPE_CHECKING:
    from ica.metrics.base import Metric


class ICAModel:
    """Orquestra carregamento, pre-processamento e estimacao de ICA sobre uma amostra.

    Fachada (Facade) que nao conhece detalhes de implementacao de nenhuma
    camada inferior: recebe, por injecao de dependencia, um
    :class:`~ica.data.base.DataTemplate` (fonte dos dados), um
    :class:`~ica.preprocessing.pipeline.Pipeline` de pre-processamento e
    um :class:`~ica.algorithms.base.ICAAlgorithm` ja configurado com sua
    :class:`~ica.nonlinearities.base.NonlinearityTemplate`.

    Parameters
    ----------
    data : DataTemplate
        Carregador da amostra a processar.
    pipeline : Pipeline
        Sequencia de passos de pre-processamento (tipicamente
        centralizacao seguida de branqueamento).
    algorithm : ICAAlgorithm
        Algoritmo de otimizacao de ICA, ja configurado.

    Attributes
    ----------
    mixtures_ : np.ndarray or None
        Misturas originais carregadas, shape ``(n_misturas, n_amostras)``.
    preprocessed_ : np.ndarray or None
        Misturas apos o pipeline de pre-processamento.
    sources_ : np.ndarray or None
        Componentes independentes recuperadas.
    unmixing_matrix_ : np.ndarray or None
        Matriz de separacao estimada pelo algoritmo, no espaco
        pre-processado (ex.: branqueado).
    full_unmixing_matrix_ : np.ndarray or None
        Matriz de separacao composta com o branqueamento, mapeando
        diretamente das misturas originais para as fontes recuperadas.
        Definida apenas quando o pipeline inclui um passo
        :class:`~ica.preprocessing.whitening.Whitening`.
    history_ : list of float or None
        Historico de convergencia do algoritmo (ver
        :attr:`ICAAlgorithm.history_ <ica.algorithms.base.ICAAlgorithm.history_>`).
    log_likelihood_history_ : list of float or None
        Log-verossimilhanca media a cada iteracao (ver
        :attr:`ICAAlgorithm.log_likelihood_history_
        <ica.algorithms.base.ICAAlgorithm.log_likelihood_history_>`).
    converged_ : bool or None
        Se o algoritmo convergiu antes do numero maximo de iteracoes.
    n_iterations_ : int or None
        Numero de iteracoes executadas pelo algoritmo.
    elapsed_time_ : float or None
        Tempo de execucao do algoritmo, em segundos.
    """

    def __init__(self, data: DataTemplate, pipeline: Pipeline, algorithm: ICAAlgorithm) -> None:
        self.data = data
        self.pipeline = pipeline
        self.algorithm = algorithm

        self.mixtures_: np.ndarray | None = None
        self.preprocessed_: np.ndarray | None = None
        self.sources_: np.ndarray | None = None
        self.unmixing_matrix_: np.ndarray | None = None
        self.full_unmixing_matrix_: np.ndarray | None = None
        self.history_: list[float] | None = None
        self.log_likelihood_history_: list[float] | None = None
        self.converged_: bool | None = None
        self.n_iterations_: int | None = None
        self.elapsed_time_: float | None = None

    def fit(self) -> "ICAModel":
        """Carrega a amostra, pre-processa e estima a separacao de fontes.

        Returns
        -------
        ICAModel
            A propria instancia, com os atributos ``*_`` preenchidos.
        """
        self.mixtures_ = self.data.load()
        self.preprocessed_ = self.pipeline.fit_transform(self.mixtures_)

        self.unmixing_matrix_ = self.algorithm.fit(self.preprocessed_)
        self.sources_ = self.unmixing_matrix_ @ self.preprocessed_

        try:
            whitening = self.pipeline.get_step(Whitening)
            self.full_unmixing_matrix_ = self.unmixing_matrix_ @ whitening.whitening_matrix_
        except ValueError:
            self.full_unmixing_matrix_ = self.unmixing_matrix_

        self.history_ = self.algorithm.history_
        self.log_likelihood_history_ = self.algorithm.log_likelihood_history_
        self.converged_ = self.algorithm.converged_
        self.n_iterations_ = self.algorithm.n_iterations_
        self.elapsed_time_ = self.algorithm.elapsed_time_
        return self

    def evaluate(self, metrics: list[Metric]) -> dict[str, float | np.ndarray]:
        """Aplica uma lista de metricas sobre o resultado de :meth:`fit`.

        Parameters
        ----------
        metrics : list of Metric
            Metricas (``ica.metrics``) a calcular sobre este modelo.

        Returns
        -------
        dict
            Mapa de ``metric.name`` para o valor calculado por
            ``metric.compute(self)``.
        """
        return {metric.name: metric.compute(self) for metric in metrics}
