"""Fachada de alto nivel que orquestra o pipeline completo de ICA.

Corresponde ao trecho ``preprocess/ -> ica/`` do fluxo de
``.claude/PIPELINE_MAP.md`` para uma unica celula (um algoritmo, um modo de
condicionamento); a resolucao de ambiguidades (``postprocess/``), a bateria
estatistica (``assess/``) e a comparacao entre celulas de uma grade
(``harness/``) sao camadas em torno desta fachada.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ica.algorithms.base import ICAAlgorithm
from ica.data.base import DataTemplate
from ica.postprocessing.ambiguity import resolve_ambiguities
from ica.preprocessing.pipeline import Pipeline

if TYPE_CHECKING:
    from ica.assess.report import AssessmentReport
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
        centralizacao, reescala robusta e branqueamento).
    algorithm : ICAAlgorithm
        Algoritmo de otimizacao de ICA, ja configurado.
    groundtruth_root : pathlib.Path, optional
        Diretorio raiz do gabarito para o tipo de amostra de ``data``
        (tipicamente ``data/groundtruth/<tipo>``). Quando informado e
        ``data`` sabe :meth:`~ica.data.base.DataTemplate.load_ground_truth`,
        :meth:`fit` popula ``mixing_matrix_true_``/``sources_true_`` --
        usados pelas metricas de validacao (Amari, PSNR/SSIM, SI-SDR) em
        ``ica.metrics``, que se auto-desabilitam (``None``) na ausencia
        deles.

    Attributes
    ----------
    mixtures_ : np.ndarray or None
        Misturas originais carregadas, shape ``(n_misturas, n_amostras)``.
    domain_ : str or None
        Dominio da amostra (``"image"``, ``"distribution"`` ou ``"audio"``),
        do :class:`~ica.interfaces.SignalMatrix` devolvido por ``data.load()``.
    signal_meta_ : dict
        Metadados de dominio do :class:`~ica.interfaces.SignalMatrix`
        (altura/largura/``is_rgb`` para imagem, taxa de amostragem para
        audio, tamanho amostral para distribuicao).
    mixing_matrix_true_ : np.ndarray or None
        Matriz de mistura verdadeira ``A``, quando ``groundtruth_root`` foi
        informado e o gabarito existe para este run.
    sources_true_ : np.ndarray or None
        Fontes verdadeiras, quando ``groundtruth_root`` foi informado e o
        gabarito existe para este run.
    preprocessed_ : np.ndarray or None
        Misturas apos o pipeline de pre-processamento.
    sources_ : np.ndarray or None
        Componentes independentes recuperadas, ja com as ambiguidades de
        escala/sinal/ordem resolvidas (:func:`~ica.postprocessing.ambiguity.resolve_ambiguities`,
        skill ica-evaluation, Secao 1) -- e o que a figura-vitrine mostra.
        O casamento hungaro contra o gabarito (usado so para pontuar) roda
        a parte, dentro de ``ica.metrics``, e nunca realimenta este
        atributo.
    unmixing_matrix_ : np.ndarray or None
        Matriz de separacao estimada pelo algoritmo, no espaco
        pre-processado (ex.: branqueado).
    full_unmixing_matrix_ : np.ndarray or None
        Matriz de separacao composta com os passos lineares do pipeline
        (ex.: :class:`~ica.preprocessing.whitening.Whitening`,
        :class:`~ica.preprocessing.pca.PCA`), mapeando diretamente das
        misturas originais para as fontes recuperadas -- ver
        :meth:`~ica.preprocessing.pipeline.Pipeline.compose_linear_matrix`.
        Passos afins (:class:`~ica.preprocessing.centering.Centering`) ou
        ``estimation_only`` nao contribuem.
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

    def __init__(
        self,
        data: DataTemplate,
        pipeline: Pipeline,
        algorithm: ICAAlgorithm,
        groundtruth_root: Path | None = None,
    ) -> None:
        self.data = data
        self.pipeline = pipeline
        self.algorithm = algorithm
        self.groundtruth_root = groundtruth_root

        self.mixtures_: np.ndarray | None = None
        self.domain_: str | None = None
        self.signal_meta_: dict = {}
        self.mixing_matrix_true_: np.ndarray | None = None
        self.sources_true_: np.ndarray | None = None
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

        O pipeline e aplicado duas vezes com papeis distintos (ver
        :attr:`~ica.preprocessing.base.PreprocessingStep.estimation_only`):
        uma vez completo (``fit_transform``), para *estimar* a matriz de
        separacao B, e outra vez pulando os passos ``estimation_only``
        (``reconstruction_transform``), partindo sempre dos dados
        originais, para *reconstruir* as fontes finais.

        Returns
        -------
        ICAModel
            A propria instancia, com os atributos ``*_`` preenchidos.
        """
        signal_matrix = self.data.load()
        self.mixtures_ = signal_matrix.data
        self.domain_ = signal_matrix.domain
        self.signal_meta_ = signal_matrix.meta

        if self.groundtruth_root is not None:
            self.mixing_matrix_true_, self.sources_true_ = self.data.load_ground_truth(
                self.groundtruth_root
            )

        self.preprocessed_ = self.pipeline.fit_transform(self.mixtures_)

        self.unmixing_matrix_ = self.algorithm.fit(self.preprocessed_)

        reconstruction_input = self.pipeline.reconstruction_transform(self.mixtures_)
        raw_sources = self.unmixing_matrix_ @ reconstruction_input
        self.sources_ = resolve_ambiguities(raw_sources, domain=self.domain_)

        self.full_unmixing_matrix_ = self.pipeline.compose_linear_matrix(self.unmixing_matrix_)

        self.history_ = self.algorithm.history_
        self.log_likelihood_history_ = self.algorithm.log_likelihood_history_
        self.converged_ = self.algorithm.converged_
        self.n_iterations_ = self.algorithm.n_iterations_
        self.elapsed_time_ = self.algorithm.elapsed_time_
        return self

    def evaluate(self, metrics: list[Metric]) -> dict[str, Any]:
        """Aplica uma lista de metricas sobre o resultado de :meth:`fit`.

        Parameters
        ----------
        metrics : list of Metric
            Metricas (``ica.metrics``) a calcular sobre este modelo.

        Returns
        -------
        dict
            Mapa de ``metric.name`` para o valor calculado por
            ``metric.compute(self)`` -- ``None`` para metricas de validacao
            sem gabarito disponivel (ver :class:`~ica.metrics.base.Metric`).
        """
        return {metric.name: metric.compute(self) for metric in metrics}

    def assess(self) -> AssessmentReport:
        """Roda a bateria estatistica pre-BSS (bloco ``assess/``) sobre este modelo.

        Usa ``mixtures_`` para a bateria de gaussianidade/HOS/atraso, e
        ``sources_`` (pos-ICA, ja preenchido por :meth:`fit`) para o
        diagnostico de nao-linearidade -- que so e confiavel medido sobre a
        melhor separacao linear obtida, nunca sobre dados so branqueados
        (skill ``bss-assessment``, Secao 3).

        Returns
        -------
        AssessmentReport
            Relatorio agregando gaussianidade, HOS, nao-linearidade e
            atraso (quando aplicavel ao dominio).

        Raises
        ------
        RuntimeError
            Se chamado antes de :meth:`fit`.
        """
        from ica.assess.report import assess_run

        if self.mixtures_ is None or self.sources_ is None:
            raise RuntimeError("assess() requer fit() ja executado.")
        return assess_run(mixtures=self.mixtures_, sources=self.sources_, domain=self.domain_)
