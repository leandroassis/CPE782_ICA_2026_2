"""Interface base para algoritmos de ICA (Template Method).

Ver ``.claude/skills/ica-ml/SKILL.md``, Secoes 6-7 (os 3 algoritmos e seus
criterios de convergencia).
"""

import time
from abc import ABC, abstractmethod

import numpy as np

from ica.interfaces import ICAResult
from ica.nonlinearities.base import NonlinearityTemplate


class ICAAlgorithm(ABC):
    """Algoritmo de otimizacao para estimar a matriz de separacao B.

    Implementa o padrao Template Method: :meth:`fit` define o esqueleto
    fixo do processo iterativo (inicializar B, repetir o passo de
    atualizacao ate convergencia ou numero maximo de iteracoes,
    cronometrando e registrando o historico), delegando as subclasses o
    passo de atualizacao (:meth:`_update_step`) e o criterio de
    convergencia (:meth:`_convergence_residual`) -- que a skill ``ica-ml``
    (Secao 7) define como **especifico por algoritmo**: GN/BS usam o
    residuo do ponto fixo ``||I + E{g(y)y^T}||_F``; FastICA-ML usa o
    desalinhamento de colunas ``1 - min_i|<b_i_novo, b_i_antigo>|``.

    Parameters
    ----------
    nonlinearity : NonlinearityTemplate
        Funcao de pontuacao g(y) usada no passo de atualizacao (injetada
        por dependencia).
    learning_rate : float, default=0.01
        Passo da atualizacao. Ignorado por algoritmos livres de taxa
        (ex.: ``FastICAML``), que documentam essa excecao explicitamente.
    max_iterations : int, default=500
        Numero maximo de iteracoes antes de parar mesmo sem convergencia.
    tolerance : float, default=1e-6
        Limiar do residuo de convergencia (:meth:`_convergence_residual`)
        abaixo do qual o algoritmo e considerado convergido.
    random_state : int or None, default=None
        Semente para a inicializacao aleatoria de B. Se ``None``, B e
        inicializada como a identidade.

    Attributes
    ----------
    unmixing_matrix_ : np.ndarray or None
        Matriz de separacao B estimada, definida apos :meth:`fit`.
    converged_ : bool or None
        Se o criterio de convergencia foi atingido antes de
        ``max_iterations``.
    n_iterations_ : int or None
        Numero de iteracoes efetivamente executadas.
    history_ : list of float
        Residuo de convergencia a cada iteracao (alias de
        ``result_.convergence_trace``, mantido por compatibilidade com o
        padrao scikit-learn ja documentado no projeto).
    log_likelihood_history_ : list of float
        Log-verossimilhanca media -- ``(1/T) log L(B)``, convencao do livro
        (skill ``ica-ml``, Secao 2) -- calculada a cada iteracao com a
        matriz B ja atualizada. Cresce (ou se mantem estavel) ao longo das
        iteracoes quando o algoritmo esta de fato ascendendo a
        verossimilhanca.
    elapsed_time_ : float or None
        Tempo de execucao de :meth:`fit`, em segundos.
    result_ : ICAResult or None
        Resultado completo (ver :class:`~ica.interfaces.ICAResult`),
        definido apos :meth:`fit`.
    """

    def __init__(
        self,
        nonlinearity: NonlinearityTemplate,
        learning_rate: float = 0.01,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        self.nonlinearity = nonlinearity
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state

        self.unmixing_matrix_: np.ndarray | None = None
        self.converged_: bool | None = None
        self.n_iterations_: int | None = None
        self.history_: list[float] = []
        self.log_likelihood_history_: list[float] = []
        self.elapsed_time_: float | None = None
        self.result_: ICAResult | None = None

    def fit(self, X: np.ndarray) -> np.ndarray:
        """Estima a matriz de separacao B que torna Y = BX o mais independente possivel.

        Esqueleto fixo (Template Method): inicializa B, itera
        :meth:`_update_step` ate a convergencia (:meth:`_convergence_residual`
        abaixo de ``tolerance``) ou ``max_iterations``, cronometrando e
        registrando o historico de convergencia e de log-verossimilhanca.

        Parameters
        ----------
        X : np.ndarray
            Dados pre-processados (tipicamente centralizados e
            branqueados), shape ``(n_componentes, n_amostras)``.

        Returns
        -------
        np.ndarray
            A matriz de separacao estimada (``unmixing_matrix_``).
        """
        start_time = time.perf_counter()
        n_components = X.shape[0]

        B = self._initialize_unmixing_matrix(n_components)
        self.history_ = []
        self.log_likelihood_history_ = []
        self.converged_ = False
        iteration = 0
        residual = float("inf")

        for iteration in range(1, self.max_iterations + 1):
            B_new = self._update_step(B, X)
            residual = self._convergence_residual(B, B_new, X)
            self.history_.append(residual)
            self.log_likelihood_history_.append(self._log_likelihood(B_new, X))
            B = B_new
            if residual < self.tolerance:
                self.converged_ = True
                break

        self.unmixing_matrix_ = B
        self.n_iterations_ = iteration
        self.elapsed_time_ = time.perf_counter() - start_time

        Y = B @ X
        self.result_ = ICAResult(
            B=B,
            Y=Y,
            log_likelihood_history=self.log_likelihood_history_,
            convergence_trace=self.history_,
            nonlinearity_per_component=self._nonlinearity_labels(Y, n_components),
            n_iterations=self.n_iterations_,
            converged=self.converged_,
            elapsed_time=self.elapsed_time_,
        )
        return self.unmixing_matrix_

    def _initialize_unmixing_matrix(self, n_components: int) -> np.ndarray:
        """Inicializa B como a identidade, ou ortogonal aleatoria se ``random_state`` for informado.

        Parameters
        ----------
        n_components : int
            Numero de componentes (linhas/colunas de B).

        Returns
        -------
        np.ndarray
            Matriz inicial B, shape ``(n_componentes, n_componentes)``.
        """
        if self.random_state is None:
            return np.eye(n_components)
        rng = np.random.default_rng(self.random_state)
        random_matrix = rng.normal(size=(n_components, n_components))
        orthogonal_matrix, _ = np.linalg.qr(random_matrix)
        return orthogonal_matrix

    @abstractmethod
    def _update_step(self, B: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Calcula a nova estimativa de B a partir da estimativa atual.

        Ponto variavel do Template Method -- cada algoritmo concreto
        (Bell-Sejnowski, Gradiente Natural, FastICA-ML) implementa aqui sua
        regra de atualizacao especifica (skill ``ica-ml``, Secao 6).

        Parameters
        ----------
        B : np.ndarray
            Estimativa atual da matriz de separacao, shape
            ``(n_componentes, n_componentes)``.
        X : np.ndarray
            Dados pre-processados, shape ``(n_componentes, n_amostras)``.

        Returns
        -------
        np.ndarray
            Nova estimativa de B.
        """

    @abstractmethod
    def _convergence_residual(self, B_old: np.ndarray, B_new: np.ndarray, X: np.ndarray) -> float:
        """Residuo de convergencia especifico do algoritmo (skill ``ica-ml``, Secao 7).

        Parameters
        ----------
        B_old : np.ndarray
            Estimativa anterior da matriz de separacao.
        B_new : np.ndarray
            Nova estimativa da matriz de separacao.
        X : np.ndarray
            Dados pre-processados, shape ``(n_componentes, n_amostras)``.

        Returns
        -------
        float
            Residuo nao-negativo; convergiu quando cai abaixo de
            ``tolerance``.
        """

    def _log_likelihood(self, B: np.ndarray, X: np.ndarray) -> float:
        """Log-verossimilhanca media no ponto ``B`` (skill ``ica-ml``, Secao 2).

        Implementa ``(1/T) log L(B) = sum_i E{log p_i(b_i^T x)} +
        log|det B|``, usando ``self.nonlinearity.log_density`` como a
        densidade suposta ``p_i``. ``log|det B|`` e calculado via
        :func:`numpy.linalg.slogdet` por estabilidade numerica.

        Este calculo permanece na propria classe do algoritmo (e nao em
        ``ica.metrics``) porque precisa da matriz de separacao
        *intermediaria* a cada iteracao -- disponivel apenas aqui dentro
        do loop de :meth:`fit`, antes de existir um ``ICAModel`` ajustado
        sobre o qual uma ``Metric`` pudesse operar. O valor final desta
        trajetoria e reexposto, sem duplicar a formula, por
        :class:`~ica.metrics.log_likelihood.LogLikelihood`.

        Parameters
        ----------
        B : np.ndarray
            Matriz de separacao no ponto a avaliar.
        X : np.ndarray
            Dados pre-processados, shape ``(n_componentes, n_amostras)``.

        Returns
        -------
        float
            Log-verossimilhanca media em ``B``.
        """
        Y = B @ X
        _, log_abs_det = np.linalg.slogdet(B)
        return float(np.sum(np.mean(self.nonlinearity.log_density(Y), axis=1)) + log_abs_det)

    def _nonlinearity_labels(self, Y: np.ndarray, n_components: int) -> list[str]:
        """Rotulos ``"super"``/``"sub"`` por componente na iteracao final.

        Usa :meth:`~ica.nonlinearities.adaptive.AdaptiveScore.labels` quando
        ``self.nonlinearity`` chaveia por componente; para uma nao-linearidade
        fixa, repete o mesmo rotulo (inferido do nome da classe) para todas
        as componentes.

        Parameters
        ----------
        Y : np.ndarray
            Componentes recuperadas na iteracao final, shape
            ``(n_componentes, n_amostras)``.
        n_components : int
            Numero de componentes.

        Returns
        -------
        list of str
            Um rotulo por componente.
        """
        if hasattr(self.nonlinearity, "labels"):
            self.nonlinearity.score(Y)
            return self.nonlinearity.labels()
        fixed_label = "super" if "Super" in type(self.nonlinearity).__name__ else "sub"
        return [fixed_label] * n_components
