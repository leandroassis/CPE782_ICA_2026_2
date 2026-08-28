"""Interface base para algoritmos de ICA (Template Method).

Ver context/ICA_BACKGROUND.md, Secao 4; context/DEVELOPMENT_GUIDELINES.md, Secao 2.4.
"""

import time
from abc import ABC, abstractmethod

import numpy as np

from ica.nonlinearities.base import NonlinearityTemplate


class ICAAlgorithm(ABC):
    """Algoritmo de otimizacao para estimar a matriz de separacao B.

    Implementa o padrao Template Method: :meth:`fit` define o esqueleto
    fixo do processo iterativo (inicializar B, repetir o passo de
    atualizacao ate convergencia ou numero maximo de iteracoes,
    cronometrando e registrando o historico), delegando as subclasses
    apenas o passo de atualizacao em si, via :meth:`_update_step`.

    Parameters
    ----------
    nonlinearity : NonlinearityTemplate
        Funcao de pontuacao g(y) usada no passo de atualizacao (injetada
        por dependencia -- ver DEVELOPMENT_GUIDELINES.md, Secao 3, "D --
        Dependency Inversion").
    learning_rate : float, default=0.01
        Passo da atualizacao. Ignorado por algoritmos livres de taxa
        (ex.: ``FastICAML``), que documentam essa excecao explicitamente.
    max_iterations : int, default=500
        Numero maximo de iteracoes antes de parar mesmo sem convergencia.
    tolerance : float, default=1e-6
        Limiar de variacao relativa de B abaixo do qual o algoritmo e
        considerado convergido (ver :meth:`_has_converged`).
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
        Variacao relativa de B a cada iteracao (para diagnostico/plot de
        convergencia).
    log_likelihood_history_ : list of float
        Log-verossimilhanca media -- ``(1/T) log L(B)``, ICA_BACKGROUND.md
        Secao 3.2 -- calculada a cada iteracao com a matriz B ja
        atualizada. Cresce (ou se mantem estavel) ao longo das iteracoes
        quando o algoritmo esta de fato ascendendo a verossimilhanca.
    elapsed_time_ : float or None
        Tempo de execucao de :meth:`fit`, em segundos.
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

    def fit(self, X: np.ndarray) -> np.ndarray:
        """Estima a matriz de separacao B que torna Y = BX o mais independente possivel.

        Esqueleto fixo (Template Method): inicializa B, itera
        :meth:`_update_step` ate a convergencia (:meth:`_has_converged`)
        ou ``max_iterations``, cronometrando e registrando o historico de
        convergencia.

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

        for iteration in range(1, self.max_iterations + 1):
            B_new = self._update_step(B, X)
            self.history_.append(self._relative_change(B, B_new))
            self.log_likelihood_history_.append(self._log_likelihood(B_new, X))
            has_converged = self._has_converged(B, B_new)
            B = B_new
            if has_converged:
                self.converged_ = True
                break

        self.unmixing_matrix_ = B
        self.n_iterations_ = iteration
        self.elapsed_time_ = time.perf_counter() - start_time
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

        Unico ponto variavel do Template Method -- cada algoritmo
        concreto (Bell-Sejnowski, Gradiente Natural, FastICA-ML)
        implementa aqui sua regra de atualizacao especifica (ver
        ICA_BACKGROUND.md, Secao 4).

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

    def _relative_change(self, B_old: np.ndarray, B_new: np.ndarray) -> float:
        """Variacao relativa de B entre duas iteracoes, na norma de Frobenius.

        Parameters
        ----------
        B_old : np.ndarray
            Estimativa anterior de B.
        B_new : np.ndarray
            Nova estimativa de B.

        Returns
        -------
        float
            ``||B_new - B_old||_F / ||B_old||_F``.
        """
        return float(np.linalg.norm(B_new - B_old) / np.linalg.norm(B_old))

    def _log_likelihood(self, B: np.ndarray, X: np.ndarray) -> float:
        """Log-verossimilhanca media no ponto ``B`` (ICA_BACKGROUND.md, Secao 3.2).

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

    def _has_converged(self, B_old: np.ndarray, B_new: np.ndarray) -> bool:
        """Verifica se a variacao relativa de B esta abaixo de ``tolerance``.

        Parameters
        ----------
        B_old : np.ndarray
            Estimativa anterior de B.
        B_new : np.ndarray
            Nova estimativa de B.

        Returns
        -------
        bool
            ``True`` se ``_relative_change(B_old, B_new) < tolerance``.
        """
        return self._relative_change(B_old, B_new) < self.tolerance
