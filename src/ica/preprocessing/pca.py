"""Reducao de dimensionalidade por Analise de Componentes Principais (PCA).

Ver livro-texto (Hyvarinen, Karhunen & Oja), Secao 13.2 (paginas 267-269) --
a mesma decomposicao em autovalores/autovetores usada pelo branqueamento
(``.claude/skills/ica-ml/SKILL.md``, Secao 6). Gancho exposto e off por
padrao (``.claude/PIPELINE_MAP.md``, bloco ``preprocess/``).
"""

import numpy as np

from ica.preprocessing.base import PreprocessingStep


class PCA(PreprocessingStep):
    """Projeta X nas direcoes de maior variancia, reduzindo sua dimensao.

    Implementa ``x~ = E_n^T x`` (Eq. 13.6 do livro-texto), com ``E_n``
    formada pelos autovetores de maior autovalor da covariancia amostral
    ``C_x = (1/T) X X^T``. Assume que X ja esta centralizado (mesma
    convencao de :class:`~ica.preprocessing.whitening.Whitening`).

    Dois usos praticos documentados na Secao 13.2 do livro-texto:

    - Igualar o numero de misturas ao numero de fontes quando ha mais
      misturas do que fontes (Secao 13.2.1, "making the mixing matrix
      square").
    - Reduzir ruido e evitar *overlearning* descartando as direcoes de
      menor variancia, tipicamente dominadas por ruido -- mas reduzir
      demais tambem pode descartar fontes fracas (Secao 13.2.2, Fig. 13.1:
      dimensao insuficiente produz separacoes espurias tipo "spike").

    Parameters
    ----------
    n_components : int or float
        Se ``int``, numero exato de componentes principais a reter. Se
        ``float`` em ``(0, 1]``, a menor quantidade de componentes cuja
        soma de autovalores explica pelo menos essa fracao da variancia
        total (Secao 13.3, "the minimum number of principal components
        that explain the data well enough").

    Attributes
    ----------
    components_ : np.ndarray or None
        Autovetores retidos como colunas, shape
        ``(n_misturas, n_componentes_retidos)``, definido apos
        :meth:`fit`.
    explained_variance_ratio_ : np.ndarray or None
        Fracao da variancia total explicada por cada componente retido,
        em ordem decrescente.
    n_components_ : int or None
        Numero de componentes efetivamente retidos apos :meth:`fit`.
    linear_matrix_ : np.ndarray or None
        Alias de ``components_.T`` (ver
        :attr:`~ica.preprocessing.base.PreprocessingStep.linear_matrix_`),
        usado por ``Pipeline.compose_linear_matrix`` para montar
        ``full_unmixing_matrix_``.
    """

    def __init__(self, n_components: int | float) -> None:
        if isinstance(n_components, float) and not (0.0 < n_components <= 1.0):
            raise ValueError(
                f"n_components como float deve estar em (0, 1], recebido {n_components}."
            )
        if isinstance(n_components, int) and n_components < 1:
            raise ValueError(f"n_components como int deve ser >= 1, recebido {n_components}.")
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.n_components_: int | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        """Estima as direcoes principais a partir da covariancia amostral de X.

        Parameters
        ----------
        X : np.ndarray
            Dados centralizados, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        PCA
            A propria instancia.

        Raises
        ------
        ValueError
            Se ``n_components`` (inteiro) exceder o numero de misturas.
        """
        n_mixtures = X.shape[0]
        covariance = (X @ X.T) / X.shape[1]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        total_variance = eigenvalues.sum()

        if isinstance(self.n_components, float):
            cumulative_ratio = np.cumsum(eigenvalues) / total_variance
            k = int(np.searchsorted(cumulative_ratio, self.n_components) + 1)
            k = min(k, n_mixtures)
        else:
            k = self.n_components
            if k > n_mixtures:
                raise ValueError(f"n_components={k} excede o numero de misturas ({n_mixtures}).")

        self.components_ = eigenvectors[:, :k]
        self.explained_variance_ratio_ = eigenvalues[:k] / total_variance
        self.n_components_ = k
        return self

    @property
    def linear_matrix_(self) -> np.ndarray | None:
        """Ver Attributes da classe. Alias de ``components_.T``."""
        return None if self.components_ is None else self.components_.T

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projeta X no subespaco principal: ``x~ = E_n^T x``.

        Parameters
        ----------
        X : np.ndarray
            Dados centralizados, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        np.ndarray
            Dados projetados, shape ``(n_componentes_retidos, n_amostras)``.
        """
        return self.components_.T @ X

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Reconstroi uma aproximacao de X a partir do subespaco reduzido.

        Reconstrucao com perdas (least-squares) quando
        ``n_componentes_retidos < n_misturas`` -- as direcoes descartadas
        nao podem ser recuperadas (Secao 6.1.2 do livro-texto, "PCA by
        minimum MSE compression").

        Parameters
        ----------
        Y : np.ndarray
            Dados no subespaco reduzido, shape
            ``(n_componentes_retidos, n_amostras)``.

        Returns
        -------
        np.ndarray
            Aproximacao de X no espaco original, shape
            ``(n_misturas, n_amostras)``.
        """
        return self.components_ @ Y
