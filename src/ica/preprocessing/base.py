"""Interface base para passos de pre-processamento encadeaveis.

Ver context/DEVELOPMENT_GUIDELINES.md, Secao 2.2.
"""

from abc import ABC, abstractmethod

import numpy as np


class PreprocessingStep(ABC):
    """Um passo de pre-processamento ajustavel e reversivel sobre X.

    Convencao de forma usada em todo o pacote: X tem shape
    ``(n_misturas, n_amostras)`` -- cada linha e uma mistura observada,
    cada coluna e uma amostra/observacao.

    Attributes
    ----------
    estimation_only : bool
        Se ``True``, o passo participa apenas da estimacao da matriz de
        separacao B -- usado por
        :meth:`~ica.preprocessing.pipeline.Pipeline.fit_transform`, cuja
        saida alimenta ``ICAAlgorithm.fit`` -- mas e ignorado tanto por
        :meth:`~ica.preprocessing.pipeline.Pipeline.reconstruction_transform`
        (que recupera as fontes finais a partir dos dados originais)
        quanto por
        :meth:`~ica.preprocessing.pipeline.Pipeline.compose_linear_matrix`
        (usado para ``full_unmixing_matrix_``). ``False`` por padrao.
        Usado por
        :class:`~ica.preprocessing.temporal_filtering.TemporalFiltering`
        (livro-texto, Secao 13.1, p.263-267): a filtragem melhora a
        estimacao de B, mas nao deve encurtar nem alterar o sinal final
        recuperado -- "we can use the filtered data in the ICA estimating
        method only. After estimating the mixing matrix, we can apply the
        same mixing matrix on the original data".
    linear_matrix_ : np.ndarray or None
        Matriz linear equivalente deste passo no espaco de misturas
        (canais), usada por
        :meth:`~ica.preprocessing.pipeline.Pipeline.compose_linear_matrix`
        para construir ``full_unmixing_matrix_`` independentemente da
        ordem em que os passos concretos aparecem no pipeline. ``None``
        por padrao -- passos afins
        (:class:`~ica.preprocessing.centering.Centering`) ou que atuam no
        eixo do tempo em vez do eixo dos canais
        (:class:`~ica.preprocessing.temporal_filtering.TemporalFiltering`)
        nao contribuem. Sobrescrita por passos que de fato representam
        uma transformacao linear no espaco de canais (ex.: ``Whitening``,
        ``PCA``).
    """

    estimation_only: bool = False

    @property
    def linear_matrix_(self) -> np.ndarray | None:
        """Ver Attributes da classe. ``None`` por padrao."""
        return None

    @abstractmethod
    def fit(self, X: np.ndarray) -> "PreprocessingStep":
        """Estima os parametros do passo a partir dos dados.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        PreprocessingStep
            A propria instancia (permite encadear ``fit(X).transform(X)``).
        """

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica a transformacao ja ajustada.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        np.ndarray
            Dados transformados, mesma shape de ``X``.
        """

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Ajusta o passo a X e em seguida o transforma.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        np.ndarray
            Dados transformados.
        """
        return self.fit(X).transform(X)

    @abstractmethod
    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Desfaz a transformacao, trazendo Y de volta ao espaco original de X.

        Parameters
        ----------
        Y : np.ndarray
            Dados no espaco transformado, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        np.ndarray
            Dados aproximados no espaco original.
        """
