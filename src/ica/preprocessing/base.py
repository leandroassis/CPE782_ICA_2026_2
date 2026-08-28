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
    """

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
