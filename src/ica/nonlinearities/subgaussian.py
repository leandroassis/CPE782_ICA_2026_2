"""Funcao de pontuacao para fontes subgaussianas (distribuicoes achatadas).

Ver context/ICA_BACKGROUND.md, Secao 3.4.
"""

import numpy as np

from ica.nonlinearities.base import NonlinearityTemplate


class SubGaussianScore(NonlinearityTemplate):
    """``g_-(s) = tanh(s) - s``, derivada da densidade log-suposta para distribuicoes achatadas.

    Adequada para fontes como a Uniforme (ICA_BACKGROUND.md, Secao 3.4).
    """

    def score(self, y: np.ndarray) -> np.ndarray:
        """``g_-(y) = tanh(y) - y``.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo.

        Returns
        -------
        np.ndarray
            ``tanh(y) - y``.
        """
        return np.tanh(y) - y

    def derivative(self, y: np.ndarray) -> np.ndarray:
        """``g_-'(y) = -tanh^2(y)``.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo.

        Returns
        -------
        np.ndarray
            ``-tanh(y)**2``.
        """
        return -np.tanh(y) ** 2
