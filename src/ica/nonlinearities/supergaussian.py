"""Funcao de pontuacao para fontes supergaussianas (caudas pesadas).

Ver context/ICA_BACKGROUND.md, Secao 3.4.
"""

import numpy as np

from ica.nonlinearities.base import NonlinearityTemplate


class SuperGaussianScore(NonlinearityTemplate):
    """``g_+(s) = 2 tanh(s)``, derivada da densidade log-suposta para caudas pesadas.

    Adequada para fontes como a Laplaciana (ICA_BACKGROUND.md, Secao 3.4).
    """

    def score(self, y: np.ndarray) -> np.ndarray:
        """``g_+(y) = 2 tanh(y)``.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo.

        Returns
        -------
        np.ndarray
            ``2 * tanh(y)``.
        """
        return 2.0 * np.tanh(y)

    def derivative(self, y: np.ndarray) -> np.ndarray:
        """``g_+'(y) = 2 (1 - tanh^2(y))``.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo.

        Returns
        -------
        np.ndarray
            ``2 * (1 - tanh(y)**2)``.
        """
        return 2.0 * (1.0 - np.tanh(y) ** 2)

    def log_density(self, y: np.ndarray) -> np.ndarray:
        """``log p_+(y) = -log(2) - 2 log(cosh(y))``, a antiderivada exata de ``-g_+``.

        Calculada de forma numericamente estavel via
        ``log(cosh(y)) = logaddexp(y, -y) - log(2)``. Diferente do caso
        subgaussiano (ver
        :meth:`SubGaussianScore.log_density
        <ica.nonlinearities.subgaussian.SubGaussianScore.log_density>`),
        esta e uma densidade de probabilidade propriamente normalizada
        (``integral exp(log_density) = 1``): e a densidade da
        distribuicao logistica padrao, verificado numericamente.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo.

        Returns
        -------
        np.ndarray
            ``-log(2) - 2*log(cosh(y))``.
        """
        log_cosh = np.logaddexp(y, -y) - np.log(2.0)
        return -np.log(2.0) - 2.0 * log_cosh
