"""Funcao de pontuacao para fontes supergaussianas (caudas pesadas).

Ver ``.claude/skills/ica-ml/SKILL.md``, Secoes 3-4.
"""

import numpy as np

from ica.nonlinearities.base import NonlinearityTemplate


class SuperGaussianScore(NonlinearityTemplate):
    """``g_+(s) = -2 tanh(s)``, funcao de pontuacao para caudas pesadas (livro, eq. 9.18).

    Densidade log suposta: ``log p_+(s) = -log(2) - 2 log(cosh(s))`` (a
    densidade logistica padrao, propriamente normalizada). A convencao do
    pacote e ``g = (log p_suposta)'`` -- **sem** sinal negativo na frente
    (Hyvarinen, Karhunen & Oja, Cap. 9); adequada para fontes como a
    Laplaciana.
    """

    def score(self, y: np.ndarray) -> np.ndarray:
        """``g_+(y) = -2 tanh(y)``.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo.

        Returns
        -------
        np.ndarray
            ``-2 * tanh(y)``.
        """
        return -2.0 * np.tanh(y)

    def derivative(self, y: np.ndarray) -> np.ndarray:
        """``g_+'(y) = -2 (1 - tanh^2(y))``.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo.

        Returns
        -------
        np.ndarray
            ``-2 * (1 - tanh(y)**2)``.
        """
        return -2.0 * (1.0 - np.tanh(y) ** 2)

    def log_density(self, y: np.ndarray) -> np.ndarray:
        """``log p_+(y) = -log(2) - 2 log(cosh(y))``, a antiderivada exata de ``g_+``.

        Calculada de forma numericamente estavel via
        ``log(cosh(y)) = logaddexp(y, -y) - log(2)``. E uma densidade de
        probabilidade propriamente normalizada (``integral exp(log_density) =
        1``): e a densidade da distribuicao logistica padrao, verificado
        numericamente.

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
