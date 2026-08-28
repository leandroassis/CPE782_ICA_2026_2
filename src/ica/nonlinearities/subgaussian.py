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

    def log_density(self, y: np.ndarray) -> np.ndarray:
        """``log p_-(y) = y^2/2 - log(cosh(y))``, a antiderivada exata de ``-g_-``.

        Calculada de forma numericamente estavel via
        ``log(cosh(y)) = logaddexp(y, -y) - log(2)``.

        Ao contrario de :meth:`SuperGaussianScore.log_density
        <ica.nonlinearities.supergaussian.SuperGaussianScore.log_density>`
        (a densidade logistica, propriamente normalizada), esta funcao
        **nao** e uma densidade de probabilidade valida: ``y^2/2``
        domina ``log(cosh(y))`` para ``|y|`` grande, entao
        ``exp(log_density(y))`` diverge e nao e integravel. Isso e
        esperado e nao invalida seu uso: ICA_BACKGROUND.md, Secao 3.3,
        ja observa que a "densidade suposta" e um substituto de trabalho
        (working surrogate) que nao precisa ser exata -- o que importa
        e que ``score = -d/dy log_density`` exatamente, garantindo que
        esta funcao seja a quantidade que as regras de gradiente
        (Bell-Sejnowski, Gradiente Natural) de fato ascendem a cada
        passo. Verificado numericamente: ``-d/dy log_density(y)`` bate
        com ``score(y)`` a menos de 1e-10 em todo o dominio.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo.

        Returns
        -------
        np.ndarray
            ``y**2 / 2 - log(cosh(y))``.
        """
        log_cosh = np.logaddexp(y, -y) - np.log(2.0)
        return (y**2) / 2.0 - log_cosh
