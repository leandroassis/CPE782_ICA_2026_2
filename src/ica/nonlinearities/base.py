"""Interface base para funcoes de pontuacao (score functions).

Ver context/ICA_BACKGROUND.md, Secao 3.3-3.4.
"""

from abc import ABC, abstractmethod

import numpy as np


class NonlinearityTemplate(ABC):
    """Funcao de pontuacao ``g_i(s) = -d/ds log(p_suposta(s))`` da estimacao ML/Infomax.

    Cada implementacao concreta corresponde a uma densidade suposta
    diferente para as fontes (ICA_BACKGROUND.md, Secao 3.3). Pelo Teorema
    da consistencia local (Secao 3.3), a densidade suposta nao precisa ser
    exata -- basta que ``g_i`` opere no lado correto da nao-gaussianidade
    da fonte.
    """

    @abstractmethod
    def score(self, y: np.ndarray) -> np.ndarray:
        """Calcula ``g(y)`` elemento a elemento.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo, shape ``(n_componentes, n_amostras)``.

        Returns
        -------
        np.ndarray
            ``g(y)``, mesma shape de ``y``.
        """

    @abstractmethod
    def derivative(self, y: np.ndarray) -> np.ndarray:
        """Calcula ``g'(y)`` elemento a elemento.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo, shape ``(n_componentes, n_amostras)``.

        Returns
        -------
        np.ndarray
            ``g'(y)``, mesma shape de ``y``.
        """

    @abstractmethod
    def log_density(self, y: np.ndarray) -> np.ndarray:
        """Calcula ``log p_suposta(y)`` elemento a elemento (antiderivada de ``-g``).

        Usada para acompanhar a log-verossimilhanca media por iteracao
        (ICA_BACKGROUND.md, Secao 3.2): ``(1/T) log L(B) = sum_i E{log
        p_i(b_i^T x)} + log|det B|``.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo, shape ``(n_componentes, n_amostras)``.

        Returns
        -------
        np.ndarray
            ``log p_suposta(y)``, mesma shape de ``y``.
        """
