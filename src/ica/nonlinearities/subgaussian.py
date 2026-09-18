"""Funcao de pontuacao para fontes subgaussianas (distribuicoes achatadas).

Ver ``.claude/skills/ica-ml/SKILL.md``, Secoes 3-4.
"""

import numpy as np

from ica.nonlinearities.base import NonlinearityTemplate


class SubGaussianScore(NonlinearityTemplate):
    """``g_-(s) = tanh(s) - s``, funcao de pontuacao subgaussiana (livro, eq. 9.19).

    Adequada para fontes achatadas ou multimodais (Uniforme, binaria,
    arcoseno). A densidade suposta correspondente e

    ``log p_-(s) = log cosh(s) - s^2/2`` (a menos de constante de
    normalizacao),

    isto e, uma log-densidade gaussiana ``-s^2/2`` "achatada" pelo termo
    ``log cosh(s)`` -- exatamente a construcao do livro-texto para o ramo
    subgaussiano da familia de duas densidades. Nas caudas, ``-s^2/2`` domina
    ``log cosh(s) ~ |s|``, entao ``exp(log_density(s))`` decai como uma
    gaussiana e e integravel.

    Toda o pacote adota a convencao ``g = (log p_suposta)'``, **sem** sinal
    negativo na frente (Hyvarinen, Karhunen & Oja, Cap. 9) -- ``g_-`` e
    ``g_+`` seguem a mesma convencao, sob pena de o algoritmo ascender a
    verossimilhanca em uma componente e descende-la em outra (skill
    ``ica-ml``, aviso de convencao).
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
        """``log p_-(y) = log(cosh(y)) - y^2/2``, a antiderivada exata de ``g_-``.

        Calculada de forma numericamente estavel via
        ``log(cosh(y)) = logaddexp(y, -y) - log(2)``.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo.

        Returns
        -------
        np.ndarray
            ``log(cosh(y)) - y**2 / 2``.
        """
        log_cosh = np.logaddexp(y, -y) - np.log(2.0)
        return log_cosh - (y**2) / 2.0
