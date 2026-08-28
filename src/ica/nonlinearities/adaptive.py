"""Chaveamento adaptativo entre nao-linearidades super e subgaussianas.

Ver context/ICA_BACKGROUND.md, Secao 3.4.
"""

import numpy as np

from ica.nonlinearities.base import NonlinearityTemplate
from ica.nonlinearities.subgaussian import SubGaussianScore
from ica.nonlinearities.supergaussian import SuperGaussianScore


class AdaptiveScore(NonlinearityTemplate):
    """Escolhe, por componente, entre ``g_+`` e ``g_-`` via o momento ``gamma_i``.

    Implementa o mecanismo de chaveamento de ICA_BACKGROUND.md, Secao 3.4:
    ``gamma_i = E{tanh(y_i) y_i - (1 - tanh^2(y_i))}``; se ``gamma_i < 0``
    a componente e supergaussiana (usa ``g_+``), caso contrario e
    subgaussiana (usa ``g_-``). Essa direcao (e nao o inverso) e a que
    corresponde a estatistica de chaveamento do Extended Infomax (Lee,
    Girolami & Sejnowski, 1999) -- confirmado empiricamente em
    ICA_BACKGROUND.md, Secao 3.4. O chaveamento e recalculado a cada
    chamada, a partir do ``y`` recebido nessa chamada.

    Notes
    -----
    ``tanh`` nao e invariante a escala: o sinal de ``gamma_i`` so reflete
    de forma confiavel a nao-gaussianidade da fonte quando ``y`` tem
    variancia proxima de 1 (ou seja, apos branqueamento -- Secao 2.2). Em
    ``y`` com escala muito maior ou menor que 1 o sinal de ``gamma_i``
    pode se inverter independentemente do formato da distribuicao.

    Parameters
    ----------
    super_gaussian : NonlinearityTemplate, optional
        Nao-linearidade usada quando ``gamma_i < 0``. Por padrao, uma nova
        ``SuperGaussianScore()``. Injetavel para testes/customizacao.
    sub_gaussian : NonlinearityTemplate, optional
        Nao-linearidade usada quando ``gamma_i >= 0``. Por padrao, uma
        nova ``SubGaussianScore()``. Injetavel para testes/customizacao.

    Attributes
    ----------
    gamma_ : np.ndarray or None
        Ultimo vetor de momentos ``gamma_i`` calculado, shape
        ``(n_componentes,)``, definido apos a primeira chamada a
        :meth:`score` ou :meth:`derivative`.
    is_super_gaussian_ : np.ndarray or None
        Mascara booleana com o resultado do ultimo chaveamento por
        componente, shape ``(n_componentes,)``.
    """

    def __init__(
        self,
        super_gaussian: NonlinearityTemplate | None = None,
        sub_gaussian: NonlinearityTemplate | None = None,
    ) -> None:
        self._super_gaussian = (
            super_gaussian if super_gaussian is not None else SuperGaussianScore()
        )
        self._sub_gaussian = sub_gaussian if sub_gaussian is not None else SubGaussianScore()
        self.gamma_: np.ndarray | None = None
        self.is_super_gaussian_: np.ndarray | None = None

    def _update_switch(self, y: np.ndarray) -> None:
        """Recalcula ``gamma_`` e ``is_super_gaussian_`` a partir de ``y``."""
        tanh_y = np.tanh(y)
        self.gamma_ = np.mean(tanh_y * y - (1.0 - tanh_y**2), axis=1)
        self.is_super_gaussian_ = self.gamma_ < 0

    def score(self, y: np.ndarray) -> np.ndarray:
        """Calcula ``g(y)`` delegando por componente conforme o sinal de ``gamma_i``.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo, shape ``(n_componentes, n_amostras)``.

        Returns
        -------
        np.ndarray
            ``g(y)``, mesma shape de ``y``, com cada linha calculada por
            ``g_+`` ou ``g_-`` conforme o chaveamento.
        """
        self._update_switch(y)
        return np.where(
            self.is_super_gaussian_[:, np.newaxis],
            self._super_gaussian.score(y),
            self._sub_gaussian.score(y),
        )

    def derivative(self, y: np.ndarray) -> np.ndarray:
        """Calcula ``g'(y)`` delegando por componente conforme o sinal de ``gamma_i``.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo, shape ``(n_componentes, n_amostras)``.

        Returns
        -------
        np.ndarray
            ``g'(y)``, mesma shape de ``y``.
        """
        self._update_switch(y)
        return np.where(
            self.is_super_gaussian_[:, np.newaxis],
            self._super_gaussian.derivative(y),
            self._sub_gaussian.derivative(y),
        )

    def log_density(self, y: np.ndarray) -> np.ndarray:
        """Calcula ``log p(y)`` delegando por componente conforme o sinal de ``gamma_i``.

        Parameters
        ----------
        y : np.ndarray
            Saida atual do modelo, shape ``(n_componentes, n_amostras)``.

        Returns
        -------
        np.ndarray
            ``log p(y)``, mesma shape de ``y``.
        """
        self._update_switch(y)
        return np.where(
            self.is_super_gaussian_[:, np.newaxis],
            self._super_gaussian.log_density(y),
            self._sub_gaussian.log_density(y),
        )
