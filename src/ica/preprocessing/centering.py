"""Centralizacao (subtracao da media amostral).

Ver context/ICA_BACKGROUND.md, Secao 2.1.
"""

import numpy as np

from ica.preprocessing.base import PreprocessingStep


class Centering(PreprocessingStep):
    """Subtrai a media amostral de cada mistura, tornando X de media zero.

    Implementa ``x <- x - E{x}`` (ICA_BACKGROUND.md, Secao 2.1). A media
    estimada e guardada em ``mean_`` para poder ser somada de volta as
    componentes recuperadas por :meth:`inverse_transform`.

    Attributes
    ----------
    mean_ : np.ndarray or None
        Vetor de medias amostrais por mistura, shape ``(n_misturas,)``,
        definido apos :meth:`fit`.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Centering":
        """Estima a media amostral de cada mistura (linha) de X.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        Centering
            A propria instancia.
        """
        self.mean_ = X.mean(axis=1)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Subtrai ``mean_`` de cada mistura.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        np.ndarray
            Dados centralizados, mesma shape de ``X``.
        """
        return X - self.mean_[:, np.newaxis]

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Soma ``mean_`` de volta, desfazendo a centralizacao.

        Parameters
        ----------
        Y : np.ndarray
            Dados centralizados, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        np.ndarray
            Dados com a media original restaurada.
        """
        return Y + self.mean_[:, np.newaxis]
