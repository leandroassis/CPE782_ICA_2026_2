"""Branqueamento (whitening) via decomposicao em autovalores da covariancia.

Ver context/ICA_BACKGROUND.md, Secao 2.2.
"""

import numpy as np

from ica.preprocessing.base import PreprocessingStep


class Whitening(PreprocessingStep):
    """Branqueia X, tornando suas componentes descorrelacionadas e de variancia unitaria.

    Implementa ``V = D^(-1/2) E^T``, onde ``E`` e ``D`` vem da decomposicao
    em autovalores/autovetores (EVD) da covariancia
    ``C_x = (1/T) X X^T`` (ICA_BACKGROUND.md, Secao 2.2, Eq. 6.33 do
    livro-texto). Assume que X ja esta centralizado (media zero); aplicar
    :class:`~ica.preprocessing.centering.Centering` antes e
    responsabilidade de quem compoe o
    :class:`~ica.preprocessing.pipeline.Pipeline`.

    Attributes
    ----------
    whitening_matrix_ : np.ndarray or None
        Matriz V, shape ``(n_misturas, n_misturas)``, definida apos
        :meth:`fit`.
    dewhitening_matrix_ : np.ndarray or None
        Inversa de V (``E D^(1/2)``), usada por :meth:`inverse_transform`.
    """

    def __init__(self) -> None:
        self.whitening_matrix_: np.ndarray | None = None
        self.dewhitening_matrix_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Whitening":
        """Estima a matriz de branqueamento a partir da covariancia amostral de X.

        Parameters
        ----------
        X : np.ndarray
            Dados centralizados, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        Whitening
            A propria instancia.
        """
        n_samples = X.shape[1]
        covariance = (X @ X.T) / n_samples
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        inverse_sqrt_eigenvalues = 1.0 / np.sqrt(eigenvalues)
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        self.whitening_matrix_ = inverse_sqrt_eigenvalues[:, np.newaxis] * eigenvectors.T
        self.dewhitening_matrix_ = eigenvectors * sqrt_eigenvalues[np.newaxis, :]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica ``z = V x``.

        Parameters
        ----------
        X : np.ndarray
            Dados centralizados, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        np.ndarray
            Dados branqueados, mesma shape de ``X``.
        """
        return self.whitening_matrix_ @ X

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Desfaz o branqueamento, aproximando x a partir de z.

        Parameters
        ----------
        Y : np.ndarray
            Dados branqueados, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        np.ndarray
            Dados aproximados no espaco (centralizado) original.
        """
        return self.dewhitening_matrix_ @ Y
