"""Branqueamento (whitening) via decomposicao em autovalores da covariancia.

Ver ``.claude/skills/ica-ml/SKILL.md``, Secao 6 (branqueamento como
pre-requisito comum dos 3 algoritmos).
"""

import numpy as np

from ica.preprocessing.base import PreprocessingStep
from ica.preprocessing.symmetric import pca_whitening_matrices


class Whitening(PreprocessingStep):
    """Branqueia X, tornando suas componentes descorrelacionadas e de variancia unitaria.

    Implementa ``V = D^(-1/2) E^T``, onde ``E`` e ``D`` vem da decomposicao
    em autovalores/autovetores (EVD) da covariancia
    ``C_x = (1/T) X X^T`` (skill ``ica-ml``, ``references/algorithms.md``,
    "pre-requisitos comuns"), via :func:`~ica.preprocessing.symmetric.pca_whitening_matrices`
    -- a mesma rotina de raiz-inversa reaproveitada pela projecao do
    FastICA-ML. Assume que X ja esta centralizado (media zero); aplicar
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
    linear_matrix_ : np.ndarray or None
        Alias de ``whitening_matrix_`` (ver
        :attr:`~ica.preprocessing.base.PreprocessingStep.linear_matrix_`),
        usado por ``Pipeline.compose_linear_matrix`` para montar
        ``full_unmixing_matrix_``.
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
        self.whitening_matrix_, self.dewhitening_matrix_ = pca_whitening_matrices(covariance)
        return self

    @property
    def linear_matrix_(self) -> np.ndarray | None:
        """Ver Attributes da classe. Alias de ``whitening_matrix_``."""
        return self.whitening_matrix_

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
