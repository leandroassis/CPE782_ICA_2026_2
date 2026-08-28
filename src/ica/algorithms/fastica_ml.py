"""Algoritmo FastICA adaptado para Maxima Verossimilhanca (ponto fixo em bloco).

Ver context/ICA_BACKGROUND.md, Secao 4.3.
"""

import numpy as np

from ica.algorithms.base import ICAAlgorithm


class FastICAML(ICAAlgorithm):
    """Ponto fixo em bloco para ML, com ortogonalizacao simetrica a cada iteracao.

    Aproxima o Hessiano da log-verossimilhanca por blocos diagonais,
    evitando o calculo exato (computacionalmente proibitivo). Livre de
    taxa de aprendizado -- ``learning_rate`` e ignorado (ICA_BACKGROUND.md,
    Secao 4.3 e tabela comparativa da Secao 4.4).
    """

    def _update_step(self, B: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Aplica um passo de ponto fixo em bloco seguido de ortogonalizacao simetrica.

        Parameters
        ----------
        B : np.ndarray
            Estimativa atual da matriz de separacao.
        X : np.ndarray
            Dados pre-processados, shape ``(n_componentes, n_amostras)``.

        Returns
        -------
        np.ndarray
            ``(B C_x B^T)^{-1/2} B'``, onde ``B'`` e a atualizacao de
            ponto fixo em bloco ``B + diag(alpha)[diag(beta) - g(Y)Y^T/T]B``.
        """
        n_samples = X.shape[1]
        Y = B @ X
        G = self.nonlinearity.score(Y)
        G_derivative = self.nonlinearity.derivative(Y)

        beta = np.mean(Y * G, axis=1)
        alpha = 1.0 / (beta + np.mean(G_derivative, axis=1))

        block_update = np.diag(beta) - (G @ Y.T) / n_samples
        B_updated = B + alpha[:, np.newaxis] * (block_update @ B)

        covariance = (X @ X.T) / n_samples
        return self._symmetric_orthogonalize(B_updated, covariance)

    @staticmethod
    def _symmetric_orthogonalize(B: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """Aplica ``B <- (B C_x B^T)^{-1/2} B``, forcando as saidas a serem descorrelacionadas.

        Parameters
        ----------
        B : np.ndarray
            Matriz de separacao candidata.
        covariance : np.ndarray
            Covariancia amostral dos dados de entrada, ``C_x``.

        Returns
        -------
        np.ndarray
            B ortogonalizado.
        """
        M = B @ covariance @ B.T
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        return inverse_sqrt @ B
