"""Algoritmo de Bell-Sejnowski para ICA por Maxima Verossimilhanca / Infomax.

Ver context/ICA_BACKGROUND.md, Secao 4.1.
"""

import numpy as np

from ica.algorithms.base import ICAAlgorithm


class BellSejnowskiICA(ICAAlgorithm):
    """Regra de atualizacao classica: ``B <- B + lr [(B^T)^-1 - g(y) x^T / T]``.

    Gradiente direto (euclidiano) da log-verossimilhanca (ICA_BACKGROUND.md,
    Secao 4.1). Requer inverter B a cada iteracao -- computacionalmente
    mais caro e com convergencia mais lenta que o Gradiente Natural
    (Secao 4.2), mas nao exige dados branqueados para funcionar
    corretamente.
    """

    def _update_step(self, B: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Aplica um passo da regra de Bell-Sejnowski.

        Parameters
        ----------
        B : np.ndarray
            Estimativa atual da matriz de separacao.
        X : np.ndarray
            Dados pre-processados, shape ``(n_componentes, n_amostras)``.

        Returns
        -------
        np.ndarray
            ``B + learning_rate * ((B^T)^-1 - (g(Y) @ X.T) / T)``, onde
            ``Y = B @ X``.
        """
        n_samples = X.shape[1]
        Y = B @ X
        G = self.nonlinearity.score(Y)
        inverse_transpose = np.linalg.inv(B.T)
        return B + self.learning_rate * (inverse_transpose - (G @ X.T) / n_samples)
