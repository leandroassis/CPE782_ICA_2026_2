"""Algoritmo de Bell-Sejnowski para ICA por Maxima Verossimilhanca / Infomax.

Ver ``.claude/skills/ica-ml/SKILL.md``, Secao 6; ``references/algorithms.md``.
"""

import numpy as np

from ica.algorithms.base import ICAAlgorithm


class BellSejnowskiICA(ICAAlgorithm):
    """Regra de atualizacao classica: ``B <- B + mu [(B^T)^-1 + g(y) x^T / T]``.

    Gradiente direto (euclidiano) da log-verossimilhanca -- skill
    ``ica-ml``, ``references/algorithms.md``, eq. 9.15/9.16. Requer
    inverter B a cada iteracao -- computacionalmente mais caro e com
    convergencia mais lenta que o Gradiente Natural, mas nao exige dados
    branqueados para funcionar corretamente.
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
            ``B + learning_rate * ((B^T)^-1 + (g(Y) @ X.T) / T)``, onde
            ``Y = B @ X``.
        """
        n_samples = X.shape[1]
        Y = B @ X
        G = self.nonlinearity.score(Y)
        inverse_transpose = np.linalg.inv(B.T)
        return B + self.learning_rate * (inverse_transpose + (G @ X.T) / n_samples)

    def _convergence_residual(self, B_old: np.ndarray, B_new: np.ndarray, X: np.ndarray) -> float:
        """Residuo do ponto fixo ``||I + E{g(y) y^T}||_F`` (skill ``ica-ml``, Secao 7).

        Mesmo residuo nativo usado pelo Gradiente Natural: no ponto fixo de
        ambos, ``E{g(y) y^T} = -I``.

        Parameters
        ----------
        B_old : np.ndarray
            Nao usado.
        B_new : np.ndarray
            Estimativa de B na qual avaliar o residuo.
        X : np.ndarray
            Dados pre-processados, shape ``(n_componentes, n_amostras)``.

        Returns
        -------
        float
            ``||I + E{g(y) y^T}||_F``.
        """
        n_samples = X.shape[1]
        n_components = X.shape[0]
        Y = B_new @ X
        G = self.nonlinearity.score(Y)
        identity = np.eye(n_components)
        return float(np.linalg.norm(identity + (G @ Y.T) / n_samples))
