"""Algoritmo de Gradiente Natural para ICA por Maxima Verossimilhanca.

Ver ``.claude/skills/ica-ml/SKILL.md``, Secao 6; ``references/algorithms.md``.
"""

import numpy as np

from ica.algorithms.base import ICAAlgorithm
from ica.nonlinearities.base import NonlinearityTemplate


class NaturalGradientICA(ICAAlgorithm):
    """Regra de atualizacao do Gradiente Natural: ``B <- B + mu [I + E{g(y) y^T}] B``.

    Deriva do gradiente de Bell-Sejnowski multiplicado pela metrica
    Riemanniana ``B^T B``, o que elimina a necessidade de inverter B a cada
    iteracao e confere equivariancia estrita -- skill ``ica-ml``,
    ``references/algorithms.md``, eq. 9.17. Ponto fixo em
    ``E{g(y) y^T} = -I`` (descorrelacao nao-linear).

    Usa um ``learning_rate`` padrao bem mais conservador que o da classe
    base (``0.0005`` em vez de ``0.01``): verificado empiricamente, tanto em
    fontes sinteticas quanto nas imagens reais de ``data/mix/imagens/run1``,
    que com a nao-linearidade subgaussiana ``g_-(y) = tanh(y) - y``
    (nao-limitada para ``|y|`` grande) o termo multiplicativo
    ``[I + g(y) y^T] B`` diverge numericamente para taxas maiores --
    diferente de Bell-Sejnowski, cuja inversao de matriz atua como um fator
    estabilizante adicional.
    """

    def __init__(
        self,
        nonlinearity: NonlinearityTemplate,
        learning_rate: float = 0.0005,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        """Ver :class:`~ica.algorithms.base.ICAAlgorithm` para a descricao dos parametros.

        A unica diferenca e o valor padrao de ``learning_rate``, reduzido
        por estabilidade numerica (ver docstring da classe).
        """
        super().__init__(
            nonlinearity=nonlinearity,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            random_state=random_state,
        )

    def _update_step(self, B: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Aplica um passo da regra de gradiente natural.

        Parameters
        ----------
        B : np.ndarray
            Estimativa atual da matriz de separacao.
        X : np.ndarray
            Dados pre-processados, shape ``(n_componentes, n_amostras)``.

        Returns
        -------
        np.ndarray
            ``B + learning_rate * (I + (g(Y) @ Y.T) / T) @ B``, onde
            ``Y = B @ X``.
        """
        n_samples = X.shape[1]
        n_components = X.shape[0]
        Y = B @ X
        G = self.nonlinearity.score(Y)
        identity = np.eye(n_components)
        return B + self.learning_rate * (identity + (G @ Y.T) / n_samples) @ B

    def _convergence_residual(self, B_old: np.ndarray, B_new: np.ndarray, X: np.ndarray) -> float:
        """Residuo do ponto fixo ``||I + E{g(y) y^T}||_F`` (skill ``ica-ml``, Secao 7).

        Parameters
        ----------
        B_old : np.ndarray
            Nao usado (o residuo e nativo do ponto fixo, nao de ``B``).
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
