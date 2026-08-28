"""Algoritmo de Gradiente Natural para ICA por Maxima Verossimilhanca.

Ver context/ICA_BACKGROUND.md, Secao 4.2.
"""

import numpy as np

from ica.algorithms.base import ICAAlgorithm
from ica.nonlinearities.base import NonlinearityTemplate


class NaturalGradientICA(ICAAlgorithm):
    """Regra de atualizacao do Gradiente Natural: ``B <- B + lr [I - g(y) y^T / T] B``.

    Deriva do gradiente de Bell-Sejnowski multiplicado pela metrica
    Riemanniana ``B^T B``, o que elimina a necessidade de inverter B a
    cada iteracao e confere equivariancia estrita (ICA_BACKGROUND.md,
    Secao 4.2).

    Usa um ``learning_rate`` padrao bem mais conservador que o da classe
    base (``0.0005`` em vez de ``0.01``): verificado empiricamente, tanto
    em fontes sinteticas quanto nas imagens reais de
    ``data/imagens/run1``, que com a nao-linearidade subgaussiana
    ``g_-(s) = tanh(s) - s`` (nao-limitada para ``|s|`` grande) o termo
    multiplicativo ``[I - g(y) y^T]B`` diverge numericamente para taxas
    maiores -- diferente de Bell-Sejnowski, cuja inversao de matriz atua
    como um fator estabilizante adicional. Mesmo em ``0.001`` o algoritmo
    ainda divergiu sobre dados reais; ``0.0005`` foi a maior taxa estavel
    encontrada nesse teste.
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
            ``B + learning_rate * (I - (g(Y) @ Y.T) / T) @ B``, onde
            ``Y = B @ X``.
        """
        n_samples = X.shape[1]
        n_components = X.shape[0]
        Y = B @ X
        G = self.nonlinearity.score(Y)
        identity = np.eye(n_components)
        return B + self.learning_rate * (identity - (G @ Y.T) / n_samples) @ B
