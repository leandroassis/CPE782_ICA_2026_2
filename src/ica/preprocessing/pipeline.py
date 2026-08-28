"""Composicao encadeada de passos de pre-processamento.

Ver context/DEVELOPMENT_GUIDELINES.md, Secao 2.2.
"""

import numpy as np

from ica.preprocessing.base import PreprocessingStep


class Pipeline:
    """Encadeia varios :class:`PreprocessingStep`, aplicando-os em sequencia.

    Parameters
    ----------
    steps : list of PreprocessingStep
        Passos a aplicar, na ordem informada.

    Attributes
    ----------
    steps : list of PreprocessingStep
        Os passos que compoem o pipeline.
    """

    def __init__(self, steps: list[PreprocessingStep]) -> None:
        self.steps = steps

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Ajusta e aplica cada passo, em ordem, sobre a saida do anterior.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        np.ndarray
            Dados apos todos os passos.
        """
        for step in self.steps:
            X = step.fit_transform(X)
        return X

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Desfaz todos os passos, na ordem inversa em que foram aplicados.

        Parameters
        ----------
        Y : np.ndarray
            Dados no espaco totalmente transformado.

        Returns
        -------
        np.ndarray
            Dados aproximados no espaco original.
        """
        for step in reversed(self.steps):
            Y = step.inverse_transform(Y)
        return Y

    def get_step(self, step_type: type[PreprocessingStep]) -> PreprocessingStep:
        """Recupera, pelo tipo, um passo ja ajustado do pipeline.

        Usado por :class:`~ica.model.ICAModel` para acessar a matriz de
        branqueamento e compor a matriz de separacao completa.

        Parameters
        ----------
        step_type : type
            Classe concreta do passo procurado (ex.: ``Whitening``).

        Returns
        -------
        PreprocessingStep
            A instancia correspondente.

        Raises
        ------
        ValueError
            Se nenhum passo do tipo pedido estiver no pipeline.
        """
        for step in self.steps:
            if isinstance(step, step_type):
                return step
        raise ValueError(f"Nenhum passo do tipo {step_type.__name__} encontrado no pipeline.")
