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

    def reconstruction_transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica somente os passos que nao sao ``estimation_only``, ja ajustados.

        Usa os parametros ja estimados por :meth:`fit_transform` (chama
        ``step.transform``, nunca ``step.fit``), pulando os passos com
        ``estimation_only = True`` (ex.:
        :class:`~ica.preprocessing.temporal_filtering.TemporalFiltering`).
        Isso implementa a recomendacao do livro-texto (Secao 13.1, p.264):
        a filtragem temporal so deve entrar na estimacao da matriz de
        separacao B, nunca na reconstrucao final das fontes, que deve
        partir dos dados originais para preservar sua forma/comprimento.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada -- tipicamente ``mixtures_``, nao filtrado
            -- shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        np.ndarray
            Dados prontos para multiplicar pela matriz de separacao B e
            obter as fontes finais.
        """
        for step in self.steps:
            if step.estimation_only:
                continue
            X = step.transform(X)
        return X

    def compose_linear_matrix(self, base_matrix: np.ndarray) -> np.ndarray:
        """Compoe ``base_matrix`` com a matriz linear equivalente de cada passo do pipeline.

        Usado por :class:`~ica.model.ICAModel` para construir
        ``full_unmixing_matrix_``: uma unica matriz que mapeia diretamente
        das misturas originais (apos centralizacao) para as fontes
        recuperadas, sem que o modelo precise conhecer quais passos
        concretos (``Whitening``, ``PCA``, ...) compoem o pipeline nem em
        que ordem. Percorre os passos em ordem inversa, multiplicando
        ``base_matrix`` a direita por ``step.linear_matrix_`` sempre que
        definido; passos afins (sem ``linear_matrix_``) ou marcados
        ``estimation_only`` (ver :class:`~ica.preprocessing.base.PreprocessingStep`)
        sao ignorados.

        Parameters
        ----------
        base_matrix : np.ndarray
            Matriz a compor -- tipicamente ``unmixing_matrix_`` (B), shape
            ``(n_componentes, n_componentes)``.

        Returns
        -------
        np.ndarray
            ``base_matrix`` composta com os passos lineares do pipeline,
            shape ``(n_componentes, n_misturas)``.
        """
        matrix = base_matrix
        for step in reversed(self.steps):
            if step.estimation_only:
                continue
            linear_matrix = step.linear_matrix_
            if linear_matrix is not None:
                matrix = matrix @ linear_matrix
        return matrix

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
