"""Filtragem temporal linear (passa-baixa, passa-alta e passa-banda).

Ver livro-texto (Hyvarinen, Karhunen & Oja), Secao 13.1 (paginas 263-267).
"""

import numpy as np

from ica.preprocessing.base import PreprocessingStep

_VALID_KINDS = {"low", "high", "band"}


class TemporalFiltering(PreprocessingStep):
    """Filtragem temporal linear de sinais com estrutura de tempo real (ex.: audio).

    Ver livro-texto, Secao 13.1 (p.263-267). So faz sentido quando a
    ordem das amostras (eixo das colunas de X) e significativa -- a
    Secao 13.1.1 alerta que filtrar dados sem ordem temporal real (ex.:
    pixels de imagem ou linhas i.i.d. de uma distribuicao) nao e
    meaningful.

    Dois efeitos documentados no livro, de sinais opostos:

    - ``kind="low"`` (passa-baixa, media movel de ``window`` amostras,
      Eq. 13.3): reduz ruido, mas pode reduzir a independencia das
      componentes ao suavizar as caracteristicas de alta frequencia que
      as distinguem (Secao 13.1.2).
    - ``kind="high"`` (passa-alta, diferenciacao de 1a ordem, Eq. 13.4):
      aproxima o processo de inovacao (Teorema 13.1), tipicamente
      aumentando a independencia e a nao-gaussianidade das componentes --
      mas pode amplificar ruido (Secao 13.1.3).
    - ``kind="band"``: aplica passa-baixa seguido de passa-alta, um
      compromisso simples entre os dois efeitos (Secao 13.1.4).

    Como qualquer filtragem linear no eixo do tempo preserva o modelo de
    ICA com a mesma matriz de mistura (``X~ = XM = ASM = A S~``, Eq.
    13.1-13.2 do livro-texto), este passo e marcado como
    :attr:`estimation_only`: ele participa apenas do ajuste da matriz de
    separacao B; a reconstrucao final das fontes
    (:meth:`~ica.preprocessing.pipeline.Pipeline.reconstruction_transform`)
    usa sempre o sinal original, nao filtrado (Secao 13.1, p.264: "we can
    use the filtered data in the ICA estimating method only. After
    estimating the mixing matrix, we can apply the same mixing matrix on
    the original data to obtain the independent components").

    Parameters
    ----------
    kind : {"low", "high", "band"}
        Tipo de filtro.
    window : int, default=5
        Tamanho da janela da media movel (passa-baixa), em amostras.
        Usado quando ``kind`` e ``"low"`` ou ``"band"``; ignorado para
        ``"high"``.

    Attributes
    ----------
    estimation_only : bool
        Sempre ``True`` nesta classe (ver acima e
        :attr:`~ica.preprocessing.base.PreprocessingStep.estimation_only`).
    """

    estimation_only = True

    def __init__(self, kind: str = "low", window: int = 5) -> None:
        if kind not in _VALID_KINDS:
            raise ValueError(f"kind deve ser um de {sorted(_VALID_KINDS)}, recebido {kind!r}.")
        if window < 1:
            raise ValueError(f"window deve ser >= 1, recebido {window}.")
        self.kind = kind
        self.window = window

    def fit(self, X: np.ndarray) -> "TemporalFiltering":
        """Nao ha parametros a ajustar: o filtro e deterministico.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        TemporalFiltering
            A propria instancia.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica o filtro escolhido a cada mistura (linha), independentemente.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        np.ndarray
            Dados filtrados, mesma shape de ``X``.
        """
        if self.kind == "low":
            return self._low_pass(X)
        if self.kind == "high":
            return self._high_pass(X)
        return self._high_pass(self._low_pass(X))

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Nao suportado: a filtragem descarta informacao e nao e reversivel.

        Passa-baixa perde as componentes de alta frequencia (Secao
        13.1.2) e passa-alta perde o nivel/tendencia original (Secao
        13.1.3); nenhum dos dois admite inversa exata. Como este passo e
        ``estimation_only`` (usado so para ajustar B), a reconstrucao das
        fontes nunca passa por aqui -- ver
        :meth:`~ica.preprocessing.pipeline.Pipeline.reconstruction_transform`.

        Raises
        ------
        NotImplementedError
            Sempre.
        """
        raise NotImplementedError(
            "TemporalFiltering nao e reversivel (livro-texto, Secao 13.1); e um passo "
            "estimation_only, usado apenas para ajustar B. Use "
            "Pipeline.reconstruction_transform para recuperar as fontes a partir dos "
            "dados originais."
        )

    def _low_pass(self, X: np.ndarray) -> np.ndarray:
        """Media movel causal-simetrica de ``window`` amostras (Eq. 13.3)."""
        kernel = np.ones(self.window) / self.window
        return np.array([np.convolve(row, kernel, mode="same") for row in X])

    def _high_pass(self, X: np.ndarray) -> np.ndarray:
        """Diferenciacao de 1a ordem (Eq. 13.4), preservando o numero de amostras."""
        return np.diff(X, axis=1, prepend=X[:, :1])
