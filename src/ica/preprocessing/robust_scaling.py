"""Reescala robusta por mistura, com recorte (winsorizing) opcional de outliers.

Motivacao pratica (nao do livro-texto): a curtose e a EVD da covariancia
usadas no branqueamento e na estimacao por ML/Infomax sao extremamente
sensiveis a outliers -- um unico valor extremo domina a curtose amostral
(Secao 8.3.1 do livro-texto, "Critique of kurtosis") e pode tornar a
covariancia numericamente singular (numero de condicao maior que o
alcance de `float64`), quebrando a decomposicao espectral usada por
:class:`~ica.preprocessing.whitening.Whitening`.
"""

import numpy as np

from ica.preprocessing.base import PreprocessingStep


class RobustScaling(PreprocessingStep):
    """Reescala cada mistura pelo seu desvio absoluto mediano (MAD), com recorte opcional.

    Para cada mistura (linha) i, estima a mediana e o MAD escalado
    ``scale_i = 1.4826 * mediana(|x_i - mediana(x_i)|)`` -- o fator
    1.4826 torna o MAD consistente com o desvio-padrao sob normalidade,
    mas, ao contrario do desvio-padrao, permanece estavel na presenca de
    caudas pesadas ou outliers extremos. Se ``clip_mads`` for informado,
    valores a mais de ``clip_mads`` MADs da mediana sao recortados
    (*winsorizing*) antes da reescala, limitando a influencia de valores
    extremos sobre a covariancia amostral usada a jusante (branqueamento)
    e sobre a funcao de pontuacao usada na otimizacao (Secao 3.3-3.4 de
    ICA_BACKGROUND.md).

    Reescala pura (``clip_mads=None``) e uma transformacao linear
    diagonal -- compativel com o modelo ``x = As`` (equivale a reescalar
    as linhas de A) -- e por isso contribui para
    :attr:`~ica.preprocessing.base.PreprocessingStep.linear_matrix_`. Com
    recorte ativo, a transformacao deixa de ser linear (o recorte e
    afim por partes), entao ``linear_matrix_`` volta a ``None`` nesse
    caso -- ``full_unmixing_matrix_`` deixaria de representar exatamente
    o mapeamento das misturas originais para as fontes recuperadas.

    Parameters
    ----------
    clip_mads : float or None, default=8.0
        Numero de MADs a partir da mediana alem do qual os valores sao
        recortados. ``None`` desativa o recorte (so reescala).

    Attributes
    ----------
    median_ : np.ndarray or None
        Mediana por mistura, shape ``(n_misturas,)``, definida apos
        :meth:`fit`.
    scale_ : np.ndarray or None
        MAD escalado por mistura, shape ``(n_misturas,)``.
    """

    def __init__(self, clip_mads: float | None = 8.0) -> None:
        self.clip_mads = clip_mads
        self.median_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "RobustScaling":
        """Estima a mediana e o MAD escalado de cada mistura (linha) de X.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        RobustScaling
            A propria instancia.
        """
        self.median_ = np.median(X, axis=1)
        mad = np.median(np.abs(X - self.median_[:, np.newaxis]), axis=1)
        self.scale_ = mad * 1.4826
        return self

    @property
    def linear_matrix_(self) -> np.ndarray | None:
        """Ver Attributes da classe. ``diag(1/scale_)`` se ``clip_mads is None``, senao ``None``."""
        if self.clip_mads is not None or self.scale_ is None:
            return None
        return np.diag(1.0 / self.scale_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Recorta (se configurado) e reescala X por ``scale_``.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        np.ndarray
            Dados reescalados, mesma shape de ``X``.
        """
        if self.clip_mads is not None:
            lower = self.median_[:, np.newaxis] - self.clip_mads * self.scale_[:, np.newaxis]
            upper = self.median_[:, np.newaxis] + self.clip_mads * self.scale_[:, np.newaxis]
            X = np.clip(X, lower, upper)
        return X / self.scale_[:, np.newaxis]

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Desfaz a reescala (``Y * scale_``); o recorte, se aplicado, e irreversivel.

        Parameters
        ----------
        Y : np.ndarray
            Dados reescalados, shape ``(n_misturas, n_amostras)``.

        Returns
        -------
        np.ndarray
            Dados aproximados no espaco original -- exato quando
            ``clip_mads is None``, aproximado (satura nos limites de
            recorte) caso contrario.
        """
        return Y * self.scale_[:, np.newaxis]
