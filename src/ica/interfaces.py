"""Interfaces e estruturas de dados centrais, compartilhadas por todo o pacote.

Duas famílias de contrato convivem aqui:

- ``Protocol``s pequenos e focados (Interface Segregation): cada capacidade
  opcional que uma implementação de :class:`~ica.data.base.DataTemplate` pode
  oferecer é expressa separadamente, verificável em tempo de execução via
  ``isinstance``.
- As estruturas de dados centrais do fluxo ``io -> assess -> preprocess -> ica
  -> postprocess -> evaluate`` descrito em ``.claude/PIPELINE_MAP.md``:
  :class:`SignalMatrix` (saída de ``io/``) e :class:`ICAResult` (saída de
  ``ica/``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

Domain = Literal["image", "distribution", "audio"]


@dataclass
class SignalMatrix:
    """Matriz de sinais mais os metadados de domínio necessários para de-matrixizar.

    Formato canônico usado em todo o pacote: ``data`` tem shape
    ``(n_sinais, n_amostras)`` -- cada linha é um sinal observado (uma
    mistura ou, após a ICA, uma componente recuperada), cada coluna é uma
    amostra. É o que :class:`~ica.data.base.DataTemplate` devolve em
    ``load()`` e o que circula por ``preprocess/`` e ``ica/``.

    Parameters
    ----------
    data : np.ndarray
        Sinais, shape ``(n_sinais, n_amostras)``.
    domain : {"image", "distribution", "audio"}
        Domínio de origem, usado por ``postprocess/`` e pelos visualizadores
        para escolher a reconstrução/rescale corretos.
    meta : dict
        Metadados específicos do domínio, necessários para de-matrixizar de
        volta à forma nativa -- ex.: ``height``/``width``/``is_rgb`` (imagem),
        ``sample_rate`` (áudio), ``sample_size`` (distribuição).

    Attributes
    ----------
    n_signals : int
        Número de sinais (linhas de ``data``).
    n_samples : int
        Número de amostras (colunas de ``data``).
    """

    data: np.ndarray
    domain: Domain
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_signals(self) -> int:
        """Número de sinais (linhas de :attr:`data`)."""
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        """Número de amostras (colunas de :attr:`data`)."""
        return self.data.shape[1]


@dataclass
class ICAResult:
    """Resultado completo de um algoritmo de ICA-ML (bloco ``ica/``).

    Reúne num só objeto tudo que ``postprocess/`` e ``evaluate/`` (aqui,
    ``metrics/``) precisam de uma otimização já concluída, em vez de atributos
    soltos espalhados pelo algoritmo -- ver ``.claude/PIPELINE_MAP.md``,
    "Interfaces centrais".

    Parameters
    ----------
    B : np.ndarray
        Matriz de separação estimada, shape ``(n_componentes, n_componentes)``.
    Y : np.ndarray
        Componentes recuperadas no espaço em que o algoritmo rodou
        (``Y = B @ X``), shape ``(n_componentes, n_amostras)``.
    log_likelihood_history : list of float
        Log-verossimilhança média -- ``(1/T) log L(B)`` -- calculada a cada
        iteração (convenção do livro, sem sinal negativo -- skill ``ica-ml``).
    convergence_trace : list of float
        Resíduo de convergência a cada iteração, na métrica própria do
        algoritmo (GN/BS: ``||I + E{g(y)y^T}||_F``; FastICA-ML: desalinhamento
        de colunas ``1 - min_i|<b_i_novo, b_i_antigo>|``) -- skill ``ica-ml``, §7.
    nonlinearity_per_component : list of str
        Rótulo (``"super"``/``"sub"``) da não-linearidade escolhida por
        componente na última iteração (chaveamento da skill ``ica-ml``, §5).
    n_iterations : int
        Número de iterações efetivamente executadas.
    converged : bool
        Se o critério de convergência foi atingido antes do máximo de
        iterações.
    elapsed_time : float
        Tempo de execução do ajuste, em segundos.
    """

    B: np.ndarray
    Y: np.ndarray
    log_likelihood_history: list[float]
    convergence_trace: list[float]
    nonlinearity_per_component: list[str]
    n_iterations: int
    converged: bool
    elapsed_time: float


@runtime_checkable
class Reconstructable(Protocol):
    """Amostras cuja saida vetorial pode ser reconstruida em uma forma visualizavel.

    Implementada por :class:`~ica.data.image_template.ImageTemplate`, que
    sabe reformatar um vetor de pixels serializados de volta em uma imagem
    2D (ou RGB).
    """

    def reconstruct(self, source_vector: np.ndarray) -> np.ndarray:
        """Reformata um vetor de fonte recuperada em uma forma visualizavel.

        Parameters
        ----------
        source_vector : np.ndarray
            Vetor 1D de forma ``(n_amostras,)``, tipicamente uma linha de
            ``ICAModel.sources_``.

        Returns
        -------
        np.ndarray
            Array reformatado, pronto para ser exibido (ex.: ``(H, W)``).
        """
        ...


@runtime_checkable
class Exportable(Protocol):
    """Amostras cuja saida vetorial pode ser exportada de volta a um arquivo de mídia.

    Implementada por :class:`~ica.data.audio_template.AudioTemplate`, que
    sabe converter um vetor de fonte recuperada de volta em um arquivo
    ``.wav``.
    """

    def export(self, signal: np.ndarray, output_path: Path) -> None:
        """Grava um sinal recuperado em disco no formato nativo da amostra.

        Parameters
        ----------
        signal : np.ndarray
            Vetor 1D de forma ``(n_amostras,)`` a ser exportado.
        output_path : pathlib.Path
            Caminho do arquivo de saida.
        """
        ...
