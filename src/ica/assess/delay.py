"""Diagnostico de atraso / nao-instantaneidade entre canais.

Skill ``bss-assessment``, Secao 4. Aplicavel a series temporais (audio);
N/A para imagem/distribuicao, que nao tem eixo temporal. Na mistura
**instantanea** ``x = As``, o pico de correlacao cruzada entre qualquer par
de canais fica em lag 0; um pico deslocado denuncia atraso de propagacao
(modelo convolutivo, fora de escopo para separacao -- so detectamos).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DelayEstimate:
    """Estimativa de atraso relativo entre dois canais.

    Attributes
    ----------
    lag_samples : int
        Deslocamento (em amostras) que maximiza a correlacao; positivo
        significa que o segundo canal esta atrasado em relacao ao primeiro.
    lag_seconds : float or None
        ``lag_samples / sample_rate``, quando ``sample_rate`` e informado.
    """

    lag_samples: int
    lag_seconds: float | None


def cross_correlation_lag(xi: np.ndarray, xj: np.ndarray) -> int:
    """Estima o lag pelo pico da correlacao cruzada no dominio do tempo.

    Parameters
    ----------
    xi, xj : np.ndarray
        Dois canais, mesmo comprimento.

    Returns
    -------
    int
        Lag (em amostras) do pico de ``R_ij(tau) = E{x_i(t) x_j(t+tau)}``.
    """
    correlation = np.correlate(xi - xi.mean(), xj - xj.mean(), mode="full")
    lags = np.arange(-(len(xj) - 1), len(xi))
    return int(lags[np.argmax(correlation)])


def gcc_phat(xi: np.ndarray, xj: np.ndarray, sample_rate: float | None = None) -> DelayEstimate:
    """GCC-PHAT (Generalized Cross-Correlation with Phase Transform), frame-free.

    ``R_hat_ij(tau) = IFFT{ X_i(f) X_j*(f) / |X_i(f) X_j*(f)| }`` -- a
    normalizacao de fase afia o pico e o torna robusto a bandas coloridas;
    roda sobre o sinal **inteiro**, sem janelamento (Knapp & Carter, 1976).

    Parameters
    ----------
    xi, xj : np.ndarray
        Dois canais, mesmo comprimento.
    sample_rate : float, optional
        Taxa de amostragem (Hz), para converter o lag para segundos.

    Returns
    -------
    DelayEstimate
        Lag estimado (amostras e, se ``sample_rate`` informado, segundos).
    """
    n = xi.shape[0] + xj.shape[0]
    Xi = np.fft.rfft(xi, n=n)
    Xj = np.fft.rfft(xj, n=n)
    cross_spectrum = Xi * np.conj(Xj)
    magnitude = np.abs(cross_spectrum)
    magnitude = np.where(magnitude > 0, magnitude, 1e-15)
    phase_transformed = cross_spectrum / magnitude

    cross_correlation = np.fft.irfft(phase_transformed, n=n)
    max_shift = n // 2
    cross_correlation = np.concatenate(
        (cross_correlation[-max_shift:], cross_correlation[: max_shift + 1])
    )
    lag_samples = int(np.argmax(np.abs(cross_correlation)) - max_shift)
    lag_seconds = lag_samples / sample_rate if sample_rate else None
    return DelayEstimate(lag_samples=lag_samples, lag_seconds=lag_seconds)


def delay_battery(X: np.ndarray, sample_rate: float | None = None) -> np.ndarray:
    """Matriz de lags GCC-PHAT entre todos os pares de canais de ``X``.

    Parameters
    ----------
    X : np.ndarray
        Canais, shape ``(n_canais, n_amostras)``.
    sample_rate : float, optional
        Taxa de amostragem (Hz).

    Returns
    -------
    np.ndarray
        Matriz ``(n_canais, n_canais)`` de lags em amostras
        (``lags[i, j]`` = atraso de ``j`` em relacao a ``i``; diagonal 0).
    """
    n = X.shape[0]
    lags = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                lags[i, j] = gcc_phat(X[i], X[j], sample_rate).lag_samples
    return lags
