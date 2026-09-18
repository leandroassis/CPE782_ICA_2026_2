"""Diagnostico de nao-linearidade (validade do modelo linear-instantaneo).

Skill ``bss-assessment``, Secao 3. **Armadilha de falso-positivo**: medir
dependencia nao-linear sobre dados so branqueados NAO diagnostica
nao-linearidade -- mistura linear branqueada e descorrelacionada mas ainda
dependente em ordem superior (e exatamente o que a ICA resolve). O
diagnostico confiavel (:func:`residual_dependence`) mede a dependencia
residual **pos-ICA** (sobre ``Y``, a melhor separacao linear obtida): se a
mistura e linear, a ICA-ML atinge independencia e a dependencia residual
tende a 0; se a mistura verdadeira e nao-linear, nenhum ``B`` linear a
alcanca e sobra dependencia.

:func:`joint_diagonalization_defect` e um pre-screen opcional e
corroborante (mais caro, e pode ser executado antes da separacao, sobre
dados so branqueados): mede o quanto as matrizes de cumulantes de 4a ordem
de dados branqueados linearmente-misturados deixam de ser conjuntamente
diagonalizaveis por uma unica rotacao ortogonal (base do JADE). E uma
aproximacao simplificada (Jacobi de duas matrizes, poucas iteracoes) --
nao um JADE completo -- e nao e informativo quando as fontes sao
proximas de gaussianas (cumulantes de ordem >= 3 nulos, "ponto cego" da
ICA): nesse caso o defeito fica artificialmente alto por ruido amostral,
nao por nao-linearidade.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ResidualDependenceResult:
    """Dependencia residual entre pares de componentes recuperadas.

    Attributes
    ----------
    abs_correlation : np.ndarray
        Matriz ``(n, n)`` de ``|corr(|y_i|, |y_j|)|`` (diagonal zerada).
    squared_correlation : np.ndarray
        Matriz ``(n, n)`` de ``|corr(y_i^2, y_j^2)|`` (diagonal zerada).
    max_residual_dependence : float
        Maior valor entre as duas matrizes -- estatistica resumo.
    is_likely_nonlinear : bool
        ``max_residual_dependence > threshold``.
    """

    abs_correlation: np.ndarray
    squared_correlation: np.ndarray
    max_residual_dependence: float
    is_likely_nonlinear: bool


def residual_dependence(Y: np.ndarray, threshold: float = 0.1) -> ResidualDependenceResult:
    """Mede a dependencia residual entre componentes recuperadas **pos-ICA**.

    Parameters
    ----------
    Y : np.ndarray
        Componentes recuperadas pela melhor separacao linear obtida
        (``ICAModel.sources_``), shape ``(n_componentes, n_amostras)``.
        Nunca passar dados so branqueados aqui (ver aviso do modulo).
    threshold : float, default=0.1
        Limiar de dependencia residual acima do qual sinaliza-se suspeita
        de nao-linearidade na mistura original.

    Returns
    -------
    ResidualDependenceResult
        Matrizes de dependencia e a flag resumo.
    """
    n = Y.shape[0]
    abs_correlation = np.corrcoef(np.abs(Y)) if n > 1 else np.zeros((n, n))
    squared_correlation = np.corrcoef(Y**2) if n > 1 else np.zeros((n, n))
    abs_correlation = np.nan_to_num(abs_correlation)
    squared_correlation = np.nan_to_num(squared_correlation)
    np.fill_diagonal(abs_correlation, 0.0)
    np.fill_diagonal(squared_correlation, 0.0)

    max_dependence = float(
        max(np.max(np.abs(abs_correlation)), np.max(np.abs(squared_correlation)))
        if n > 1
        else 0.0
    )
    return ResidualDependenceResult(
        abs_correlation=abs_correlation,
        squared_correlation=squared_correlation,
        max_residual_dependence=max_dependence,
        is_likely_nonlinear=max_dependence > threshold,
    )


def distance_correlation(x: np.ndarray, y: np.ndarray, max_samples: int = 2000) -> float:
    """Correlacao de distancia (Szekely, Rizzo & Bakirov, 2007) entre dois canais 1D.

    Captura *qualquer* forma de dependencia (linear ou nao); ``= 0`` sse
    independencia. Custo ``O(T^2)`` -- subamostra ate ``max_samples`` pontos
    quando ``T`` excede esse limite.

    Parameters
    ----------
    x, y : np.ndarray
        Canais 1D de mesmo comprimento.
    max_samples : int, default=2000
        Numero maximo de amostras usadas (subamostragem aleatoria acima
        disso, com semente fixa para reprodutibilidade).

    Returns
    -------
    float
        Correlacao de distancia, em ``[0, 1]``.
    """
    if x.shape[0] > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
        x, y = x[idx], y[idx]

    def _double_centered_distances(v: np.ndarray) -> np.ndarray:
        d = np.abs(v[:, np.newaxis] - v[np.newaxis, :])
        return d - d.mean(axis=0, keepdims=True) - d.mean(axis=1, keepdims=True) + d.mean()

    A = _double_centered_distances(x)
    B = _double_centered_distances(y)
    dcov2 = np.mean(A * B)
    dvar_x = np.mean(A * A)
    dvar_y = np.mean(B * B)
    denominator = np.sqrt(dvar_x * dvar_y)
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0) / denominator))


def _fourth_order_cumulant_matrices(Z: np.ndarray) -> list[np.ndarray]:
    """Matrizes-pai de cumulantes de 4a ordem de dados branqueados (base do JADE).

    ``M_ij[k,l] = E{Z_i Z_j Z_k Z_l} - delta_ij delta_kl - delta_ik delta_jl - delta_il delta_jk``,
    para cada par ``i <= j``. Assume ``E{Z Z^T} = I`` (``Z`` ja branqueado).
    """
    n, n_samples = Z.shape
    matrices = []
    for i in range(n):
        for j in range(i, n):
            weight = Z[i] * Z[j]
            M = (Z * weight) @ Z.T / n_samples
            correction = (1.0 if i == j else 0.0) * np.eye(n)
            correction[i, j] += 1.0
            correction[j, i] += 1.0
            matrices.append(M - correction)
    return matrices


def _joint_diagonalize(
    matrices: list[np.ndarray], n_sweeps: int = 30, tolerance: float = 1e-9
) -> list[np.ndarray]:
    """Diagonaliza ``matrices`` por rotacoes de Jacobi por pares (Cardoso-Souloumiac)."""
    n = matrices[0].shape[0]
    current = [M.copy() for M in matrices]
    for _ in range(n_sweeps):
        max_angle = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                a = np.array([M[p, p] - M[q, q] for M in current])
                b = np.array([2.0 * M[p, q] for M in current])
                g = np.array([[a @ a, a @ b], [a @ b, b @ b]])
                _, eigenvectors = np.linalg.eigh(g)
                x, y = eigenvectors[:, -1]
                theta = 0.5 * np.arctan2(y, x)
                # O autovetor tem sinal arbitrario (v e -v sao ambos validos), e
                # theta(-v) = theta(v) -/+ pi/2: sem essa correcao, o angulo pode
                # "saltar" para uma rotacao de ~90 graus (que so permuta eixos,
                # sem reduzir energia fora da diagonal) em vez do pequeno passo
                # de fato otimo. Restringe a (-pi/4, pi/4], o intervalo em que
                # a rotacao de Jacobi por par e unica.
                if theta > np.pi / 4:
                    theta -= np.pi / 2
                elif theta <= -np.pi / 4:
                    theta += np.pi / 2
                max_angle = max(max_angle, abs(theta))
                if abs(theta) < 1e-12:
                    continue
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                rotation = np.eye(n)
                rotation[p, p] = cos_t
                rotation[q, q] = cos_t
                rotation[p, q] = -sin_t
                rotation[q, p] = sin_t
                current = [rotation.T @ M @ rotation for M in current]
        if max_angle < tolerance:
            break
    return current


def joint_diagonalization_defect(Z: np.ndarray) -> float:
    """Defeito de diagonalizacao conjunta das matrizes de cumulantes de 4a ordem de ``Z``.

    Perto de 0: um unico ``A`` linear explica os dados (base do JADE). Alto:
    nenhum ``A`` linear unico serve -- evidencia de nao-linearidade, mais de
    uma gaussiana, ou ruido forte (skill bss-assessment, Secao 3). Ver aviso
    do modulo sobre a limitacao em fontes proximas de gaussianas.

    Parameters
    ----------
    Z : np.ndarray
        Dados **branqueados** (``E{ZZ^T} approx I``), shape
        ``(n_componentes, n_amostras)``.

    Returns
    -------
    float
        Fracao de energia fora da diagonal apos a melhor rotacao comum
        encontrada, em ``[0, 1]``.
    """
    matrices = _fourth_order_cumulant_matrices(Z)
    diagonalized = _joint_diagonalize(matrices)
    total_energy = sum(np.sum(M**2) for M in diagonalized)
    if total_energy <= 0:
        return 0.0
    diagonal_energy = sum(np.sum(np.diag(M) ** 2) for M in diagonalized)
    return float((total_energy - diagonal_energy) / total_energy)
