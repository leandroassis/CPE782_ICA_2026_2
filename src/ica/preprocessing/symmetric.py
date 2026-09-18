"""Raiz-inversa (simetrica ou nao) de uma matriz simetrica positiva-definida, via EVD.

Rotina unica compartilhada entre o branqueamento
(:class:`~ica.preprocessing.whitening.Whitening`) e a projecao/ortogonalizacao
simetrica do FastICA-ML (:class:`~ica.algorithms.fastica_ml.FastICAML`) --
skill ``ica-ml``, ``references/algorithms.md``: "a rotina de
raiz-inversa-simetrica e unica e compartilhada entre o branqueamento e a
projecao do FastICA-ML". As duas composicoes finais (``D^(-1/2) E^T`` para
branqueamento PCA, ``E D^(-1/2) E^T`` para a projecao simetrica do
FastICA-ML) partem da mesma decomposicao em autovalores/autovetores e do
mesmo piso numerico contra autovalores quase nulos.
"""

import numpy as np


def _eigh_inverse_sqrt_eigenvalues(
    matrix: np.ndarray, eigenvalue_floor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Decompoe ``matrix`` (EVD) e devolve os autovetores e ``1/sqrt(autovalor)``.

    Autovalores abaixo de ``eigenvalue_floor * autovalor_maximo`` sao
    saturados nesse piso antes de inverter a raiz, evitando ``1/sqrt(~0)``
    (NaN/inf) quando ``matrix`` e proxima de singular (covariancia amostral
    com outliers extremos, ex. ``data/mix/dist/run8``).

    Parameters
    ----------
    matrix : np.ndarray
        Matriz simetrica positiva-(semi)definida, shape ``(n, n)``.
    eigenvalue_floor : float
        Piso relativo ao maior autovalor.

    Returns
    -------
    tuple of np.ndarray
        ``(eigenvectors, inverse_sqrt_eigenvalues)``, shapes ``(n, n)`` e
        ``(n,)``.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = eigenvalue_floor * np.max(eigenvalues)
    safe_eigenvalues = np.maximum(eigenvalues, floor)
    return eigenvectors, 1.0 / np.sqrt(safe_eigenvalues)


def symmetric_inverse_sqrt(matrix: np.ndarray, eigenvalue_floor: float = 1e-12) -> np.ndarray:
    """Calcula a raiz-inversa **simetrica** ``E D^(-1/2) E^T`` de ``matrix``.

    Usada pela projecao do FastICA-ML, ``B <- (B C_x B^T)^(-1/2) B``, que
    mantem ``y = Bx`` branco e descorrelacionado a cada iteracao (skill
    ``ica-ml``, eq. 9.25).

    Parameters
    ----------
    matrix : np.ndarray
        Matriz simetrica positiva-(semi)definida, shape ``(n, n)``.
    eigenvalue_floor : float, default=1e-12
        Ver :func:`_eigh_inverse_sqrt_eigenvalues`.

    Returns
    -------
    np.ndarray
        ``matrix^(-1/2)`` (forma simetrica), shape ``(n, n)``.
    """
    eigenvectors, inverse_sqrt_eigenvalues = _eigh_inverse_sqrt_eigenvalues(
        matrix, eigenvalue_floor
    )
    return eigenvectors @ (inverse_sqrt_eigenvalues[:, np.newaxis] * eigenvectors.T)


def pca_whitening_matrices(
    covariance: np.ndarray, eigenvalue_floor: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula a matriz de branqueamento PCA ``V = D^(-1/2) E^T`` e sua inversa.

    Usada por :class:`~ica.preprocessing.whitening.Whitening` (skill
    ``ica-ml``, ``references/algorithms.md``: "V = D^(-1/2) E^T, onde E, D
    vem da EVD de C_x").

    Parameters
    ----------
    covariance : np.ndarray
        Covariancia amostral ``C_x``, shape ``(n, n)``.
    eigenvalue_floor : float, default=1e-12
        Ver :func:`_eigh_inverse_sqrt_eigenvalues`.

    Returns
    -------
    tuple of np.ndarray
        ``(whitening_matrix, dewhitening_matrix)`` -- ``V`` e sua inversa
        ``E D^(1/2)``, ambas shape ``(n, n)``.
    """
    eigenvectors, inverse_sqrt_eigenvalues = _eigh_inverse_sqrt_eigenvalues(
        covariance, eigenvalue_floor
    )
    whitening_matrix = inverse_sqrt_eigenvalues[:, np.newaxis] * eigenvectors.T
    dewhitening_matrix = eigenvectors * (1.0 / inverse_sqrt_eigenvalues)[np.newaxis, :]
    return whitening_matrix, dewhitening_matrix
