"""Algoritmo FastICA adaptado para Maxima Verossimilhanca (ponto fixo em bloco).

Ver ``.claude/skills/ica-ml/SKILL.md``, Secao 6; ``references/algorithms.md``,
eq. 9.22-9.25 e Tabela 9.2.
"""

import numpy as np

from ica.algorithms.base import ICAAlgorithm
from ica.preprocessing.symmetric import symmetric_inverse_sqrt


class FastICAML(ICAAlgorithm):
    """Ponto fixo em bloco para ML, com ortogonalizacao simetrica a cada iteracao.

    Implementa a forma matricial sobre dados nao-branqueados (skill
    ``ica-ml``, eq. 9.24-9.25)::

        y  = B x
        beta_i  = -E{y_i g(y_i)}
        alpha_i = -1 / (beta_i + E{g'(y_i)})
        B <- B + diag(alpha_i) [diag(beta_i) + E{g(y) y^T}] B
        B <- (B C_x B^T)^(-1/2) B                      # projecao (eq. 9.25)

    onde ``C_x = E{x x^T}`` e a projecao usa a raiz-inversa-simetrica
    compartilhada (:func:`~ica.preprocessing.symmetric.symmetric_inverse_sqrt`).
    Aproxima o Hessiano da log-verossimilhanca por blocos diagonais,
    evitando o calculo exato (computacionalmente proibitivo). Livre de
    taxa de aprendizado -- ``learning_rate`` e ignorado. Exige dados
    branqueados (skill ``ica-ml``, Secao 6: "obrigatorio para FastICA-ML").
    Com ``tanh`` fixo, separa fontes sub e super automaticamente -- o
    chaveamento explicito da Secao 5 nao e necessario aqui, `diag(beta_i)`
    absorve a natureza de cada fonte.
    """

    def _update_step(self, B: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Aplica um passo de ponto fixo em bloco seguido de ortogonalizacao simetrica.

        Parameters
        ----------
        B : np.ndarray
            Estimativa atual da matriz de separacao.
        X : np.ndarray
            Dados pre-processados (branqueados), shape
            ``(n_componentes, n_amostras)``.

        Returns
        -------
        np.ndarray
            ``(B C_x B^T)^(-1/2) B'``, onde ``B'`` e a atualizacao de ponto
            fixo em bloco ``B + diag(alpha)[diag(beta) + E{g(y)y^T}/T]B``.
        """
        n_samples = X.shape[1]
        Y = B @ X
        G = self.nonlinearity.score(Y)
        G_derivative = self.nonlinearity.derivative(Y)

        beta = -np.mean(Y * G, axis=1)
        alpha = -1.0 / (beta + np.mean(G_derivative, axis=1))

        block_update = np.diag(beta) + (G @ Y.T) / n_samples
        B_updated = B + alpha[:, np.newaxis] * (block_update @ B)

        covariance = (X @ X.T) / n_samples
        return symmetric_inverse_sqrt(B_updated @ covariance @ B_updated.T) @ B_updated

    def _convergence_residual(self, B_old: np.ndarray, B_new: np.ndarray, X: np.ndarray) -> float:
        """Desalinhamento de colunas ``1 - min_i |<b_i_novo, b_i_antigo>|`` (skill, Secao 7).

        Parameters
        ----------
        B_old : np.ndarray
            Estimativa anterior de B (linhas ja ortogonalizadas na iteracao
            anterior).
        B_new : np.ndarray
            Nova estimativa de B, ja ortogonalizada por
            :meth:`_update_step`.
        X : np.ndarray
            Nao usado -- o criterio do FastICA-ML depende so de B.

        Returns
        -------
        float
            ``1 - min_i |cos(b_i_novo, b_i_antigo)|``; 0 quando todas as
            linhas de B estabilizaram a menos de sinal.
        """
        old_norms = np.linalg.norm(B_old, axis=1)
        new_norms = np.linalg.norm(B_new, axis=1)
        cosines = np.abs(np.sum(B_old * B_new, axis=1)) / (old_norms * new_norms)
        return float(1.0 - np.min(cosines))
