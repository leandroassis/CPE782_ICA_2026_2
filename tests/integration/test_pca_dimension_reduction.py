"""Teste de integracao: PCA reduz misturas em excesso antes da ICA.

Ver livro-texto, Secao 13.2.1 (p.267): quando ha mais misturas do que
fontes (m > n), o modelo basico de ICA nao vale diretamente sobre as
misturas -- PCA reduz a dimensao para igualar m a n antes do branqueamento.
"""

import numpy as np

from ica.algorithms.fastica_ml import FastICAML
from ica.model import ICAModel
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.pca import PCA
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.whitening import Whitening


def test_pca_reduces_five_mixtures_to_two_sources(
    rng, make_sources, best_match_correlation, array_data_template
):
    """5 misturas de 2 fontes: PCA(n_components=2) deve permitir recuperar as 2 fontes.

    Sem ruido, os dados ficam inteiramente contidos no subespaco de
    dimensao 2 gerado pelas colunas de A -- PCA encontra esse subespaco
    exatamente (Secao 13.2.1), reduzindo o problema ao caso quadrado
    usual antes do branqueamento.
    """
    S = make_sources(["laplace", "uniform"], 5000, rng)
    # Colunas ortonormais (via QR) garantem bom condicionamento do subespaco de
    # mistura de dimensao 2 dentro do espaco de 5 misturas -- make_mixing_matrix
    # so gera matrizes quadradas, entao a matriz "gorda" e construida aqui.
    A, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    X = A @ S

    model = ICAModel(
        data=array_data_template(X),
        pipeline=Pipeline([Centering(), PCA(n_components=2), Whitening()]),
        algorithm=FastICAML(nonlinearity=AdaptiveScore(), max_iterations=500),
    )
    model.fit()

    assert model.sources_.shape == (2, 5000)
    assert best_match_correlation(S, model.sources_) > 0.9

    assert model.full_unmixing_matrix_.shape == (2, 5)
    recovered_from_raw = model.full_unmixing_matrix_ @ X
    assert best_match_correlation(S, recovered_from_raw) > 0.9
