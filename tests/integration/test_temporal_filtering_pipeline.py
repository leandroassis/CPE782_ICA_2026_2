"""Teste de integracao: TemporalFiltering so entra na estimacao de B, nunca na reconstrucao.

Ver livro-texto, Secao 13.1 (p.264): "we can use the filtered data in the
ICA estimating method only. After estimating the mixing matrix, we can
apply the same mixing matrix on the original data to obtain the
independent components."
"""

import numpy as np

from ica.algorithms.natural_gradient import NaturalGradientICA
from ica.model import ICAModel
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.temporal_filtering import TemporalFiltering
from ica.preprocessing.whitening import Whitening


def test_sources_are_reconstructed_from_unfiltered_data(
    rng, make_sources, make_mixing_matrix, array_data_template
):
    """model.sources_ deve vir do sinal original, mesmo com TemporalFiltering no pipeline.

    Verifica a mecanica exata: sources_ == unmixing_matrix_ @
    Whitening.transform(Centering.transform(mixtures_)) -- pulando o
    filtro -- e nao unmixing_matrix_ @ preprocessed_ (que passaria pelo
    filtro). As duas entradas de reconstrucao sao explicitamente
    comparadas para confirmar que o filtro de fato alterou os dados
    (o teste nao seria significativo se preprocessed_ e o caminho de
    reconstrucao coincidissem por acaso).
    """
    S = make_sources(["laplace", "uniform"], 4000, rng)
    A = make_mixing_matrix(rng, 2)
    X = A @ S

    pipeline = Pipeline([Centering(), TemporalFiltering(kind="high"), Whitening()])
    model = ICAModel(
        data=array_data_template(X),
        pipeline=pipeline,
        algorithm=NaturalGradientICA(nonlinearity=AdaptiveScore(), max_iterations=200),
    )
    model.fit()

    whitening = pipeline.get_step(Whitening)
    centered = X - X.mean(axis=1, keepdims=True)
    expected_reconstruction_input = whitening.transform(centered)
    expected_sources = model.unmixing_matrix_ @ expected_reconstruction_input

    assert model.sources_.shape[1] == X.shape[1]
    assert np.allclose(model.sources_, expected_sources)
    assert not np.allclose(model.preprocessed_, expected_reconstruction_input)


def test_full_unmixing_matrix_excludes_temporal_filtering(
    rng, make_sources, make_mixing_matrix, array_data_template
):
    """full_unmixing_matrix_ nao deve incluir o efeito de TemporalFiltering (nao-linear no tempo).

    Deve ser identico ao caso sem filtragem: unmixing_matrix_ @
    whitening_matrix_, apesar de a estimacao de B ter usado dados
    filtrados.
    """
    S = make_sources(["laplace", "uniform"], 4000, rng)
    A = make_mixing_matrix(rng, 2)
    X = A @ S

    pipeline = Pipeline([Centering(), TemporalFiltering(kind="low", window=3), Whitening()])
    model = ICAModel(
        data=array_data_template(X),
        pipeline=pipeline,
        algorithm=NaturalGradientICA(nonlinearity=AdaptiveScore(), max_iterations=200),
    )
    model.fit()

    whitening = pipeline.get_step(Whitening)
    expected = model.unmixing_matrix_ @ whitening.whitening_matrix_

    assert np.array_equal(model.full_unmixing_matrix_, expected)
