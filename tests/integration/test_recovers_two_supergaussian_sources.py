"""Teste de integracao: ICAModel completo recupera 2 fontes supergaussianas.

Ver context/ICA_BACKGROUND.md, Secoes 2 e 4.2.
"""

from ica.algorithms.natural_gradient import NaturalGradientICA
from ica.model import ICAModel
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.whitening import Whitening


def test_ica_model_recovers_two_laplace_sources(
    rng, make_sources, make_mixing_matrix, best_match_correlation, array_data_template
):
    """Pipeline completo (Centering + Whitening + NaturalGradientICA/Adaptive) deve separar bem."""
    S = make_sources(["laplace", "laplace"], 3000, rng)
    A = make_mixing_matrix(rng, 2)
    X = A @ S

    model = ICAModel(
        data=array_data_template(X),
        pipeline=Pipeline([Centering(), Whitening()]),
        algorithm=NaturalGradientICA(nonlinearity=AdaptiveScore(), max_iterations=500),
    )
    model.fit()

    assert best_match_correlation(S, model.sources_) > 0.9
    assert model.mixtures_.shape == X.shape
    assert model.full_unmixing_matrix_.shape == (2, 2)

    recovered_from_raw = model.full_unmixing_matrix_ @ X
    assert best_match_correlation(S, recovered_from_raw) > 0.9
    assert isinstance(model.n_iterations_, int)
    assert model.elapsed_time_ is not None and model.elapsed_time_ >= 0.0
