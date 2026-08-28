"""Teste de integracao: os 3 ICAAlgorithm sao intercambiaveis (Liskov Substitution).

Ver context/DEVELOPMENT_GUIDELINES.md, Secao 3 ("L -- Liskov Substitution").
"""

import pytest

from ica.algorithms.bell_sejnowski import BellSejnowskiICA
from ica.algorithms.fastica_ml import FastICAML
from ica.algorithms.natural_gradient import NaturalGradientICA
from ica.model import ICAModel
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.whitening import Whitening


@pytest.mark.parametrize(
    "algorithm_factory",
    [
        lambda: NaturalGradientICA(nonlinearity=AdaptiveScore(), max_iterations=500),
        lambda: BellSejnowskiICA(
            nonlinearity=AdaptiveScore(), learning_rate=0.01, max_iterations=500
        ),
        lambda: FastICAML(nonlinearity=AdaptiveScore(), max_iterations=100),
    ],
    ids=["NaturalGradientICA", "BellSejnowskiICA", "FastICAML"],
)
def test_any_ica_algorithm_recovers_the_same_synthetic_problem(
    algorithm_factory,
    rng,
    make_sources,
    make_mixing_matrix,
    best_match_correlation,
    array_data_template,
):
    """Qualquer ICAAlgorithm deve poder substituir outro no mesmo ICAModel e separar bem.

    Todos compartilham a mesma interface publica (``fit``,
    ``converged_``, ``n_iterations_``, ``history_``) e podem ser
    injetados de forma intercambiavel no mesmo :class:`~ica.model.ICAModel`
    sem alterar quem os utiliza.
    """
    S = make_sources(["laplace", "laplace"], 3000, rng)
    A = make_mixing_matrix(rng, 2)
    X = A @ S

    model = ICAModel(
        data=array_data_template(X),
        pipeline=Pipeline([Centering(), Whitening()]),
        algorithm=algorithm_factory(),
    )
    model.fit()

    assert best_match_correlation(S, model.sources_) > 0.9
    assert isinstance(model.converged_, bool)
    assert isinstance(model.n_iterations_, int)
    assert len(model.history_) == model.n_iterations_
