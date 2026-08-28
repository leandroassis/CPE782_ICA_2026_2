"""Teste de integracao: fit() + evaluate() ponta a ponta com as 4 metricas reais."""

import numpy as np

from ica.algorithms.natural_gradient import NaturalGradientICA
from ica.metrics.convergence_iterations import ConvergenceIterations
from ica.metrics.execution_time import ExecutionTime
from ica.metrics.log_likelihood import LogLikelihood
from ica.metrics.non_gaussianity import NonGaussianityScore
from ica.model import ICAModel
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.whitening import Whitening


def test_evaluate_with_real_metrics_returns_sane_values(
    rng, make_sources, make_mixing_matrix, array_data_template
):
    """As 4 metricas devem retornar valores plausiveis apos um fit() real."""
    S = make_sources(["laplace", "laplace"], 3000, rng)
    A = make_mixing_matrix(rng, 2)
    X = A @ S

    model = ICAModel(
        data=array_data_template(X),
        pipeline=Pipeline([Centering(), Whitening()]),
        algorithm=NaturalGradientICA(nonlinearity=AdaptiveScore(), max_iterations=500),
    )
    model.fit()

    results = model.evaluate(
        [ConvergenceIterations(), ExecutionTime(), NonGaussianityScore(), LogLikelihood()]
    )

    assert 0 < results["convergence_iterations"] <= 500
    assert results["execution_time_seconds"] >= 0.0
    assert results["non_gaussianity_kurtosis"].shape == (2,)
    assert np.all(results["non_gaussianity_kurtosis"] > 0.5)
    assert results["log_likelihood"] == model.log_likelihood_history_[-1]
