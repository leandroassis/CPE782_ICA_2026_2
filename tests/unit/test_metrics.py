"""Testes unitarios para as metricas de ica.metrics (DEVELOPMENT_GUIDELINES.md, Secao 2.6)."""

import numpy as np

from ica.metrics.convergence_iterations import ConvergenceIterations
from ica.metrics.execution_time import ExecutionTime
from ica.metrics.non_gaussianity import NonGaussianityScore


class _FakeModel:
    """Duble minimo de ICAModel, expondo apenas os atributos que as metricas leem."""

    def __init__(self, n_iterations=None, elapsed_time=None, sources=None) -> None:
        self.n_iterations_ = n_iterations
        self.elapsed_time_ = elapsed_time
        self.sources_ = sources


def test_convergence_iterations_reads_n_iterations():
    """ConvergenceIterations deve retornar exatamente model.n_iterations_."""
    model = _FakeModel(n_iterations=42)
    assert ConvergenceIterations().compute(model) == 42.0


def test_execution_time_reads_elapsed_time():
    """ExecutionTime deve retornar exatamente model.elapsed_time_."""
    model = _FakeModel(elapsed_time=1.5)
    assert ExecutionTime().compute(model) == 1.5


def test_metrics_expose_stable_names():
    """O atributo name de cada metrica deve ser um identificador estavel para evaluate()."""
    assert ConvergenceIterations().name == "convergence_iterations"
    assert ExecutionTime().name == "execution_time_seconds"
    assert NonGaussianityScore().name == "non_gaussianity_kurtosis"


def test_non_gaussianity_score_signs_match_known_distributions(rng):
    """Curtose deve ser positiva p/ Laplaciana, negativa p/ Uniforme, ~0 p/ Gaussiana."""
    n = 200_000
    laplace = rng.laplace(size=n)
    uniform = rng.uniform(-1, 1, size=n)
    gaussian = rng.normal(size=n)
    model = _FakeModel(sources=np.vstack([laplace, uniform, gaussian]))

    scores = NonGaussianityScore().compute(model)

    assert scores[0] > 1.0
    assert scores[1] < -0.5
    assert abs(scores[2]) < 0.1
