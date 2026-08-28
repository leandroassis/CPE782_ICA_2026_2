"""Testes unitarios para ICAModel (DEVELOPMENT_GUIDELINES.md, Secao 2.5).

Colaboradores completamente mockados: verifica apenas a orquestracao do
fluxo fit()/evaluate(), nao a matematica real de ICA (ja coberta pelos
testes de ``algorithms/`` e ``integration/``).
"""

import numpy as np

from ica.model import ICAModel


class _FakeDataTemplate:
    """Duble: retorna uma matriz de misturas fixa."""

    def __init__(self, X: np.ndarray) -> None:
        self._X = X
        self.load_call_count = 0

    def load(self) -> np.ndarray:
        self.load_call_count += 1
        return self._X


class _FakePipelineWithoutWhitening:
    """Duble: pipeline que nao contem nenhum passo de Whitening."""

    def __init__(self, transform_offset: float = 10.0) -> None:
        self._transform_offset = transform_offset
        self.fit_transform_calls: list[np.ndarray] = []

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit_transform_calls.append(X)
        return X + self._transform_offset

    def get_step(self, step_type):
        raise ValueError("nenhum passo deste tipo")


class _FakeWhiteningStep:
    """Duble: expoe apenas whitening_matrix_, como ica.preprocessing.whitening.Whitening."""

    def __init__(self, whitening_matrix: np.ndarray) -> None:
        self.whitening_matrix_ = whitening_matrix


class _FakePipelineWithWhitening:
    """Duble: pipeline cujo get_step(Whitening) devolve um passo com whitening_matrix_ conhecida."""

    def __init__(self, whitening_matrix: np.ndarray) -> None:
        self._whitening_step = _FakeWhiteningStep(whitening_matrix)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def get_step(self, step_type):
        return self._whitening_step


class _FakeAlgorithm:
    """Duble: algoritmo cuja matriz de separacao e diagnosticos sao fixos e conhecidos."""

    def __init__(self, unmixing_matrix: np.ndarray) -> None:
        self._unmixing_matrix = unmixing_matrix
        self.history_ = [0.5, 0.1, 0.01]
        self.log_likelihood_history_ = [-3.2, -1.5, -1.1]
        self.converged_ = True
        self.n_iterations_ = 3
        self.elapsed_time_ = 0.001
        self.fit_calls: list[np.ndarray] = []

    def fit(self, X: np.ndarray) -> np.ndarray:
        self.fit_calls.append(X)
        return self._unmixing_matrix


class _FakeMetric:
    """Duble de ica.metrics.base.Metric."""

    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self._value = value

    def compute(self, model: ICAModel) -> float:
        return self._value


def test_fit_calls_collaborators_in_order_and_wires_outputs():
    """fit() deve encadear data.load() -> pipeline.fit_transform() -> algorithm.fit()."""
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    data = _FakeDataTemplate(X)
    pipeline = _FakePipelineWithoutWhitening(transform_offset=10.0)
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    algorithm = _FakeAlgorithm(B)

    model = ICAModel(data=data, pipeline=pipeline, algorithm=algorithm)
    model.fit()

    assert data.load_call_count == 1
    assert np.array_equal(pipeline.fit_transform_calls[0], X)
    assert np.array_equal(algorithm.fit_calls[0], X + 10.0)
    assert np.array_equal(model.mixtures_, X)
    assert np.array_equal(model.preprocessed_, X + 10.0)
    assert np.array_equal(model.sources_, B @ (X + 10.0))
    assert np.array_equal(model.unmixing_matrix_, B)


def test_fit_falls_back_to_unmixing_matrix_when_no_whitening_step():
    """Sem passo de Whitening, full_unmixing_matrix_ deve igualar unmixing_matrix_."""
    X = np.eye(2)
    data = _FakeDataTemplate(X)
    pipeline = _FakePipelineWithoutWhitening()
    B = np.array([[2.0, 0.0], [0.0, 2.0]])
    algorithm = _FakeAlgorithm(B)

    model = ICAModel(data=data, pipeline=pipeline, algorithm=algorithm).fit()

    assert np.array_equal(model.full_unmixing_matrix_, B)


def test_fit_composes_full_unmixing_matrix_with_whitening_step():
    """Com um passo de Whitening, full_unmixing_matrix_ deve ser B @ whitening_matrix_."""
    X = np.eye(2)
    data = _FakeDataTemplate(X)
    V = np.array([[3.0, 0.0], [0.0, 3.0]])
    pipeline = _FakePipelineWithWhitening(whitening_matrix=V)
    B = np.array([[2.0, 0.0], [0.0, 2.0]])
    algorithm = _FakeAlgorithm(B)

    model = ICAModel(data=data, pipeline=pipeline, algorithm=algorithm).fit()

    assert np.array_equal(model.full_unmixing_matrix_, B @ V)


def test_fit_copies_algorithm_diagnostics_onto_model():
    """history_, log_likelihood_history_, converged_ etc. devem ser copiados do algoritmo."""
    X = np.eye(2)
    data = _FakeDataTemplate(X)
    pipeline = _FakePipelineWithoutWhitening()
    algorithm = _FakeAlgorithm(np.eye(2))

    model = ICAModel(data=data, pipeline=pipeline, algorithm=algorithm).fit()

    assert model.history_ == algorithm.history_
    assert model.log_likelihood_history_ == algorithm.log_likelihood_history_
    assert model.converged_ == algorithm.converged_
    assert model.n_iterations_ == algorithm.n_iterations_
    assert model.elapsed_time_ == algorithm.elapsed_time_


def test_evaluate_calls_each_metric_and_wires_dict_by_name():
    """evaluate() deve chamar metric.compute(model) para cada metrica e indexar pelo nome."""
    X = np.eye(2)
    data = _FakeDataTemplate(X)
    pipeline = _FakePipelineWithoutWhitening()
    algorithm = _FakeAlgorithm(np.eye(2))
    model = ICAModel(data=data, pipeline=pipeline, algorithm=algorithm).fit()

    metrics = [_FakeMetric("a", 1.23), _FakeMetric("b", 4.56)]
    result = model.evaluate(metrics)

    assert result == {"a": 1.23, "b": 4.56}
