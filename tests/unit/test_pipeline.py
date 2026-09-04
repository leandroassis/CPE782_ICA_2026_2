"""Testes unitarios para Pipeline (DEVELOPMENT_GUIDELINES.md, Secao 2.2)."""

import numpy as np
import pytest

from ica.preprocessing.base import PreprocessingStep
from ica.preprocessing.centering import Centering
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.whitening import Whitening


class _RecordingStep(PreprocessingStep):
    """Passo de teste que apenas registra a ordem em que foi chamado."""

    def __init__(self, call_order: list[str], name: str) -> None:
        self._call_order = call_order
        self._name = name

    def fit(self, X: np.ndarray) -> "_RecordingStep":
        self._call_order.append(self._name)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X + 1.0

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        return Y - 1.0


class _EstimationOnlyStep(PreprocessingStep):
    """Passo de teste que marca estimation_only=True (como TemporalFiltering)."""

    estimation_only = True

    def fit(self, X: np.ndarray) -> "_EstimationOnlyStep":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * 100.0

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _LinearStep(PreprocessingStep):
    """Passo de teste que expoe linear_matrix_ (como Whitening/PCA)."""

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def fit(self, X: np.ndarray) -> "_LinearStep":
        return self

    @property
    def linear_matrix_(self) -> np.ndarray:
        return self._matrix

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._matrix @ X

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        return np.linalg.inv(self._matrix) @ Y


def test_fit_transform_applies_steps_in_order():
    """Os passos devem ser ajustados/aplicados na ordem em que foram informados."""
    call_order: list[str] = []
    pipeline = Pipeline(
        [_RecordingStep(call_order, "primeiro"), _RecordingStep(call_order, "segundo")]
    )
    pipeline.fit_transform(np.zeros((2, 5)))
    assert call_order == ["primeiro", "segundo"]


def test_fit_transform_matches_manual_chain(rng):
    """fit_transform do pipeline deve coincidir com encadear os passos manualmente."""
    X = rng.normal(loc=3.0, size=(3, 200))
    pipeline_result = Pipeline([Centering(), Whitening()]).fit_transform(X)

    centering = Centering()
    manual = centering.fit_transform(X)
    whitening = Whitening()
    manual = whitening.fit_transform(manual)

    assert np.allclose(pipeline_result, manual)


def test_inverse_transform_reverses_steps_in_reverse_order():
    """inverse_transform deve desfazer os passos na ordem inversa da aplicacao."""
    call_order: list[str] = []
    pipeline = Pipeline(
        [_RecordingStep(call_order, "primeiro"), _RecordingStep(call_order, "segundo")]
    )
    transformed = pipeline.fit_transform(np.zeros((2, 5)))
    reconstructed = pipeline.inverse_transform(transformed)
    assert np.allclose(reconstructed, np.zeros((2, 5)))


def test_get_step_returns_matching_instance(rng):
    """get_step deve retornar a instancia ja ajustada do tipo pedido."""
    X = rng.normal(size=(3, 100))
    whitening = Whitening()
    pipeline = Pipeline([Centering(), whitening])
    pipeline.fit_transform(X)
    assert pipeline.get_step(Whitening) is whitening


def test_get_step_raises_when_type_absent():
    """get_step deve levantar ValueError se o tipo pedido nao estiver no pipeline."""
    pipeline = Pipeline([Centering()])
    with pytest.raises(ValueError):
        pipeline.get_step(Whitening)


def test_reconstruction_transform_skips_estimation_only_steps(rng):
    """reconstruction_transform deve pular passos estimation_only, aplicando so os demais."""
    X = rng.normal(size=(2, 10))
    pipeline = Pipeline([Centering(), _EstimationOnlyStep()])
    pipeline.fit_transform(X)

    reconstructed_input = pipeline.reconstruction_transform(X)

    assert np.allclose(reconstructed_input, Centering().fit_transform(X))


def test_reconstruction_transform_uses_already_fitted_parameters(rng):
    """reconstruction_transform nao deve re-ajustar os passos, so aplicar transform."""
    X = rng.normal(loc=5.0, size=(2, 50))
    centering = Centering()
    pipeline = Pipeline([centering])
    pipeline.fit_transform(X)

    X_new = rng.normal(loc=5.0, size=(2, 50))
    reconstructed = pipeline.reconstruction_transform(X_new)

    assert np.allclose(reconstructed, X_new - centering.mean_[:, np.newaxis])


def test_compose_linear_matrix_ignores_steps_without_linear_matrix():
    """Passos sem linear_matrix_ (ex.: Centering) nao devem alterar base_matrix."""
    pipeline = Pipeline([Centering()])
    B = np.array([[2.0, 0.0], [0.0, 2.0]])
    assert np.array_equal(pipeline.compose_linear_matrix(B), B)


def test_compose_linear_matrix_ignores_estimation_only_steps():
    """Passos estimation_only nao devem contribuir para compose_linear_matrix."""
    pipeline = Pipeline([_EstimationOnlyStep()])
    B = np.array([[2.0, 0.0], [0.0, 2.0]])
    assert np.array_equal(pipeline.compose_linear_matrix(B), B)


def test_compose_linear_matrix_multiplies_in_reverse_pipeline_order():
    """compose_linear_matrix deve compor base_matrix @ V @ P, respeitando a ordem do pipeline.

    Para um pipeline [Centering, P, V] (P aplicado antes de V), a
    composicao correta e B @ V @ P -- a mesma ordem em que as matrizes
    seriam aplicadas manualmente aos dados originais.
    """
    P = np.array([[2.0, 0.0], [0.0, 2.0]])
    V = np.array([[1.0, 1.0], [0.0, 1.0]])
    pipeline = Pipeline([Centering(), _LinearStep(P), _LinearStep(V)])
    B = np.array([[5.0, 0.0], [0.0, 5.0]])

    result = pipeline.compose_linear_matrix(B)

    assert np.array_equal(result, B @ V @ P)
