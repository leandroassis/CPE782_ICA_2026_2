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
