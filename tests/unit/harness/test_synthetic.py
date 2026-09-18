"""Testes unitarios para ica.harness.synthetic."""

import numpy as np
import pytest

from ica.algorithms.fastica_ml import FastICAML
from ica.harness.synthetic import (
    generate_synthetic_run,
    make_synthetic_sources,
    make_well_conditioned_mixing_matrix,
)
from ica.model import ICAModel
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.postprocessing.matching import best_match_correlation
from ica.preprocessing import Centering, Pipeline, Whitening


def test_make_synthetic_sources_are_standardized():
    """Cada fonte gerada deve ter media ~0 e variancia ~1."""
    rng = np.random.default_rng(0)
    S = make_synthetic_sources(["laplace", "uniform"], 5000, rng)
    assert np.allclose(S.mean(axis=1), 0.0, atol=1e-8)
    assert np.allclose(S.std(axis=1), 1.0, atol=1e-8)


def test_make_synthetic_sources_rejects_unknown_kind():
    """Um tipo de fonte desconhecido deve levantar ValueError."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        make_synthetic_sources(["nao-existe"], 100, rng)


def test_make_well_conditioned_mixing_matrix_respects_condition_bound():
    """A matriz sorteada deve ter numero de condicao abaixo do limite pedido."""
    rng = np.random.default_rng(0)
    A = make_well_conditioned_mixing_matrix(rng, 3, max_condition_number=5.0)
    assert np.linalg.cond(A) < 5.0


def test_generate_synthetic_run_populates_ground_truth_and_meta():
    """generate_synthetic_run deve popular o gabarito e os metadados de dominio."""
    run = generate_synthetic_run(
        domain="distribution", kinds=["laplace", "uniform"], n_samples=3000, seed=0
    )
    assert run.sources_true.shape == (2, 3000)
    assert run.mixing_matrix_true.shape == (2, 2)
    signal_matrix = run.data.load()
    assert signal_matrix.domain == "distribution"
    assert signal_matrix.meta["sample_size"] == 3000
    mixing_matrix_true, sources_true = run.data.load_ground_truth("ignored")
    assert mixing_matrix_true is run.mixing_matrix_true
    assert sources_true is run.sources_true


def test_make_synthetic_sources_accepts_gaussian_kind():
    """O tipo 'gaussian' deve gerar uma fonte padronizada, como os demais tipos."""
    rng = np.random.default_rng(0)
    S = make_synthetic_sources(["gaussian"], 5000, rng)
    assert S.shape == (1, 5000)
    assert abs(S.mean()) < 0.05
    assert abs(S.std() - 1.0) < 0.05


def test_generate_synthetic_run_populates_image_metadata():
    """Para domain='image', a metadata deve incluir height/width/is_rgb/n_images."""
    run = generate_synthetic_run(domain="image", kinds=["laplace", "uniform"], n_samples=16, seed=0)
    meta = run.data.load().meta
    assert meta["height"] == 4
    assert meta["width"] == 4
    assert meta["is_rgb"] is False
    assert meta["n_images"] == 2


def test_generate_synthetic_run_is_reproducible_with_same_seed():
    """A mesma semente deve gerar exatamente a mesma mistura."""
    run_a = generate_synthetic_run("audio", ["laplace", "uniform"], 2000, seed=7)
    run_b = generate_synthetic_run("audio", ["laplace", "uniform"], 2000, seed=7)
    assert np.array_equal(run_a.data.load().data, run_b.data.load().data)


def test_generate_synthetic_run_is_recoverable_end_to_end():
    """Uma mistura sintetica deve ser recuperavel de ponta a ponta via ICAModel."""
    run = generate_synthetic_run(
        domain="distribution", kinds=["laplace", "uniform"], n_samples=5000, seed=1
    )
    model = ICAModel(
        data=run.data,
        pipeline=Pipeline([Centering(), Whitening()]),
        algorithm=FastICAML(nonlinearity=AdaptiveScore(), max_iterations=200),
        groundtruth_root="ignored",
    )
    model.fit()
    assert best_match_correlation(run.sources_true, model.sources_) > 0.9
