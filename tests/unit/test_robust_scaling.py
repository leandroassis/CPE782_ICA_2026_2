"""Testes unitarios para RobustScaling (mitigacao de outliers via MAD)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ica.preprocessing.centering import Centering
from ica.preprocessing.robust_scaling import RobustScaling

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_median_and_scale_shape_match_n_mixtures(rng):
    """median_ e scale_ devem ter shape (n_misturas,)."""
    X = rng.normal(size=(4, 500))
    step = RobustScaling(clip_mads=None).fit(X)
    assert step.median_.shape == (4,)
    assert step.scale_.shape == (4,)


def test_scale_matches_std_for_gaussian_data(rng):
    """Para dados gaussianos, o MAD escalado (x1.4826) deve aproximar o desvio-padrao."""
    X = rng.normal(scale=3.0, size=(2, 200_000))
    step = RobustScaling(clip_mads=None).fit(X)
    assert np.allclose(step.scale_, 3.0, atol=0.05)


def test_transform_without_clipping_is_pure_rescale(rng):
    """Sem clip_mads, transform deve ser exatamente X / scale_ (nenhum recorte)."""
    X = rng.normal(size=(3, 300))
    step = RobustScaling(clip_mads=None).fit(X)
    Z = step.transform(X)
    assert np.allclose(Z, X / step.scale_[:, np.newaxis])


def test_transform_clips_extreme_outlier(rng):
    """Com clip_mads, um outlier extremo deve ser recortado, nao passar integralmente."""
    X = rng.normal(size=(1, 1000))
    X[0, 0] = 1000.0  # outlier grosseiro
    step = RobustScaling(clip_mads=5.0).fit(X)
    Z = step.transform(X)
    upper_bound_scaled = (step.median_[0] + 5.0 * step.scale_[0]) / step.scale_[0]
    assert Z[0, 0] <= upper_bound_scaled + 1e-8
    assert Z[0, 0] < 1000.0 / step.scale_[0]


def test_clipping_reduces_condition_number_under_heavy_tailed_contamination(rng):
    """Recorte deve melhorar drasticamente o condicionamento da covariancia sob outliers.

    Reproduz o padrao observado na run8 de dist: uma mistura com poucos
    valores extremos (varias ordens de grandeza acima do grosso dos
    dados) deixa a covariancia numericamente quase singular; o recorte
    deve conter esse efeito.
    """
    n = 5000
    base = rng.normal(size=(2, n))
    contaminated = base.copy()
    outlier_idx = rng.choice(n, size=20, replace=False)
    contaminated[1, outlier_idx] *= 1e8

    cond_before = np.linalg.cond(np.cov(contaminated))
    step = RobustScaling(clip_mads=8.0).fit(contaminated)
    scaled = step.transform(contaminated)
    cond_after = np.linalg.cond(np.cov(scaled))

    assert cond_after < cond_before / 1000


def test_inverse_transform_round_trips_without_clipping(rng):
    """Sem recorte, inverse_transform(transform(X)) deve recuperar X exatamente."""
    X = rng.normal(size=(3, 400))
    step = RobustScaling(clip_mads=None)
    Z = step.fit_transform(X)
    reconstructed = step.inverse_transform(Z)
    assert np.allclose(reconstructed, X, atol=1e-8)


def test_linear_matrix_is_diagonal_inverse_scale_without_clipping(rng):
    """linear_matrix_ deve ser diag(1/scale_) quando clip_mads is None."""
    X = rng.normal(size=(3, 300))
    step = RobustScaling(clip_mads=None).fit(X)
    expected = np.diag(1.0 / step.scale_)
    assert np.allclose(step.linear_matrix_, expected)


def test_linear_matrix_is_none_when_clipping_enabled(rng):
    """linear_matrix_ deve ser None quando clip_mads esta ativo (transformacao nao-linear)."""
    X = rng.normal(size=(3, 300))
    step = RobustScaling(clip_mads=8.0).fit(X)
    assert step.linear_matrix_ is None


def test_linear_matrix_is_none_before_fit():
    """linear_matrix_ deve ser None antes de fit (scale_ ainda nao definido)."""
    assert RobustScaling(clip_mads=None).linear_matrix_ is None


def test_estimation_only_is_false():
    """RobustScaling participa tanto da estimacao quanto da reconstrucao (nao e estimation_only)."""
    assert RobustScaling().estimation_only is False


def test_real_run8_covariance_is_much_better_conditioned_with_robust_scaling():
    """Regressao: data/mix/dist/run8 e quase singular (outliers extremos, cond ~1e19).

    E o caso real que motivou ``RobustScaling`` entrar no pipeline padrao
    (``ica.harness.grid.standard_pipeline``): sem ela, o branqueamento (EVD)
    so evita gerar NaN gracas ao piso numerico de
    ``ica.preprocessing.symmetric``, mas a maioria das componentes
    branqueadas fica degenerada (variancia quase nula, sem informacao).
    Com ``RobustScaling``, a covariancia fica ordens de magnitude melhor
    condicionada.
    """
    csv_path = _REPO_ROOT / "data" / "mix" / "dist" / "run8" / "mix_100000_stats.csv"
    if not csv_path.exists():
        pytest.skip("dados reais de data/mix/dist/run8 nao disponiveis")

    X = pd.read_csv(csv_path).to_numpy(dtype=np.float64).T
    centered = Centering().fit_transform(X)

    condition_without = np.linalg.cond(np.cov(centered))
    robustly_scaled = RobustScaling().fit_transform(centered)
    condition_with = np.linalg.cond(np.cov(robustly_scaled))

    assert condition_without > 1e15
    assert condition_with < condition_without / 1e6
