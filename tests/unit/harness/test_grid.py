"""Testes unitarios para ica.harness.grid."""

import numpy as np
import pandas as pd
import pytest

from ica.harness.grid import ALGORITHMS, conditioning_modes, run_cell, run_grid


class _FakeImageData:
    def __init__(self, is_rgb):
        self.is_rgb_ = is_rgb


def test_conditioning_modes_for_grayscale_image_is_a_only():
    """Imagem em escala de cinza: so o modo A (B==C nesse caso)."""
    assert conditioning_modes("imagens", _FakeImageData(is_rgb=False)) == ["A"]


def test_conditioning_modes_for_rgb_image_is_b_and_c():
    """Imagem RGB: modos B e C, nunca A."""
    assert conditioning_modes("imagens", _FakeImageData(is_rgb=True)) == ["B", "C"]


def test_conditioning_modes_for_distribution_and_audio_is_single_mode():
    """Distribuicao e audio: um unico modo (sem eixo de condicionamento)."""
    assert conditioning_modes("dist", None) == ["unico"]
    assert conditioning_modes("audio", None) == ["unico"]


def _write_synthetic_distribution_run(tmp_path, rng, n_samples=500):
    """Escreve um run sintetico de distribuicao (mix/ + groundtruth/) em tmp_path."""
    laplace = rng.laplace(size=n_samples)
    uniform = rng.uniform(-1, 1, size=n_samples)
    S = np.vstack(
        [
            (laplace - laplace.mean()) / laplace.std(),
            (uniform - uniform.mean()) / uniform.std(),
        ]
    )
    A = np.array([[1.0, 0.5], [0.3, 1.0]])
    X = A @ S

    mix_dir = tmp_path / "mix" / "run1"
    mix_dir.mkdir(parents=True)
    pd.DataFrame(X.T, columns=["mistura1", "mistura2"]).to_csv(
        mix_dir / f"mix_{n_samples}_stats.csv", index=False
    )

    gt_dir = tmp_path / "groundtruth" / "run1"
    gt_dir.mkdir(parents=True)
    pd.DataFrame(A, columns=["A1", "A2"]).to_csv(gt_dir / "mix_matrix_run1.csv", index=False)
    pd.DataFrame(S.T, columns=["Laplace", "Uniform"]).to_csv(
        gt_dir / f"sources_{n_samples}_stats.csv", index=False
    )
    return tmp_path / "mix", tmp_path / "groundtruth"


def test_run_cell_fits_and_evaluates_a_distribution_cell(tmp_path):
    """run_cell deve ajustar o modelo e computar as metricas padrao (com gabarito)."""
    rng = np.random.default_rng(0)
    data_root, groundtruth_root = _write_synthetic_distribution_run(tmp_path, rng)

    cell = run_cell(
        "dist",
        "run1",
        "fastica_ml",
        "unico",
        data_root,
        groundtruth_root,
        sample_size=500,
        max_iterations=200,
    )

    assert cell.sample == "dist"
    assert cell.model.converged_ is not None
    assert cell.metrics["amari_index"] is not None
    assert cell.metrics["amari_index"] < 0.3
    assert cell.log_likelihood_per_sample == cell.model.log_likelihood_history_[-1]


def test_run_grid_runs_all_algorithms_for_a_single_mode_domain(tmp_path):
    """run_grid sobre distribuicao (modo unico) deve rodar os 3 algoritmos."""
    rng = np.random.default_rng(1)
    data_root, groundtruth_root = _write_synthetic_distribution_run(tmp_path, rng)

    cells = run_grid(
        "dist",
        "run1",
        data_root,
        groundtruth_root,
        sample_size=500,
        max_workers=2,
        max_iterations=100,
    )

    assert len(cells) == len(ALGORITHMS)
    assert {cell.algorithm for cell in cells} == set(ALGORITHMS)
    assert all(cell.mode == "unico" for cell in cells)


def test_run_cell_rejects_unknown_algorithm(tmp_path):
    """Um nome de algoritmo desconhecido deve levantar KeyError."""
    rng = np.random.default_rng(0)
    data_root, groundtruth_root = _write_synthetic_distribution_run(tmp_path, rng)
    with pytest.raises(KeyError):
        run_cell(
            "dist", "run1", "nao_existe", "unico", data_root, groundtruth_root, sample_size=500
        )


def _write_synthetic_rgb_image_run(tmp_path, rng, n_pixels=16, n_images=3):
    """Escreve um run RGB sintetico (mix/ + groundtruth/) em tmp_path, com A conhecida."""
    n_sources = n_images * 3
    sources = rng.uniform(0.0, 1.0, size=(n_sources, n_pixels))
    A = np.eye(n_sources) + 0.05 * rng.normal(size=(n_sources, n_sources))
    mixtures = A @ sources

    mix_dir = tmp_path / "mix" / "imagens" / "run_rgb"
    mix_dir.mkdir(parents=True)
    pd.DataFrame(
        mixtures.T, columns=[f"mistura{i + 1}" for i in range(n_sources)]
    ).to_csv(mix_dir / "mix_imagens_rgb.csv", index=False)

    gt_dir = tmp_path / "groundtruth" / "imagens" / "run_rgb"
    gt_dir.mkdir(parents=True)
    pd.DataFrame(A, columns=[f"A{i + 1}" for i in range(n_sources)]).to_csv(
        gt_dir / "mix_matrix_rgb.csv", index=False
    )
    source_columns = [
        f"Imagem{i + 1}_{channel}" for i in range(n_images) for channel in ("R", "G", "B")
    ]
    pd.DataFrame(sources.T, columns=source_columns).to_csv(
        gt_dir / "sources_imagens_rgb.csv", index=False
    )
    return tmp_path / "mix" / "imagens", tmp_path / "groundtruth" / "imagens"


def test_run_cell_mode_c_produces_rgb_composites_and_psnr_ssim(tmp_path):
    """Modo C: um unico fit, sources_ ja concatenado por imagem, com PSNR/SSIM contra o gabarito."""
    rng = np.random.default_rng(0)
    data_root, groundtruth_root = _write_synthetic_rgb_image_run(tmp_path, rng)

    cell = run_cell(
        "imagens", "run_rgb", "fastica_ml", "C", data_root, groundtruth_root, max_iterations=200
    )

    assert cell.mode == "C"
    assert len(cell.models) == 1
    assert cell.rgb_composites is not None
    assert len(cell.rgb_composites) == 3
    assert cell.rgb_composites[0].shape == (3, 16)
    assert cell.metrics["psnr_db_per_source"] is not None
    assert cell.metrics["ssim_per_source"].shape == (3,)


def test_run_cell_mode_b_fits_three_independent_planes_and_regroups(tmp_path):
    """Modo B: 3 fits independentes (um por plano), reagrupados em trincas RGB."""
    rng = np.random.default_rng(1)
    data_root, groundtruth_root = _write_synthetic_rgb_image_run(tmp_path, rng)

    cell = run_cell(
        "imagens", "run_rgb", "fastica_ml", "B", data_root, groundtruth_root, max_iterations=200
    )

    assert cell.mode == "B"
    assert len(cell.models) == 3
    assert cell.rgb_composites is not None
    assert len(cell.rgb_composites) == 3
    assert isinstance(cell.metrics["convergence_iterations"], list)
    assert len(cell.metrics["convergence_iterations"]) == 3
    assert cell.metrics["psnr_db_per_source"] is not None
    assert cell.log_likelihood_per_sample == pytest.approx(
        float(np.mean([m.log_likelihood_history_[-1] for m in cell.models]))
    )


def test_run_cell_image_conditioned_modes_tolerate_missing_ground_truth(tmp_path):
    """Sem gabarito, modos B/C ainda devem rodar, so sem as metricas de validacao."""
    rng = np.random.default_rng(2)
    data_root, groundtruth_root = _write_synthetic_rgb_image_run(tmp_path, rng)
    missing_groundtruth_root = groundtruth_root.parent / "does_not_exist"

    cell = run_cell(
        "imagens",
        "run_rgb",
        "fastica_ml",
        "C",
        data_root,
        missing_groundtruth_root,
        max_iterations=50,
    )

    assert "psnr_db_per_source" not in cell.metrics
    assert cell.rgb_composites is not None
