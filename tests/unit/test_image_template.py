"""Testes unitarios para ImageTemplate (DEVELOPMENT_GUIDELINES.md, Secao 2.1).

Usa CSVs sinteticos em ``tmp_path``, no formato real de
``data/imagens/``, nunca os dados reais do trabalho.
"""

import numpy as np
import pandas as pd
import pytest

from ica.data.image_template import ImageTemplate


def _write_grayscale_csv(run_dir, n_pixels=16, n_mixtures=3):
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {f"mistura{i + 1}": np.arange(n_pixels, dtype=float) + i for i in range(n_mixtures)}
    pd.DataFrame(data).to_csv(run_dir / "mix_imagens_grayscale.csv", index=False)


def _write_rgb_csv(run_dir, n_pixels=16, n_mixtures=9):
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {f"mistura{i + 1}": np.arange(n_pixels, dtype=float) + i for i in range(n_mixtures)}
    pd.DataFrame(data).to_csv(run_dir / "mix_imagens_rgb.csv", index=False)


def test_load_returns_mixtures_by_pixels_shape(tmp_path):
    """load() deve retornar X com shape (n_misturas, n_pixels), transposto do CSV."""
    _write_grayscale_csv(tmp_path / "run1", n_pixels=16, n_mixtures=3)
    template = ImageTemplate(run="run1", data_root=tmp_path)
    X = template.load()
    assert X.shape == (3, 16)


def test_is_rgb_detection_from_filename(tmp_path):
    """is_rgb_ deve refletir qual arquivo CSV esta presente no run."""
    _write_grayscale_csv(tmp_path / "run1")
    _write_rgb_csv(tmp_path / "run3")

    grayscale_template = ImageTemplate(run="run1", data_root=tmp_path)
    rgb_template = ImageTemplate(run="run3", data_root=tmp_path)

    assert grayscale_template.is_rgb_ is False
    assert rgb_template.is_rgb_ is True


def test_height_and_width_inferred_from_pixel_count(tmp_path):
    """height_/width_ devem ser inferidos como sqrt(n_pixels) quando nao informados."""
    _write_grayscale_csv(tmp_path / "run1", n_pixels=16, n_mixtures=3)
    template = ImageTemplate(run="run1", data_root=tmp_path)
    template.load()
    assert (template.height_, template.width_) == (4, 4)


def test_reconstruct_reshapes_vector_to_height_width(tmp_path):
    """Reconstruct deve reformatar um vetor plano em uma matriz (height_, width_)."""
    _write_grayscale_csv(tmp_path / "run1", n_pixels=16, n_mixtures=3)
    template = ImageTemplate(run="run1", data_root=tmp_path)
    template.load()
    vector = np.arange(16, dtype=float)
    image = template.reconstruct(vector)
    assert image.shape == (4, 4)
    assert np.array_equal(image, vector.reshape(4, 4))


def test_load_raises_when_pixel_count_is_not_a_perfect_square(tmp_path):
    """load() deve levantar ValueError se n_pixels nao for quadrado e height/width faltarem."""
    _write_grayscale_csv(tmp_path / "run1", n_pixels=15, n_mixtures=3)
    template = ImageTemplate(run="run1", data_root=tmp_path)
    with pytest.raises(ValueError):
        template.load()


def test_explicit_height_width_override_bypasses_square_inference(tmp_path):
    """Informar height/width explicitamente deve funcionar mesmo com n_pixels nao-quadrado."""
    _write_grayscale_csv(tmp_path / "run1", n_pixels=12, n_mixtures=3)
    template = ImageTemplate(run="run1", data_root=tmp_path, height=3, width=4)
    template.load()
    vector = np.arange(12, dtype=float)
    assert template.reconstruct(vector).shape == (3, 4)


def test_n_mixtures_reflects_csv_column_count(tmp_path):
    """n_mixtures deve refletir o numero de colunas misturaN, sem precisar de load()."""
    _write_rgb_csv(tmp_path / "run3", n_pixels=16, n_mixtures=9)
    template = ImageTemplate(run="run3", data_root=tmp_path)
    assert template.n_mixtures == 9


def test_reconstruct_rgb_triplet_composes_three_channels(tmp_path):
    """reconstruct_rgb_triplet deve empilhar 3 vetores em um painel (H, W, 3)."""
    _write_rgb_csv(tmp_path / "run3", n_pixels=16, n_mixtures=9)
    template = ImageTemplate(run="run3", data_root=tmp_path)
    template.load()
    vectors = [np.arange(16, dtype=float) for _ in range(3)]
    rgb = template.reconstruct_rgb_triplet(vectors)
    assert rgb.shape == (4, 4, 3)


def test_reconstruct_rgb_triplet_rejects_wrong_number_of_vectors(tmp_path):
    """reconstruct_rgb_triplet deve exigir exatamente 3 vetores."""
    _write_rgb_csv(tmp_path / "run3", n_pixels=16, n_mixtures=9)
    template = ImageTemplate(run="run3", data_root=tmp_path)
    template.load()
    with pytest.raises(ValueError):
        template.reconstruct_rgb_triplet([np.zeros(16), np.zeros(16)])


def test_discover_runs_finds_both_grayscale_and_rgb(tmp_path):
    """discover_runs deve encontrar runs com CSV grayscale OU rgb."""
    _write_grayscale_csv(tmp_path / "run1")
    _write_rgb_csv(tmp_path / "run3")
    (tmp_path / "run_empty").mkdir()

    assert ImageTemplate.discover_runs(tmp_path) == ["run1", "run3"]


def test_discover_runs_returns_empty_list_for_missing_directory(tmp_path):
    """discover_runs nao deve levantar excecao se data_root nao existir."""
    assert ImageTemplate.discover_runs(tmp_path / "does_not_exist") == []
