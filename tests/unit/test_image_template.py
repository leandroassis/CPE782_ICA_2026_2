"""Testes unitarios para ImageTemplate.

Usa CSVs sinteticos em ``tmp_path``, no formato real de
``data/mix/imagens/``, exceto quando explicitamente indicado que o teste
verifica algo contra os dados reais do trabalho (``data/groundtruth/``).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ica.data.image_template import ImageTemplate

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_grayscale_csv(run_dir, n_pixels=16, n_mixtures=3):
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {f"mistura{i + 1}": np.arange(n_pixels, dtype=float) + i for i in range(n_mixtures)}
    pd.DataFrame(data).to_csv(run_dir / "mix_imagens_grayscale.csv", index=False)


def _write_rgb_csv(run_dir, n_pixels=16, n_mixtures=9):
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {f"mistura{i + 1}": np.arange(n_pixels, dtype=float) + i for i in range(n_mixtures)}
    pd.DataFrame(data).to_csv(run_dir / "mix_imagens_rgb.csv", index=False)


def test_load_returns_mixtures_by_pixels_shape(tmp_path):
    """load() deve retornar um SignalMatrix com data shape (n_misturas, n_pixels)."""
    _write_grayscale_csv(tmp_path / "run1", n_pixels=16, n_mixtures=3)
    template = ImageTemplate(run="run1", data_root=tmp_path)
    signal_matrix = template.load()
    assert signal_matrix.data.shape == (3, 16)
    assert signal_matrix.domain == "image"
    assert signal_matrix.meta["is_rgb"] is False
    assert signal_matrix.meta["n_images"] == 3


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


def test_mode_b_plane_matrices_groups_by_channel(tmp_path):
    """mode_b_plane_matrices deve devolver 3 matrizes (n_images, n_pixels), uma por canal."""
    n_pixels = 4
    run_dir = tmp_path / "run3"
    run_dir.mkdir(parents=True)
    # 3 imagens x 3 canais = 9 misturas; cada mistura e uma constante para
    # que o agrupamento seja facil de verificar por valor.
    data = {f"mistura{i + 1}": np.full(n_pixels, i, dtype=float) for i in range(9)}
    pd.DataFrame(data).to_csv(run_dir / "mix_imagens_rgb.csv", index=False)

    template = ImageTemplate(run="run3", data_root=tmp_path)
    planes = template.mode_b_plane_matrices()

    assert len(planes) == 3
    for plane in planes:
        assert plane.shape == (3, n_pixels)
    # canal R (indice 0): misturas 0, 3, 6 -> imagens 0, 1, 2
    assert np.allclose(planes[0][:, 0], [0, 3, 6])
    # canal G (indice 1): misturas 1, 4, 7
    assert np.allclose(planes[1][:, 0], [1, 4, 7])
    # canal B (indice 2): misturas 2, 5, 8
    assert np.allclose(planes[2][:, 0], [2, 5, 8])


def test_mode_b_plane_matrices_rejects_grayscale(tmp_path):
    """mode_b_plane_matrices so se aplica a RGB."""
    _write_grayscale_csv(tmp_path / "run1")
    template = ImageTemplate(run="run1", data_root=tmp_path)
    with pytest.raises(ValueError):
        template.mode_b_plane_matrices()


def test_mode_c_matrix_concatenates_channels_per_image(tmp_path):
    """mode_c_matrix deve concatenar R,G,B de cada imagem numa linha de comprimento 3P."""
    n_pixels = 4
    run_dir = tmp_path / "run3"
    run_dir.mkdir(parents=True)
    data = {f"mistura{i + 1}": np.full(n_pixels, i, dtype=float) for i in range(9)}
    pd.DataFrame(data).to_csv(run_dir / "mix_imagens_rgb.csv", index=False)

    template = ImageTemplate(run="run3", data_root=tmp_path)
    matrix = template.mode_c_matrix()

    assert matrix.shape == (3, 3 * n_pixels)
    # imagem 0 (misturas 0,1,2) concatenadas
    assert np.allclose(matrix[0], np.repeat([0, 1, 2], n_pixels))


def test_load_ground_truth_returns_none_when_run_missing(tmp_path):
    """load_ground_truth deve tolerar a ausencia do diretorio de gabarito."""
    _write_grayscale_csv(tmp_path / "run1")
    template = ImageTemplate(run="run1", data_root=tmp_path)
    mixing_matrix_true, sources_true = template.load_ground_truth(tmp_path / "does_not_exist")
    assert mixing_matrix_true is None
    assert sources_true is None


def test_load_ground_truth_against_real_rgb_run():
    """O agrupamento consecutivo-por-imagem deve bater com o gabarito real de run3.

    ``data/groundtruth/imagens/run3/sources_imagens_rgb.csv`` tem cabecalho
    ``Cachorro_R,Cachorro_G,Cachorro_B,Gato_R,...`` -- confirma que os
    canais de uma mesma imagem sao consecutivos, a mesma convencao usada
    por :meth:`mode_b_plane_matrices`/:meth:`mode_c_matrix`.
    """
    data_root = _REPO_ROOT / "data" / "mix" / "imagens"
    groundtruth_root = _REPO_ROOT / "data" / "groundtruth" / "imagens"
    if not (data_root / "run3").exists():
        pytest.skip("dados reais de data/mix/imagens/run3 nao disponiveis")

    template = ImageTemplate(run="run3", data_root=data_root)
    mixing_matrix_true, sources_true = template.load_ground_truth(groundtruth_root)

    assert mixing_matrix_true.shape == (9, 9)
    assert sources_true.shape[0] == 9
    assert template.n_mixtures == 9
