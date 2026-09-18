"""Testes unitarios para ica.postprocessing.regroup (modo B, skill ica-evaluation)."""

import numpy as np
import pytest

from ica.postprocessing.regroup import regroup_rgb_planes


def test_regroup_recovers_original_grouping_despite_permutation_and_sign(rng):
    """Cada plano tem sua propria permutacao/sinal; regroup deve desfazer isso e reagrupar."""
    n_pixels = 500
    # 3 "imagens-fonte" espacialmente distintas, uma por linha.
    base_images = rng.normal(size=(3, n_pixels))

    r_plane = base_images.copy()
    g_plane = base_images[[2, 0, 1]] * -1.0  # permutado e com sinal trocado
    b_plane = base_images[[1, 2, 0]] * 2.0  # permutado e reescalado

    triplets = regroup_rgb_planes([r_plane, g_plane, b_plane])

    assert len(triplets) == 3
    for i, triplet in enumerate(triplets):
        assert triplet.shape == (3, n_pixels)
        # os 3 canais da trinca devem estar fortemente correlacionados entre si
        # (mesma imagem-fonte espacial) e com a imagem-fonte i original.
        correlation_with_source = [
            abs(np.corrcoef(triplet[c], base_images[i])[0, 1]) for c in range(3)
        ]
        assert min(correlation_with_source) > 0.99


def test_regroup_rejects_wrong_number_of_planes():
    """regroup_rgb_planes deve exigir exatamente 3 planos."""
    with pytest.raises(ValueError):
        regroup_rgb_planes([np.zeros((2, 10)), np.zeros((2, 10))])


def test_regroup_rejects_mismatched_component_counts():
    """regroup_rgb_planes deve exigir o mesmo numero de componentes em cada plano."""
    with pytest.raises(ValueError):
        regroup_rgb_planes([np.zeros((2, 10)), np.zeros((3, 10)), np.zeros((2, 10))])
