"""Teste de integracao: pipeline completo sobre um CSV de imagens sintetico.

Usa um CSV no formato real de ``data/imagens/`` (colunas ``misturaN``,
uma linha por pixel), nunca os dados reais do trabalho.
"""

import numpy as np
import pandas as pd

from ica.algorithms.fastica_ml import FastICAML
from ica.data.image_template import ImageTemplate
from ica.model import ICAModel
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.whitening import Whitening


def test_image_csv_pipeline_recovers_sources_and_reconstructs_shape(
    tmp_path, rng, best_match_correlation
):
    """CSV sintetico (8x8) -> ImageTemplate -> ICAModel deve separar bem e reconstruir (8, 8)."""
    height, width = 8, 8
    n_pixels = height * width

    source_1 = rng.laplace(size=n_pixels)
    source_1 = (source_1 - source_1.mean()) / source_1.std()
    source_2 = rng.uniform(-1, 1, size=n_pixels)
    source_2 = (source_2 - source_2.mean()) / source_2.std()
    S = np.vstack([source_1, source_2])

    A = rng.normal(size=(2, 2))
    while np.linalg.cond(A) > 10:
        A = rng.normal(size=(2, 2))
    X = A @ S

    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    pd.DataFrame({"mistura1": X[0], "mistura2": X[1]}).to_csv(
        run_dir / "mix_imagens_grayscale.csv", index=False
    )

    data = ImageTemplate(run="run1", data_root=tmp_path)
    model = ICAModel(
        data=data,
        pipeline=Pipeline([Centering(), Whitening()]),
        algorithm=FastICAML(nonlinearity=AdaptiveScore(), max_iterations=200),
    )
    model.fit()

    assert data.height_ == height
    assert data.width_ == width
    assert best_match_correlation(S, model.sources_) > 0.9

    reconstructed = data.reconstruct(model.sources_[0])
    assert reconstructed.shape == (height, width)
