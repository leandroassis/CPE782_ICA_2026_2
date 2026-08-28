"""Teste de integracao: efeito do tamanho amostral na qualidade da separacao.

Ver context/TASK_DESCRIPTION.md ("compara como o tamanho da amostra
afeta a qualidade da separacao"). Usa CSVs no formato real de
``data/dist/`` (um arquivo ``mix_{T}_stats.csv`` por tamanho amostral),
nunca os dados reais do trabalho.
"""

import numpy as np
import pandas as pd

from ica.algorithms.fastica_ml import FastICAML
from ica.data.distribution_template import DistributionTemplate
from ica.model import ICAModel
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.whitening import Whitening


def _standardize(raw: np.ndarray) -> np.ndarray:
    return (raw - raw.mean()) / raw.std()


def test_larger_sample_sizes_achieve_at_least_as_good_recovery(
    tmp_path, rng, make_mixing_matrix, best_match_correlation
):
    """Mesma mistura subamostrada em T=100/1000/100000 -- a maior amostra deve separar melhor."""
    source_1_pool = _standardize(rng.laplace(size=100_000))
    source_2_pool = _standardize(rng.uniform(-1, 1, size=100_000))
    A = make_mixing_matrix(rng, 2)

    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    sample_sizes = [100, 1000, 100000]
    for sample_size in sample_sizes:
        X = A @ np.vstack([source_1_pool[:sample_size], source_2_pool[:sample_size]])
        pd.DataFrame({"mistura1": X[0], "mistura2": X[1]}).to_csv(
            run_dir / f"mix_{sample_size}_stats.csv", index=False
        )

    scores = {}
    for sample_size in sample_sizes:
        data = DistributionTemplate(run="run1", data_root=tmp_path, sample_size=sample_size)
        model = ICAModel(
            data=data,
            pipeline=Pipeline([Centering(), Whitening()]),
            algorithm=FastICAML(nonlinearity=AdaptiveScore(), max_iterations=200),
        )
        model.fit()
        S = np.vstack(
            [source_1_pool[:sample_size], source_2_pool[:sample_size]]
        )
        scores[sample_size] = best_match_correlation(S, model.sources_)

    assert scores[100] > 0.9
    assert scores[1000] > 0.95
    assert scores[100000] > 0.999
    assert scores[100000] >= scores[100]
