"""Teste de integracao: chaveamento adaptativo importa fim-a-fim com fontes mistas.

Ver ``.claude/skills/ica-ml/SKILL.md``, Secao 5 (chaveamento) e Secao 6
(GN/BS exigem o chaveamento explicito; FastICA-ML nao -- ``diag(beta_i)``
absorve a natureza da fonte, por isso o teste usa Gradiente Natural, no qual
o efeito de uma nao-linearidade fixa errada e mensuravel).
"""

import numpy as np

from ica.algorithms.natural_gradient import NaturalGradientICA
from ica.model import ICAModel
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.nonlinearities.subgaussian import SubGaussianScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.whitening import Whitening


def test_adaptive_score_beats_fixed_wrong_nonlinearity(
    rng, make_sources, make_mixing_matrix, best_match_correlation, array_data_template
):
    """Com 1 fonte Laplaciana + 1 Uniforme, AdaptiveScore deve superar g_- fixa.

    ``AdaptiveScore`` chaveia por componente via o momento nao-polinomial
    ``m_i`` (skill ica-ml, Secao 5) e escolhe, para cada uma, o ramo que
    satisfaz a condicao de estabilidade do Teor. 9.1. Forcar
    ``SubGaussianScore`` também sobre a componente supergaussiana (a
    Laplaciana) viola essa condicao para ela e degrada a qualidade da
    separacao de forma mensuravel e reprodutivel entre sementes.
    """
    S = make_sources(["laplace", "uniform"], 3000, rng)
    A = make_mixing_matrix(rng, 2)
    X = A @ S

    adaptive_model = ICAModel(
        data=array_data_template(X),
        pipeline=Pipeline([Centering(), Whitening()]),
        algorithm=NaturalGradientICA(
            nonlinearity=AdaptiveScore(), learning_rate=0.001, max_iterations=500
        ),
    )
    adaptive_model.fit()

    wrong_model = ICAModel(
        data=array_data_template(X),
        pipeline=Pipeline([Centering(), Whitening()]),
        algorithm=NaturalGradientICA(
            nonlinearity=SubGaussianScore(), learning_rate=0.001, max_iterations=500
        ),
    )
    with np.errstate(all="ignore"):
        wrong_model.fit()

    assert np.all(np.isfinite(adaptive_model.sources_))
    adaptive_quality = best_match_correlation(S, adaptive_model.sources_)
    assert adaptive_quality > 0.9

    if np.all(np.isfinite(wrong_model.sources_)):
        wrong_quality = best_match_correlation(S, wrong_model.sources_)
        assert adaptive_quality > wrong_quality
