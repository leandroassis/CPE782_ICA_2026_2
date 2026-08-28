"""Teste de integracao: chaveamento adaptativo importa fim-a-fim com fontes mistas.

Ver context/ICA_BACKGROUND.md, Secao 3.4; context/DEVELOPMENT_GUIDELINES.md, Secao 5.
"""

import numpy as np

from ica.algorithms.natural_gradient import NaturalGradientICA
from ica.model import ICAModel
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.nonlinearities.subgaussian import SubGaussianScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.whitening import Whitening


def test_adaptive_score_succeeds_where_fixed_wrong_nonlinearity_diverges(
    rng, make_sources, make_mixing_matrix, best_match_correlation, array_data_template
):
    """Com 1 fonte Laplaciana + 1 Uniforme, AdaptiveScore deve separar bem.

    Uma unica nao-linearidade fixa errada para ambas (``SubGaussianScore``
    aplicada tambem a componente supergaussiana) deve divergir -- prova,
    atraves da fachada completa :class:`~ica.model.ICAModel`, que o
    chaveamento por componente (ICA_BACKGROUND.md, Secao 3.4) nao e um
    detalhe cosmetico.

    Usa ``learning_rate=0.001`` explicitamente (maior que o default de
    ``NaturalGradientICA``, ``0.0005``) para tornar a divergencia visivel
    de forma confiavel -- o default mais conservador existe justamente
    para evitar essa instabilidade na pratica.
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
    assert np.all(np.isfinite(adaptive_model.sources_))
    assert best_match_correlation(S, adaptive_model.sources_) > 0.9

    wrong_model = ICAModel(
        data=array_data_template(X),
        pipeline=Pipeline([Centering(), Whitening()]),
        algorithm=NaturalGradientICA(
            nonlinearity=SubGaussianScore(), learning_rate=0.001, max_iterations=500
        ),
    )
    with np.errstate(all="ignore"):
        wrong_model.fit()
    assert not np.all(np.isfinite(wrong_model.sources_))
