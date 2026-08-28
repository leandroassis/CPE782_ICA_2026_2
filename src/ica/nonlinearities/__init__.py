"""Funcoes de pontuacao (score functions) sub/supergaussianas e adaptativa.

Ver context/ICA_BACKGROUND.md, Secao 3.3-3.4.
"""

from ica.nonlinearities.adaptive import AdaptiveScore
from ica.nonlinearities.base import NonlinearityTemplate
from ica.nonlinearities.subgaussian import SubGaussianScore
from ica.nonlinearities.supergaussian import SuperGaussianScore

__all__ = ["NonlinearityTemplate", "SuperGaussianScore", "SubGaussianScore", "AdaptiveScore"]
