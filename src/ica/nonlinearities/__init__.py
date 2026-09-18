"""Funcoes de pontuacao (score functions) sub/supergaussianas e adaptativa.

Ver ``.claude/skills/ica-ml/SKILL.md``, Secoes 3-5.
"""

from ica.nonlinearities.adaptive import AdaptiveScore
from ica.nonlinearities.base import NonlinearityTemplate
from ica.nonlinearities.subgaussian import SubGaussianScore
from ica.nonlinearities.supergaussian import SuperGaussianScore

__all__ = ["NonlinearityTemplate", "SuperGaussianScore", "SubGaussianScore", "AdaptiveScore"]
