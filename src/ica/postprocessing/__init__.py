"""Bloco ``postprocess/`` de ``.claude/PIPELINE_MAP.md``.

Resolve as ambiguidades de escala/sinal/permutacao da ICA
(:mod:`ica.postprocessing.ambiguity`), casa componentes recuperadas as fontes
verdadeiras so para pontuar (:mod:`ica.postprocessing.matching`) e reagrupa
planos RGB por correlacao espacial no modo de condicionamento B
(:mod:`ica.postprocessing.regroup`). Roda depois de ``ica/`` (skill
``ica-evaluation``).
"""

from ica.postprocessing.ambiguity import (
    fix_scale,
    fix_sign,
    resolve_ambiguities,
    stable_permutation_order,
)
from ica.postprocessing.matching import MatchResult, best_match_correlation, hungarian_match
from ica.postprocessing.regroup import regroup_rgb_planes

__all__ = [
    "fix_scale",
    "fix_sign",
    "stable_permutation_order",
    "resolve_ambiguities",
    "MatchResult",
    "hungarian_match",
    "best_match_correlation",
    "regroup_rgb_planes",
]
