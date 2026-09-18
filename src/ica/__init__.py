"""Separacao Cega de Fontes via Analise de Componentes Independentes (ICA).

Implementa a familia de algoritmos Infomax / Maxima Verossimilhanca (ML)
descrita na skill ``ica-ml`` (``.claude/skills/ica-ml/SKILL.md``):
pre-processamento (centralizacao e branqueamento), funcoes de pontuacao
(score functions) sub/supergaussianas e tres algoritmos de otimizacao
(Bell-Sejnowski, Gradiente Natural, FastICA-ML) intercambiaveis por injecao
de dependencia. Fluxo completo: ``io -> assess -> preprocess -> ica ->
postprocess -> evaluate``, orquestrado por ``harness`` -- ver
``.claude/PIPELINE_MAP.md``.
"""

from ica.model import ICAModel

__all__ = ["ICAModel"]
__version__ = "0.1.0"
