"""Visualizacoes qualitativas e quantitativas de um ICAModel ajustado.

``ShowcaseVisualizer`` implementa a figura-vitrine de
``.claude/PIPELINE_MAP.md`` (mistura|esperado|obtido + metricas escritas);
os demais visualizadores cobrem diagnosticos auxiliares (diagrama de
mistura, curva de log-verossimilhanca) reaproveitados por
``ica.harness.grid``.
"""

from ica.visualization.audio_visualizer import AudioVisualizer
from ica.visualization.base import Visualizer
from ica.visualization.histogram_visualizer import HistogramVisualizer
from ica.visualization.image_visualizer import ImageVisualizer
from ica.visualization.log_likelihood_visualizer import LogLikelihoodVisualizer
from ica.visualization.mixing_diagram_3d_visualizer import MixingDiagram3DVisualizer
from ica.visualization.mixing_diagram_visualizer import MixingDiagramVisualizer
from ica.visualization.showcase_visualizer import ShowcaseVisualizer

__all__ = [
    "Visualizer",
    "ImageVisualizer",
    "HistogramVisualizer",
    "AudioVisualizer",
    "MixingDiagramVisualizer",
    "MixingDiagram3DVisualizer",
    "LogLikelihoodVisualizer",
    "ShowcaseVisualizer",
]
