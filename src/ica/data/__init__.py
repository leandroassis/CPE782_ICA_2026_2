"""Carregadores de amostras (DataTemplate) para cada tipo de dado do trabalho.

Bloco ``io/`` de ``.claude/PIPELINE_MAP.md``: adapters dominio <->
``SignalMatrix``.
"""

from ica.data.audio_template import AudioTemplate
from ica.data.base import DataTemplate
from ica.data.distribution_template import DistributionTemplate
from ica.data.image_template import ImageTemplate
from ica.data.in_memory_template import InMemorySignalMatrixTemplate

__all__ = [
    "DataTemplate",
    "ImageTemplate",
    "DistributionTemplate",
    "AudioTemplate",
    "InMemorySignalMatrixTemplate",
]
