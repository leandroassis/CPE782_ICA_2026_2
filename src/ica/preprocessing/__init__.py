"""Passos de pre-processamento encadeaveis (centralizacao, branqueamento, PCA, filtragem).

Ver context/ICA_BACKGROUND.md, Secao 2; livro-texto, Secao 13 (paginas
263-272) para PCA e filtragem temporal.
"""

from ica.preprocessing.base import PreprocessingStep
from ica.preprocessing.centering import Centering
from ica.preprocessing.pca import PCA
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.temporal_filtering import TemporalFiltering
from ica.preprocessing.whitening import Whitening

__all__ = [
    "PreprocessingStep",
    "Centering",
    "PCA",
    "TemporalFiltering",
    "Whitening",
    "Pipeline",
]
