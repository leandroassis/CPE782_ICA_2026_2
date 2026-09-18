"""Passos de pre-processamento encadeaveis (centralizacao, reescala robusta, branqueamento, PCA).

Ver ``.claude/PIPELINE_MAP.md``, bloco ``preprocess/``: centralizacao sempre,
branqueamento sempre (raiz-inversa-simetrica compartilhada, ver
:mod:`ica.preprocessing.symmetric`), gancho de PCA exposto e off por padrao.
``RobustScaling`` nao vem do livro-texto, mas mitiga a fragilidade da EVD da
covariancia a outliers extremos (ex.: ``data/mix/dist/run8``, quase singular
sem ela) -- faz parte do pipeline padrao entre ``Centering`` e ``Whitening``.
"""

from ica.preprocessing.base import PreprocessingStep
from ica.preprocessing.centering import Centering
from ica.preprocessing.pca import PCA
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.robust_scaling import RobustScaling
from ica.preprocessing.whitening import Whitening

__all__ = [
    "PreprocessingStep",
    "Centering",
    "RobustScaling",
    "PCA",
    "Whitening",
    "Pipeline",
]
