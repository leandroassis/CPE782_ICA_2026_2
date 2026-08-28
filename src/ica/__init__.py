"""Separacao Cega de Fontes via Analise de Componentes Independentes (ICA).

Implementa a familia de algoritmos Infomax / Maxima Verossimilhanca (ML)
descrita em ``context/ICA_BACKGROUND.md``: pre-processamento (centralizacao e
branqueamento), funcoes de pontuacao (score functions) sub/supergaussianas e
tres algoritmos de otimizacao (Bell-Sejnowski, Gradiente Natural, FastICA-ML)
intercambiaveis por injecao de dependencia.
"""

from ica.model import ICAModel

__all__ = ["ICAModel"]
__version__ = "0.1.0"
