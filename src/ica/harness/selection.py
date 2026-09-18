"""Selecao da "melhor separacao" de uma grade de celulas de um run.

Skill ``ica-evaluation``, Secao 5: ``melhor = argmax log-L/amostra``,
``desempate = menor indice de Amari`` (validacao). Guarda-se **todos** os
resultados da grade (para as secoes comparativas do trabalho); so o
vencedor vai para a figura-vitrine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ica.harness.grid import CellResult


def log_likelihood_per_sample(cell: CellResult) -> float:
    """Ultimo valor da trajetoria de log-verossimilhanca media de uma celula.

    Parameters
    ----------
    cell : CellResult
        Resultado de uma celula ja ajustada.

    Returns
    -------
    float
        ``(1/T) log L(B)`` na iteracao final.
    """
    return cell.model.log_likelihood_history_[-1]


def select_best(cells: list[CellResult]) -> CellResult:
    """Escolhe a melhor celula: argmax log-L/amostra, desempate por menor Amari.

    Parameters
    ----------
    cells : list of CellResult
        Todas as celulas de um run (algoritmos x modos de condicionamento
        aplicaveis).

    Returns
    -------
    CellResult
        A celula vencedora.

    Raises
    ------
    ValueError
        Se ``cells`` estiver vazia.
    """
    if not cells:
        raise ValueError("select_best requer ao menos uma celula.")

    def sort_key(cell: CellResult) -> tuple[float, float]:
        amari = cell.metrics.get("amari_index")
        amari_for_tiebreak = amari if amari is not None else float("inf")
        return (-log_likelihood_per_sample(cell), amari_for_tiebreak)

    return min(cells, key=sort_key)
