"""Bloco ``harness/`` de ``.claude/PIPELINE_MAP.md`` -- orquestra a grade de execucao.

``synthetic.py`` gera misturas sinteticas com fundo de verdade conhecido
(off por padrao); ``grid.py`` roda uma celula (:func:`~ica.harness.grid.run_cell`)
ou a grade inteira de um run (:func:`~ica.harness.grid.run_grid`),
paralelizada; ``store.py`` cacheia celulas para retomada; ``selection.py``
escolhe a "melhor separacao" (skill ica-evaluation, Secao 5).
"""

from ica.harness.grid import (
    ALGORITHMS,
    CellResult,
    build_algorithm,
    conditioning_modes,
    run_cell,
    run_grid,
    standard_metrics,
    standard_pipeline,
)
from ica.harness.selection import log_likelihood_per_sample, select_best
from ica.harness.store import ResumableStore, cell_key
from ica.harness.synthetic import (
    SyntheticRun,
    generate_synthetic_run,
    make_synthetic_sources,
    make_well_conditioned_mixing_matrix,
)

__all__ = [
    "ALGORITHMS",
    "CellResult",
    "build_algorithm",
    "conditioning_modes",
    "run_cell",
    "run_grid",
    "standard_metrics",
    "standard_pipeline",
    "log_likelihood_per_sample",
    "select_best",
    "ResumableStore",
    "cell_key",
    "SyntheticRun",
    "generate_synthetic_run",
    "make_synthetic_sources",
    "make_well_conditioned_mixing_matrix",
]
