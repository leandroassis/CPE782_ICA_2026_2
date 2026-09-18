"""Reagrupamento de planos RGB por correlacao espacial (modo de condicionamento B).

Skill ``ica-evaluation``, Secao 1: cada plano R/G/B de
:meth:`~ica.data.image_template.ImageTemplate.mode_b_plane_matrices` e
separado por uma ICA **independente** -- tem permutacao/sinal proprios.
Reagrupa os planos da mesma fonte pela correlacao espacial entre as
componentes recuperadas dos tres planos (as versoes R/G/B de uma mesma
imagem-fonte sao espacialmente muito correlacionadas), reaproveitando o
mesmo problema de atribuicao de :mod:`ica.postprocessing.matching`.
"""

import numpy as np

from ica.postprocessing.matching import hungarian_match


def regroup_rgb_planes(plane_components: list[np.ndarray]) -> list[np.ndarray]:
    """Reagrupa componentes recuperadas de 3 planos (R, G, B) em trincas por imagem.

    Usa o plano R como ancora; G e B sao casados a ele por correlacao
    espacial absoluta (:func:`~ica.postprocessing.matching.hungarian_match`).
    O sinal de cada canal dentro da trinca **nao** e alinhado aqui -- isso
    e responsabilidade de :func:`~ica.postprocessing.ambiguity.fix_sign`,
    aplicado por plano antes do reagrupamento.

    Parameters
    ----------
    plane_components : list of np.ndarray
        Exatamente 3 arrays ``(k, n_pixels)``, um por canal, na ordem R, G, B
        -- tipicamente ``ICAModel.sources_`` de 3 ajustes independentes.

    Returns
    -------
    list of np.ndarray
        ``k`` trincas, cada uma shape ``(3, n_pixels)`` (linhas R, G, B da
        mesma imagem-fonte), na ordem das linhas do plano-ancora (R).

    Raises
    ------
    ValueError
        Se ``plane_components`` nao tiver exatamente 3 elementos, ou se os
        planos nao tiverem o mesmo numero de componentes ``k``.
    """
    if len(plane_components) != 3:
        raise ValueError("regroup_rgb_planes espera exatamente 3 planos (R, G, B).")

    anchor = plane_components[0]
    n_components = anchor.shape[0]
    if any(plane.shape[0] != n_components for plane in plane_components):
        raise ValueError("Todos os planos devem ter o mesmo numero de componentes.")

    matched_planes = [anchor]
    for plane in plane_components[1:]:
        match = hungarian_match(anchor, plane)
        order = np.empty(n_components, dtype=int)
        order[match.reference_indices] = match.matched_indices
        matched_planes.append(plane[order])

    return [np.stack([plane[i] for plane in matched_planes]) for i in range(n_components)]
