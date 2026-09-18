"""Gerador sintetico -- espelha os 3 dominios com fundo de verdade conhecido.

Bloco ``harness/`` de ``.claude/PIPELINE_MAP.md``: "gerador sintetico
(espelha os 3 discentes; T parametrizavel, off por default)". Usado para
testar ``postprocess``/``assess``/``evaluate`` fim-a-fim com gabarito
sempre disponivel, e como base pronta (mas nao ligada por padrao) para um
eventual estudo de influencia do tamanho amostral ``T`` (deferido).
"""

from dataclasses import dataclass

import numpy as np

from ica.data.in_memory_template import InMemorySignalMatrixTemplate
from ica.interfaces import Domain, SignalMatrix

_VALID_KINDS = {"laplace", "uniform", "gaussian"}


def make_synthetic_sources(
    kinds: list[str], n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Gera fontes independentes i.i.d., padronizadas (media 0, variancia 1).

    Parameters
    ----------
    kinds : list of str
        Um de ``"laplace"`` (super), ``"uniform"`` (sub) ou ``"gaussian"``
        por fonte.
    n_samples : int
        Tamanho amostral ``T``.
    rng : numpy.random.Generator
        Gerador com semente fixa, para reprodutibilidade.

    Returns
    -------
    np.ndarray
        Fontes, shape ``(len(kinds), n_samples)``.

    Raises
    ------
    ValueError
        Se algum tipo em ``kinds`` nao for reconhecido.
    """
    rows = []
    for kind in kinds:
        if kind not in _VALID_KINDS:
            raise ValueError(f"Tipo de fonte desconhecido: {kind!r} (validos: {_VALID_KINDS}).")
        if kind == "laplace":
            raw = rng.laplace(size=n_samples)
        elif kind == "uniform":
            raw = rng.uniform(-1.0, 1.0, size=n_samples)
        else:
            raw = rng.normal(size=n_samples)
        rows.append((raw - raw.mean()) / raw.std())
    return np.vstack(rows)


def make_well_conditioned_mixing_matrix(
    rng: np.random.Generator, n: int, max_condition_number: float = 10.0
) -> np.ndarray:
    """Sorteia uma matriz de mistura ``A`` quadrada e bem-condicionada.

    Parameters
    ----------
    rng : numpy.random.Generator
        Gerador com semente fixa.
    n : int
        Dimensao (numero de fontes = numero de misturas).
    max_condition_number : float, default=10.0
        Numero de condicao maximo aceito -- evita misturas quase
        singulares, que inviabilizariam a separacao mesmo com o algoritmo
        correto.

    Returns
    -------
    np.ndarray
        Matriz ``A``, shape ``(n, n)``.

    Raises
    ------
    RuntimeError
        Se nenhuma matriz bem-condicionada for encontrada em 100 tentativas.
    """
    for _ in range(100):
        candidate = rng.normal(size=(n, n))
        if np.linalg.cond(candidate) < max_condition_number:
            return candidate
    raise RuntimeError("Nao foi possivel gerar uma matriz de mistura bem-condicionada.")


@dataclass
class SyntheticRun:
    """Uma mistura sintetica com fundo de verdade conhecido.

    Attributes
    ----------
    data : InMemorySignalMatrixTemplate
        Pronto para uso direto em ``ICAModel(data=..., ...)``, ja com o
        gabarito embutido (``load_ground_truth`` sempre o devolve).
    sources_true : np.ndarray
        Fontes verdadeiras, shape ``(n, n_samples)``.
    mixing_matrix_true : np.ndarray
        Matriz de mistura verdadeira ``A``, shape ``(n, n)``.
    """

    data: InMemorySignalMatrixTemplate
    sources_true: np.ndarray
    mixing_matrix_true: np.ndarray


def generate_synthetic_run(
    domain: Domain,
    kinds: list[str],
    n_samples: int,
    seed: int,
    domain_meta: dict | None = None,
) -> SyntheticRun:
    """Gera uma mistura sintetica de um dominio, com fundo de verdade conhecido.

    Parameters
    ----------
    domain : {"image", "distribution", "audio"}
        Dominio a simular (so afeta os metadados anexados ao
        :class:`~ica.interfaces.SignalMatrix`, nao a geracao em si -- o
        modelo ``x = As`` e agnostico ao tipo de dado, skill ica-ml).
    kinds : list of str
        Tipos de fonte (ver :func:`make_synthetic_sources`); ``len(kinds)``
        define ``n`` (fontes = misturas).
    n_samples : int
        Tamanho amostral ``T`` -- parametro do estudo de influencia de T
        (deferido; aqui so o hook, off por default).
    seed : int
        Semente do gerador, para reprodutibilidade.
    domain_meta : dict, optional
        Metadados extras de dominio (ex.: ``height``/``width`` para
        imagem); mesclados por cima dos padroes inferidos.

    Returns
    -------
    SyntheticRun
        Mistura + gabarito, pronta para ``ICAModel``.
    """
    rng = np.random.default_rng(seed)
    n = len(kinds)
    sources_true = make_synthetic_sources(kinds, n_samples, rng)
    mixing_matrix_true = make_well_conditioned_mixing_matrix(rng, n)
    mixtures = mixing_matrix_true @ sources_true

    meta: dict = {}
    if domain == "distribution":
        meta["sample_size"] = n_samples
    elif domain == "audio":
        meta["sample_rate"] = 8000
    elif domain == "image":
        side = int(round(np.sqrt(n_samples)))
        meta["height"] = side
        meta["width"] = side
        meta["is_rgb"] = False
        meta["n_images"] = n
    meta.update(domain_meta or {})

    signal_matrix = SignalMatrix(data=mixtures, domain=domain, meta=meta)
    data = InMemorySignalMatrixTemplate(
        signal_matrix,
        ground_truth=(mixing_matrix_true, sources_true),
        run=f"synthetic_{domain}",
    )
    return SyntheticRun(data=data, sources_true=sources_true, mixing_matrix_true=mixing_matrix_true)
