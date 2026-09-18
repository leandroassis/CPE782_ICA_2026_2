"""Grade de execucao: celula = (dominio, run, algoritmo, modo de condicionamento).

Skill ``ica-ml`` (os 3 algoritmos) + ``.claude/PIPELINE_MAP.md`` ("Grade de
execucao"): unidade de paralelismo = a celula (embarrassingly parallel via
``ProcessPoolExecutor`` -- stdlib, sem depender de ``joblib``). Poda do eixo
de condicionamento: imagem-grayscale -> ``{A}``; imagem-cor -> ``{B, C}``;
distribuicao/audio -> modo unico. Cada fit consome as T amostras inteiras
do run -- nunca fatiar por amostra (quebraria a estimativa global de ML).
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ica.algorithms.base import ICAAlgorithm
from ica.algorithms.bell_sejnowski import BellSejnowskiICA
from ica.algorithms.fastica_ml import FastICAML
from ica.algorithms.natural_gradient import NaturalGradientICA
from ica.data.audio_template import AudioTemplate
from ica.data.base import DataTemplate
from ica.data.distribution_template import DistributionTemplate
from ica.data.image_template import ImageTemplate
from ica.data.in_memory_template import InMemorySignalMatrixTemplate
from ica.interfaces import SignalMatrix
from ica.metrics.amari import AmariIndex
from ica.metrics.convergence_iterations import ConvergenceIterations
from ica.metrics.distribution_metrics import DistributionFitMetric
from ica.metrics.execution_time import ExecutionTime
from ica.metrics.family_identification import FamilyIdentificationMetric
from ica.metrics.image_metrics import PSNRMetric, SSIMMetric, psnr, ssim
from ica.metrics.log_likelihood import LogLikelihood
from ica.metrics.non_gaussianity import NonGaussianityScore
from ica.metrics.sir_sdr import SIRSDRMetric
from ica.model import ICAModel
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.postprocessing.regroup import regroup_rgb_planes
from ica.preprocessing.centering import Centering
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.robust_scaling import RobustScaling
from ica.preprocessing.whitening import Whitening

ALGORITHMS: dict[str, type[ICAAlgorithm]] = {
    "natural_gradient": NaturalGradientICA,
    "bell_sejnowski": BellSejnowskiICA,
    "fastica_ml": FastICAML,
}

_SAMPLE_TEMPLATES: dict[str, type[DataTemplate]] = {
    "imagens": ImageTemplate,
    "dist": DistributionTemplate,
    "audio": AudioTemplate,
}


def standard_pipeline() -> Pipeline:
    """Pipeline padrao da grade: ``Centering -> RobustScaling -> Whitening``.

    ``RobustScaling(clip_mads=8.0)`` entra sempre (nao so-off-by-default):
    e aproximadamente no-op em dados bem-condicionados e evita que runs com
    outliers extremos (ex.: ``data/mix/dist/run8``, covariancia quase
    singular) quebrem o branqueamento. ``PCA`` continua fora do padrao,
    exposta como gancho opcional (``PIPELINE_MAP.md`` pede isso
    especificamente so para ela).

    Returns
    -------
    Pipeline
        Pipeline pronto para uso em qualquer celula.
    """
    return Pipeline([Centering(), RobustScaling(), Whitening()])


def standard_metrics() -> list:
    """Metricas padrao aplicadas a toda celula (cego + validacao, auto-desabilitada sem gabarito).

    Returns
    -------
    list of Metric
        Instancias novas a cada chamada (metricas sao stateless entre
        ``compute()``, mas evita compartilhar instancia entre processos).
    """
    return [
        ConvergenceIterations(),
        ExecutionTime(),
        NonGaussianityScore(),
        LogLikelihood(),
        AmariIndex(),
        SIRSDRMetric(),
        PSNRMetric(),
        SSIMMetric(),
        DistributionFitMetric(),
        FamilyIdentificationMetric(),
    ]


def build_algorithm(name: str, **kwargs: Any) -> ICAAlgorithm:
    """Instancia um ``ICAAlgorithm`` pelo nome, com chaveamento adaptativo (skill ica-ml).

    Parameters
    ----------
    name : {"natural_gradient", "bell_sejnowski", "fastica_ml"}
        Algoritmo desejado.
    **kwargs
        Repassados ao construtor (ex.: ``max_iterations``, ``tolerance``).

    Returns
    -------
    ICAAlgorithm
        Algoritmo configurado com :class:`~ica.nonlinearities.adaptive.AdaptiveScore`.

    Raises
    ------
    KeyError
        Se ``name`` nao for um algoritmo conhecido.
    """
    return ALGORITHMS[name](nonlinearity=AdaptiveScore(), **kwargs)


def conditioning_modes(sample: str, data: DataTemplate) -> list[str]:
    """Poda do eixo de condicionamento (``.claude/PIPELINE_MAP.md``).

    Parameters
    ----------
    sample : {"imagens", "dist", "audio"}
        Tipo de amostra.
    data : DataTemplate
        Instancia ja construida (para inspecionar ``is_rgb_`` em imagens).

    Returns
    -------
    list of str
        ``["A"]`` (grayscale, B==C), ``["B", "C"]`` (cor), ou
        ``["unico"]`` (distribuicao/audio).
    """
    if sample != "imagens":
        return ["unico"]
    return ["B", "C"] if getattr(data, "is_rgb_", False) else ["A"]


@dataclass
class CellResult:
    """Resultado de uma celula da grade (um algoritmo, um modo de condicionamento).

    Attributes
    ----------
    sample, run, algorithm, mode : str
        Identificacao da celula.
    models : list of ICAModel
        Um ``ICAModel`` ajustado -- ou, no modo B de imagem, 3 (um por
        plano R/G/B, ja reagrupados em :attr:`rgb_composites`).
    metrics : dict
        Metricas agregadas da celula (ver :func:`run_cell`).
    rgb_composites : list of np.ndarray or None
        So no modo B: as ``k`` trincas RGB reagrupadas
        (:func:`~ica.postprocessing.regroup.regroup_rgb_planes`), shape
        ``(3, n_pixels)`` cada.
    """

    sample: str
    run: str
    algorithm: str
    mode: str
    models: list[ICAModel]
    metrics: dict[str, Any] = field(default_factory=dict)
    rgb_composites: list[np.ndarray] | None = None

    @property
    def model(self) -> ICAModel:
        """Atalho para ``models[0]`` -- valido para os modos A/C (um unico fit)."""
        return self.models[0]

    @property
    def log_likelihood_per_sample(self) -> float:
        """Media do log-L/amostra final de cada modelo da celula.

        Para A/C e a log-verossimilhanca do unico fit; para B (3 fits
        independentes por plano) e a **media** dos 3 -- uma aproximacao
        pratica para comparar entre modos (a base de normalizacao de T
        difere entre "3 problemas n x P" e "1 problema n x 3P"; ver
        ``.claude/PIPELINE_MAP.md``, secao de condicionamento).
        """
        return float(np.mean([m.log_likelihood_history_[-1] for m in self.models]))


def _build_data_template(
    sample: str, run: str, data_root: Path, sample_size: int | None
) -> DataTemplate:
    """Instancia o ``DataTemplate`` correto para ``sample``."""
    if sample == "dist":
        return DistributionTemplate(run=run, data_root=data_root, sample_size=sample_size)
    return _SAMPLE_TEMPLATES[sample](run=run, data_root=data_root)


def _fit_one(
    data: DataTemplate, groundtruth_root: Path | None, algorithm_name: str, **algorithm_kwargs: Any
) -> ICAModel:
    """Constroi o pipeline padrao, ajusta um ``ICAModel`` e devolve-o (nao avaliado ainda)."""
    algorithm = build_algorithm(algorithm_name, **algorithm_kwargs)
    model = ICAModel(
        data=data,
        pipeline=standard_pipeline(),
        algorithm=algorithm,
        groundtruth_root=groundtruth_root,
    )
    model.fit()
    return model


def run_cell(
    sample: str,
    run: str,
    algorithm: str,
    mode: str,
    data_root: Path,
    groundtruth_root: Path | None = None,
    sample_size: int | None = None,
    **algorithm_kwargs: Any,
) -> CellResult:
    """Roda uma unica celula ``(sample, run, algorithm, mode)`` de ponta a ponta.

    Modo A/C (grayscale, ou RGB achatado em 3P): um unico ``ICAModel``.
    Modo B (RGB por plano): 3 ``ICAModel`` independentes (um por plano),
    reagrupados por :func:`~ica.postprocessing.regroup.regroup_rgb_planes`;
    metricas de validacao (PSNR/SSIM) sao recalculadas sobre os compostos
    RGB reagrupados, nao por plano isolado.

    Parameters
    ----------
    sample : {"imagens", "dist", "audio"}
        Tipo de amostra.
    run : str
        Identificador do run.
    algorithm : {"natural_gradient", "bell_sejnowski", "fastica_ml"}
        Algoritmo de ICA-ML.
    mode : {"A", "B", "C", "unico"}
        Modo de condicionamento (ver :func:`conditioning_modes`).
    data_root : pathlib.Path
        Raiz das misturas (``data/mix/<sample>``).
    groundtruth_root : pathlib.Path, optional
        Raiz do gabarito (``data/groundtruth/<sample>``).
    sample_size : int, optional
        Obrigatorio para ``sample="dist"``.
    **algorithm_kwargs
        Repassados ao construtor do algoritmo (ex.: ``max_iterations``).

    Returns
    -------
    CellResult
        Resultado completo da celula.
    """
    if sample == "imagens" and mode in ("B", "C"):
        return _run_image_conditioned_cell(
            run, algorithm, mode, data_root, groundtruth_root, **algorithm_kwargs
        )

    data = _build_data_template(sample, run, data_root, sample_size)
    model = _fit_one(data, groundtruth_root, algorithm, **algorithm_kwargs)
    metrics = model.evaluate(standard_metrics())
    return CellResult(
        sample=sample, run=run, algorithm=algorithm, mode=mode, models=[model], metrics=metrics
    )


def _group_true_rgb_triplets(sources_true: np.ndarray) -> list[np.ndarray]:
    """Agrupa as 9 linhas (1 canal cada) do gabarito de imagem RGB em 3 trincas.

    ``sources_true`` vem sempre no formato nativo do gabarito -- 9 linhas
    de 1 canal cada, agrupadas consecutivamente por imagem (``Cachorro_R,
    Cachorro_G, Cachorro_B, Gato_R, ...``, confirmado contra o cabecalho
    real de ``data/groundtruth/imagens/run3/sources_imagens_rgb.csv``) --
    independentemente do modo de condicionamento usado para separar.
    """
    return [sources_true[3 * i : 3 * i + 3] for i in range(sources_true.shape[0] // 3)]


def _run_image_conditioned_cell(
    run: str,
    algorithm: str,
    mode: str,
    data_root: Path,
    groundtruth_root: Path | None,
    **algorithm_kwargs: Any,
) -> CellResult:
    """Roda a celula de imagem RGB nos modos B (por plano) ou C (achatado em 3P)."""
    image_data = ImageTemplate(run=run, data_root=data_root)
    mixing_matrix_true, sources_true = (
        image_data.load_ground_truth(groundtruth_root) if groundtruth_root else (None, None)
    )
    height, width = image_data.height_ or 0, image_data.width_ or 0
    if height == 0:
        image_data.load()
        height, width = image_data.height_, image_data.width_

    if mode == "C":
        matrix = image_data.mode_c_matrix()
        template = InMemorySignalMatrixTemplate(
            SignalMatrix(data=matrix, domain="image", meta={"height": height, "width": width}),
            ground_truth=(None, None),
        )
        model = _fit_one(template, None, algorithm, **algorithm_kwargs)
        metrics = model.evaluate(
            [ConvergenceIterations(), ExecutionTime(), NonGaussianityScore(), LogLikelihood()]
        )
        # cada linha de model.sources_ e uma imagem inteira com R,G,B ja
        # concatenados (convencao de mode_c_matrix()); sources_true, porem,
        # vem sempre como 9 linhas de 1 canal cada (formato nativo do
        # gabarito, agrupado 3 a 3 por _group_true_rgb_triplets).
        rgb_composites = [row.reshape(3, height * width) for row in model.sources_]
        if sources_true is not None:
            true_composites = _group_true_rgb_triplets(sources_true)
            metrics.update(_rgb_composite_metrics(true_composites, rgb_composites, height, width))
        return CellResult(
            sample="imagens",
            run=run,
            algorithm=algorithm,
            mode=mode,
            models=[model],
            metrics=metrics,
            rgb_composites=rgb_composites,
        )

    # modo B: 3 fits independentes, um por plano
    planes = image_data.mode_b_plane_matrices()
    models = []
    for plane_matrix in planes:
        template = InMemorySignalMatrixTemplate(
            SignalMatrix(data=plane_matrix, domain="image", meta={"height": height, "width": width})
        )
        models.append(_fit_one(template, None, algorithm, **algorithm_kwargs))

    rgb_composites = regroup_rgb_planes([m.sources_ for m in models])
    metrics: dict[str, Any] = {
        "convergence_iterations": [m.n_iterations_ for m in models],
        "execution_time_seconds": sum(m.elapsed_time_ for m in models),
        "log_likelihood": float(np.mean([m.log_likelihood_history_[-1] for m in models])),
    }
    if sources_true is not None:
        true_composites = _group_true_rgb_triplets(sources_true)
        metrics.update(_rgb_composite_metrics(true_composites, rgb_composites, height, width))

    return CellResult(
        sample="imagens",
        run=run,
        algorithm=algorithm,
        mode=mode,
        models=models,
        metrics=metrics,
        rgb_composites=rgb_composites,
    )


def _rgb_composite_metrics(
    true_composites: list[np.ndarray],
    estimated_composites: list[np.ndarray],
    height: int,
    width: int,
) -> dict[str, Any]:
    """PSNR/SSIM medio (sobre os 3 canais) entre compostos RGB verdadeiros e recuperados.

    Casa os compostos por correlacao total (achatando os 3 canais) via
    casamento hungaro, reaproveitando a mesma logica de
    :mod:`ica.postprocessing.matching`.
    """
    from ica.postprocessing.matching import hungarian_match

    true_flat = np.array([c.reshape(-1) for c in true_composites])
    estimated_flat = np.array([c.reshape(-1) for c in estimated_composites])
    match = hungarian_match(true_flat, estimated_flat)
    order = np.argsort(match.reference_indices)

    psnr_values, ssim_values = [], []
    for i in order:
        true_composite = true_composites[match.reference_indices[i]]
        estimated_composite = estimated_composites[match.matched_indices[i]]
        per_channel_psnr, per_channel_ssim = [], []
        for channel in range(3):
            true_image = true_composite[channel].reshape(height, width)
            true_image = (true_image - true_image.min()) / max(
                true_image.max() - true_image.min(), 1e-12
            )
            estimated_image = estimated_composite[channel].reshape(height, width)
            per_channel_psnr.append(psnr(true_image, estimated_image))
            per_channel_ssim.append(ssim(true_image, estimated_image))
        psnr_values.append(float(np.mean(per_channel_psnr)))
        ssim_values.append(float(np.mean(per_channel_ssim)))
    return {"psnr_db_per_source": np.array(psnr_values), "ssim_per_source": np.array(ssim_values)}


def run_grid(
    sample: str,
    run: str,
    data_root: Path,
    groundtruth_root: Path | None = None,
    sample_size: int | None = None,
    algorithms: list[str] | None = None,
    max_workers: int | None = None,
    **algorithm_kwargs: Any,
) -> list[CellResult]:
    """Roda a grade completa (todos os algoritmos x modos aplicaveis) de um run.

    Paraleliza as celulas via ``ProcessPoolExecutor`` (embarrassingly
    parallel -- ``.claude/PIPELINE_MAP.md``).

    Parameters
    ----------
    sample : {"imagens", "dist", "audio"}
        Tipo de amostra.
    run : str
        Identificador do run.
    data_root : pathlib.Path
        Raiz das misturas.
    groundtruth_root : pathlib.Path, optional
        Raiz do gabarito.
    sample_size : int, optional
        Obrigatorio para ``sample="dist"``.
    algorithms : list of str, optional
        Por padrao, os 3 (``ALGORITHMS``).
    max_workers : int, optional
        Repassado a ``ProcessPoolExecutor`` (``None`` = padrao do SO).
    **algorithm_kwargs
        Repassados a cada :func:`run_cell`.

    Returns
    -------
    list of CellResult
        Uma entrada por ``(algoritmo, modo)``.
    """
    algorithms = algorithms or list(ALGORITHMS)
    probe_data = _build_data_template(sample, run, data_root, sample_size)
    modes = conditioning_modes(sample, probe_data)

    jobs = [(algorithm, mode) for algorithm in algorithms for mode in modes]
    # "spawn" evita o aviso/risco de deadlock do fork() de processo com
    # multiplas threads (numpy/BLAS, matplotlib) -- mais lento para
    # inicializar que "fork", mas seguro por padrao.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        futures = [
            executor.submit(
                run_cell,
                sample,
                run,
                algorithm,
                mode,
                data_root,
                groundtruth_root,
                sample_size,
                **algorithm_kwargs,
            )
            for algorithm, mode in jobs
        ]
        return [future.result() for future in futures]
