"""Interface de linha de comando: ``python -m ica --sample ... --run ...``.

Ver justfile (``just run``) e context/DEVELOPMENT_GUIDELINES.md, Secao 8.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ica.algorithms.base import ICAAlgorithm
from ica.algorithms.bell_sejnowski import BellSejnowskiICA
from ica.algorithms.fastica_ml import FastICAML
from ica.algorithms.natural_gradient import NaturalGradientICA
from ica.data.audio_template import AudioTemplate
from ica.data.base import DataTemplate
from ica.data.distribution_template import DistributionTemplate
from ica.data.image_template import ImageTemplate
from ica.metrics.convergence_iterations import ConvergenceIterations
from ica.metrics.execution_time import ExecutionTime
from ica.metrics.non_gaussianity import NonGaussianityScore
from ica.model import ICAModel
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.nonlinearities.subgaussian import SubGaussianScore
from ica.nonlinearities.supergaussian import SuperGaussianScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.whitening import Whitening
from ica.visualization.audio_visualizer import AudioVisualizer
from ica.visualization.histogram_visualizer import HistogramVisualizer
from ica.visualization.image_visualizer import ImageVisualizer
from ica.visualization.mixing_diagram_visualizer import MixingDiagramVisualizer

_TEMPLATE_FACTORIES = {
    "imagens": ImageTemplate,
    "dist": DistributionTemplate,
    "audio": AudioTemplate,
}

_ALGORITHM_FACTORIES = {
    "bell_sejnowski": BellSejnowskiICA,
    "natural_gradient": NaturalGradientICA,
    "fastica_ml": FastICAML,
}

_NONLINEARITY_FACTORIES = {
    "super": SuperGaussianScore,
    "sub": SubGaussianScore,
    "adaptive": AdaptiveScore,
}


def _build_parser() -> argparse.ArgumentParser:
    """Constroi o parser de argumentos da CLI.

    Returns
    -------
    argparse.ArgumentParser
        Parser configurado com as opcoes documentadas em
        DEVELOPMENT_GUIDELINES.md, Secao 8.
    """
    parser = argparse.ArgumentParser(
        prog="python -m ica",
        description="Separacao Cega de Fontes via ICA (Infomax / Maxima Verossimilhanca).",
    )
    parser.add_argument("--sample", choices=sorted(_TEMPLATE_FACTORIES), required=True)
    parser.add_argument("--run")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--algorithm", choices=sorted(_ALGORITHM_FACTORIES), default="fastica_ml")
    parser.add_argument(
        "--nonlinearity", choices=sorted(_NONLINEARITY_FACTORIES), default="adaptive"
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--list-runs", action="store_true")
    return parser


def _build_data_template(args: argparse.Namespace, type_root: Path) -> DataTemplate:
    """Instancia o DataTemplate correspondente a ``--sample``.

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos ja parseados.
    type_root : pathlib.Path
        Diretorio raiz das amostras deste tipo (``data_root/sample``).

    Returns
    -------
    DataTemplate
        Carregador configurado para o run pedido.
    """
    if args.sample == "dist":
        return DistributionTemplate(
            run=args.run, data_root=type_root, sample_size=args.sample_size
        )
    return _TEMPLATE_FACTORIES[args.sample](run=args.run, data_root=type_root)


def _build_algorithm(args: argparse.Namespace) -> ICAAlgorithm:
    """Instancia o ICAAlgorithm e a NonlinearityTemplate pedidos por linha de comando.

    O ``learning_rate`` so e repassado ao algoritmo quando informado
    explicitamente pelo usuario -- caso contrario, prevalece o default
    proprio de cada classe concreta de ``ICAAlgorithm`` (ex.:
    ``NaturalGradientICA`` usa um default mais conservador por
    estabilidade numerica, ver ICA_BACKGROUND.md, Secao 4.2).

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos ja parseados.

    Returns
    -------
    ICAAlgorithm
        Algoritmo configurado, com a nao-linearidade injetada.
    """
    nonlinearity = _NONLINEARITY_FACTORIES[args.nonlinearity]()
    algorithm_cls = _ALGORITHM_FACTORIES[args.algorithm]
    kwargs = {
        "nonlinearity": nonlinearity,
        "max_iterations": args.max_iterations,
        "tolerance": args.tolerance,
    }
    if args.learning_rate is not None:
        kwargs["learning_rate"] = args.learning_rate
    return algorithm_cls(**kwargs)


def _build_visualizers(sample: str, data: DataTemplate) -> list:
    """Escolhe os Visualizer apropriados para o tipo de amostra.

    Parameters
    ----------
    sample : str
        Um de ``"imagens"``, ``"dist"``, ``"audio"``.
    data : DataTemplate
        Carregador ja usado no ajuste do modelo (reutilizado por
        visualizadores que precisam reconstruir/exportar, ex.:
        ``ImageVisualizer``, ``AudioVisualizer``).

    Returns
    -------
    list of Visualizer
        Visualizadores a executar, sempre incluindo
        ``MixingDiagramVisualizer`` (Secao "Decisoes de projeto" do
        plano: informativo mesmo sem matriz de mistura verdadeira).
    """
    visualizers = [MixingDiagramVisualizer()]
    if sample == "imagens":
        visualizers.append(ImageVisualizer(data=data))
    elif sample == "dist":
        visualizers.append(HistogramVisualizer())
    elif sample == "audio":
        visualizers.append(AudioVisualizer(data=data))
    return visualizers


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da CLI: ``python -m ica --sample ... --run ...``.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos de linha de comando; por padrao, ``sys.argv[1:]``.

    Returns
    -------
    int
        Codigo de saida (``0`` em caso de sucesso).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    type_root = args.data_root / args.sample
    template_cls = _TEMPLATE_FACTORIES[args.sample]
    available_runs = template_cls.discover_runs(type_root)

    if args.list_runs:
        for run in available_runs:
            print(run)
        return 0

    if args.sample == "dist" and args.sample_size is None:
        parser.error("--sample-size e obrigatorio para --sample dist.")
    if args.sample != "dist" and args.sample_size is not None:
        parser.error("--sample-size so e valido para --sample dist.")
    if args.run is None:
        parser.error("--run e obrigatorio (use --list-runs para ver os runs disponiveis).")
    if args.run not in available_runs:
        parser.error(
            f"Run {args.run!r} nao encontrado em {type_root}. Disponiveis: {available_runs}."
        )

    data = _build_data_template(args, type_root)
    pipeline = Pipeline([Centering(), Whitening()])
    algorithm = _build_algorithm(args)

    model = ICAModel(data=data, pipeline=pipeline, algorithm=algorithm)
    model.fit()

    output_dir = args.output_dir or Path("output") / args.sample / args.run
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = model.evaluate([ConvergenceIterations(), ExecutionTime(), NonGaussianityScore()])
    serializable_metrics = {
        name: value.tolist() if hasattr(value, "tolist") else value
        for name, value in metrics.items()
    }
    (output_dir / "metrics.json").write_text(json.dumps(serializable_metrics, indent=2))
    for name, value in serializable_metrics.items():
        print(f"{name}: {value}")

    for visualizer in _build_visualizers(args.sample, data):
        visualizer.plot(model, output_dir)

    return 0
