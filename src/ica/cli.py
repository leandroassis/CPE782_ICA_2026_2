"""Interface de linha de comando: ``python -m ica --sample ... --run ...``.

Roda a grade completa de uma celula (``ica.harness.grid``) por
``(sample, run)`` -- os 3 algoritmos de ICA-ML x os modos de
condicionamento aplicaveis (``.claude/PIPELINE_MAP.md``) --, escolhe o
vencedor (``ica.harness.selection``, skill ica-evaluation Secao 5) e gera a
figura-vitrine dele. ``--algorithm`` continua disponivel como atalho para
depurar uma unica celula (todos os modos aplicaveis desse algoritmo), fora
da grade completa.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from ica.data.audio_template import AudioTemplate
from ica.data.base import DataTemplate
from ica.data.distribution_template import DistributionTemplate
from ica.data.image_template import ImageTemplate
from ica.harness.grid import ALGORITHMS, CellResult, conditioning_modes, run_cell, run_grid
from ica.harness.selection import select_best
from ica.visualization.showcase_visualizer import ShowcaseVisualizer

_TEMPLATE_FACTORIES: dict[str, type[DataTemplate]] = {
    "imagens": ImageTemplate,
    "dist": DistributionTemplate,
    "audio": AudioTemplate,
}


def _build_parser() -> argparse.ArgumentParser:
    """Constroi o parser de argumentos da CLI.

    Returns
    -------
    argparse.ArgumentParser
        Parser configurado.
    """
    parser = argparse.ArgumentParser(
        prog="python -m ica",
        description=(
            "Separacao Cega de Fontes via ICA-ML: roda a grade de algoritmos x "
            "modos de condicionamento sobre um run e mostra a melhor separacao."
        ),
    )
    parser.add_argument("--sample", choices=sorted(_TEMPLATE_FACTORIES), required=True)
    parser.add_argument("--run")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALGORITHMS),
        default=None,
        help="Roda so este algoritmo (todos os modos aplicaveis), fora da grade completa.",
    )
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--data-root", type=Path, default=Path("data/mix"))
    parser.add_argument(
        "--groundtruth-root",
        type=Path,
        default=None,
        help="Por padrao, o irmao 'groundtruth' de --data-root (data/mix -> data/groundtruth).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--list-runs", action="store_true")
    return parser


def _serialize(value: Any) -> Any:
    """Converte o valor de uma metrica para algo serializavel em JSON.

    Trata ``np.ndarray``/``np.floating``/``np.integer``, dataclasses
    (recursivamente, por campo), listas/tuplas/dicionarios e ``float``
    nao-finito (``inf``/``nan`` -> string).

    Parameters
    ----------
    value : Any
        Valor devolvido por ``Metric.compute`` (ou aninhado dentro dele).

    Returns
    -------
    Any
        Estrutura pronta para ``json.dumps``.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return _serialize(value.tolist())
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _serialize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, list | tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Filtra ``metrics`` para so os valores numericos escalares (para escrever na figura)."""
    return {
        name: float(value)
        for name, value in metrics.items()
        if isinstance(value, int | float | np.floating) and np.isfinite(value)
    }


def _showcase_data(sample: str, run: str, data_root: Path, winner: CellResult) -> Any:
    """Escolhe o objeto ``data`` a passar ao ``ShowcaseVisualizer`` para a celula vencedora.

    Para o modo A (unico fit, imagem/audio) reaproveita ``winner.model.data``
    (ja carregado); para os modos B/C de imagem (fits em cima de matrizes em
    memoria, sem ``reconstruct``) reconstroi um ``ImageTemplate`` real,
    apontado para o mesmo run em disco.
    """
    if sample == "imagens" and winner.mode in ("B", "C"):
        return ImageTemplate(run=run, data_root=data_root / sample)
    return winner.model.data


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
    if args.groundtruth_root is None:
        # Segue --data-root: "data/mix" -> "data/groundtruth" (irmaos). Nunca
        # cai de volta num caminho fixo, para nao ler gabarito de outro lugar
        # quando --data-root e sobrescrito (ex.: testes com tmp_path).
        args.groundtruth_root = args.data_root.parent / "groundtruth"

    sample_data_root = args.data_root / args.sample
    template_cls = _TEMPLATE_FACTORIES[args.sample]
    available_runs = template_cls.discover_runs(sample_data_root)

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
            f"Run {args.run!r} nao encontrado em {sample_data_root}. Disponiveis: {available_runs}."
        )

    groundtruth_root = args.groundtruth_root / args.sample
    algorithm_kwargs = {"max_iterations": args.max_iterations, "tolerance": args.tolerance}

    if args.algorithm is not None:
        if args.sample == "dist":
            probe_data = template_cls(
                run=args.run, data_root=sample_data_root, sample_size=args.sample_size
            )
        else:
            probe_data = template_cls(run=args.run, data_root=sample_data_root)
        modes = conditioning_modes(args.sample, probe_data)
        cells = [
            run_cell(
                args.sample,
                args.run,
                args.algorithm,
                mode,
                sample_data_root,
                groundtruth_root,
                args.sample_size,
                **algorithm_kwargs,
            )
            for mode in modes
        ]
    else:
        cells = run_grid(
            args.sample,
            args.run,
            sample_data_root,
            groundtruth_root,
            args.sample_size,
            max_workers=args.max_workers,
            **algorithm_kwargs,
        )

    winner = select_best(cells)

    output_dir = args.output_dir or Path("output") / args.sample / args.run
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {f"{cell.algorithm}/{cell.mode}": _serialize(cell.metrics) for cell in cells}
    (output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))

    winner_summary = {
        "algorithm": winner.algorithm,
        "mode": winner.mode,
        "log_likelihood_per_sample": winner.log_likelihood_per_sample,
    }
    (output_dir / "winner.json").write_text(json.dumps(winner_summary, indent=2))
    print(
        f"Vencedor: {winner.algorithm}/{winner.mode} "
        f"(log-L/amostra={winner.log_likelihood_per_sample:.4f})"
    )
    for name, value in all_metrics.items():
        print(f"{name}: {value}")

    visualizer = ShowcaseVisualizer(
        data=_showcase_data(args.sample, args.run, args.data_root, winner),
        metrics=_numeric_metrics(winner.metrics),
    )
    if winner.rgb_composites is not None:
        image_data = ImageTemplate(run=args.run, data_root=sample_data_root)
        image_data.load()  # popula height_/width_
        true_composites = None
        if groundtruth_root.exists():
            _, sources_true = image_data.load_ground_truth(groundtruth_root)
            if sources_true is not None:
                true_composites = [
                    sources_true[3 * i : 3 * i + 3] for i in range(sources_true.shape[0] // 3)
                ]
        visualizer.plot_rgb_composites(
            winner.rgb_composites,
            height=image_data.height_ or 0,
            width=image_data.width_ or 0,
            output_dir=output_dir,
            true_composites=true_composites,
        )
    else:
        visualizer.plot(winner.model, output_dir)

    return 0
