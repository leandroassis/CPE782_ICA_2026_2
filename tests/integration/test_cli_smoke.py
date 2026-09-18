"""Teste de integracao (smoke): CLI completa (ica.cli.main) para os 3 tipos de amostra.

Usa arvores de arquivos sinteticas em ``tmp_path``, no formato real de
``data/mix/``+``data/groundtruth/``, nunca os dados reais do trabalho. Nao
valida qualidade de separacao (ja coberta por outros testes de integracao)
-- apenas que a CLI executa de ponta a ponta e produz os artefatos da
figura-vitrine (skill ica-evaluation, Secao 6).
"""

import json

import numpy as np
import pandas as pd
from scipy.io import wavfile

from ica.cli import main


def _write_mixture_csv(path, n_mixtures, n_rows, rng):
    columns = {f"mistura{i + 1}": rng.normal(size=n_rows) for i in range(n_mixtures)}
    pd.DataFrame(columns).to_csv(path, index=False)


def test_cli_smoke_imagens(tmp_path):
    """CLI deve rodar de ponta a ponta para --sample imagens e gerar metrics.json + a vitrine."""
    rng = np.random.default_rng(0)
    run_dir = tmp_path / "imagens" / "run1"
    run_dir.mkdir(parents=True)
    _write_mixture_csv(run_dir / "mix_imagens_grayscale.csv", n_mixtures=3, n_rows=16, rng=rng)

    output_dir = tmp_path / "output"
    exit_code = main(
        [
            "--sample",
            "imagens",
            "--run",
            "run1",
            "--algorithm",
            "fastica_ml",
            "--max-iterations",
            "50",
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "winner.json").exists()
    assert (output_dir / "vitrine_imagens.png").exists()
    winner = json.loads((output_dir / "winner.json").read_text())
    assert winner["algorithm"] == "fastica_ml"
    assert winner["mode"] == "A"


def test_cli_smoke_dist(tmp_path):
    """CLI deve rodar de ponta a ponta para --sample dist e gerar metrics.json + a vitrine."""
    rng = np.random.default_rng(0)
    run_dir = tmp_path / "dist" / "run1"
    run_dir.mkdir(parents=True)
    _write_mixture_csv(run_dir / "mix_100_stats.csv", n_mixtures=3, n_rows=100, rng=rng)

    output_dir = tmp_path / "output"
    exit_code = main(
        [
            "--sample",
            "dist",
            "--run",
            "run1",
            "--sample-size",
            "100",
            "--algorithm",
            "fastica_ml",
            "--max-iterations",
            "50",
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "vitrine_distribuicoes.png").exists()


def test_cli_smoke_audio(tmp_path):
    """CLI deve rodar de ponta a ponta para --sample audio e exportar os .wav recuperados."""
    rng = np.random.default_rng(0)
    run_dir = tmp_path / "audio" / "run1"
    run_dir.mkdir(parents=True)
    for i in range(2):
        pcm = (rng.normal(size=500) * 3000).astype(np.int16)
        wavfile.write(run_dir / f"mixture_{i + 1}.wav", 8000, pcm)

    output_dir = tmp_path / "output"
    exit_code = main(
        [
            "--sample",
            "audio",
            "--run",
            "run1",
            "--algorithm",
            "fastica_ml",
            "--max-iterations",
            "50",
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "vitrine_audio_espectros.png").exists()
    assert (output_dir / "fonte_separada_1.wav").exists()
    assert (output_dir / "fonte_separada_2.wav").exists()


def test_cli_smoke_full_grid_selects_a_winner(tmp_path):
    """Sem --algorithm, a CLI deve rodar a grade completa (3 algoritmos) e escolher um vencedor."""
    rng = np.random.default_rng(0)
    run_dir = tmp_path / "dist" / "run1"
    run_dir.mkdir(parents=True)
    _write_mixture_csv(run_dir / "mix_100_stats.csv", n_mixtures=2, n_rows=200, rng=rng)

    output_dir = tmp_path / "output"
    exit_code = main(
        [
            "--sample",
            "dist",
            "--run",
            "run1",
            "--sample-size",
            "100",
            "--max-iterations",
            "50",
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--max-workers",
            "2",
        ]
    )

    assert exit_code == 0
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert set(metrics) == {"natural_gradient/unico", "bell_sejnowski/unico", "fastica_ml/unico"}
