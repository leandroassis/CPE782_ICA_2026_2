"""Teste de integracao (smoke): CLI completa (ica.cli.main) para os 3 tipos de amostra.

Usa arvores de arquivos sinteticas em ``tmp_path``, no formato real de
``data/``, nunca os dados reais do trabalho. Nao valida qualidade de
separacao (ja coberta por outros testes de integracao) -- apenas que a
CLI executa de ponta a ponta e produz os artefatos esperados.
"""

import numpy as np
import pandas as pd
from scipy.io import wavfile

from ica.cli import main


def _write_mixture_csv(path, n_mixtures, n_rows, rng):
    columns = {f"mistura{i + 1}": rng.normal(size=n_rows) for i in range(n_mixtures)}
    pd.DataFrame(columns).to_csv(path, index=False)


def test_cli_smoke_imagens(tmp_path):
    """CLI deve rodar de ponta a ponta para --sample imagens e gerar metrics.json + PNGs."""
    rng = np.random.default_rng(0)
    run_dir = tmp_path / "imagens" / "run1"
    run_dir.mkdir(parents=True)
    _write_mixture_csv(run_dir / "mix_imagens_grayscale.csv", n_mixtures=2, n_rows=16, rng=rng)

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
    assert (output_dir / "imagens_misturas_vs_fontes.png").exists()
    assert (output_dir / "diagrama_de_mistura.png").exists()


def test_cli_smoke_dist(tmp_path):
    """CLI deve rodar de ponta a ponta para --sample dist e gerar metrics.json + histogramas."""
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
    assert (output_dir / "histogramas_misturas_vs_fontes.png").exists()
    assert (output_dir / "diagrama_de_mistura.png").exists()


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
    assert (output_dir / "audio_formas_de_onda_e_espectrogramas.png").exists()
    assert (output_dir / "fonte_recuperada_1.wav").exists()
    assert (output_dir / "fonte_recuperada_2.wav").exists()
