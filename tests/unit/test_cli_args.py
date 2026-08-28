"""Testes unitarios para a validacao de argumentos da CLI (ica.cli)."""

import pytest

from ica.cli import _build_parser, main


def test_parser_requires_sample():
    """--sample deve ser obrigatorio."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--run", "run1"])


def test_parser_restricts_sample_to_known_choices():
    """--sample deve aceitar apenas imagens, dist ou audio."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--sample", "invalido", "--run", "run1"])


def test_main_rejects_sample_size_for_non_distribution_sample(tmp_path):
    """--sample-size so deve ser aceito quando --sample dist."""
    image_run = tmp_path / "imagens" / "run1"
    image_run.mkdir(parents=True)
    (image_run / "mix_imagens_grayscale.csv").write_text("mistura1\n0.1\n")

    with pytest.raises(SystemExit):
        main(
            [
                "--sample",
                "imagens",
                "--run",
                "run1",
                "--sample-size",
                "100",
                "--data-root",
                str(tmp_path),
            ]
        )


def test_main_requires_sample_size_for_distribution_sample(tmp_path):
    """--sample-size deve ser obrigatorio quando --sample dist."""
    dist_run = tmp_path / "dist" / "run1"
    dist_run.mkdir(parents=True)
    (dist_run / "mix_100_stats.csv").write_text("mistura1\n0.1\n")

    with pytest.raises(SystemExit):
        main(["--sample", "dist", "--run", "run1", "--data-root", str(tmp_path)])


def test_main_rejects_unknown_run(tmp_path):
    """--run deve ser validado contra os runs descobertos em disco."""
    image_run = tmp_path / "imagens" / "run1"
    image_run.mkdir(parents=True)
    (image_run / "mix_imagens_grayscale.csv").write_text("mistura1\n0.1\n")

    with pytest.raises(SystemExit):
        main(
            [
                "--sample",
                "imagens",
                "--run",
                "run_inexistente",
                "--data-root",
                str(tmp_path),
            ]
        )


def test_main_list_runs_prints_discovered_runs(tmp_path, capsys):
    """--list-runs deve imprimir os runs descobertos e sair sem exigir --run."""
    run1 = tmp_path / "imagens" / "run1"
    run1.mkdir(parents=True)
    (run1 / "mix_imagens_grayscale.csv").write_text("mistura1\n0.1\n")
    run3 = tmp_path / "imagens" / "run3"
    run3.mkdir(parents=True)
    (run3 / "mix_imagens_rgb.csv").write_text("mistura1\n0.1\n")

    exit_code = main(["--sample", "imagens", "--list-runs", "--data-root", str(tmp_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "run1" in output
    assert "run3" in output
