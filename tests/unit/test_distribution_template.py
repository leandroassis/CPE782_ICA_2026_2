"""Testes unitarios para DistributionTemplate.

Usa CSVs sinteticos em ``tmp_path``, no formato real de ``data/mix/dist/``,
nunca os dados reais do trabalho.
"""

import numpy as np
import pandas as pd
import pytest

from ica.data.distribution_template import DistributionTemplate


def _write_stats_csv(run_dir, sample_size, n_mixtures=3):
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {f"mistura{i + 1}": np.arange(sample_size, dtype=float) + i for i in range(n_mixtures)}
    pd.DataFrame(data).to_csv(run_dir / f"mix_{sample_size}_stats.csv", index=False)


def test_load_transposes_rows_to_columns(tmp_path):
    """load() deve retornar um SignalMatrix com data shape (n_misturas, sample_size)."""
    _write_stats_csv(tmp_path / "run1", sample_size=10, n_mixtures=3)
    template = DistributionTemplate(run="run1", data_root=tmp_path, sample_size=10)
    signal_matrix = template.load()
    assert signal_matrix.data.shape == (3, 10)
    assert signal_matrix.domain == "distribution"
    assert signal_matrix.meta["sample_size"] == 10


def test_load_raises_file_not_found_for_missing_sample_size(tmp_path):
    """load() deve levantar FileNotFoundError se o tamanho amostral pedido nao existir."""
    _write_stats_csv(tmp_path / "run1", sample_size=100)
    template = DistributionTemplate(run="run1", data_root=tmp_path, sample_size=999)
    with pytest.raises(FileNotFoundError):
        template.load()


def test_n_mixtures_reflects_column_count(tmp_path):
    """n_mixtures deve refletir o numero de colunas misturaN do tamanho amostral configurado."""
    _write_stats_csv(tmp_path / "run4", sample_size=100, n_mixtures=5)
    template = DistributionTemplate(run="run4", data_root=tmp_path, sample_size=100)
    assert template.n_mixtures == 5


def test_discover_runs_finds_directories_with_stats_csv(tmp_path):
    """discover_runs deve encontrar apenas subdiretorios com CSV mix_*_stats.csv."""
    _write_stats_csv(tmp_path / "run1", sample_size=100)
    _write_stats_csv(tmp_path / "run2", sample_size=1000)
    (tmp_path / "not_a_run").mkdir()

    assert DistributionTemplate.discover_runs(tmp_path) == ["run1", "run2"]


def test_discover_sample_sizes_parses_filenames(tmp_path):
    """discover_sample_sizes deve extrair os tamanhos amostrais dos nomes dos arquivos."""
    run_dir = tmp_path / "run1"
    _write_stats_csv(run_dir, sample_size=100)
    _write_stats_csv(run_dir, sample_size=1000)
    _write_stats_csv(run_dir, sample_size=100000)

    sizes = DistributionTemplate.discover_sample_sizes(tmp_path, "run1")

    assert sizes == [100, 1000, 100000]


def test_discover_sample_sizes_returns_empty_list_for_missing_run(tmp_path):
    """discover_sample_sizes nao deve levantar excecao se o run nao existir."""
    assert DistributionTemplate.discover_sample_sizes(tmp_path, "does_not_exist") == []


def test_load_ground_truth_returns_none_when_run_missing(tmp_path):
    """load_ground_truth deve tolerar a ausencia do diretorio de gabarito."""
    template = DistributionTemplate(run="run1", data_root=tmp_path, sample_size=100)
    mixing_matrix_true, sources_true = template.load_ground_truth(tmp_path / "does_not_exist")
    assert mixing_matrix_true is None
    assert sources_true is None


def test_load_ground_truth_tolerates_missing_mixing_matrix(tmp_path):
    """Alguns runs (ex.: dist/run6) so tem fontes verdadeiras, sem a matriz A."""
    groundtruth_run_dir = tmp_path / "groundtruth" / "run6"
    groundtruth_run_dir.mkdir(parents=True)
    pd.DataFrame({"Uniform": np.zeros(100), "Gauss": np.zeros(100)}).to_csv(
        groundtruth_run_dir / "sources_100_stats.csv", index=False
    )

    template = DistributionTemplate(run="run6", data_root=tmp_path / "mix", sample_size=100)
    mixing_matrix_true, sources_true = template.load_ground_truth(tmp_path / "groundtruth")

    assert mixing_matrix_true is None
    assert sources_true.shape == (2, 100)


def test_load_ground_truth_reads_mix_matrix_and_matching_sample_size(tmp_path):
    """load_ground_truth deve achar mix_matrix_run{k}.csv por glob e casar o sample_size."""
    groundtruth_run_dir = tmp_path / "groundtruth" / "run1"
    groundtruth_run_dir.mkdir(parents=True)
    pd.DataFrame({"A1": [1.0, 0.0], "A2": [0.0, 1.0]}).to_csv(
        groundtruth_run_dir / "mix_matrix_run1.csv", index=False
    )
    pd.DataFrame({"Uniform": np.zeros(100), "Gauss": np.zeros(100)}).to_csv(
        groundtruth_run_dir / "sources_100_stats.csv", index=False
    )

    template = DistributionTemplate(run="run1", data_root=tmp_path / "mix", sample_size=100)
    mixing_matrix_true, sources_true = template.load_ground_truth(tmp_path / "groundtruth")

    assert mixing_matrix_true.shape == (2, 2)
    assert sources_true.shape == (2, 100)
