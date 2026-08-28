"""Testes unitarios para DistributionTemplate (DEVELOPMENT_GUIDELINES.md, Secao 2.1).

Usa CSVs sinteticos em ``tmp_path``, no formato real de ``data/dist/``,
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
    """load() deve retornar X com shape (n_misturas, sample_size), transposto do CSV."""
    _write_stats_csv(tmp_path / "run1", sample_size=10, n_mixtures=3)
    template = DistributionTemplate(run="run1", data_root=tmp_path, sample_size=10)
    X = template.load()
    assert X.shape == (3, 10)


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
