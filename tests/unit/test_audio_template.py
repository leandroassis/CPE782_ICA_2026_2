"""Testes unitarios para AudioTemplate.

Usa arquivos ``.wav`` sinteticos em ``tmp_path``, no formato real de
``data/mix/audio/``, nunca os dados reais do trabalho.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from ica.data.audio_template import AudioTemplate


def _write_wav(path, n_samples=100, sample_rate=44100, amplitude=1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 1, n_samples, endpoint=False)
    signal = (amplitude * np.sin(2 * np.pi * 5 * t)).astype(np.int16)
    wavfile.write(path, sample_rate, signal)


def test_load_returns_one_row_per_mixture_file(tmp_path):
    """load() deve retornar um SignalMatrix com uma linha por mixture_*.wav, ordem numerica."""
    run_dir = tmp_path / "run1"
    _write_wav(run_dir / "mixture_1.wav", n_samples=200)
    _write_wav(run_dir / "mixture_2.wav", n_samples=200)

    template = AudioTemplate(run="run1", data_root=tmp_path)
    signal_matrix = template.load()

    assert signal_matrix.data.shape == (2, 200)
    assert signal_matrix.domain == "audio"


def test_n_mixtures_reflects_two_vs_three_files(tmp_path):
    """n_mixtures deve refletir 2 ou 3 conforme os arquivos presentes (ver run2 de audio)."""
    run1 = tmp_path / "run1"
    _write_wav(run1 / "mixture_1.wav")
    _write_wav(run1 / "mixture_2.wav")

    run2 = tmp_path / "run2"
    _write_wav(run2 / "mixture_1.wav")
    _write_wav(run2 / "mixture_2.wav")
    _write_wav(run2 / "mixture_3.wav")

    assert AudioTemplate(run="run1", data_root=tmp_path).n_mixtures == 2
    assert AudioTemplate(run="run2", data_root=tmp_path).n_mixtures == 3


def test_load_normalizes_to_unit_range(tmp_path):
    """Os valores carregados devem estar normalizados em [-1, 1]."""
    run_dir = tmp_path / "run1"
    _write_wav(run_dir / "mixture_1.wav", amplitude=32000)

    template = AudioTemplate(run="run1", data_root=tmp_path)
    signal_matrix = template.load()

    assert signal_matrix.data.max() <= 1.0
    assert signal_matrix.data.min() >= -1.0


def test_sample_rate_is_preserved(tmp_path):
    """sample_rate_ deve refletir a taxa de amostragem lida do arquivo .wav."""
    run_dir = tmp_path / "run1"
    _write_wav(run_dir / "mixture_1.wav", sample_rate=22050)

    template = AudioTemplate(run="run1", data_root=tmp_path)
    template.load()

    assert template.sample_rate_ == 22050


def test_export_round_trips_signal(tmp_path):
    """export() seguido de leitura direta deve preservar aproximadamente o sinal exportado."""
    run_dir = tmp_path / "run1"
    _write_wav(run_dir / "mixture_1.wav", n_samples=500, sample_rate=44100)

    template = AudioTemplate(run="run1", data_root=tmp_path)
    template.load()

    original_signal = np.sin(2 * np.pi * 3 * np.linspace(0, 1, 500, endpoint=False))
    output_path = tmp_path / "recovered.wav"
    template.export(original_signal, output_path)

    sample_rate, exported = wavfile.read(output_path)
    exported_normalized = exported.astype(np.float64) / np.iinfo(np.int16).max

    assert sample_rate == 44100
    assert np.allclose(exported_normalized, original_signal, atol=1e-3)


def test_export_raises_before_load(tmp_path):
    """export() deve levantar RuntimeError se chamado antes de load() (sample_rate_ ausente)."""
    template = AudioTemplate(run="run1", data_root=tmp_path)
    with pytest.raises(RuntimeError):
        template.export(np.zeros(10), tmp_path / "out.wav")


def test_load_raises_on_inconsistent_sample_lengths(tmp_path):
    """load() deve levantar ValueError se as misturas do run tiverem tamanhos diferentes."""
    run_dir = tmp_path / "run1"
    _write_wav(run_dir / "mixture_1.wav", n_samples=100)
    _write_wav(run_dir / "mixture_2.wav", n_samples=200)

    template = AudioTemplate(run="run1", data_root=tmp_path)
    with pytest.raises(ValueError):
        template.load()


def test_discover_runs_finds_directories_with_mixture_files(tmp_path):
    """discover_runs deve encontrar apenas subdiretorios com arquivo mixture_*.wav."""
    _write_wav(tmp_path / "run1" / "mixture_1.wav")
    (tmp_path / "not_a_run").mkdir()

    assert AudioTemplate.discover_runs(tmp_path) == ["run1"]


def test_load_ground_truth_returns_none_when_run_missing(tmp_path):
    """load_ground_truth deve tolerar a ausencia do diretorio de gabarito."""
    template = AudioTemplate(run="run1", data_root=tmp_path)
    mixing_matrix_true, sources_true = template.load_ground_truth(tmp_path / "does_not_exist")
    assert mixing_matrix_true is None
    assert sources_true is None


def test_load_ground_truth_reads_mix_matrix_and_sources(tmp_path):
    """load_ground_truth deve ler mix_matrix*.csv e source_*.wav, normalizando as fontes."""
    groundtruth_run_dir = tmp_path / "groundtruth" / "run1"
    groundtruth_run_dir.mkdir(parents=True)
    pd.DataFrame({"A1": [1.0, 0.0], "A2": [0.0, 1.0]}).to_csv(
        groundtruth_run_dir / "mix_matrix.csv", index=False
    )
    _write_wav(groundtruth_run_dir / "source_1.wav", n_samples=100, amplitude=16000)
    _write_wav(groundtruth_run_dir / "source_2.wav", n_samples=100, amplitude=16000)

    template = AudioTemplate(run="run1", data_root=tmp_path / "mix")
    mixing_matrix_true, sources_true = template.load_ground_truth(tmp_path / "groundtruth")

    assert mixing_matrix_true.shape == (2, 2)
    assert sources_true.shape == (2, 100)
    assert sources_true.max() <= 1.0 and sources_true.min() >= -1.0
