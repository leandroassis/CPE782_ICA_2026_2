"""Testes unitarios para os Visualizer (DEVELOPMENT_GUIDELINES.md, Secao 2.6).

Usa fixtures sinteticas minimas; nao valida o conteudo visual das
figuras, apenas que os arquivos esperados sao efetivamente escritos.
"""

from pathlib import Path

import numpy as np
import pytest

from ica.visualization.audio_visualizer import AudioVisualizer
from ica.visualization.histogram_visualizer import HistogramVisualizer
from ica.visualization.image_visualizer import ImageVisualizer
from ica.visualization.log_likelihood_visualizer import LogLikelihoodVisualizer
from ica.visualization.mixing_diagram_3d_visualizer import MixingDiagram3DVisualizer
from ica.visualization.mixing_diagram_visualizer import MixingDiagramVisualizer


class _FakeModel:
    """Duble minimo de ICAModel, expondo mixtures_/sources_/log_likelihood_history_."""

    def __init__(
        self,
        mixtures: np.ndarray,
        sources: np.ndarray,
        log_likelihood_history: list[float] | None = None,
    ) -> None:
        self.mixtures_ = mixtures
        self.sources_ = sources
        self.log_likelihood_history_ = log_likelihood_history or []


class _FakeImageData:
    """Duble minimo de ImageTemplate: reconstroi um vetor plano em (H, W)."""

    def __init__(self, height: int, width: int, is_rgb: bool = False) -> None:
        self.height_ = height
        self.width_ = width
        self.is_rgb_ = is_rgb

    def reconstruct(self, vector: np.ndarray) -> np.ndarray:
        return vector.reshape(self.height_, self.width_)

    def reconstruct_rgb_triplet(self, vectors: list[np.ndarray]) -> np.ndarray:
        channels = [self.reconstruct(v) for v in vectors]
        return np.stack(channels, axis=-1)


class _FakeAudioData:
    """Duble minimo de AudioTemplate: registra as chamadas a export."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate_ = sample_rate
        self.export_calls: list[tuple[np.ndarray, Path]] = []

    def export(self, signal: np.ndarray, output_path: Path) -> None:
        self.export_calls.append((signal, output_path))
        Path(output_path).write_bytes(b"RIFF....WAVEfake")


def test_image_visualizer_writes_grid_png(tmp_path):
    """ImageVisualizer.plot deve escrever o PNG do grid misturas vs fontes."""
    rng = np.random.default_rng(0)
    mixtures = rng.normal(size=(3, 16))
    sources = rng.normal(size=(3, 16))
    model = _FakeModel(mixtures, sources)
    data = _FakeImageData(height=4, width=4)

    written = ImageVisualizer(data=data).plot(model, tmp_path)

    assert len(written) == 1
    assert written[0].exists()


def test_image_visualizer_writes_rgb_composite_for_rgb_sample(tmp_path):
    """Para amostras RGB (9 misturas), ImageVisualizer deve tambem salvar o composto colorido."""
    rng = np.random.default_rng(0)
    mixtures = rng.normal(size=(9, 16))
    sources = rng.normal(size=(9, 16))
    model = _FakeModel(mixtures, sources)
    data = _FakeImageData(height=4, width=4, is_rgb=True)

    written = ImageVisualizer(data=data).plot(model, tmp_path)

    assert len(written) == 2
    assert all(path.exists() for path in written)


def test_histogram_visualizer_writes_png(tmp_path):
    """HistogramVisualizer.plot deve escrever o PNG com os histogramas."""
    rng = np.random.default_rng(0)
    mixtures = rng.normal(size=(3, 500))
    sources = rng.laplace(size=(3, 500))
    model = _FakeModel(mixtures, sources)

    written = HistogramVisualizer().plot(model, tmp_path)

    assert len(written) == 1
    assert written[0].exists()


def test_audio_visualizer_writes_png_and_wav_per_source(tmp_path):
    """AudioVisualizer.plot deve escrever o PNG e um .wav por fonte, via data.export."""
    rng = np.random.default_rng(0)
    mixtures = rng.normal(size=(2, 1000))
    sources = rng.normal(size=(2, 1000))
    model = _FakeModel(mixtures, sources)
    data = _FakeAudioData(sample_rate=8000)

    written = AudioVisualizer(data=data).plot(model, tmp_path)

    assert len(written) == 3
    assert all(path.exists() for path in written)
    assert len(data.export_calls) == 2


@pytest.mark.parametrize("n_sources", [2, 4, 5])
def test_mixing_diagram_visualizer_handles_various_component_counts(tmp_path, n_sources):
    """MixingDiagramVisualizer nao deve quebrar para diferentes quantidades de componentes."""
    rng = np.random.default_rng(0)
    mixtures = rng.normal(size=(n_sources, 200))
    sources = rng.normal(size=(n_sources, 200))
    model = _FakeModel(mixtures, sources)

    written = MixingDiagramVisualizer(max_pairs=3).plot(model, tmp_path)

    assert len(written) == 1
    assert written[0].exists()


def test_log_likelihood_visualizer_writes_png(tmp_path):
    """LogLikelihoodVisualizer.plot deve escrever o PNG da curva de convergencia."""
    rng = np.random.default_rng(0)
    mixtures = rng.normal(size=(2, 100))
    sources = rng.normal(size=(2, 100))
    history = list(np.linspace(-5.0, -1.0, 50))
    model = _FakeModel(mixtures, sources, log_likelihood_history=history)

    written = LogLikelihoodVisualizer().plot(model, tmp_path)

    assert len(written) == 1
    assert written[0].exists()
    assert written[0].name == "log_verossimilhanca.png"


@pytest.mark.parametrize("n_sources", [2, 3, 4, 5])
def test_mixing_diagram_3d_visualizer_only_writes_for_exactly_three_components(
    tmp_path, n_sources
):
    """MixingDiagram3DVisualizer so deve escrever arquivo quando ha exatamente 3 componentes."""
    rng = np.random.default_rng(0)
    mixtures = rng.normal(size=(n_sources, 200))
    sources = rng.normal(size=(n_sources, 200))
    model = _FakeModel(mixtures, sources)

    written = MixingDiagram3DVisualizer().plot(model, tmp_path)

    if n_sources == 3:
        assert len(written) == 1
        assert written[0].exists()
        assert written[0].name == "nuvem_3d_misturas_vs_fontes.png"
    else:
        assert written == []


def test_mixing_diagram_3d_visualizer_subsamples_large_point_clouds(tmp_path):
    """Com mais amostras que max_points, o visualizador nao deve falhar (subamostra)."""
    rng = np.random.default_rng(0)
    mixtures = rng.normal(size=(3, 10_000))
    sources = rng.normal(size=(3, 10_000))
    model = _FakeModel(mixtures, sources)

    written = MixingDiagram3DVisualizer(max_points=500).plot(model, tmp_path)

    assert len(written) == 1
    assert written[0].exists()
