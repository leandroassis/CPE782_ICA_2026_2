"""Testes unitarios para ShowcaseVisualizer (skill ica-evaluation, Secao 6)."""

import numpy as np

from ica.visualization.showcase_visualizer import ShowcaseVisualizer


class _FakeImageData:
    def reconstruct(self, vector):
        side = int(round(np.sqrt(vector.shape[0])))
        return vector.reshape(side, side)


class _FakeAudioData:
    def __init__(self):
        self.exported = []

    def export(self, signal, output_path):
        self.exported.append((signal, output_path))
        output_path.touch()


class _FakeModel:
    def __init__(self, domain, mixtures, sources, sources_true=None, meta=None):
        self.domain_ = domain
        self.mixtures_ = mixtures
        self.sources_ = sources
        self.sources_true_ = sources_true
        self.signal_meta_ = meta or {}


def test_plot_distribution_writes_a_png(tmp_path, rng):
    """Para o dominio distribuicao, deve escrever exatamente 1 PNG."""
    model = _FakeModel(
        "distribution",
        mixtures=rng.normal(size=(2, 1000)),
        sources=rng.laplace(size=(2, 1000)),
        sources_true=rng.laplace(size=(2, 1000)),
    )
    paths = ShowcaseVisualizer(metrics={"amari_index": 0.1}).plot(model, tmp_path)
    assert len(paths) == 1
    assert paths[0].exists()


def test_plot_distribution_without_ground_truth_still_works(tmp_path, rng):
    """Sem sources_true_, a figura deve degradar para 2 faixas sem erro."""
    model = _FakeModel(
        "distribution", mixtures=rng.normal(size=(2, 500)), sources=rng.laplace(size=(2, 500))
    )
    paths = ShowcaseVisualizer().plot(model, tmp_path)
    assert paths[0].exists()


def test_plot_image_writes_a_png(tmp_path):
    """Para o dominio imagem, deve escrever 1 PNG usando data.reconstruct()."""
    rng = np.random.default_rng(0)
    n_pixels = 16
    model = _FakeModel(
        "image",
        mixtures=rng.uniform(size=(2, n_pixels)),
        sources=rng.uniform(size=(2, n_pixels)),
        sources_true=rng.uniform(size=(2, n_pixels)),
    )
    paths = ShowcaseVisualizer(data=_FakeImageData()).plot(model, tmp_path)
    assert len(paths) == 1
    assert paths[0].exists()


def test_plot_rgb_composites_writes_a_png_with_and_without_expected(tmp_path, rng):
    """plot_rgb_composites deve funcionar com e sem trincas verdadeiras."""
    composites = [rng.uniform(size=(3, 16)) for _ in range(2)]
    true_composites = [rng.uniform(size=(3, 16)) for _ in range(2)]

    visualizer = ShowcaseVisualizer()
    paths_with_expected = visualizer.plot_rgb_composites(
        composites, height=4, width=4, output_dir=tmp_path / "a", true_composites=true_composites
    )
    paths_without_expected = visualizer.plot_rgb_composites(
        composites, height=4, width=4, output_dir=tmp_path / "b"
    )

    assert paths_with_expected[0].exists()
    assert paths_without_expected[0].exists()


def test_plot_audio_exports_wavs_and_writes_spectrum_png(tmp_path, rng):
    """Para audio, deve exportar 1 wav por fonte e escrever a figura de espectros."""
    model = _FakeModel(
        "audio",
        mixtures=rng.normal(size=(2, 2000)),
        sources=rng.normal(size=(2, 2000)),
        meta={"sample_rate": 8000},
    )
    data = _FakeAudioData()
    paths = ShowcaseVisualizer(data=data).plot(model, tmp_path)

    assert len(data.exported) == 2
    assert any(p.name == "vitrine_audio_espectros.png" and p.exists() for p in paths)


def test_metrics_caption_ignores_non_numeric_and_nan_values():
    """A legenda de metricas deve ignorar valores None/nao-numericos/NaN."""
    visualizer = ShowcaseVisualizer(
        metrics={"amari_index": 0.25, "family": "laplaciana", "missing": None, "bad": float("nan")}
    )
    caption = visualizer._metrics_caption()
    assert "amari_index=0.25" in caption
    assert "family" not in caption
    assert "missing" not in caption
    assert "bad" not in caption
