"""Testes unitarios para ShowcaseVisualizer (skill ica-evaluation, Secao 6)."""

import numpy as np

from ica.visualization.showcase_visualizer import (
    ShowcaseVisualizer,
    _convergence_curves_by_algorithm,
    _match_true_to_estimated,
)


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


def test_plot_distribution_with_correlated_but_permuted_ground_truth_renders_without_error(
    tmp_path, rng
):
    """Fim a fim: gabarito permutado e correlacionado nao deve quebrar o overlay.

    Caso real de ICA (a saida vem numa ordem permutada e nao alinhada ao
    gabarito) -- nao deve quebrar o overlay de ajuste de distribuicao nem a
    anotacao de KS/correlacao.
    """
    n = 2000
    true_a = rng.laplace(size=n)
    true_b = rng.uniform(-1.0, 1.0, size=n)
    model = _FakeModel(
        "distribution",
        mixtures=rng.normal(size=(2, n)),
        sources=np.vstack(
            [true_b + rng.normal(scale=0.05, size=n), true_a + rng.normal(scale=0.05, size=n)]
        ),
        sources_true=np.vstack([true_a, true_b]),
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


def test_plot_image_with_height_width_meta_annotates_psnr_ssim_without_error(tmp_path):
    """Com height/width em signal_meta_, o titulo 'obtido' deve poder anotar PSNR/SSIM."""
    rng = np.random.default_rng(0)
    n_pixels = 16
    model = _FakeModel(
        "image",
        mixtures=rng.uniform(size=(2, n_pixels)),
        sources=rng.uniform(size=(2, n_pixels)),
        sources_true=rng.uniform(size=(2, n_pixels)),
        meta={"height": 4, "width": 4},
    )
    paths = ShowcaseVisualizer(data=_FakeImageData()).plot(model, tmp_path)
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


def test_plot_audio_spectrogram_matches_expected_to_correlated_obtained(tmp_path):
    """O espectrograma tambem deve casar esperado<->obtido por correlacao, nao por indice."""
    rng = np.random.default_rng(0)
    n = 4000
    t = np.arange(n) / 8000.0
    true_a = np.sin(2 * np.pi * 220 * t)
    true_b = rng.normal(size=n)
    model = _FakeModel(
        "audio",
        mixtures=rng.normal(size=(2, n)),
        # sources_ chega na ordem trocada em relacao ao gabarito.
        sources=np.vstack([true_b, true_a]),
        sources_true=np.vstack([true_a, true_b]),
        meta={"sample_rate": 8000},
    )
    data = _FakeAudioData()
    paths = ShowcaseVisualizer(data=data).plot(model, tmp_path)
    assert any(p.name == "vitrine_audio_espectros.png" and p.exists() for p in paths)


class _FakeICAModelHistory:
    def __init__(self, log_likelihood_history):
        self.log_likelihood_history_ = log_likelihood_history


class _FakeCellResult:
    def __init__(self, algorithm, mode, log_likelihood_history):
        self.algorithm = algorithm
        self.mode = mode
        self.models = [_FakeICAModelHistory(log_likelihood_history)]


def test_convergence_curves_by_algorithm_filters_by_mode_and_sorts_alphabetically():
    """So as celulas do modo pedido entram, ordenadas por nome do algoritmo."""
    cells = [
        _FakeCellResult("fastica_ml", "unico", [-2.0, -1.0]),
        _FakeCellResult("natural_gradient", "unico", [-2.0, -1.5, -1.2, -1.1]),
        _FakeCellResult("bell_sejnowski", "unico", [-2.1, -1.6, -1.3]),
        _FakeCellResult("natural_gradient", "outro_modo", [-5.0, -4.0]),
    ]

    curves = _convergence_curves_by_algorithm(cells, mode="unico")

    assert [algorithm for algorithm, _ in curves] == [
        "bell_sejnowski",
        "fastica_ml",
        "natural_gradient",
    ]
    natural_gradient_curve = dict(curves)["natural_gradient"]
    assert list(natural_gradient_curve) == [-2.0, -1.5, -1.2, -1.1]


def test_convergence_curves_by_algorithm_averages_multiple_models_per_cell():
    """Para celulas com varios modelos (ex.: modo B), usa a media truncada ao menor historico."""
    cell = _FakeCellResult("fastica_ml", "B", [-2.0, -1.0, -0.5])
    cell.models.append(_FakeICAModelHistory([-3.0, -2.0]))  # historico mais curto

    curves = _convergence_curves_by_algorithm([cell], mode="B")

    assert len(curves) == 1
    algorithm, curve = curves[0]
    assert algorithm == "fastica_ml"
    assert list(curve) == [-2.5, -1.5]


def test_plot_convergence_comparison_writes_a_png(tmp_path):
    """Smoke test: deve escrever o PNG comparativo sem erro."""
    cells = [
        _FakeCellResult("natural_gradient", "unico", [-2.0, -1.5, -1.2, -1.1]),
        _FakeCellResult("bell_sejnowski", "unico", [-2.1, -1.6, -1.3]),
        _FakeCellResult("fastica_ml", "unico", [-2.0, -1.0]),
    ]
    path = ShowcaseVisualizer().plot_convergence_comparison(
        cells, mode="unico", output_dir=tmp_path
    )
    assert path.exists()


def test_plot_system_matrix_writes_a_png_for_a_permuted_scaled_recovery(tmp_path):
    """G = B@A de uma recuperacao perfeita a menos de permutacao/escala deve renderizar."""
    A = np.array([[1.0, 0.5], [0.3, 1.0]])
    # B recupera as fontes trocadas de ordem e com escala diferente (ambiguidades da ICA).
    B = np.linalg.inv(A) @ np.array([[0.0, 2.0], [3.0, 0.0]])
    path = ShowcaseVisualizer().plot_system_matrix(B, A, output_dir=tmp_path)
    assert path.exists()


def test_match_true_to_estimated_finds_permuted_pairs_by_correlation(rng):
    """O casamento deve achar o par certo por correlacao, nunca por indice literal.

    A ICA permuta livremente -- a coluna ``i`` do gabarito nao precisa
    corresponder a coluna ``i`` do obtido.
    """
    n = 2000
    true_a = rng.laplace(size=n)
    true_b = rng.uniform(-1.0, 1.0, size=n)
    sources_true = np.vstack([true_a, true_b])
    # sources_ chega na ordem trocada (permutacao tipica da saida da ICA),
    # com um pouco de ruido para nao ser uma correlacao perfeita/degenerada.
    sources_estimated = np.vstack(
        [true_b + rng.normal(scale=0.01, size=n), true_a + rng.normal(scale=0.01, size=n)]
    )

    true_by_estimated, correlation_by_estimated = _match_true_to_estimated(
        sources_true, sources_estimated
    )

    assert true_by_estimated == {0: 1, 1: 0}
    assert correlation_by_estimated[0] > 0.99
    assert correlation_by_estimated[1] > 0.99


def test_match_true_to_estimated_returns_empty_dicts_without_ground_truth():
    """Sem gabarito (reference=None), os dois dicts devem vir vazios."""
    true_by_estimated, correlation_by_estimated = _match_true_to_estimated(
        None, np.zeros((2, 10))
    )
    assert true_by_estimated == {}
    assert correlation_by_estimated == {}


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
