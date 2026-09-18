"""Testes unitarios para ica.metrics.image_metrics (skill ica-evaluation, Secao 4)."""

import numpy as np

from ica.metrics.image_metrics import PSNRMetric, SSIMMetric, image_metrics_battery, psnr, ssim


def test_psnr_is_infinite_for_identical_images():
    """PSNR deve ser infinito para imagens identicas (MSE = 0)."""
    image = np.random.default_rng(0).uniform(size=(8, 8))
    assert psnr(image, image) == float("inf")


def test_psnr_decreases_with_more_noise(rng):
    """Mais ruido deve reduzir o PSNR."""
    image = rng.uniform(size=(16, 16))
    low_noise = np.clip(image + rng.normal(scale=0.01, size=(16, 16)), 0, 1)
    high_noise = np.clip(image + rng.normal(scale=0.3, size=(16, 16)), 0, 1)
    assert psnr(image, low_noise) > psnr(image, high_noise)


def test_ssim_is_close_to_one_for_identical_images(rng):
    """SSIM deve ser proximo de 1 para imagens identicas."""
    image = rng.uniform(size=(20, 20))
    assert ssim(image, image) > 0.999


def test_ssim_decreases_with_more_noise(rng):
    """Mais ruido deve reduzir o SSIM."""
    image = rng.uniform(size=(20, 20))
    low_noise = image + rng.normal(scale=0.01, size=(20, 20))
    high_noise = image + rng.normal(scale=0.5, size=(20, 20))
    assert ssim(image, low_noise) > ssim(image, high_noise)


def test_image_metrics_battery_matches_and_orders_by_true_index(rng):
    """image_metrics_battery deve casar (hungaro) e ordenar pelo indice verdadeiro."""
    height, width = 8, 8
    n_pixels = height * width
    image_a = rng.uniform(size=n_pixels)
    image_b = rng.uniform(size=n_pixels)
    sources_true = np.vstack([image_a, image_b])
    sources_estimated = np.vstack([image_b, image_a])  # ordem trocada

    battery = image_metrics_battery(sources_true, sources_estimated, height, width)

    assert len(battery) == 2
    for psnr_value, ssim_value in battery:
        assert psnr_value > 30
        assert ssim_value > 0.95


class _FakeModel:
    def __init__(self, sources_true, sources, domain, meta):
        self.sources_true_ = sources_true
        self.sources_ = sources
        self.domain_ = domain
        self.signal_meta_ = meta


def test_psnr_metric_returns_none_without_ground_truth():
    """PSNRMetric.compute deve devolver None sem sources_true_."""
    model = _FakeModel(None, np.zeros((2, 64)), "image", {"height": 8, "width": 8})
    assert PSNRMetric().compute(model) is None


def test_psnr_metric_returns_none_outside_image_domain():
    """PSNRMetric.compute deve devolver None fora do dominio imagem."""
    model = _FakeModel(np.zeros((2, 64)), np.zeros((2, 64)), "distribution", {})
    assert PSNRMetric().compute(model) is None


def test_ssim_metric_computes_array_with_ground_truth(rng):
    """SSIMMetric.compute deve devolver um array quando ha gabarito de imagem."""
    height, width = 8, 8
    image = rng.uniform(size=height * width)
    model = _FakeModel(
        np.vstack([image, image]), np.vstack([image, image]), "image", {"height": 8, "width": 8}
    )
    result = SSIMMetric().compute(model)
    assert result.shape == (2,)
    assert np.all(result > 0.99)
