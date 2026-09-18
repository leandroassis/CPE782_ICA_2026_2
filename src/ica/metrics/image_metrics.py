"""PSNR e SSIM -- metricas perceptuais para imagem, por fonte casada.

Skill ``ica-evaluation``, Secao 4, e ``references/metrics-formulas.md``,
Secao 4. Implementadas a mao (numpy/scipy), sem ``scikit-image`` (nao e
dependencia do projeto).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import uniform_filter

from ica.metrics.base import Metric
from ica.postprocessing.matching import hungarian_match

if TYPE_CHECKING:
    from ica.model import ICAModel

_SSIM_K1 = 0.01
_SSIM_K2 = 0.03


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """Normaliza ``image`` para ``[0, 1]`` -- rescale de imagem da skill ica-evaluation."""
    minimum, maximum = image.min(), image.max()
    span = maximum - minimum if maximum > minimum else 1.0
    return (image - minimum) / span


def psnr(image_a: np.ndarray, image_b: np.ndarray, max_value: float = 1.0) -> float:
    """``PSNR = 10 log10(MAX^2 / MSE)``.

    Parameters
    ----------
    image_a, image_b : np.ndarray
        Imagens de mesma shape, tipicamente ja normalizadas para
        ``[0, max_value]``.
    max_value : float, default=1.0
        Valor maximo da faixa dinamica (``MAX_I``).

    Returns
    -------
    float
        PSNR em dB (``inf`` se as imagens forem identicas).
    """
    mse = float(np.mean((image_a - image_b) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10((max_value**2) / mse))


def ssim(
    image_a: np.ndarray, image_b: np.ndarray, window_size: int = 7, dynamic_range: float = 1.0
) -> float:
    """Similaridade estrutural (SSIM), janelada por filtro de media movel local.

    ``references/metrics-formulas.md``, Secao 4:
    ``SSIM = [(2 mu_a mu_b + c1)(2 sigma_ab + c2)] / [(mu_a^2+mu_b^2+c1)(sigma_a^2+sigma_b^2+c2)]``,
    com ``c1=(k1 L)^2``, ``c2=(k2 L)^2``, ``L`` a faixa dinamica. Usa uma
    janela de media movel (``scipy.ndimage.uniform_filter``) em vez da
    janela gaussiana do artigo original -- simplificacao documentada, ainda
    localmente sensivel a estrutura (nao so a erro pixel-a-pixel).

    Parameters
    ----------
    image_a, image_b : np.ndarray
        Imagens 2D de mesma shape.
    window_size : int, default=7
        Tamanho da janela local (lado do quadrado).
    dynamic_range : float, default=1.0
        Faixa dinamica ``L`` (``1.0`` para imagens normalizadas em ``[0,1]``).

    Returns
    -------
    float
        SSIM medio sobre a imagem, em ``[-1, 1]`` (``1`` = identico).
    """
    c1 = (_SSIM_K1 * dynamic_range) ** 2
    c2 = (_SSIM_K2 * dynamic_range) ** 2

    mu_a = uniform_filter(image_a, window_size)
    mu_b = uniform_filter(image_b, window_size)
    mu_a_sq, mu_b_sq, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b

    sigma_a_sq = uniform_filter(image_a**2, window_size) - mu_a_sq
    sigma_b_sq = uniform_filter(image_b**2, window_size) - mu_b_sq
    sigma_ab = uniform_filter(image_a * image_b, window_size) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return float(np.mean(numerator / denominator))


def image_metrics_battery(
    sources_true: np.ndarray, sources_estimated: np.ndarray, height: int, width: int
) -> list[tuple[float, float]]:
    """Casa (hungaro) e calcula (PSNR, SSIM) por par de imagens-fonte.

    Ambas as imagens sao normalizadas para ``[0, 1]`` antes da comparacao
    (a fonte verdadeira nao vem necessariamente nessa faixa; a recuperada
    ja vem, via :func:`~ica.postprocessing.ambiguity.fix_scale`).

    Parameters
    ----------
    sources_true : np.ndarray
        Fontes-imagem verdadeiras, shape ``(n_fontes, n_pixels)``.
    sources_estimated : np.ndarray
        Componentes recuperadas, shape ``(n_componentes, n_pixels)``.
    height, width : int
        Dimensoes para reformatar cada vetor em imagem 2D.

    Returns
    -------
    list of (float, float)
        ``(psnr_db, ssim)`` por fonte verdadeira casada, na ordem de
        ``sources_true``.
    """
    match = hungarian_match(sources_true, sources_estimated)
    order = np.argsort(match.reference_indices)
    results = []
    for i in order:
        true_image = min_max_normalize(
            sources_true[match.reference_indices[i]].reshape(height, width)
        )
        estimated_image = sources_estimated[match.matched_indices[i]].reshape(height, width)
        results.append((psnr(true_image, estimated_image), ssim(true_image, estimated_image)))
    return results


class PSNRMetric(Metric):
    """PSNR por fonte-imagem casada, quando o gabarito esta disponivel."""

    name = "psnr_db_per_source"

    def compute(self, model: ICAModel) -> np.ndarray | None:
        """Calcula o PSNR de cada fonte-imagem casada de ``model``.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado sobre uma amostra de imagem.

        Returns
        -------
        np.ndarray or None
            PSNR (dB) por fonte, ou ``None`` sem gabarito/fora do dominio imagem.
        """
        if model.sources_true_ is None or model.domain_ != "image":
            return None
        battery = image_metrics_battery(
            model.sources_true_,
            model.sources_,
            model.signal_meta_["height"],
            model.signal_meta_["width"],
        )
        return np.array([psnr_value for psnr_value, _ in battery])


class SSIMMetric(Metric):
    """SSIM por fonte-imagem casada, quando o gabarito esta disponivel."""

    name = "ssim_per_source"

    def compute(self, model: ICAModel) -> np.ndarray | None:
        """Calcula o SSIM de cada fonte-imagem casada de ``model``.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado sobre uma amostra de imagem.

        Returns
        -------
        np.ndarray or None
            SSIM por fonte, ou ``None`` sem gabarito/fora do dominio imagem.
        """
        if model.sources_true_ is None or model.domain_ != "image":
            return None
        battery = image_metrics_battery(
            model.sources_true_,
            model.sources_,
            model.signal_meta_["height"],
            model.signal_meta_["width"],
        )
        return np.array([ssim_value for _, ssim_value in battery])
