"""Figura-vitrine: mistura | esperado (gabarito) | obtido, com metricas escritas.

Skill ``ica-evaluation``, Secao 6. Por run, uma figura da melhor separacao
(escolhida em ``ica.harness.selection``), **sem alinhar** esperado<->obtido
(a ICA permuta; a ordem mostrada e a de ``model.sources_``, ja resolvida
por :func:`~ica.postprocessing.ambiguity.resolve_ambiguities`, nunca a do
casamento hungaro). Quando nao ha gabarito (``model.sources_true_ is
None``), a faixa "esperado" e omitida e a figura degrada para 2 faixas
(mistura | obtido).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from ica.metrics.family_identification import identify_family
from ica.visualization.base import Visualizer

if TYPE_CHECKING:
    from ica.data.audio_template import AudioTemplate
    from ica.data.image_template import ImageTemplate
    from ica.model import ICAModel


class ShowcaseVisualizer(Visualizer):
    """Monta a figura-vitrine por dominio, com as metricas do run escritas nela.

    Parameters
    ----------
    data : DataTemplate
        Template da amostra (usado para ``reconstruct``/``export`` por
        dominio -- so precisa existir para imagem/audio).
    metrics : dict, optional
        Metricas ja calculadas (ex.: via ``model.evaluate(...)``) a
        escrever como texto na figura. Valores ``None``/nao-numericos sao
        ignorados silenciosamente.
    """

    def __init__(self, data: Any = None, metrics: dict[str, Any] | None = None) -> None:
        self._data = data
        self._metrics = metrics or {}

    def plot(self, model: ICAModel, output_dir: Path) -> list[Path]:
        """Gera a figura-vitrine (e, para audio, os ``.wav`` separados) do dominio de ``model``.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado (o vencedor da grade, escolhido por
            ``ica.harness.selection``).
        output_dir : pathlib.Path
            Diretorio de saida.

        Returns
        -------
        list of pathlib.Path
            Caminhos dos arquivos gerados.

        Raises
        ------
        ValueError
            Se ``model.domain_`` nao for um dos 3 dominios suportados.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if model.domain_ == "image":
            return self._plot_image(model, output_dir)
        if model.domain_ == "distribution":
            return self._plot_distribution(model, output_dir)
        if model.domain_ == "audio":
            return self._plot_audio(model, output_dir)
        raise ValueError(f"Dominio desconhecido: {model.domain_!r}.")

    def _metrics_caption(self) -> str:
        """Formata as metricas numericas do run como uma linha de legenda."""
        parts = []
        for name, value in self._metrics.items():
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                parts.append(f"{name}={value:.4g}")
        return " | ".join(parts)

    # -- imagem ------------------------------------------------------------

    def _plot_image(self, model: ICAModel, output_dir: Path) -> list[Path]:
        data: ImageTemplate = self._data
        has_expected = model.sources_true_ is not None
        n_rows = 3 if has_expected else 2
        n_cols = model.mixtures_.shape[0]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
        row_labels = ["mistura"]
        if has_expected:
            row_labels.append("esperado")
        row_labels.append("obtido")

        for col in range(n_cols):
            axes[0, col].imshow(data.reconstruct(model.mixtures_[col]), cmap="gray")
            row = 1
            if has_expected and col < model.sources_true_.shape[0]:
                axes[row, col].imshow(data.reconstruct(model.sources_true_[col]), cmap="gray")
                row += 1
            axes[row, col].imshow(data.reconstruct(model.sources_[col]), cmap="gray")

        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label, fontsize=11)
        for row in range(n_rows):
            for col in range(n_cols):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

        fig.suptitle(self._metrics_caption(), fontsize=9)
        fig.tight_layout()
        path = output_dir / "vitrine_imagens.png"
        fig.savefig(path)
        plt.close(fig)
        return [path]

    def plot_rgb_composites(
        self,
        estimated_composites: list[np.ndarray],
        height: int,
        width: int,
        output_dir: Path,
        true_composites: list[np.ndarray] | None = None,
    ) -> list[Path]:
        """Figura-vitrine para os modos de condicionamento B/C de imagem (RGB ja reagrupado).

        Usada em vez de :meth:`plot` porque os modos B/C
        (:mod:`ica.harness.grid`) produzem trincas RGB ja reagrupadas
        (:func:`~ica.postprocessing.regroup.regroup_rgb_planes` ou
        :meth:`~ica.data.image_template.ImageTemplate.mode_c_matrix`), nao
        um unico ``ICAModel`` com ``sources_`` de 1 canal.

        Parameters
        ----------
        estimated_composites : list of np.ndarray
            Trincas RGB recuperadas, cada uma shape ``(3, n_pixels)``.
        height, width : int
            Dimensoes para reformatar cada canal em imagem 2D.
        output_dir : pathlib.Path
            Diretorio de saida.
        true_composites : list of np.ndarray, optional
            Trincas RGB verdadeiras (gabarito), mesma shape -- omite a
            faixa "esperado" quando ``None``.

        Returns
        -------
        list of pathlib.Path
            Caminho do PNG gerado.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        has_expected = true_composites is not None
        n_rows = 2 if has_expected else 1
        n_cols = len(estimated_composites)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
        for col in range(n_cols):
            row = 0
            if has_expected:
                axes[0, col].imshow(_composite_to_rgb_image(true_composites[col], height, width))
                axes[0, col].set_title(f"esperado {col + 1}")
                row = 1
            axes[row, col].imshow(
                _composite_to_rgb_image(estimated_composites[col], height, width)
            )
            axes[row, col].set_title(f"obtido {col + 1}")

        for row in range(n_rows):
            for col in range(n_cols):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

        fig.suptitle(self._metrics_caption(), fontsize=9)
        fig.tight_layout()
        path = output_dir / "vitrine_imagens_rgb.png"
        fig.savefig(path)
        plt.close(fig)
        return [path]

    # -- distribuicao --------------------------------------------------------

    def _plot_distribution(self, model: ICAModel, output_dir: Path) -> list[Path]:
        has_expected = model.sources_true_ is not None
        n_rows = 3 if has_expected else 2
        n_cols = model.mixtures_.shape[0]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows), squeeze=False)
        for col in range(n_cols):
            axes[0, col].hist(model.mixtures_[col], bins=50, density=True, alpha=0.7)
            axes[0, col].set_title(f"mistura {col + 1}")
            row = 1
            if has_expected and col < model.sources_true_.shape[0]:
                axes[row, col].hist(
                    model.sources_true_[col], bins=50, density=True, alpha=0.7, color="tab:green"
                )
                axes[row, col].set_title(f"esperado {col + 1}")
                row += 1
            family = identify_family(model.sources_[col])
            axes[row, col].hist(
                model.sources_[col], bins=50, density=True, alpha=0.7, color="tab:orange"
            )
            axes[row, col].set_title(f"obtido {col + 1} ({family.best.name})")

        fig.suptitle(self._metrics_caption(), fontsize=9)
        fig.tight_layout()
        path = output_dir / "vitrine_distribuicoes.png"
        fig.savefig(path)
        plt.close(fig)
        return [path]

    # -- audio ---------------------------------------------------------------

    def _magnitude_spectrum(
        self, signal: np.ndarray, sample_rate: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Espectro de magnitude do sinal inteiro (sem framing/STFT)."""
        spectrum = np.abs(np.fft.rfft(signal))
        frequencies = np.fft.rfftfreq(signal.shape[0], d=1.0 / sample_rate)
        return frequencies, spectrum

    def _plot_audio(self, model: ICAModel, output_dir: Path) -> list[Path]:
        data: AudioTemplate = self._data
        sample_rate = model.signal_meta_["sample_rate"]
        written: list[Path] = []

        for i, source in enumerate(model.sources_):
            wav_path = output_dir / f"fonte_separada_{i + 1}.wav"
            data.export(source, wav_path)
            written.append(wav_path)

        has_expected = model.sources_true_ is not None
        n_rows = 3 if has_expected else 2
        n_cols = model.mixtures_.shape[0]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False)
        for col in range(n_cols):
            freq, mag = self._magnitude_spectrum(model.mixtures_[col], sample_rate)
            axes[0, col].plot(freq, mag)
            axes[0, col].set_title(f"espectro mistura {col + 1}")
            row = 1
            if has_expected and col < model.sources_true_.shape[0]:
                freq, mag = self._magnitude_spectrum(model.sources_true_[col], sample_rate)
                axes[row, col].plot(freq, mag, color="tab:green")
                axes[row, col].set_title(f"espectro esperado {col + 1}")
                row += 1
            freq, mag = self._magnitude_spectrum(model.sources_[col], sample_rate)
            axes[row, col].plot(freq, mag, color="tab:orange")
            axes[row, col].set_title(f"espectro obtido {col + 1}")

        fig.suptitle(self._metrics_caption(), fontsize=9)
        fig.tight_layout()
        spectrum_path = output_dir / "vitrine_audio_espectros.png"
        fig.savefig(spectrum_path)
        plt.close(fig)
        written.append(spectrum_path)
        return written


def _composite_to_rgb_image(composite: np.ndarray, height: int, width: int) -> np.ndarray:
    """Reformata uma trinca ``(3, n_pixels)`` em uma imagem ``(H, W, 3)`` normalizada em ``[0, 1]``.

    Parameters
    ----------
    composite : np.ndarray
        Trinca RGB, shape ``(3, height * width)``.
    height, width : int
        Dimensoes de cada canal.

    Returns
    -------
    np.ndarray
        Imagem RGB, shape ``(height, width, 3)``, cada canal normalizado
        independentemente para ``[0, 1]`` (fins de visualizacao).
    """
    channels = [composite[c].reshape(height, width) for c in range(3)]
    rgb = np.stack(channels, axis=-1)
    channel_min = rgb.min(axis=(0, 1), keepdims=True)
    channel_max = rgb.max(axis=(0, 1), keepdims=True)
    span = np.where(channel_max > channel_min, channel_max - channel_min, 1.0)
    return (rgb - channel_min) / span
