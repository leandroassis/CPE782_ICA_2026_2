"""Visualizacao qualitativa de audio: forma de onda, espectrograma e exportacao das fontes.

Ver context/TASK_DESCRIPTION.md ("inteligibilidade dos audios separados").
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from ica.visualization.base import Visualizer

if TYPE_CHECKING:
    from ica.data.audio_template import AudioTemplate
    from ica.model import ICAModel


class AudioVisualizer(Visualizer):
    """Plota formas de onda e espectrogramas, e exporta as fontes recuperadas como ``.wav``.

    Parameters
    ----------
    data : AudioTemplate
        Template da amostra de audio, usado para exportar as fontes
        recuperadas via ``export`` (Protocol ``Exportable``, ver
        ``ica.interfaces``).
    """

    def __init__(self, data: AudioTemplate) -> None:
        self._data = data

    def plot(self, model: ICAModel, output_dir: Path) -> list[Path]:
        """Salva um grid PNG (forma de onda + espectrograma) e um ``.wav`` por fonte recuperada.

        Parameters
        ----------
        model : ICAModel
            Modelo ja ajustado.
        output_dir : pathlib.Path
            Diretorio de saida.

        Returns
        -------
        list of pathlib.Path
            Caminho do PNG seguido dos caminhos dos ``.wav`` exportados.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        n = model.sources_.shape[0]
        sample_rate = self._data.sample_rate_

        fig, axes = plt.subplots(2, n, figsize=(4 * n, 6), squeeze=False)
        for i in range(n):
            signal = model.sources_[i]
            axes[0, i].plot(signal, linewidth=0.5)
            axes[0, i].set_title(f"fonte recuperada {i + 1} -- forma de onda")

            frequencies, times, power = spectrogram(signal, fs=sample_rate)
            axes[1, i].pcolormesh(
                times, frequencies, 10 * np.log10(power + 1e-12), shading="auto"
            )
            axes[1, i].set_title(f"fonte recuperada {i + 1} -- espectrograma")
            axes[1, i].set_xlabel("tempo (s)")
            axes[1, i].set_ylabel("frequencia (Hz)")
        fig.tight_layout()

        plot_path = output_dir / "audio_formas_de_onda_e_espectrogramas.png"
        fig.savefig(plot_path)
        plt.close(fig)
        written = [plot_path]

        for i in range(n):
            wav_path = output_dir / f"fonte_recuperada_{i + 1}.wav"
            self._data.export(model.sources_[i], wav_path)
            written.append(wav_path)

        return written
