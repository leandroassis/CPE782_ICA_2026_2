"""Figura-vitrine: mistura | esperado (gabarito) | obtido, com metricas escritas.

Skill ``ica-evaluation``, Secao 6. Por run, uma figura da melhor separacao
(escolhida em ``ica.harness.selection``). A **ordem das colunas exibidas**
(``model.sources_``, ja resolvida por
:func:`~ica.postprocessing.ambiguity.resolve_ambiguities`) nunca muda; o
que o casamento hungaro (:mod:`ica.postprocessing.matching`) resolve aqui e
so **qual coluna "esperado" ilustra cada coluna "obtido"** -- a ICA permuta
livremente, entao mostrar a fonte verdadeira ``i`` ao lado do componente
que menos se parece com ela renderizaria a figura enganosa. Cada par
casado ganha a metrica de similaridade correspondente escrita junto (KS
para distribuicao, PSNR/SSIM para imagem). Quando nao ha gabarito
(``model.sources_true_ is None``), a faixa "esperado" e omitida e a figura
degrada para 2 faixas (mistura | obtido).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

from ica.metrics.distribution_metrics import distribution_fit_quality
from ica.metrics.family_identification import family_pdf, identify_family
from ica.metrics.image_metrics import min_max_normalize, psnr, ssim
from ica.postprocessing.matching import hungarian_match
from ica.visualization.base import Visualizer

if TYPE_CHECKING:
    from ica.data.audio_template import AudioTemplate
    from ica.data.image_template import ImageTemplate
    from ica.harness.grid import CellResult
    from ica.model import ICAModel

# Paleta categorica fixa (3 primeiros slots do tema validado -- distinguivel
# sob deuteranopia/protanopia nas 3 combinacoes, ``skill dataviz``,
# ``references/palette.md``), em ordem fixa por algoritmo -- nunca ciclada
# nem dependente da ordem de iteracao, para a mesma cor identificar o mesmo
# algoritmo em toda a vitrine (Secao 6) e no comparativo de convergencia.
_ALGORITHM_COLORS = {
    "natural_gradient": "#2a78d6",  # azul
    "bell_sejnowski": "#eb6834",  # laranja
    "fastica_ml": "#1baf7a",  # agua
}
_SPECTROGRAM_CMAP = "magma"  # sequencial, perceptualmente uniforme (nunca "jet")


def _match_true_to_estimated(
    reference: np.ndarray | None, candidates: np.ndarray
) -> tuple[dict[int, int], dict[int, float]]:
    """Casa ``reference`` (gabarito) a ``candidates`` (obtido) por correlacao (hungaro).

    A coluna **candidata** nunca muda de ordem na figura-vitrine (ver
    docstring do modulo); o casamento so decide qual indice do gabarito
    ilustra cada coluna. Devolve dois dicts indexados pela coluna
    candidata: ``estimado -> verdadeiro`` e ``estimado -> correlacao`` do
    par -- ambos vazios quando ``reference`` e ``None`` (sem gabarito).

    Parameters
    ----------
    reference : np.ndarray or None
        Fontes/componentes verdadeiras, shape ``(n_referencia, n_amostras)``.
    candidates : np.ndarray
        Componentes recuperadas, shape ``(n_candidatos, n_amostras)`` --
        mesma ordem de ``model.sources_``.

    Returns
    -------
    (dict, dict)
        ``true_by_estimated``, ``correlation_by_estimated``.
    """
    if reference is None:
        return {}, {}
    match = hungarian_match(reference, candidates)
    true_by_estimated = dict(
        zip(match.matched_indices.tolist(), match.reference_indices.tolist(), strict=True)
    )
    correlation_by_estimated = dict(
        zip(match.matched_indices.tolist(), match.correlations.tolist(), strict=True)
    )
    return true_by_estimated, correlation_by_estimated


def _convergence_curves_by_algorithm(
    cells: list[CellResult], mode: str
) -> list[tuple[str, np.ndarray]]:
    """Extrai 1 curva de log-L/amostra por algoritmo, so das celulas do modo pedido.

    Pura (sem matplotlib) para ser testavel isoladamente de
    :meth:`ShowcaseVisualizer.plot_convergence_comparison`. Para celulas
    com varios modelos (ex.: modo B de imagem, 3 planos), usa a media
    elemento-a-elemento truncada ao menor historico entre eles -- mesma
    convencao de :attr:`~ica.harness.grid.CellResult.log_likelihood_per_sample`.

    Parameters
    ----------
    cells : list of CellResult
        Grade completa de um run.
    mode : str
        Modo de condicionamento a filtrar.

    Returns
    -------
    list of (str, np.ndarray)
        Pares ``(algoritmo, curva)``, ordenados por nome do algoritmo.
    """
    relevant = [cell for cell in cells if cell.mode == mode]
    curves = []
    for cell in sorted(relevant, key=lambda c: c.algorithm):
        histories = [m.log_likelihood_history_ for m in cell.models]
        min_len = min(len(history) for history in histories)
        curve = np.mean([history[:min_len] for history in histories], axis=0)
        curves.append((cell.algorithm, curve))
    return curves


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

    def plot_convergence_comparison(
        self, cells: list[CellResult], mode: str, output_dir: Path
    ) -> Path:
        """Compara, num unico grafico, a curva de convergencia dos 3 algoritmos de ICA-ML.

        Skill ``ica-ml``, Secao 7: cada algoritmo tem seu proprio criterio
        de convergencia interno (GN/BS: residuo ``||I+E{g(y)y^T}||_F``;
        FastICA-ML: alinhamento de colunas) -- nao comparaveis diretamente
        entre si. A log-verossimilhanca media por amostra, porem, e a
        **mesma grandeza para os 3** (e o proprio criterio de
        ``argmax`` usado por :func:`~ica.harness.selection.select_best`),
        entao serve de moeda comum para ilustrar velocidade de convergencia
        lado a lado (skill ``ica-evaluation``, Secao 5).

        Parameters
        ----------
        cells : list of CellResult
            Grade completa de um run (``ica.harness.run_grid``).
        mode : str
            Modo de condicionamento a comparar (tipicamente o do
            vencedor, ``ica.harness.select_best``) -- filtra para as
            celulas desse modo, uma por algoritmo, garantindo uma
            comparacao like-for-like (nunca entre modos diferentes).
        output_dir : pathlib.Path
            Diretorio de saida.

        Returns
        -------
        pathlib.Path
            Caminho do PNG gerado.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        for algorithm, curve in _convergence_curves_by_algorithm(cells, mode):
            ax.plot(
                np.arange(1, len(curve) + 1),
                curve,
                color=_ALGORITHM_COLORS.get(algorithm, "gray"),
                linewidth=2,
                label=algorithm,
            )

        ax.set_xlabel("iteração")
        ax.set_ylabel("log-verossimilhança / amostra")
        ax.set_title(f"Convergência por algoritmo (modo {mode})", fontsize=10)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = output_dir / "convergencia_algoritmos.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_system_matrix(
        self, unmixing_matrix: np.ndarray, mixing_matrix_true: np.ndarray, output_dir: Path
    ) -> Path:
        """Visualiza a matriz global ``G = B @ A`` -- quantifica a qualidade de separacao.

        Skill ``ica-evaluation``, Secao 4 (indice de Amari): separacao
        perfeita equivale a ``G`` ser uma permutacao x escala (um unico
        entrada dominante por linha/coluna, resto perto de zero) -- **nao**
        precisa ser a identidade, ja que a ICA nunca resolve a ordem/escala
        das fontes. Cada linha e normalizada pelo seu maior valor absoluto,
        entao a estrutura de permutacao fica visualmente evidente
        independente da escala de cada componente; o indice de Amari
        (``ica.metrics.amari.amari_index``) e o resumo escalar do mesmo
        ``G``.

        Parameters
        ----------
        unmixing_matrix : np.ndarray
            ``B``, matriz de separacao estimada no espaco das misturas
            originais (``ICAModel.full_unmixing_matrix_``).
        mixing_matrix_true : np.ndarray
            ``A``, matriz de mistura verdadeira (gabarito).
        output_dir : pathlib.Path
            Diretorio de saida.

        Returns
        -------
        pathlib.Path
            Caminho do PNG gerado.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        system_matrix = unmixing_matrix @ mixing_matrix_true
        row_scale = np.max(np.abs(system_matrix), axis=1, keepdims=True)
        row_scale = np.where(row_scale > 0, row_scale, 1.0)
        normalized = np.abs(system_matrix) / row_scale

        figsize = (1.2 * system_matrix.shape[1] + 1, 1.2 * system_matrix.shape[0] + 1.4)
        fig, ax = plt.subplots(figsize=figsize)
        mesh = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
        for i in range(system_matrix.shape[0]):
            for j in range(system_matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{system_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if normalized[i, j] < 0.6 else "black",
                    fontsize=8,
                )
        ax.set_xlabel("coluna (fonte verdadeira)")
        ax.set_ylabel("linha (componente recuperada)")
        ax.set_title("Matriz global G = B·A", fontsize=10, pad=10)
        ax.set_xticks(range(system_matrix.shape[1]))
        ax.set_yticks(range(system_matrix.shape[0]))
        fig.colorbar(mesh, ax=ax, label="|G_ij| / max|linha|", shrink=0.8)
        fig.tight_layout()
        path = output_dir / "matriz_sistema_global.png"
        fig.savefig(path)
        plt.close(fig)
        return path

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

        true_for_estimated, _ = _match_true_to_estimated(model.sources_true_, model.sources_)
        height = model.signal_meta_.get("height")
        width = model.signal_meta_.get("width")

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
        row_labels = ["mistura"]
        if has_expected:
            row_labels.append("esperado")
        row_labels.append("obtido")

        # Cada painel e normalizado para [0,1] so para exibicao (mesma escala
        # de cinza em mistura/esperado/obtido, apesar de estarem em unidades
        # distintas) -- nao afeta nenhuma metrica, que usa os dados crus.
        image = None
        for col in range(n_cols):
            image = axes[0, col].imshow(
                min_max_normalize(data.reconstruct(model.mixtures_[col])),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            row = 1
            true_idx = true_for_estimated.get(col)
            if true_idx is not None:
                axes[row, col].imshow(
                    min_max_normalize(data.reconstruct(model.sources_true_[true_idx])),
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                axes[row, col].set_title(f"esperado {true_idx + 1}", fontsize=9)
                row += 1
            axes[row, col].imshow(
                data.reconstruct(model.sources_[col]), cmap="gray", vmin=0, vmax=1
            )
            title = f"obtido {col + 1}"
            if true_idx is not None and height and width:
                true_norm = min_max_normalize(
                    model.sources_true_[true_idx].reshape(height, width)
                )
                estimated = model.sources_[col].reshape(height, width)
                title += (
                    f"\nPSNR={psnr(true_norm, estimated):.1f}dB "
                    f"SSIM={ssim(true_norm, estimated):.2f}"
                )
            axes[row, col].set_title(title, fontsize=9)

        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label, fontsize=11)
        for row in range(n_rows):
            for col in range(n_cols):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

        fig.suptitle(self._metrics_caption(), fontsize=9)
        fig.colorbar(
            image, ax=axes.ravel().tolist(), shrink=0.6, label="intensidade (normalizada)"
        )
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

        true_for_estimated = {}
        if has_expected:
            true_flat = np.array([c.reshape(-1) for c in true_composites])
            estimated_flat = np.array([c.reshape(-1) for c in estimated_composites])
            true_for_estimated, _ = _match_true_to_estimated(true_flat, estimated_flat)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
        for col in range(n_cols):
            row = 0
            true_idx = true_for_estimated.get(col)
            if has_expected and true_idx is not None:
                axes[0, col].imshow(
                    _composite_to_rgb_image(true_composites[true_idx], height, width)
                )
                axes[0, col].set_title(f"esperado {true_idx + 1}")
                row = 1
            axes[row, col].imshow(
                _composite_to_rgb_image(estimated_composites[col], height, width)
            )
            title = f"obtido {col + 1}"
            if true_idx is not None:
                per_channel_psnr, per_channel_ssim = [], []
                for channel in range(3):
                    true_channel = min_max_normalize(
                        true_composites[true_idx][channel].reshape(height, width)
                    )
                    estimated_channel = estimated_composites[col][channel].reshape(height, width)
                    per_channel_psnr.append(psnr(true_channel, estimated_channel))
                    per_channel_ssim.append(ssim(true_channel, estimated_channel))
                title += (
                    f"\nPSNR={np.mean(per_channel_psnr):.1f}dB SSIM={np.mean(per_channel_ssim):.2f}"
                )
            axes[row, col].set_title(title)

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

        true_for_estimated, correlation_for_estimated = _match_true_to_estimated(
            model.sources_true_, model.sources_
        )

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows), squeeze=False)
        for col in range(n_cols):
            axes[0, col].hist(model.mixtures_[col], bins=50, density=True, alpha=0.7)
            axes[0, col].set_title(f"mistura {col + 1}")
            row = 1
            true_idx = true_for_estimated.get(col)
            if true_idx is not None:
                true_source = model.sources_true_[true_idx]
                axes[row, col].hist(
                    true_source, bins=50, density=True, alpha=0.7, color="tab:green", label="dados"
                )
                expected_family = identify_family(true_source)
                grid = np.linspace(true_source.min(), true_source.max(), 200)
                axes[row, col].plot(
                    grid,
                    family_pdf(expected_family, grid),
                    color="black",
                    linewidth=1.5,
                    label=f"ajuste: {expected_family.best.name}",
                )
                axes[row, col].set_title(
                    f"esperado {true_idx + 1} (fit: {expected_family.best.name})"
                )
                axes[row, col].legend(fontsize=7, loc="upper right")
                row += 1
            estimated_source = model.sources_[col]
            family = identify_family(estimated_source)
            axes[row, col].hist(
                estimated_source,
                bins=50,
                density=True,
                alpha=0.7,
                color="tab:orange",
                label="dados",
            )
            estimated_grid = np.linspace(estimated_source.min(), estimated_source.max(), 200)
            axes[row, col].plot(
                estimated_grid,
                family_pdf(family, estimated_grid),
                color="black",
                linewidth=1.5,
                label=f"ajuste: {family.best.name}",
            )
            axes[row, col].legend(fontsize=7, loc="upper right")
            title = f"obtido {col + 1} ({family.best.name})"
            if true_idx is not None:
                ks = distribution_fit_quality(estimated_source, model.sources_true_[true_idx])
                title += (
                    f"\ncorr={correlation_for_estimated[col]:.2f} "
                    f"KS={ks.ks_statistic:.3f}"
                )
            axes[row, col].set_title(title)

        for row in range(n_rows):
            for col in range(n_cols):
                axes[row, col].grid(alpha=0.3, linewidth=0.5)
        for col in range(n_cols):
            axes[n_rows - 1, col].set_xlabel("valor")
        for row in range(n_rows):
            axes[row, 0].set_ylabel("densidade")

        fig.suptitle(self._metrics_caption(), fontsize=9)
        fig.tight_layout()
        path = output_dir / "vitrine_distribuicoes.png"
        fig.savefig(path)
        plt.close(fig)
        return [path]

    # -- audio ---------------------------------------------------------------

    def _spectrogram(self, ax, signal: np.ndarray, sample_rate: float, title: str):
        """Espectrograma (STFT) de ``signal``: frequencia (eixo y) x tempo (eixo x).

        So para exibicao -- a separacao em si continua sobre o sinal
        inteiro no dominio do tempo, sem framing (CLAUDE.md, "Sem
        framing/STFT" refere-se ao modelo de mistura/separacao, nao a
        como um sinal ja separado e mostrado).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Eixo onde desenhar.
        signal : np.ndarray
            Sinal 1D, shape ``(n_amostras,)``.
        sample_rate : float
            Taxa de amostragem (Hz).
        title : str
            Titulo do painel.

        Returns
        -------
        matplotlib.collections.QuadMesh
            O mesh desenhado (para alimentar uma colorbar compartilhada).
        """
        nperseg = min(1024, signal.shape[0])
        frequencies, times, stft_matrix = stft(
            signal, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2
        )
        magnitude_db = 20.0 * np.log10(np.abs(stft_matrix) + 1e-10)
        mesh = ax.pcolormesh(
            times, frequencies, magnitude_db, shading="gouraud", cmap=_SPECTROGRAM_CMAP
        )
        ax.set_title(title, fontsize=9)
        return mesh

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

        true_for_estimated, correlation_for_estimated = _match_true_to_estimated(
            model.sources_true_, model.sources_
        )

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4.5 * n_cols, 2.8 * n_rows), squeeze=False
        )
        mesh = None
        for col in range(n_cols):
            mesh = self._spectrogram(
                axes[0, col], model.mixtures_[col], sample_rate, f"mistura {col + 1}"
            )
            row = 1
            true_idx = true_for_estimated.get(col)
            if true_idx is not None:
                self._spectrogram(
                    axes[row, col],
                    model.sources_true_[true_idx],
                    sample_rate,
                    f"esperado {true_idx + 1}",
                )
                row += 1
            title = f"obtido {col + 1}"
            if true_idx is not None:
                title += f" (corr={correlation_for_estimated[col]:.2f})"
            mesh = self._spectrogram(axes[row, col], model.sources_[col], sample_rate, title)

        for col in range(n_cols):
            axes[n_rows - 1, col].set_xlabel("tempo (s)", fontsize=8)
        for row in range(n_rows):
            axes[row, 0].set_ylabel("frequência (Hz)", fontsize=8)

        fig.suptitle(self._metrics_caption(), fontsize=9)
        fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.6, label="magnitude (dB)")
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
