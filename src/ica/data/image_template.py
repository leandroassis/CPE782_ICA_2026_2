"""Carregador de amostras de imagens (DataTemplate).

Trata tambem os 3 modos de condicionamento de imagens coloridas do bloco
``harness/`` (``.claude/PIPELINE_MAP.md``, "Modos de condicionamento"):
**A** (grayscale, uma unica mistura por imagem), **B** (ICA por plano RGB,
resolve 3 problemas menores k x k, um por canal) e **C** (imagem RGB inteira
achatada num vetor 3P, um solve k x k conjunto). Para imagens em escala de
cinza, B e C coincidem (usa-se so A).
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ica.data.base import DataTemplate
from ica.interfaces import SignalMatrix

_GRAYSCALE_FILENAME = "mix_imagens_grayscale.csv"
_RGB_FILENAME = "mix_imagens_rgb.csv"
_SHUFFLED_FILENAME = "mix_imagens_shuffled.csv"
_MIX_FILENAMES = (_GRAYSCALE_FILENAME, _RGB_FILENAME, _SHUFFLED_FILENAME)
_N_CHANNELS = 3


class ImageTemplate(DataTemplate):
    """Carrega misturas de imagens (escala de cinza ou RGB) serializadas como CSV.

    O CSV de cada run tem uma coluna por mistura (``misturaN``) e uma
    linha por pixel; :meth:`reconstruct` reverte essa serializacao para
    uma forma visualizavel (Protocol ``Reconstructable``, ver
    ``ica.interfaces``). Para RGB, a convencao (confirmada contra o
    cabecalho real de ``data/groundtruth/imagens/run3/sources_imagens_rgb.csv``,
    ``Cachorro_R,Cachorro_G,Cachorro_B,Gato_R,...``) e o agrupamento
    **consecutivo por imagem**: as colunas ``3*i, 3*i+1, 3*i+2`` sao os
    canais R, G, B da i-esima imagem.

    Parameters
    ----------
    run : str
        Identificador do run (ex.: ``"run1"``).
    data_root : pathlib.Path
        Diretorio raiz das amostras de imagem (``data/mix/imagens``).
    height : int, optional
        Altura da imagem em pixels. Se omitido, inferida como
        ``sqrt(n_pixels)`` (assumindo imagem quadrada) durante
        :meth:`load`.
    width : int, optional
        Largura da imagem em pixels. Se omitido, inferida como
        ``sqrt(n_pixels)`` durante :meth:`load`.

    Attributes
    ----------
    is_rgb_ : bool
        Se este run e uma amostra RGB (9 misturas = 3 imagens x 3
        canais) ou em escala de cinza.
    height_ : int or None
        Altura inferida ou informada; definida apos :meth:`load` quando
        nao informada no construtor.
    width_ : int or None
        Largura inferida ou informada; definida apos :meth:`load` quando
        nao informada no construtor.
    """

    def __init__(
        self,
        run: str,
        data_root: Path,
        height: int | None = None,
        width: int | None = None,
    ) -> None:
        super().__init__(run=run, data_root=data_root)
        self._height_override = height
        self._width_override = width
        self.height_: int | None = height
        self.width_: int | None = width
        self.is_rgb_ = self._csv_path().name in (_RGB_FILENAME, _SHUFFLED_FILENAME)

    def _csv_path(self) -> Path:
        run_dir = self.data_root / self.run
        for filename in (_RGB_FILENAME, _SHUFFLED_FILENAME):
            candidate = run_dir / filename
            if candidate.exists():
                return candidate
        return run_dir / _GRAYSCALE_FILENAME

    def _load_raw(self) -> np.ndarray:
        """Le o CSV de misturas e devolve X cru, shape ``(n_misturas, n_pixels)``."""
        pixels_by_mixture = pd.read_csv(self._csv_path()).to_numpy(dtype=np.float64)
        n_pixels = pixels_by_mixture.shape[0]
        if self._height_override is None or self._width_override is None:
            side = int(round(math.sqrt(n_pixels)))
            if side * side != n_pixels:
                raise ValueError(
                    f"Numero de pixels ({n_pixels}) nao e um quadrado perfeito; "
                    "informe height/width explicitamente."
                )
            self.height_, self.width_ = side, side
        return pixels_by_mixture.T

    def load(self) -> SignalMatrix:
        """Le o CSV de misturas (modo A: uma mistura por imagem/plano, sem reagrupar).

        Para escala de cinza, e a unica representacao possivel. Para RGB,
        e a base sobre a qual :meth:`mode_b_plane_matrices` e
        :meth:`mode_c_matrix` constroem os modos B e C
        (``.claude/PIPELINE_MAP.md``).

        Returns
        -------
        SignalMatrix
            ``data`` shape ``(n_misturas, n_pixels)``, ``domain="image"``,
            ``meta={"height", "width", "is_rgb", "n_images"}``.

        Raises
        ------
        ValueError
            Se ``height``/``width`` nao foram informados no construtor e
            o numero de pixels nao for um quadrado perfeito.
        """
        X = self._load_raw()
        n_images = X.shape[0] // _N_CHANNELS if self.is_rgb_ else X.shape[0]
        return SignalMatrix(
            data=X,
            domain="image",
            meta={
                "height": self.height_,
                "width": self.width_,
                "is_rgb": self.is_rgb_,
                "n_images": n_images,
            },
        )

    def mode_b_plane_matrices(self) -> list[np.ndarray]:
        """Constroi as 3 matrizes menores k x k do modo de condicionamento B.

        Uma matriz por canal (R, G, B), cada uma reunindo o mesmo canal das
        ``k`` imagens-mistura -- resolvida por uma ICA k x k independente;
        os planos recuperados sao reagrupados depois por correlacao
        espacial (``ica.postprocessing.regroup``).

        Returns
        -------
        list of np.ndarray
            3 matrizes, cada uma shape ``(n_images, n_pixels)``, na ordem
            R, G, B.

        Raises
        ------
        ValueError
            Se esta amostra nao for RGB (modo B so se aplica a cor; para
            escala de cinza, B coincide com A).
        """
        if not self.is_rgb_:
            raise ValueError("mode_b_plane_matrices() so se aplica a imagens RGB.")
        X = self._load_raw()
        return [X[channel::_N_CHANNELS, :] for channel in range(_N_CHANNELS)]

    def mode_c_matrix(self) -> np.ndarray:
        """Constroi a matriz k x k do modo de condicionamento C (RGB achatado em 3P).

        Cada imagem-mistura vira uma unica linha de comprimento ``3 *
        n_pixels`` (canais R, G, B concatenados) -- um unico solve k x k
        conjunto, sem precisar reagrupar depois.

        Returns
        -------
        np.ndarray
            Shape ``(n_images, 3 * n_pixels)``.

        Raises
        ------
        ValueError
            Se esta amostra nao for RGB.
        """
        if not self.is_rgb_:
            raise ValueError("mode_c_matrix() so se aplica a imagens RGB.")
        X = self._load_raw()
        n_images = X.shape[0] // _N_CHANNELS
        return X.reshape(n_images, _N_CHANNELS * X.shape[1])

    @property
    def n_mixtures(self) -> int:
        """Numero de colunas ``misturaN`` no CSV deste run.

        Returns
        -------
        int
            Numero de misturas.
        """
        header = pd.read_csv(self._csv_path(), nrows=0)
        return len(header.columns)

    def reconstruct(self, source_vector: np.ndarray) -> np.ndarray:
        """Reformata um vetor de pixels de volta em uma imagem 2D.

        Parameters
        ----------
        source_vector : np.ndarray
            Vetor 1D de forma ``(n_pixels,)``.

        Returns
        -------
        np.ndarray
            Imagem 2D, shape ``(height_, width_)``.

        Raises
        ------
        RuntimeError
            Se chamado antes de :meth:`load` (quando ``height``/``width``
            nao foram informados no construtor).
        """
        if self.height_ is None or self.width_ is None:
            raise RuntimeError(
                "height_/width_ ainda nao definidos; chame load() antes de reconstruct()."
            )
        return source_vector.reshape(self.height_, self.width_)

    def reconstruct_rgb_triplet(self, source_vectors: list[np.ndarray]) -> np.ndarray:
        """Compoe 3 vetores de fonte recuperada em um painel RGB ``(H, W, 3)``.

        Assume que os 3 vetores correspondem, na ordem informada, aos
        canais R, G e B de uma mesma imagem original -- convencao de
        agrupamento consecutivo por imagem, confirmada contra o cabecalho
        real de ``data/groundtruth/imagens/run3/sources_imagens_rgb.csv``.

        Parameters
        ----------
        source_vectors : list of np.ndarray
            Exatamente 3 vetores 1D de forma ``(n_pixels,)``, um por
            canal (R, G, B).

        Returns
        -------
        np.ndarray
            Imagem RGB, shape ``(height_, width_, 3)``, normalizada por
            canal para ``[0, 1]`` para fins de visualizacao.

        Raises
        ------
        ValueError
            Se ``source_vectors`` nao tiver exatamente 3 elementos.
        """
        if len(source_vectors) != 3:
            raise ValueError("reconstruct_rgb_triplet espera exatamente 3 vetores (R, G, B).")
        channels = [self.reconstruct(vector) for vector in source_vectors]
        rgb = np.stack(channels, axis=-1)
        channel_min = rgb.min(axis=(0, 1), keepdims=True)
        channel_max = rgb.max(axis=(0, 1), keepdims=True)
        span = np.where(channel_max > channel_min, channel_max - channel_min, 1.0)
        return (rgb - channel_min) / span

    def load_ground_truth(
        self, groundtruth_root: Path
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Carrega ``A`` verdadeira e as fontes-imagem verdadeiras deste run.

        Descobre os arquivos por glob (``mix_matrix*.csv``,
        ``sources_imagens*.csv``), pois o sufixo varia por variante
        (``grayscale``/``rgb``/``shuffled``).

        Parameters
        ----------
        groundtruth_root : pathlib.Path
            Diretorio raiz do gabarito de imagens (``data/groundtruth/imagens``).

        Returns
        -------
        tuple of (np.ndarray or None, np.ndarray or None)
            ``(mixing_matrix_true, sources_true)``, ``sources_true`` shape
            ``(n_misturas, n_pixels)`` (mesma convencao de :meth:`load`).
            ``None`` no lugar do que faltar.
        """
        run_dir = Path(groundtruth_root) / self.run
        if not run_dir.exists():
            return None, None

        mixing_matrix_true = None
        mix_matrix_paths = sorted(run_dir.glob("mix_matrix*.csv"))
        if mix_matrix_paths:
            mixing_matrix_true = pd.read_csv(mix_matrix_paths[0]).to_numpy(dtype=np.float64)

        sources_true = None
        sources_paths = sorted(run_dir.glob("sources_imagens*.csv"))
        if sources_paths:
            sources_true = pd.read_csv(sources_paths[0]).to_numpy(dtype=np.float64).T

        return mixing_matrix_true, sources_true

    @classmethod
    def discover_runs(cls, data_root: Path) -> list[str]:
        """Lista os runs de imagem disponiveis (com CSV de mistura grayscale, RGB ou shuffled).

        Parameters
        ----------
        data_root : pathlib.Path
            Diretorio raiz das amostras de imagem (``data/mix/imagens``).

        Returns
        -------
        list of str
            Identificadores de run disponiveis, ordenados.
        """
        return cls._discover_runs_with_matching_file(
            data_root, lambda f: f.name in _MIX_FILENAMES
        )
