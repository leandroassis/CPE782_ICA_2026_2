"""Teste de integracao: pipeline completo sobre arquivos .wav sinteticos.

Usa arquivos no formato real de ``data/audio/`` (mono, PCM 16-bit, um
arquivo ``mixture_N.wav`` por microfone), nunca os dados reais do
trabalho.
"""

import numpy as np
from scipy.io import wavfile

from ica.algorithms.fastica_ml import FastICAML
from ica.data.audio_template import AudioTemplate
from ica.model import ICAModel
from ica.nonlinearities.adaptive import AdaptiveScore
from ica.preprocessing.centering import Centering
from ica.preprocessing.pipeline import Pipeline
from ica.preprocessing.whitening import Whitening


def test_audio_wav_pipeline_recovers_sources_and_exports_wav(
    tmp_path, rng, best_match_correlation
):
    """WAVs sinteticos -> AudioTemplate -> ICAModel deve separar bem e exportar .wav validos."""
    n_samples = 4000
    sample_rate = 8000

    source_1 = rng.laplace(size=n_samples)
    source_1 = (source_1 - source_1.mean()) / source_1.std()
    source_2 = rng.uniform(-1, 1, size=n_samples)
    source_2 = (source_2 - source_2.mean()) / source_2.std()
    S = np.vstack([source_1, source_2])

    A = rng.normal(size=(2, 2))
    while np.linalg.cond(A) > 10:
        A = rng.normal(size=(2, 2))
    X = A @ S

    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    peak = np.max(np.abs(X))
    for i in range(2):
        pcm = ((X[i] / peak) * np.iinfo(np.int16).max * 0.9).astype(np.int16)
        wavfile.write(run_dir / f"mixture_{i + 1}.wav", sample_rate, pcm)

    data = AudioTemplate(run="run1", data_root=tmp_path)
    model = ICAModel(
        data=data,
        pipeline=Pipeline([Centering(), Whitening()]),
        algorithm=FastICAML(nonlinearity=AdaptiveScore(), max_iterations=200),
    )
    model.fit()

    assert data.sample_rate_ == sample_rate
    assert best_match_correlation(S, model.sources_) > 0.9

    output_path = tmp_path / "fonte_recuperada_1.wav"
    data.export(model.sources_[0], output_path)
    exported_rate, exported_signal = wavfile.read(output_path)
    assert exported_rate == sample_rate
    assert exported_signal.shape[0] == n_samples
