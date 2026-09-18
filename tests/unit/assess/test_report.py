"""Testes unitarios para ica.assess.report (agregacao da bateria assess/)."""


from ica.assess.report import assess_run


def test_assess_run_flags_viability_risk_when_all_channels_are_gaussian_like(rng):
    """Se todos os canais da mistura parecem gaussianos, viability_flag deve ser 'risco'."""
    X = rng.normal(size=(3, 20000))
    report = assess_run(mixtures=X, sources=X, domain="distribution")
    assert report.n_gaussian_like_channels == 3
    assert report.viability_flag == "risco"


def test_assess_run_flags_viability_ok_with_nongaussian_channels(rng, make_sources):
    """Com misturas claramente nao-gaussianas, viability_flag deve ser 'viavel'."""
    S = make_sources(["laplace", "uniform"], 20000, rng)
    A = rng.normal(size=(2, 2))
    X = A @ S
    report = assess_run(mixtures=X, sources=S, domain="distribution")
    assert report.viability_flag == "viavel"
    assert len(report.channels) == 2


def test_assess_run_computes_delay_only_for_audio(rng):
    """delay_lags deve ser None para dominios sem eixo temporal, preenchido para audio."""
    X = rng.normal(size=(2, 5000))
    report_dist = assess_run(mixtures=X, sources=X, domain="distribution")
    report_audio = assess_run(mixtures=X, sources=X, domain="audio", sample_rate=8000.0)
    assert report_dist.delay_lags is None
    assert report_audio.delay_lags is not None
    assert report_audio.delay_lags.shape == (2, 2)
