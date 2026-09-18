"""Testes unitarios para ica.assess.gaussianity (skill bss-assessment, Secao 1)."""


from ica.assess.gaussianity import gaussianity_battery, gaussianity_tests


def test_gaussian_channel_has_high_pvalues(rng):
    """Para uma amostra genuinamente gaussiana, os p-valores devem ser altos (nao rejeita)."""
    y = rng.normal(size=20000)
    result = gaussianity_tests(y)
    assert result.dagostino_pearson_pvalue > 0.01
    assert result.jarque_bera_pvalue > 0.01


def test_laplace_channel_has_low_pvalues(rng):
    """Para uma amostra claramente nao-gaussiana (Laplaciana), os testes devem rejeitar."""
    y = rng.laplace(size=20000)
    result = gaussianity_tests(y)
    assert result.dagostino_pearson_pvalue < 0.01
    assert result.jarque_bera_pvalue < 0.01
    assert result.anderson_darling_statistic > 1.0


def test_gaussianity_battery_returns_one_result_per_channel(rng):
    """gaussianity_battery deve devolver um resultado por linha (canal) de X."""
    X = rng.normal(size=(3, 5000))
    results = gaussianity_battery(X)
    assert len(results) == 3
