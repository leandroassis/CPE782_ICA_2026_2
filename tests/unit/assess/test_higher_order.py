"""Testes unitarios para ica.assess.higher_order (skill bss-assessment, Secao 2)."""


from ica.assess.higher_order import higher_order_battery, higher_order_statistics


def test_laplace_has_positive_excess_kurtosis(rng):
    """Laplaciana e supergaussiana: curtose excedente > 0."""
    y = rng.laplace(size=20000)
    result = higher_order_statistics(y)
    assert result.excess_kurtosis > 0


def test_uniform_has_negative_excess_kurtosis(rng):
    """Uniforme e subgaussiana: curtose excedente < 0 (proximo de -1.2)."""
    y = rng.uniform(-1, 1, size=20000)
    result = higher_order_statistics(y)
    assert result.excess_kurtosis < 0
    assert abs(result.excess_kurtosis - (-1.2)) < 0.2


def test_gaussian_has_near_zero_negentropy(rng):
    """Uma amostra gaussiana deve ter negentropia proxima de 0 (nas duas aproximacoes)."""
    y = rng.normal(size=50000)
    result = higher_order_statistics(y)
    assert result.negentropy_cumulant < 0.02
    assert result.negentropy_contrast < 0.02


def test_laplace_has_higher_negentropy_than_gaussian(rng):
    """Negentropia deve ser maior para uma fonte claramente nao-gaussiana."""
    gaussian = higher_order_statistics(rng.normal(size=50000))
    laplace = higher_order_statistics(rng.laplace(size=50000))
    assert laplace.negentropy_contrast > gaussian.negentropy_contrast


def test_higher_order_battery_returns_one_result_per_channel(rng):
    """higher_order_battery deve devolver um resultado por linha (canal) de X."""
    X = rng.normal(size=(3, 5000))
    results = higher_order_battery(X)
    assert len(results) == 3
