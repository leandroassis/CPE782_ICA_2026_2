"""Testes unitarios para ica.harness.store (cache retomavel por celula)."""

from ica.harness.store import ResumableStore, cell_key


def test_cell_key_is_deterministic():
    """A mesma celula deve sempre produzir a mesma chave."""
    key_a = cell_key("dist", "run1", "fastica_ml", "unico", sample_size=100)
    key_b = cell_key("dist", "run1", "fastica_ml", "unico", sample_size=100)
    assert key_a == key_b


def test_cell_key_differs_for_different_parameters():
    """Celulas com parametros diferentes nunca devem colidir."""
    key_a = cell_key("dist", "run1", "fastica_ml", "unico", sample_size=100)
    key_b = cell_key("dist", "run1", "fastica_ml", "unico", sample_size=1000)
    key_c = cell_key("dist", "run1", "natural_gradient", "unico", sample_size=100)
    assert len({key_a, key_b, key_c}) == 3


def test_store_has_is_false_before_save(tmp_path):
    """Sem nada salvo, has() deve ser False."""
    store = ResumableStore(cache_dir=tmp_path)
    assert not store.has("dist", "run1", "fastica_ml", "unico")


def test_store_save_and_load_round_trips(tmp_path):
    """Salvar e depois carregar deve devolver o mesmo objeto (por valor)."""
    store = ResumableStore(cache_dir=tmp_path)
    store.save("dist", "run1", "fastica_ml", "unico", result={"log_likelihood": -1.5})
    assert store.has("dist", "run1", "fastica_ml", "unico")
    assert store.load("dist", "run1", "fastica_ml", "unico") == {"log_likelihood": -1.5}


def test_get_or_compute_only_calls_compute_once(tmp_path):
    """Uma segunda chamada com os mesmos parametros deve ler do cache, sem recomputar."""
    store = ResumableStore(cache_dir=tmp_path)
    call_count = {"n": 0}

    def compute():
        call_count["n"] += 1
        return {"value": 42}

    first = store.get_or_compute("dist", "run1", "fastica_ml", "unico", compute)
    second = store.get_or_compute("dist", "run1", "fastica_ml", "unico", compute)

    assert first == {"value": 42}
    assert second == {"value": 42}
    assert call_count["n"] == 1
