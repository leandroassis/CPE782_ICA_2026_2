venv := ".venv"
python := venv / "bin" / "python"
pip := venv / "bin" / "pip"

# Lista os comandos disponíveis
default:
    @just --list

# Cria o ambiente virtual (se necessário) e instala as dependências de desenvolvimento
install:
    test -d {{venv}} || python3 -m venv {{venv}}
    {{pip}} install --upgrade pip
    {{pip}} install -r requirements-dev.txt

# Roda a suíte de testes com relatório de cobertura
test:
    {{python}} -m pytest tests/ --cov=src/ica --cov-report=term-missing

# Executa o pipeline de ICA sobre um run (argumentos posicionais, ex:
# just run imagens run1 bell_sejnowski). sample_size so e usado quando sample=dist.
run sample="imagens" run="run1" algorithm="fastica_ml" sample_size="100000":
    {{python}} -m ica --sample {{sample}} --run {{run}} --algorithm {{algorithm}} \
        {{ if sample == "dist" { "--sample-size " + sample_size } else { "" } }}

# Executa o pipeline de ICA sobre todos os runs de um tipo de amostra
# (argumentos posicionais, ex: just run-all audio natural_gradient)
run-all sample="imagens" algorithm="fastica_ml" sample_size="100000":
    #!/usr/bin/env bash
    set -uo pipefail
    runs=$({{python}} -m ica --sample {{sample}} --list-runs)
    failed=""
    for run in $runs; do
        echo "=== {{sample}} / $run (algorithm={{algorithm}}) ==="
        if [ "{{sample}}" = "dist" ]; then
            {{python}} -m ica --sample {{sample}} --run "$run" --algorithm {{algorithm}} --sample-size {{sample_size}} || failed="$failed $run"
        else
            {{python}} -m ica --sample {{sample}} --run "$run" --algorithm {{algorithm}} || failed="$failed $run"
        fi
        echo ""
    done
    if [ -n "$failed" ]; then
        echo "Runs que falharam:$failed"
        exit 1
    fi

# Gera a documentação HTML a partir dos docstrings (NumPy style) em docs/
docs:
    {{python}} -m pdoc src/ica --docformat numpy --output-directory docs

# Verifica estilo e problemas estáticos do código
lint:
    {{python}} -m ruff check src/ tests/

# Remove caches, artefatos de build, cobertura, documentação e saídas geradas
clean:
    rm -rf docs output .coverage .pytest_cache .ruff_cache htmlcov build dist *.egg-info
    find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
