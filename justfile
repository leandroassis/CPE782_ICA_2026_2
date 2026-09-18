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

# Executa a grade de ICA-ML (3 algoritmos x modos de condicionamento
# aplicaveis) sobre um run e mostra a melhor separacao (argumentos
# posicionais, ex: just run imagens run1). algorithm="" roda a grade
# completa; informe um algoritmo (ex.: bell_sejnowski) para rodar so ele.
# sample_size so e usado quando sample=dist.
run sample="imagens" run="run1" algorithm="" sample_size="100000":
    {{python}} -m ica --sample {{sample}} --run {{run}} \
        {{ if algorithm != "" { "--algorithm " + algorithm } else { "" } }} \
        {{ if sample == "dist" { "--sample-size " + sample_size } else { "" } }}

# Executa a grade de ICA-ML sobre todos os runs de um tipo de amostra
# (argumentos posicionais, ex: just run-all audio). Ver `run` para algorithm.
run-all sample="imagens" algorithm="" sample_size="100000":
    #!/usr/bin/env bash
    set -uo pipefail
    runs=$({{python}} -m ica --sample {{sample}} --list-runs)
    algorithm_flag=""
    if [ -n "{{algorithm}}" ]; then
        algorithm_flag="--algorithm {{algorithm}}"
    fi
    failed=""
    for run in $runs; do
        echo "=== {{sample}} / $run (algorithm={{algorithm}}) ==="
        if [ "{{sample}}" = "dist" ]; then
            {{python}} -m ica --sample {{sample}} --run "$run" $algorithm_flag --sample-size {{sample_size}} || failed="$failed $run"
        else
            {{python}} -m ica --sample {{sample}} --run "$run" $algorithm_flag || failed="$failed $run"
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
