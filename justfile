export ENV := "test"
git_ref := `git rev-parse --abbrev-ref HEAD`

default:
    @just --list

format:
    uv run ruff check --fix
    uv run ruff format

lint:
    uv run ruff check

typecheck:
    uv run pyright

test *FLAGS: format lint typecheck
    uv run pytest {{FLAGS}}

test-coverage +REPORT='term':
    just test --cov=audioset_classification --cov-report={{REPORT}}
