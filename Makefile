.PHONY: setup lint typecheck test results report quick all

setup:
	uv sync --frozen

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

results:
	uv run python scripts/run_experiments.py

report:
	uv run python scripts/build_report.py --readme README.md

quick:
	uv run python scripts/run_experiments.py --seeds 4 --players 8 --budget 300 --out build/quick --card build/quick/card.json --allow-dirty
	uv run python scripts/build_report.py --results build/quick/results.json --out build/quick/index.html

all: lint typecheck test results report
