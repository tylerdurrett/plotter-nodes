.PHONY: serve test lint fmt

serve:
	uv run python scripts/run_pipeline.py serve

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

fmt:
	uv run ruff format src/ tests/
