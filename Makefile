REQ = requirements.txt
MAIN = src

install:
	uv sync

run:
	uv run python $(MAIN)

debug:
	uv run python -m pdb $(MAIN)

clean:
	find . -name "__pycache__" -print -exec rm -rf {} +
	find . -name ".mypy_cache" -print -exec rm -rf {} +
	find . -name "*.pyc" -print -delete

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

.PHONY: install run debug clean lint lint-strict
