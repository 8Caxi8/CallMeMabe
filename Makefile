MAIN = src
USER = dansimoe
HF_HOME=/sgoinfre/$(USER)/.cache/huggingface

ifeq ($(wildcard /sgoinfre/$(USER)),/sgoinfre/$(USER))
    export UV_CACHE_DIR := /sgoinfre/$(USER)/.uv-cache
    export HF_HOME := /sgoinfre/$(USER)/.cache/huggingface
    export TRANSFORMERS_CACHE := /sgoinfre/$(USER)/.cache/huggingface
endif

install:
	uv sync

run:
	export HF_HOME=$(HOMW)
	mkdir -p $HF_HOME
	clear
	uv run python -m $(MAIN)

debug:
	uv run python -m pdb -m $(MAIN)

clean:
	find . -name "__pycache__" -print -exec rm -rf {} +
	find . -name ".mypy_cache" -print -exec rm -rf {} +
	find . -name "*.pyc" -print -delete

lint:
	uv run flake8 --exclude=.venv,llm_sdk .
	uv run mypy . --exclude '\.venv|llm_sdk' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 --exclude=.venv,llm_sdk .
	uv run mypy . --exclude '\.venv|llm_sdk' --strict

.PHONY: install run debug clean lint lint-strict
