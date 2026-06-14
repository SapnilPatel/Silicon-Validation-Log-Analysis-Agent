.PHONY: help install dev-install data process index eval app test lint format typecheck check clean

PY ?= .svla/bin/python
PIP ?= .svla/bin/pip

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	$(PIP) install -e .

dev-install: ## Install with dev + app extras
	$(PIP) install -e ".[dev,app]"

data: ## Generate synthetic silicon logs (RUNS=100)
	$(PY) -m svla_agent.cli generate --runs $(or $(RUNS),100)

process: ## Run the full analysis pipeline over all runs
	$(PY) -m svla_agent.cli process

index: ## Index labelled historical failures into the vector store
	$(PY) -m svla_agent.cli index

eval: ## Evaluate pipeline output against ground-truth labels
	$(PY) -m svla_agent.cli eval

app: ## Launch the Streamlit dashboard
	.svla/bin/streamlit run app/streamlit_app.py

test: ## Run the test suite
	$(PY) -m pytest

lint: ## Lint with ruff
	$(PY) -m ruff check .

format: ## Auto-format with ruff
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

typecheck: ## Type-check with mypy
	$(PY) -m mypy

check: lint typecheck test ## Run lint + typecheck + tests

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
