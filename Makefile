.PHONY: help install dev-install data process index eval app serve sample test lint format typecheck check docker-build docker-api docker-app deploy-hf clean

# Use the project venv when present, else whatever python is on PATH.
PY  ?= $(if $(wildcard .svla/bin/python),.svla/bin/python,python)
PIP ?= $(PY) -m pip

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	$(PIP) install -e .

dev-install: ## Install with dev + app + api extras
	$(PIP) install -e ".[dev,app,api]"

data: ## Generate synthetic silicon logs (RUNS=100)
	$(PY) -m svla_agent.cli generate --runs $(or $(RUNS),100)

process: ## Run the full analysis pipeline over all runs
	$(PY) -m svla_agent.cli process

index: ## Index labelled historical failures into the vector store
	$(PY) -m svla_agent.cli index

eval: ## Evaluate pipeline output against ground-truth labels
	$(PY) -m svla_agent.cli eval --out reports/metrics.json

sample: ## Analyse the bundled 24-run sample dataset into examples/reports
	SVLA_REPORTS_DIR=examples/reports SVLA_CHROMA_DIR=.chroma_examples \
	  $(PY) -m svla_agent.cli process --logs examples/sample_logs --labels examples/sample_labels

app: ## Launch the Streamlit dashboard
	$(PY) -m streamlit run app/streamlit_app.py

serve: ## Serve the REST API on :8000
	$(PY) -m svla_agent.cli serve

test: ## Run the test suite
	SVLA_OFFLINE=1 $(PY) -m pytest

lint: ## Lint with ruff (+ format check)
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format: ## Auto-format with ruff
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

typecheck: ## Type-check with mypy
	$(PY) -m mypy

check: lint typecheck test ## Run lint + typecheck + tests

docker-build: ## Build the Docker image
	docker build -t svla-agent .

docker-api: docker-build ## Run the REST API in Docker on :8000
	docker run --rm -p 8000:8000 svla-agent

docker-app: docker-build ## Run the dashboard in Docker on :8501
	docker run --rm -p 8501:8501 svla-agent streamlit run app/streamlit_app.py --server.address 0.0.0.0

deploy-hf: ## Deploy the dashboard to Hugging Face Spaces (needs HF token)
	$(PY) scripts/deploy_hf_space.py

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info htmlcov .coverage .chroma_ci .chroma_examples
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
