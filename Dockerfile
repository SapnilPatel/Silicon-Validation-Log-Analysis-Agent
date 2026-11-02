# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SVLA_LOGS_DIR=/app/examples/sample_logs \
    SVLA_LABELS_DIR=/app/examples/sample_labels \
    SVLA_REPORTS_DIR=/app/examples/reports \
    SVLA_CHROMA_DIR=/data/store

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[app,api]"

COPY app ./app
COPY examples ./examples
RUN mkdir -p /data/store

# 8000: REST API (svla serve)   8501: Streamlit dashboard
EXPOSE 8000 8501

# Default: serve the REST API. Override, e.g.
#   docker run -p 8501:8501 svla-agent streamlit run app/streamlit_app.py --server.address 0.0.0.0
#   docker run -v $PWD/logs:/logs svla-agent svla process --logs /logs
ENTRYPOINT []
CMD ["svla", "serve", "--host", "0.0.0.0", "--port", "8000"]
