# Hugging Face Space (Docker SDK): serves the Streamlit dashboard on port 7860.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app/src \
    HF_HOME=/home/user/.cache/huggingface \
    SVLA_VECTOR_BACKEND=memory \
    SVLA_CHROMA_DIR=/home/user/app/.store \
    SVLA_LOGS_DIR=/home/user/app/examples/sample_logs \
    SVLA_LABELS_DIR=/home/user/app/examples/sample_labels \
    SVLA_REPORTS_DIR=/home/user/app/examples/reports
WORKDIR /home/user/app

COPY --chown=user requirements.txt ./
RUN pip install --user --upgrade pip && pip install --user -r requirements.txt

COPY --chown=user . .
RUN mkdir -p .store

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "7860", "--server.headless", "true"]
