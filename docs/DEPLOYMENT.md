# Deployment

## Hugging Face Spaces (live demo)

The dashboard runs as a Streamlit Space. `scripts/deploy_hf_space.py` stages the
files a Space needs (app, package source, bundled sample data/reports,
`requirements.txt`, Space README) and uploads them with `huggingface_hub`.

```bash
# one-time: hf auth login   (write token)
make deploy-hf                   # or: python scripts/deploy_hf_space.py --repo <user>/svla-agent
```

Space variables set by the script:

| Variable | Value | Why |
|---|---|---|
| `SVLA_VECTOR_BACKEND` | `memory` | no native sqlite/Chroma dependency on the Space |
| `SVLA_LOG_LEVEL` | `INFO` | |

No Anthropic key is stored on the Space. Visitors paste their own key in the
sidebar (session-only) to switch the hypothesis agent to Claude.

## Docker

```bash
make docker-build
make docker-api      # REST API  -> http://localhost:8000/docs
make docker-app      # dashboard -> http://localhost:8501
docker run --rm -v $PWD/my_logs:/logs svla-agent svla process --logs /logs
```

The image ships the sample dataset and reports under `/app/examples`; mount your
own logs and set `SVLA_LOGS_DIR` / `SVLA_REPORTS_DIR` to point elsewhere.

## REST API

```bash
svla serve --port 8000
curl localhost:8000/health
curl -X POST localhost:8000/analyze/000003
curl -X POST -F "files=@run_000003.pcie.log" -F "files=@run_000003.system.log" localhost:8000/analyze/upload
curl -H "X-Anthropic-Key: sk-ant-..." -X POST localhost:8000/analyze/000003   # Claude for this request
```

Interactive docs: `/docs` (Swagger) and `/redoc`.

## Streamlit Community Cloud (alternative)

Point share.streamlit.io at this repo with main file `app/streamlit_app.py` and
add `SVLA_VECTOR_BACKEND=memory` plus `SVLA_OFFLINE=1` as secrets if the 1 GB
tier cannot load the embedding model.
