# Deployment

## Hugging Face Spaces

> Hugging Face requires a **PRO subscription** to host Docker/Gradio Spaces on the
> free CPU tier (static Spaces are the only free kind), and new Streamlit-SDK
> Spaces can no longer be created. With a PRO account the script below deploys
> in one command.

The dashboard runs as a **Docker Space** (`deploy/hf/Dockerfile`, Streamlit on
port 7860; Hugging Face no longer accepts new Streamlit-SDK Spaces).
`scripts/deploy_hf_space.py` stages the files a Space needs (Dockerfile, app,
package source, bundled sample data/reports, `requirements.txt`, Space README)
and uploads them with `huggingface_hub`.

```bash
# one-time: hf auth login   (write token)
make deploy-hf                   # or: python scripts/deploy_hf_space.py --repo <user>/svla-agent
```

Environment baked into the Space image: `SVLA_VECTOR_BACKEND=memory` (no native
Chroma dependency), paths pointing at the bundled `examples/`, a writable store
dir and HF cache under the non-root user.

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

## Streamlit Community Cloud (free, recommended for the public demo)

The public instance lives at **https://svla-agent.streamlit.app/**.

1. Sign in at https://share.streamlit.io with the GitHub account that owns the repo.
2. **Create app** → *Deploy a public app from GitHub*:
   - Repository: `SapnilPatel/Silicon-Validation-Log-Analysis-Agent`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
   - App URL (subdomain): `svla-agent` (or any free name)
3. **Advanced settings**:
   - Python version: `3.12`
   - Secrets (they are exported as environment variables):

     ```toml
     SVLA_VECTOR_BACKEND = "memory"
     SVLA_LOG_LEVEL = "INFO"
     ```

     Optional: `SVLA_OFFLINE = "1"` forces the lightweight hashing embeddings if the
     sentence-transformers model does not fit the instance; the app falls back to
     them automatically if the model cannot load. The dashboard defaults to the
     in-memory vector store anyway, and falls back to it if Chroma's native
     bindings are unavailable (they are not on Python 3.14), so the app also runs
     if you forget the secrets or the Python version.
4. **Deploy**. The first build installs `requirements.txt` (a few minutes); the app
   then serves the bundled `examples/` dataset, reports and 400-run metrics, with
   live analysis and per-visitor Claude keys working as in the local app.

Every push to `main` redeploys automatically.
