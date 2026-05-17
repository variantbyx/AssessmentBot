Deployment Guide

This repository includes a basic `Dockerfile` and instructions to deploy the API to common hosts.

Render
------
- Create a new Web Service on Render.
- Connect the repository and set the build command to:

```bash
pip install -r requirements.txt
```

- Set the start command to:

```bash
uvicorn app:app --host 0.0.0.0 --port 10000
```

- Add environment variable: `USE_SEMANTIC_SEARCH=0` to avoid large model downloads.

Railway / Fly / Heroku
----------------------
- Railway: Use the Dockerfile or the Python template. Ensure `USE_SEMANTIC_SEARCH=0`.
- Fly: `flyctl launch` and set the Dockerfile; configure env vars.
- Heroku: Use container registry or Procfile.

HuggingFace Spaces
------------------
- Not recommended unless the Space has sufficient resources and the index/model artifacts are provided.

Notes
-----
- If you require semantic search in production, upload `shl_index.faiss` and ensure the environment has sufficient memory and the required model artifacts.
- For quick public submission, deploy with `USE_SEMANTIC_SEARCH=0` so the service uses BM25 only.
