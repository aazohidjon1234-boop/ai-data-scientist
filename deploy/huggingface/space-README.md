---
title: AI Data Scientist API
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# AI Data Scientist — backend API

FastAPI backend for the AI Data Scientist Agent: upload a CSV and it profiles,
cleans, analyses and models the data with pandas / scikit-learn, then explains
the results. Every number is computed in Python — the language model only
interprets values that already exist in the results.

Interactive API docs: **`/docs`** · health check: **`/api/health`**

The web interface for this API is deployed separately (Vercel).

## Configuration

Set these as **Space secrets** (Settings → Variables and secrets):

| Name | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | recommended | External Postgres. Without it a SQLite file is used and **is wiped on every restart**. |
| `LLM_API_KEY` | optional | Enables LLM interpretation. Without it the deterministic local engine is used. |
| `LLM_BASE_URL` | optional | e.g. `https://api.groq.com/openai/v1` |
| `LLM_MODEL` | optional | e.g. `openai/gpt-oss-120b` |
| `CORS_ORIGINS` | optional | Your frontend origin, if it calls the API directly. |

Storage on the free tier is ephemeral: uploaded files and generated reports do
not survive a restart or rebuild.
