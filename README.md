# 🤖 AI Data Scientist Agent

> Upload a CSV. The agent profiles it, detects the problem, cleans the data, trains and compares real machine-learning models, answers ad-hoc questions about the data, and explains everything in plain language.

**A full-stack web application where a tool-calling AI agent orchestrates a real Python data-science pipeline — FastAPI · scikit-learn · Plotly · PostgreSQL · Next.js.**

---

## Overview

Most "AI data analyst" demos either mock the results or let the language model do the math. This project takes the opposite, professional approach:

- **All calculations are done by Python tools** (pandas, NumPy, scikit-learn, Plotly). Correlations, imputation, train/test splits, metrics, silhouettes — every number is computed for real.
- **The AI agent is an orchestrator and an explainer.** It inspects the data, builds a plan, calls tools (`analyze_dataset()`, `detect_missing_values()`, `clean_dataset()`, `train_model()`, …), observes the results, and then explains what happened.
- **The agent can never invent a metric.** Explanations are generated from the actual result JSON. With no LLM configured, a deterministic local explanation engine produces the narrative; with an OpenAI-compatible `LLM_API_KEY`, the model is given the same JSON under a strict "only cite numbers that exist in the data" system prompt — and the local engine remains the guaranteed fallback.

The result is a serious, end-to-end working product you can put in front of a real dataset:

1. **Upload a CSV** (or load a bundled example: housing regression, churn classification, customer clustering).
2. The agent **profiles** the data — shape, dtypes, missing values, duplicates, statistics, correlations, outliers.
3. It **detects the target and the task**: regression, classification, or clustering. If it can't tell, the UI asks you to pick.
4. It **cleans and prepares** the data — imputation, dedup, one-hot encoding, scaling, dropping unusable columns, stratified train/test split.
5. It **trains a full model zoo** — 11 regression models (Linear, Ridge, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, KNN, SVR), 11 classification models (Logistic, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, SVM, KNN, Naive Bayes, AdaBoost), or a cluster zoo (K-Means, Agglomerative/Ward, DBSCAN) — and evaluates everything on held-out data.
6. It **compares them in a ranked table + charts**, picks the best by the appropriate primary metric (R², macro-F1, silhouette), and reports feature importances.
7. It **explains everything** in a chat-style panel — and writes a full downloadable Markdown report.

## Features

| Area | Details |
|---|---|
| 📤 Ingestion | CSV upload (up to 100 MB / 1M rows) with type/size validation, comma/semicolon/tab/pipe separators detected on every read, not only on upload, safe UUID storage, bundled example datasets |
| 🔍 EDA | Profile, per-column stats (mean/median/std/quartiles), missing-value audit, duplicate detection, IQR outliers, Pearson correlation matrix |
| 🎯 Task detection | Heuristic target-column detection with scored candidates; auto-falls back to clustering; user override in the UI |
| ✨ Column advice | Ask the agent which inputs to use: every column is screened (identifier, mostly empty, too many categories) and scored against the target with mutual information — then the recommendation is **cross-validated against the full set and both scores reported**, so "this improves accuracy" is a measurement, not a claim |
| 🎚️ Run configuration | Before training, pick the **target (y)** and tick exactly which **input columns (X)** the models may use — with each column's type and missing-value share shown. The selection persists across re-runs — the page restores what the last run actually used instead of resetting to every column. The target can never be selected as its own feature |
| 🧹 Cleaning | Drop fully-empty columns, dedupe rows, and fill gaps with a **choice of strategy** — median (default, resists outliers), mean or zero for numbers; most-common or an explicit `missing` label for text. Offered in the UI only for columns that actually have gaps, and testable in the improve panel so the choice is measured rather than guessed. Every operation is logged |
| 🧪 ML pipeline | Automatic feature prep (imputation → encode → scale → split, stratified for classification), per-model error isolation |
| 📈 Models | Regression (11): Linear, Ridge, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, **XGBoost**, **LightGBM**, KNN, SVR · Classification (11): Logistic, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, **XGBoost**, **LightGBM**, SVM, KNN, Naive Bayes, AdaBoost · Clustering: K-Means k-search, Agglomerative (Ward), DBSCAN eps-search — all scored by silhouette |
| 📊 Metrics | MAE/MSE/RMSE/R² · Accuracy/Precision/Recall/F1 + confusion matrix · Silhouette/Inertia/cluster sizes — all from held-out test data |
| 📉 Visualizations | Plotly histograms, box plots, correlation heatmap, target scatter pairs, category bars, missing-value chart, target distribution |
| ⏱️ Live progress | A horizontal stepper shows the stage the run is on while it works — named steps, per-step seconds, a real progress bar and the last observation — instead of a spinner that hides a minute of computation |
| 🧾 Numeric text | Columns like `€110.5M`, `5'7`, `159lbs` and FIFA's `88+2` ratings are parsed into numbers during cleaning, so they are no longer mistaken for high-cardinality text and discarded |
| 🤖 Agent UX | Live tool-call timeline (the agent's actual activity), plan display, progress indicator, chat with grounded answers |
| 🎯 Improve the score | Turns "remove the outliers and it'll be better" into a measurement: each option (drop weak columns, drop 1.5×IQR outlier rows, both) is cross-validated against the untouched baseline and the deltas are shown. Options that make things **worse** are reported as worse; **Improve automatically** tests everything and retrains on the winner in one click (or says plainly that nothing helped); per-option Apply is still there. Sits directly under the model table |
| 🎛️ Hyperparameter tuning | A randomised search (12 combinations, 3-fold) over a grid centred on each model's defaults — run only on the model that won, since searching all 22 would cost more than the pipeline. The tuned model is kept **only if it also beats the defaults on the held-out split**, so a result that merely pleased the search folds is discarded |
| 📊 Dashboard | An **auto-composed** dashboard for any CSV: the schema decides what becomes a measure, a dimension or the timeline. KPI tiles, a trend line, breakdown bars, a distribution, a share donut and a cross-tab heatmap — all recomputed from pandas whenever you click a filter chip |
| ❓ Ask the data | Natural-language questions answered by a **validated query spec** (filter / group / aggregate / sort), executed by pandas — never generated code. "Which region has the highest average revenue?" returns a sentence, a table and a chart |
| 💡 Auto-insights | Unprompted ranked findings: strong correlations, dominant categories, significant segment gaps, trends, anomalies, skew and quality risks — each scored by effect size |
| 📅 Time series | Date-column detection, auto-resampling, trend with R²/p-value, period-over-period growth, moving average, month/weekday seasonality, residual-based anomaly flags |
| 🔬 Segment testing | Group comparison with the right test — Welch t-test, one-way ANOVA, chi-square — plus a rank-based robustness check and effect size (Cohen's d, eta², Cramér's V), so a gap is reported as real or as noise |
| 🔢 Pivot tables | Cross-tab any two dimensions against a measure |
| 📄 Reports | One-click Markdown report (overview → quality → stats → correlations → model table → AI interpretation) with download |
| 📦 Model artifacts | Every trained model is saved as a joblib `.pkl` and downloadable per-model (⬇ in the table), plus a `metadata.json` with features, class labels and preprocessing notes for reloading in Python |
| 💾 Persistence | PostgreSQL via SQLAlchemy: datasets, analyses, model results, reports (SQLite fallback for zero-setup local dev) |
| 🌗 UI | Responsive Next.js dashboard, sidebar nav, cards/tables, dark/light mode, empty/loading/error states |
| 🔒 Security | File-type + size limits, sanitized UUID file storage, no code execution from uploads, env-based secrets, configurable CORS, server-side API proxy (no keys in the frontend) |

## Screenshots

| Dashboard | Analysis | Model comparison |
|---|---|---|
| ![Dashboard](docs/screenshots/dashboard.png) | ![Analysis](docs/screenshots/analysis.png) | ![Models](docs/screenshots/models.png) |

*(Add your own screenshots to `docs/screenshots/` — the app runs locally in under a minute.)*

## Architecture

```
┌──────────────────────────────  Browser  ──────────────────────────────┐
│  Next.js 14 (TypeScript, Tailwind) — dashboard, tables, Plotly charts │
└───────────────┬───────────────────────────────────────────────────────┘
                │  /api/* (same-origin; no CORS, no secrets in the client)
┌───────────────▼──────────────────── Next.js server-side proxy ────────┐
│                    rewrite: /api/:path* → BACKEND_URL/api/:path*      │
└───────────────┬───────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────── FastAPI backend ──────────────────┐
│  API layer (Pydantic schemas)                                         │
│  Services: dataset · analysis · ML · report · chat                    │
│  ┌──────────────────────── AI Agent (DataScientistAgent) ─────────┐   │
│  │  plan from observations → call tools → record trace → explain  │   │
│  └───────────────────────────────┬────────────────────────────────┘   │
│        tools/  (the only place math happens)                          │
│  analyze_dataset · detect_missing_values · clean_dataset ·            │
│  detect_outliers · generate_statistics · generate_correlation_matrix· │
│  create_visualization (Plotly) · detect_problem_type · prepare_       │
│  features · train_model · evaluate_model · compare_models ·           │
│  generate_report                                                       │
│        pandas · NumPy · scikit-learn · Plotly                         │
└───────────────┬──────────────────────────────┬────────────────────────┘
                │ SQLAlchemy 2.0               │ files
        ┌───────▼────────┐             ┌───────▼──────────┐
        │  PostgreSQL 16 │             │ datasets/uploads │
        │ (SQLite for    │             │ datasets/samples │
        │  local dev)    │             │ reports/*.md     │
        └────────────────┘             └──────────────────┘
```

### How the AI agent works

The agent (`app/agents/pipeline_agent.py`) behaves like a data scientist:

1. **Plan** — it builds a step plan from observations (e.g. it notes missing values before training, and adapts when the target changes).
2. **Tool calls** — each step calls a Python tool from `app/tools/`. Tool calls are recorded in a **trace** (tool, args, status, observation, duration) that the UI renders as an activity timeline.
3. **Adaptation** — if the data is already clean the cleaning step reports so; if a model fails, the run continues and the failure is surfaced instead of crashing the whole pipeline.
4. **Interpretation** — after the tools finish, the agent writes the explanation. Two sources, in priority order:
   - **LLM (optional)** — any OpenAI-compatible endpoint via `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`. The system prompt forbids inventing numbers; the model may only cite values present in the result JSON.
   - **Local engine (always available)** — deterministic templates that compose the narrative from the actual result dicts. This guarantees a fully working product with zero external services, and makes every explanation reproducible.

The chat panel works the same way: intents (best-model, features, missing values, metrics, correlations, outliers, improvements) are answered from the stored analysis/training JSON — with the same LLM-first, local-fallback strategy.

### How the analyst layer works

Modelling answers "what predicts the target?". The analyst layer answers the questions
that come first: *what is going on in this data, is it moving, and is the difference real?*

An ad-hoc question becomes a **validated `QuerySpec`** — a closed grammar of filters,
`group_by`, aggregations, sort and limit — which pandas executes:

```
"Which region had the highest average revenue for orders over $100?"
        │
        ├─ LLM route ──► JSON spec ──► Pydantic + column validation ──┐
        │                    (rejected? feed the error back once)     │
        └─ Heuristic route ─► column/keyword matching ────────────────┤
                                                                      ▼
                                             pandas: filter → group → aggregate → sort
                                                                      │
                                          sentence + table + Plotly chart, all from the result
```

The model never writes Python or SQL, and nothing it emits is executed. An unknown column
fails loudly with the list of real ones instead of silently answering about nothing. The
narrative sentence is composed from the result table, so it cannot cite a number pandas did
not produce — and with no API key the heuristic route keeps the whole feature working.

`generate_insights()` runs the same tools unprompted and ranks what it finds by effect
size, so "80% of rows are one category" outranks "3 values are missing".

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12+, FastAPI, Uvicorn, Pydantic v2, pydantic-settings |
| Data & ML | pandas, NumPy, scikit-learn (1.5+), Plotly |
| Database | PostgreSQL 16 (docker) / SQLite (local dev) via SQLAlchemy 2.0 |
| Frontend | Next.js 14 (App Router), TypeScript, Tailwind CSS, react-plotly.js |
| Infra | Docker, docker-compose |
| Stats | SciPy (t-test, ANOVA, chi-square, Kruskal-Wallis, linear regression for trends) |
| Tests | pytest (162 tests: upload, analysis, problem-type detection, training, metrics, API, query spec, time series, significance, insights) |

## Project structure

```
ai-data-scientist/
├── backend/
│   ├── app/
│   │   ├── agents/            # pipeline agent, explanation engine, LLM client,
│   │   │                      #   analyst (question → QuerySpec), insights
│   │   ├── tools/             # the real math: data tools, query engine, analytics
│   │   │                      #   (time series + significance), Plotly viz, report builder
│   │   ├── services/          # dataset, analysis, ML, report, chat orchestration
│   │   ├── models/            # SQLAlchemy models
│   │   ├── schemas/           # Pydantic API schemas
│   │   ├── api/               # FastAPI routers
│   │   ├── core/              # config, database, security helpers
│   │   ├── utils/             # JSON/numpy conversion, dataframe cache
│   │   └── main.py            # app factory, CORS, error handlers
│   ├── tests/                 # pytest suite
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/                   # layout, dashboard, analysis/[id]
│   ├── components/            # sidebar, tables, charts, chat, timeline, …
│   ├── lib/                   # api client, types, theme, formatters
│   └── Dockerfile
├── datasets/samples/          # bundled example CSVs + metadata
├── reports/                   # generated Markdown reports
├── deploy/                    # free hosting: HF Spaces image, CPU-only deps, DEPLOY.md
├── docker-compose.yml
├── .env.example
└── README.md
```

## Installation

### 1. Prerequisites

- Python 3.12+ and Node 20+ (for running without Docker)
- Docker + Docker Compose (easiest path)

### 2. Configuration

```bash
cp .env.example .env       # then edit secrets / limits / optional LLM key
```

### 3a. Run with Docker (recommended)

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- API + interactive docs (Swagger): http://localhost:8000/docs
- PostgreSQL: `db` container (data persisted in the `pgdata` volume)

### 3b. Everyday use

After the first setup, this is the whole thing:

```bash
cd ~/Downloads/ai-data-scientist
./start.sh
```

Then open <http://localhost:3000>. `Ctrl+C` stops both processes.

`start.sh` uses `backend/.venv`, waits for the API before starting the
dashboard, tells you whether the LLM is configured, and refuses to start if a
port is already taken (instead of failing halfway).

To let other devices on your Wi-Fi use it, run `./scripts/serve-lan.sh`
instead — it prints the address they should open.

### 3c. Run the pieces separately

```bash
# backend (defaults to SQLite — zero setup)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# frontend (second terminal)
cd frontend
npm install
npm run dev          # http://localhost:3000, proxies /api → localhost:8000
```

Point the backend at Postgres locally by setting `DATABASE_URL` in `backend/.env`.

### 3c. Deploy for free

Backend on an Oracle Cloud Always Free ARM VM, frontend on Vercel.
One script sets up the VM: `./deploy/oracle/setup.sh`.
See **[deploy/DEPLOY.md](deploy/DEPLOY.md)** and **[deploy/oracle/README.md](deploy/oracle/README.md)**.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///backend/data/app.db` | SQLAlchemy URL (Postgres in Docker) |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `UPLOAD_DIR` | `<root>/datasets` | Upload + sample storage root |
| `REPORT_DIR` | `<root>/reports` | Where Markdown reports are written |
| `MAX_UPLOAD_MB` | `100` | Upload size limit |
| `MAX_ROWS` | `1000000` | Max dataset rows (training is subsampled to 50k for speed) |
| `MAX_ROWS` / `MAX_COLUMNS` | `100000` / `150` | Dataset size limits |
| `MAX_CARDINALITY` | `50` | Categorical columns above this are dropped |
| `LLM_API_KEY` | *(empty)* | Enables LLM interpretation (OpenAI-compatible) |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Any compatible endpoint works |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `BACKEND_URL` (frontend build) | `http://localhost:8000` | Server-side proxy target |
| `POSTGRES_USER/PASSWORD/DB` | `ai` / `ai_secret_change_me` / `ai_data_scientist` | compose DB credentials — change these |

## API

Base URL: `/api` — full interactive docs at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness + version |
| `POST` | `/datasets/upload` | Upload a CSV (multipart `file`) |
| `POST` | `/datasets/load-sample` | Load a bundled sample `{name}` |
| `GET` | `/datasets/samples` | List sample datasets |
| `GET` | `/datasets` | List datasets (with status/best model) |
| `GET` | `/datasets/{id}` | Dataset + analysis + training detail |
| `POST` | `/datasets/{id}/analyze` | Run the agent's EDA pipeline `{target?}` |
| `POST` | `/datasets/{id}/train` | Train & compare models `{target?, problem_type?, k?, features?}` |
| `GET` | `/datasets/{id}/models` | Stored training run |
| `GET` | `/datasets/{id}/models/{name}/download` | Download a trained model (`.pkl`, joblib) |
| `GET` | `/datasets/{id}/models/{name}/metadata` | Run metadata: features, labels, preprocessing notes |
| `GET` | `/datasets/{id}/visualizations` | Plotly figures |
| `POST` | `/datasets/{id}/chat` | Ask the agent about the analysis/training `{message}` |
| `GET` | `/datasets/{id}/schema` | Columns with roles, plus suggested dimensions/measures/date columns |
| `GET` | `/datasets/{id}/suggested-questions` | Starter questions built from this dataset's columns |
| `POST` | `/datasets/{id}/ask` | Answer a natural-language question about the data `{question}` |
| `POST` | `/datasets/{id}/query` | Run a structured QuerySpec (filters/group_by/aggregations/sort/limit) |
| `POST` | `/datasets/{id}/pivot` | Pivot table `{index, columns, values, aggfunc}` |
| `POST` | `/datasets/{id}/trend` | Time series `{date_column?, value_column?, agg, freq?}` |
| `POST` | `/datasets/{id}/segments` | Compare groups with a significance test `{dimension, metric?}` |
| `POST` | `/datasets/{id}/suggest-features` | Screen and rank input columns, with a measured comparison `{target?, problem_type?}` |
| `POST` | `/datasets/{id}/dashboard` | KPI tiles + charts for a filtered slice `{filters?, measure?, dimension?, date_column?}` |
| `GET` | `/datasets/{id}/insights` | Ranked automatic findings |
| `POST` | `/datasets/{id}/report` | Generate the Markdown report |
| `GET` | `/reports/{id}/download` | Download the report file |

### Example workflow

```bash
# load the housing sample
ID=$(curl -s -X POST localhost:8000/api/datasets/load-sample \
  -H 'Content-Type: application/json' -d '{"name":"housing_sales"}' | jq -r .id)

# let the agent analyze it
curl -s -X POST localhost:8000/api/datasets/$ID/analyze -d '{}' -H 'Content-Type: application/json' | jq .target_column, .problem_type

# train + compare models
curl -s -X POST localhost:8000/api/datasets/$ID/train -d '{}' -H 'Content-Type: application/json' | jq '.best_model, .models[].metrics.r2'

# ask a question, then export the report
curl -s -X POST localhost:8000/api/datasets/$ID/chat -d '{"message":"Which features matter most?"}' -H 'Content-Type: application/json' | jq .reply
curl -s -X POST localhost:8000/api/datasets/$ID/report | jq .download_url
```

## Testing

```bash
cd backend
pip install -r requirements.txt
pytest
```

162 tests cover: CSV upload validation (valid/empty/wrong-extension/binary/header-only/all-NaN), profile & missing-value detection, statistics & correlations, target/task detection (regression/classification/clustering + overrides), real model training (metric ranges, best-model selection, confusion matrices, silhouette, feature importances, retrain semantics), and full API happy paths including reports and chat.

The analyst suite additionally asserts that the query engine **rejects** unknown columns,
non-numeric aggregations and near-unique groupings; that aggregation results match pandas
directly; that a rising series is called rising and pure noise is not; that a real group
difference is significant while identical groups are not; and that insights come back
ranked and JSON-serialisable.

## Error handling

The API returns structured `{error: {code, message}}` payloads for: invalid/empty/oversized uploads, unparseable CSV, header-only files, all-empty columns, row/column limits, missing or invalid target columns, high-cardinality targets, no-usable-features datasets, per-model training failures (the run continues and reports the failure), and unknown dataset/report IDs. The UI surfaces these in error banners and empty states.

## Security

- **Uploads**: `.csv`-only (extension + content parsing), size/row/column caps, bytes stored under random UUID paths, original filename never used for path building, no code execution (pandas only).
- **Secrets**: all secrets live in environment variables; the frontend never talks to third parties and holds no keys — it only calls the same-origin `/api` proxy.
- **CORS**: allow-list from `CORS_ORIGINS`.
- **DB**: parameterized SQLAlchemy only; JSON payloads, never eval of user data.

## Future improvements

- Cross-validation + hyperparameter tuning (e.g. `RandomizedSearchCV`) as an agent tool
- Async training jobs (Celery/ARQ) with live per-step progress over WebSockets
- Dataset deletion + retention policy, and Parquet support for larger files
- Unsupervised extras: PCA/t-SNE projections, DBSCAN, feature-selection report
- Multi-turn agent memory (persist chat context per dataset)
- AuthN/Z for multi-user deployments, and report sharing links
- End-to-end integration tests against a real Postgres in CI

---

Built as a portfolio project demonstrating a production-grade pattern: **LLM as orchestrator/explainer, Python as the source of truth**. Every number you see in the UI was computed by the pipeline — the AI just makes it understandable.
