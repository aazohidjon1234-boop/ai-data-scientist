# Free deployment guide

Backend on an **Oracle Cloud Always Free** ARM VM, frontend on **Vercel**.

```
Browser ──► Vercel (Next.js) ──/api/*──► Caddy (TLS) ──► FastAPI ──► Postgres
                                         └───────── Oracle A1 VM ─────────┘
```

The browser only ever talks to Vercel. Next.js proxies `/api/*` server-side, so
there is no CORS setup and no API key ever reaches the client.

**Setup lives in [oracle/README.md](oracle/README.md).** One script does the VM
side: `./deploy/oracle/setup.sh`.

## Why this split, and why not something simpler

Measured peak memory while training the model zoo:

| Dataset | Peak RSS |
|---|---|
| 500 rows | 264 MB |
| 5,000 rows | 401 MB |
| 10,000 rows | 717 MB |
| 50,000 rows | 1,051 MB |

That rules out most free tiers:

| Option | Verdict |
|---|---|
| **Oracle Always Free** | 4 ARM cores, 24 GB RAM, free forever. Card required for identity only. **Used here.** |
| Hugging Face Spaces | Docker Spaces now require a paid PRO plan; only Static Spaces are free. See [huggingface/](huggingface/) if you have PRO. |
| Render free | 512 MB **and 0.1 CPU** — fails above ~5,000 rows and trains very slowly. Spins down after 15 min. |
| Google Cloud Run | Technically fine (configurable RAM, scales to zero) and free in practice for this load. A reasonable alternative. |
| Vercel (backend) | Cannot host it: scipy + pandas + sklearn + plotly far exceed the serverless bundle limit. Frontend only. |

Vercel remains the best free host for the Next.js frontend.

---

## Appendix: managed Postgres on Neon (optional)

The Oracle setup runs Postgres on the VM. Use Neon instead by setting
`DATABASE_URL` in `deploy/oracle/.env`.

The Space filesystem is **wiped on every restart and rebuild**. Without an
external database, every uploaded dataset and analysis disappears. Skip this
only if you are just demoing.

1. Sign up at <https://neon.tech> (free tier, no card).
2. Create a project; copy the connection string. It looks like:
   `postgresql://user:pass@ep-xxx.eu-central-1.aws.neon.tech/neondb?sslmode=require`
3. Change the scheme to the driver this project uses:
   `postgresql+psycopg2://user:pass@ep-xxx.../neondb?sslmode=require`

Keep that string for step 2. Treat it as a password — it is one.

## Appendix: Hugging Face Space (requires a PRO plan)

1. Create a Space: <https://huggingface.co/new-space> → **SDK: Docker**,
   template **Blank**, hardware **CPU basic (free)**.
2. Build the deployable directory locally:

   ```bash
   ./deploy/huggingface/prepare-space.sh
   ```

   It writes `build/hf-space/` (~360 KB) with the Dockerfile and Space card at
   the root, which is the layout a Space requires.

3. Push it:

   ```bash
   cd build/hf-space
   git init && git branch -M main
   git remote add origin https://huggingface.co/spaces/<user>/<space>
   git add -A && git commit -m "Deploy backend"
   git push -u origin main
   ```

   Use a Hugging Face **access token** (Settings → Access Tokens, write scope)
   as the password when git asks.

4. In the Space: **Settings → Variables and secrets**, add:

   | Name | Value |
   |---|---|
   | `DATABASE_URL` | the Neon string from step 1 |
   | `LLM_API_KEY` | optional — enables LLM interpretation |
   | `LLM_BASE_URL` | e.g. `https://api.groq.com/openai/v1` |
   | `LLM_MODEL` | e.g. `openai/gpt-oss-120b` |

   Add `DATABASE_URL` and `LLM_API_KEY` as **secrets**, not plain variables.

5. Wait for the build (first one takes several minutes), then check:
   `https://<user>-<space>.hf.space/api/health` → `{"status":"ok",...}`
   and `https://<user>-<space>.hf.space/docs` for the interactive API.

## Frontend (Vercel)

1. Push this repository to GitHub.
2. <https://vercel.com/new> → import the repo.
3. **Root Directory: `frontend`** (important — the repo root is not the app).
   Framework preset: Next.js, detected automatically.
4. Environment variable:

   | Name | Value |
   |---|---|
   | `BACKEND_URL` | `https://<dashed-ip>.sslip.io` (from `setup.sh`) |

   No trailing slash. This is read at **build time** by `next.config.mjs`, so
   after changing it you must redeploy, not just restart.
5. Deploy. Open the Vercel URL and upload a CSV.

## Verify

```bash
API=https://<dashed-ip>.sslip.io
curl -s $API/api/health

ID=$(curl -s -X POST $API/api/datasets/load-sample \
      -H 'Content-Type: application/json' -d '{"name":"housing_sales"}' | jq -r .id)
curl -s -X POST $API/api/datasets/$ID/analyze -H 'Content-Type: application/json' -d '{}' \
  | jq '.problem_type, .target_column'
```

Then do the same through the Vercel URL to confirm the proxy works.

## Notes and limits

* **ARM.** The A1 shape is `aarch64`. Every dependency has an ARM wheel —
  including LightGBM, which publishes under the older `manylinux2014_aarch64`
  tag — so nothing compiles from source. The image is built on the VM itself,
  so no cross-compilation is needed.
* **libgomp.** The image installs `libgomp1`. LightGBM links against the system
  copy while scikit-learn bundles its own; without it two of the 22 models fail
  behind a misleading "LightGBM is not installed" message.
* **Persistence.** Uploads, models and reports live in named Docker volumes and
  survive `docker compose down` (but not `down -v`). Nothing is backed up
  automatically.
* **Capacity.** Oracle frequently answers "Out of host capacity" for free ARM
  instances. Retry at other times or in another availability domain.
* **Training time.** 22 models on 4 ARM cores is slower than a modern laptop but
  workable. `MAX_TRAIN_ROWS` (default 50 000) subsamples large datasets; lower
  it if requests time out.
* **xgboost-cpu.** `deploy/requirements-deploy.txt` (used by the Hugging Face
  image) swaps `xgboost` for `xgboost-cpu`: same API and version, 5.8 MB instead
  of ~370 MB, because the default wheel bundles unusable CUDA libraries. The
  Oracle image uses `backend/requirements.txt`; switch it too if image size
  matters to you.

## Updating after code changes

```bash
# backend — on the VM
cd ~/ai-data-scientist && git pull
docker compose -f deploy/oracle/docker-compose.yml up -d --build

# frontend — Vercel redeploys automatically from GitHub
git push
```
