# Free deployment guide

Backend on **Hugging Face Spaces**, frontend on **Vercel**, database on
**Neon**. All three have a free tier that needs no credit card.

```
Browser ──► Vercel (Next.js)  ──/api/*──►  HF Space (FastAPI)  ──►  Neon (Postgres)
```

The browser only ever talks to Vercel. Next.js proxies `/api/*` to the Space
server-side, so there is no CORS setup and no API key ever reaches the client.

## Why this split

The backend needs ~190 MB of RAM just to import pandas + scikit-learn, and more
while training. Free tiers with 512 MB (Render, Fly) sit uncomfortably close to
that ceiling; a Space gets 16 GB. Vercel, meanwhile, is the best free host for
Next.js but cannot run this backend — scikit-learn, XGBoost and LightGBM blow
past its serverless bundle limit.

---

## 1. Database (Neon) — 5 minutes

The Space filesystem is **wiped on every restart and rebuild**. Without an
external database, every uploaded dataset and analysis disappears. Skip this
only if you are just demoing.

1. Sign up at <https://neon.tech> (free tier, no card).
2. Create a project; copy the connection string. It looks like:
   `postgresql://user:pass@ep-xxx.eu-central-1.aws.neon.tech/neondb?sslmode=require`
3. Change the scheme to the driver this project uses:
   `postgresql+psycopg2://user:pass@ep-xxx.../neondb?sslmode=require`

Keep that string for step 2. Treat it as a password — it is one.

## 2. Backend (Hugging Face Space)

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

## 3. Frontend (Vercel)

1. Push this repository to GitHub.
2. <https://vercel.com/new> → import the repo.
3. **Root Directory: `frontend`** (important — the repo root is not the app).
   Framework preset: Next.js, detected automatically.
4. Environment variable:

   | Name | Value |
   |---|---|
   | `BACKEND_URL` | `https://<user>-<space>.hf.space` |

   No trailing slash. This is read at **build time** by `next.config.mjs`, so
   after changing it you must redeploy, not just restart.
5. Deploy. Open the Vercel URL and upload a CSV.

## 4. Verify

```bash
SPACE=https://<user>-<space>.hf.space
curl -s $SPACE/api/health

ID=$(curl -s -X POST $SPACE/api/datasets/load-sample \
      -H 'Content-Type: application/json' -d '{"name":"housing_sales"}' | jq -r .id)
curl -s -X POST $SPACE/api/datasets/$ID/analyze -H 'Content-Type: application/json' -d '{}' | jq '.problem_type, .target_column'
```

Then do the same through the Vercel URL to confirm the proxy works.

## Notes and limits

* **Sleep.** A free Space pauses after ~48 h without traffic and takes ~30 s to
  wake on the next request. The first request after a sleep may time out in the
  browser — reload once.
* **Ephemeral files.** Uploaded CSVs, generated `.pkl` models and Markdown
  reports live on the Space disk and are lost on restart. Rows in Postgres
  survive; the files they point at do not. For durable files, add object
  storage (e.g. Cloudflare R2 free tier) — not wired up yet.
* **CPU only.** `deploy/requirements-deploy.txt` installs `xgboost-cpu`
  instead of `xgboost`: identical API and version, but 5.8 MB instead of
  ~370 MB, because the default wheel bundles CUDA libraries a CPU host can
  never use. Verified as a drop-in (fit/predict on both regressor and
  classifier).
* **Training time.** 22 models on 2 free vCPUs is slower than your laptop.
  `MAX_TRAIN_ROWS` (default 50 000) already subsamples large datasets; lower it
  if requests time out.
* **Rebuilding.** Re-run `prepare-space.sh`, then commit and push in
  `build/hf-space` again.

## Updating after code changes

```bash
./deploy/huggingface/prepare-space.sh          # backend → refresh build/hf-space, commit, push
git push                                        # frontend → Vercel redeploys from GitHub
```
