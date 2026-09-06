# Backend on Oracle Cloud (Always Free ARM)

Backend + Postgres + TLS on a free-forever Oracle VM; frontend on Vercel.

Chosen because the app genuinely needs the headroom. Measured peak memory
while training the model zoo:

| Dataset | Peak RSS |
|---|---|
| 500 rows | 264 MB |
| 5,000 rows | 401 MB |
| 10,000 rows | 717 MB |
| 50,000 rows | 1,051 MB |

A 512 MB free tier fails above ~5,000 rows. The Always Free A1 shape gives
**4 ARM cores and 24 GB RAM**, which clears this with room to spare.

## 1. Create the VM

<https://cloud.oracle.com> → Compute → Instances → **Create instance**

| Setting | Value |
|---|---|
| Image | Canonical Ubuntu 24.04 |
| Shape | **VM.Standard.A1.Flex** (Ampere, ARM) |
| OCPUs / Memory | 4 / 24 GB (the whole free allowance) |
| SSH keys | upload your public key |

> **"Out of host capacity"** is the usual first response — free ARM capacity is
> genuinely scarce in popular regions. Retry at different hours, or pick a less
> busy availability domain. It is not a mistake on your side.

## 2. Open the ports in the VCN

Networking → Virtual Cloud Networks → your VCN → Subnet → **Security List** →
Add Ingress Rules:

| Source CIDR | Protocol | Destination port |
|---|---|---|
| 0.0.0.0/0 | TCP | 80 |
| 0.0.0.0/0 | TCP | 443 |

This is only half the firewall. Oracle's Ubuntu images also carry an iptables
rule that rejects everything except SSH — `setup.sh` fixes that side for you.

## 3. Deploy

```bash
ssh ubuntu@<public-ip>
git clone https://github.com/<user>/ai-data-scientist.git
cd ai-data-scientist
./deploy/oracle/setup.sh
```

The script installs Docker, opens the local firewall, adds 4 GB of swap,
generates `deploy/oracle/.env` (random database password, `SITE_ADDRESS` from
your public IP), then builds and starts everything.

The first ARM build takes roughly ten minutes.

### About the hostname

Without a domain, Let's Encrypt cannot certify a bare IP. The script uses
[sslip.io](https://sslip.io): `129-146-0-1.sslip.io` resolves to `129.146.0.1`,
so Caddy gets a real certificate for free. Own a domain? Point an A record at
the VM and set `SITE_ADDRESS` to it in `.env` instead.

## 4. Connect the frontend

Vercel → your project → Settings → Environment Variables:

```
BACKEND_URL = https://<dashed-ip>.sslip.io
```

`BACKEND_URL` is read at **build** time, so redeploy after changing it.

## 5. Add the LLM key (optional)

Edit `deploy/oracle/.env`:

```
LLM_API_KEY=<your key>
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
```

Then `docker compose -f deploy/oracle/docker-compose.yml up -d`.

Without a key the app still works end to end — the deterministic local engine
handles explanations and question routing.

## Operating it

```bash
cd ~/ai-data-scientist/deploy/oracle

docker compose logs -f backend        # follow the API
docker compose logs -f caddy          # certificate / proxy issues
docker compose ps                     # what is running
docker compose restart backend        # restart after an .env change
docker compose down                   # stop (data survives in volumes)

git pull && docker compose up -d --build   # deploy new code
```

Data lives in named Docker volumes (`pgdata`, `appdata`, `reports`, `models`),
so `docker compose down` does not lose it. `down -v` does.

## Notes

* **ARM wheels.** Every dependency ships an `aarch64` wheel — including
  LightGBM, which publishes under the older `manylinux2014_aarch64` tag.
  Nothing compiles from source.
* **libgomp.** The image installs `libgomp1`; LightGBM links against the system
  copy while scikit-learn bundles its own. Without it two of the 22 models fail
  behind a misleading "not installed" message.
* **Postgres is local** here rather than Neon — with 24 GB of RAM there is no
  reason to add a network hop. To use Neon anyway, set `DATABASE_URL` in `.env`.
* **CORS.** The browser only talks to Vercel, which proxies server-side, so
  `CORS_ORIGINS` can stay empty. Set it only if you call the API directly.
* **Backups.** Nothing is backed up automatically. For anything you care about:
  `docker compose exec db pg_dump -U ai ai_data_scientist > backup.sql`.
