#!/usr/bin/env bash
# Serve the app to everyone on your Wi-Fi / LAN, straight from this machine.
#
#   ./scripts/serve-lan.sh          production build (fast for visitors)
#   ./scripts/serve-lan.sh --dev    dev server with hot reload (slower)
#
# Only the dashboard port is exposed. Next.js proxies /api/* to the backend
# server-side, from this same machine, so the API stays bound to localhost and
# is never reachable from the network directly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-3000}"
API_PORT="${API_PORT:-8000}"
MODE="prod"
[ "${1:-}" = "--dev" ] && MODE="dev"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# The address other devices will use. Taken at runtime because DHCP hands out a
# different one each time you rejoin the network.
lan_ip() {
    ip -4 -o addr show scope global 2>/dev/null \
        | awk '{split($4,a,"/"); print a[1]; exit}'
}
IP="$(lan_ip)"
[ -n "$IP" ] || die "No network address found. Are you connected to Wi-Fi?"

# ------------------------------------------------------------------ backend
cd "$ROOT/backend"
PY="$ROOT/backend/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
"$PY" -c "import fastapi" 2>/dev/null || die "Backend deps missing. Run: cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"

log "Starting the API on 127.0.0.1:$API_PORT (not exposed to the network)"
"$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!
cleanup() {
    kill "$API_PID" 2>/dev/null || true
    [ -n "${WEB_PID:-}" ] && kill "$WEB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 40); do
    curl -fsS --max-time 2 "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1 && break
    sleep 0.5
done
curl -fsS --max-time 2 "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1 \
    || die "The API did not start. Run it alone to see the error: cd backend && .venv/bin/python -m uvicorn app.main:app"

# ----------------------------------------------------------------- frontend
cd "$ROOT/frontend"
[ -d node_modules ] || { log "Installing frontend dependencies"; npm install; }

# BACKEND_URL stays on localhost on purpose: the proxy runs here, not in the
# visitor's browser, so the API never needs a network-visible address.
export BACKEND_URL="http://127.0.0.1:$API_PORT"

if [ "$MODE" = "prod" ]; then
    log "Building the dashboard (one-off, ~30s)"
    npm run build
fi

cat <<EOF

$(printf '\033[1;32m%s\033[0m' "Ready. Anyone on this Wi-Fi can open:")

    http://$IP:$PORT

On this machine:  http://localhost:$PORT
API docs:         http://localhost:$API_PORT/docs

$(warn "Note: there is no login. Everyone on the network can upload data,")
$(warn "read every uploaded dataset and delete nothing. Keep this to a")
$(warn "network you trust, and stop it with Ctrl+C when you are done.")

The address changes when you rejoin the Wi-Fi — rerun this script to get it.

EOF

log "Serving (Ctrl+C stops both)"
if [ "$MODE" = "prod" ]; then
    npx next start -p "$PORT" -H 0.0.0.0 &
else
    npx next dev -p "$PORT" -H 0.0.0.0 &
fi
WEB_PID=$!
wait "$WEB_PID"
