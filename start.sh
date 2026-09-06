#!/usr/bin/env bash
# Start the AI Data Scientist Agent on this machine.
#
#   ./start.sh          API on :8000, dashboard on :3000
#   Ctrl+C stops both.
#
# To share it with other devices on your Wi-Fi, use ./scripts/serve-lan.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${PORT:-3000}"

die() { printf '\n\033[1;31m%s\033[0m\n' "$*" >&2; exit 1; }

# The dependencies live in backend/.venv, not in the system Python.
PY="$ROOT/backend/.venv/bin/python"
if [ ! -x "$PY" ]; then
    die "Backend virtualenv missing. Create it once with:
  cd $ROOT/backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
fi
"$PY" -c "import fastapi" 2>/dev/null || die "Backend dependencies missing. Run:
  cd $ROOT/backend && .venv/bin/pip install -r requirements.txt"

for port in "$API_PORT" "$WEB_PORT"; do
    if ss -ltn 2>/dev/null | grep -q ":$port "; then
        die "Port $port is already in use. Another copy may still be running:
  pkill -f 'uvicorn app.main' ; pkill -f next-server"
    fi
done

cd "$ROOT/backend"
"$PY" -m uvicorn app.main:app --reload --port "$API_PORT" &
API_PID=$!
cleanup() {
    kill "$API_PID" 2>/dev/null || true
    [ -n "${WEB_PID:-}" ] && kill "$WEB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

printf '\nWaiting for the API'
for _ in $(seq 1 40); do
    curl -fsS --max-time 2 "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1 && break
    printf '.'; sleep 0.5
done
curl -fsS --max-time 2 "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1 \
    || die "
The API did not start. Run it on its own to see why:
  cd $ROOT/backend && .venv/bin/python -m uvicorn app.main:app"

LLM=$(curl -fsS "http://127.0.0.1:$API_PORT/api/health" | grep -o '"llm_enabled":[a-z]*' | cut -d: -f2)

cd "$ROOT/frontend"
[ -d node_modules ] || { echo; echo "Installing frontend dependencies (first run only)…"; npm install; }

cat <<EOF


  Dashboard   http://localhost:$WEB_PORT
  API docs    http://localhost:$API_PORT/docs
  LLM         $([ "$LLM" = "true" ] && echo "on (backend/.env)" || echo "off — using the built-in engine")

  Ctrl+C stops both.

EOF

BACKEND_URL="http://127.0.0.1:$API_PORT" npm run dev &
WEB_PID=$!
wait "$WEB_PID"
