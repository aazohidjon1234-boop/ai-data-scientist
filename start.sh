#!/usr/bin/env bash
# Start the AI Data Scientist Agent: API (port 8000) + dashboard (port 3000)
# Usage: ./start.sh        (Ctrl+C stops both)
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT/backend"
python3 -m uvicorn app.main:app --reload --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM

echo "API starting on http://localhost:8000  (docs: /docs)"
sleep 2

cd "$ROOT/frontend"
[ -d node_modules ] || npm install
echo "Dashboard: http://localhost:3000"
npm run dev
