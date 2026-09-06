#!/usr/bin/env bash
# Assemble a directory that can be pushed straight to a Hugging Face Space.
#
#   ./deploy/huggingface/prepare-space.sh [output-dir]
#
# Default output: build/hf-space (git-ignored). Nothing in the project is
# modified — the Space needs the Dockerfile and README at ITS root, which is a
# different layout from this repository.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/build/hf-space}"

rm -rf "$OUT"
mkdir -p "$OUT/backend" "$OUT/deploy" "$OUT/datasets"

# Dockerfile and the Space card must be at the Space root.
cp "$ROOT/deploy/huggingface/Dockerfile" "$OUT/Dockerfile"
cp "$ROOT/deploy/huggingface/space-README.md" "$OUT/README.md"

# Build inputs, in the paths the Dockerfile expects.
cp "$ROOT/deploy/requirements-deploy.txt" "$OUT/deploy/requirements-deploy.txt"
cp -r "$ROOT/backend/app" "$OUT/backend/app"
cp -r "$ROOT/datasets/samples" "$OUT/datasets/samples"

# Never ship caches, local databases or uploaded user data.
find "$OUT" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$OUT" -name '*.pyc' -delete 2>/dev/null || true

cat > "$OUT/.gitignore" <<'EOF'
__pycache__/
*.pyc
EOF

echo "Space directory ready: $OUT"
echo
echo "Next:"
echo "  1. Create a Docker Space at https://huggingface.co/new-space (SDK: Docker)"
echo "  2. cd $OUT"
echo "  3. git init && git remote add origin https://huggingface.co/spaces/<user>/<space>"
echo "  4. git add -A && git commit -m 'Deploy backend' && git push -u origin main"
echo
du -sh "$OUT"
