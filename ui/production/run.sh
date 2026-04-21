#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="$HOME/.deno/bin:$PATH"
export PORT=3000

echo "Starting parko-ui on port $PORT (respawning on crash)..."
while true; do
  cd "$REPO_ROOT/ui"
  deno run -A server.ts || true
  echo "Server exited — restarting in 3s..."
  sleep 3
done
