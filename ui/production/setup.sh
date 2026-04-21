#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Install Deno if absent
if ! command -v deno &>/dev/null; then
  curl -fsSL https://deno.land/install.sh | sh
  export DENO_INSTALL="$HOME/.deno"
  export PATH="$DENO_INSTALL/bin:$PATH"
fi

# Grant port 80 to deno without running as root
sudo setcap cap_net_bind_service=+ep "$(which deno)"

# Pre-cache Deno dependencies (fast startup)
cd "$REPO_ROOT/ui"
deno cache server.ts

# Python venv + minimal deps for ALA sightings fetch
cd "$REPO_ROOT"
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet geopandas requests

# Fetch ALA sightings
mkdir -p "$REPO_ROOT/outputs/ala_cache"
bash "$REPO_ROOT/ui/production/fetch-sightings.sh"

echo ""
echo "Setup complete. Copy ranking CSVs into outputs/ then start:"
echo "  screen -S parko bash ui/production/run.sh"
