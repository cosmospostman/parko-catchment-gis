#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Install Deno if absent
if ! command -v deno &>/dev/null; then
  curl -fsSL https://deno.land/install.sh | sh
  export DENO_INSTALL="$HOME/.deno"
  export PATH="$DENO_INSTALL/bin:$PATH"
fi

# Install Caddy if absent
if ! command -v caddy &>/dev/null; then
  sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | sudo tee /etc/apt/sources.list.d/caddy-stable.list
  sudo apt-get update && sudo apt-get install -y caddy
fi

# Write Caddyfile (prompts for domain if not set)
DOMAIN="${PARKO_DOMAIN:-}"
if [[ -z "$DOMAIN" ]]; then
  read -r -p "Enter domain name (e.g. parkinsonia.hello-mlj.net): " DOMAIN
fi

sudo tee /etc/caddy/Caddyfile > /dev/null <<EOF
$DOMAIN {
    reverse_proxy localhost:3000
}
EOF

sudo systemctl enable caddy
sudo systemctl restart caddy

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
