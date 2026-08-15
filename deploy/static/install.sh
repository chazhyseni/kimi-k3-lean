#!/usr/bin/env bash
# kimi-k3-lean laptop installer.
#
# Usage:
#   curl -fsSL https://<server>/client/install.sh | bash
#
# After install, edit ~/llm-client/secrets.env with the keys the admin
# gives you, then `source ~/llm-client/remote-env.sh` and you're set.
#
# What this does:
#   1. Creates ~/llm-client/
#   2. Downloads remote-env.sh, secrets.env.example, the CA cert.
#   3. Prints the next-step instructions.
#
# Why a separate installer (not just env vars)?
#   * Caddy serves an internal CA; clients need the cert.
#   * env vars have to set the right values per-host (server URL, key).
#   * Agent CLIs (Claude Code, Pi, OpenCode, aider) need different env
#     vars per tool; remote-env.sh sets them all at once.
#
# This is the kimi-k3-lean analogue of /mnt/scratch/agents/install-remote-client.sh
# in the llm-server deployment.

set -euo pipefail

CLIENT_DIR="${HOME}/llm-client"
SERVER="${KIMI_SERVER:-${K3_SERVER:-https://YOUR_SERVER_IP}}"

note() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

# Preflight
command -v curl >/dev/null 2>&1 || die "missing required tool: curl"
mkdir -p "${CLIENT_DIR}/bin"

note "fetching client bundle from ${SERVER}"

# 1. remote-env.sh (the env var preset for the agent CLIs)
curl -fsSL -k "${SERVER}/client/remote-env.sh" -o "${CLIENT_DIR}/remote-env.sh" \
    || die "could not download remote-env.sh from ${SERVER}"

# 2. secrets.env.example (template, not the real keys)
curl -fsSL -k "${SERVER}/client/secrets.env.example" -o "${CLIENT_DIR}/secrets.env.example" \
    || die "could not download secrets.env.example"

# 3. CA cert for the server's internal Caddy TLS
curl -fsSL -k "${SERVER}/client/caddy-root.crt" -o "${CLIENT_DIR}/caddy-root.crt" \
    || warn "could not download CA cert (server may have real certs instead)"

# 4. secrets.env — copy the example so the user has a starting point
[ -f "${CLIENT_DIR}/secrets.env" ] || cp "${CLIENT_DIR}/secrets.env.example" "${CLIENT_DIR}/secrets.env"

chmod 600 "${CLIENT_DIR}/secrets.env" 2>/dev/null || true

ok "client installed to ${CLIENT_DIR}/"
echo ""
echo "Next steps:"
echo ""
echo "  1. Edit \${CLIENT_DIR}/secrets.env and set the keys your admin gave you:"
echo "       nano \${CLIENT_DIR}/secrets.env"
echo ""
echo "  2. Source the env every session:"
echo "       source \${CLIENT_DIR}/remote-env.sh"
echo ""
echo "  3. Optional: trust the CA cert system-wide:"
echo "       Linux:    sudo cp \${CLIENT_DIR}/caddy-root.crt /usr/local/share/ca-certificates/ && sudo update-ca-certificates"
echo "       macOS:    sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain \${CLIENT_DIR}/caddy-root.crt"
echo ""
echo "Test the connection:"
echo "       curl -k https://${SERVER#https://}/v1/models -H \"Authorization: Bearer \\\$INTERNAL_API_KEY\""
echo ""