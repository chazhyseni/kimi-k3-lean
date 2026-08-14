# remote-env.sh — env vars the agent CLIs need to talk to kimi-k3-lean.
#
# Source this every session before launching Claude Code / Pi / OpenCode /
# aider / Qwen Code. It is shipped by the laptop installer and lives at
# ~/llm-client/remote-env.sh.
#
# Required: secrets.env (next to this file) with INTERNAL_API_KEY set.

if [ -z "${BASH_SOURCE[0]}" ]; then
    echo "remote-env.sh must be sourced, not executed." >&2
    return 1
fi

CLIENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS="${CLIENT_DIR}/secrets.env"
[ -f "${SECRETS}" ] || { echo "missing ${SECRETS}; copy secrets.env.example and edit it" >&2; return 1; }
set -a
. "${SECRETS}"
set +a

SERVER="${K3_SERVER:-${KIMI_SERVER:-https://10.10.4.104}}"
CA="${CLIENT_DIR}/caddy-root.crt"

if [ -f "${CA}" ]; then
    export NODE_EXTRA_CA_CERTS="${CA}"
    export SSL_CERT_FILE="${CA}"
fi

# OpenAI-style clients (Python openai SDK, aider, OpenCode, Qwen, Pi):
export OPENAI_BASE_URL="${SERVER}/v1"
export OPENAI_API_KEY="${INTERNAL_API_KEY:-}"

# Anthropic-style clients (Claude Code):
export ANTHROPIC_BASE_URL="${SERVER}/anthropic"
export ANTHROPIC_API_KEY="${INTERNAL_API_KEY:-}"

# Verbose mode so `claude --debug-mode` shows the upstream URL on startup.
export CLAUDE_CODE_DEBUG_MODE="${CLAUDE_CODE_DEBUG_MODE:-1}"

echo "kimi-k3-lean env loaded:"
echo "  OPENAI_BASE_URL      = ${OPENAI_BASE_URL}"
echo "  ANTHROPIC_BASE_URL   = ${ANTHROPIC_BASE_URL}"
echo "  OPENAI_API_KEY       = ${OPENAI_API_KEY:0:8}..."
[ -f "${CA}" ] && echo "  NODE_EXTRA_CA_CERTS  = ${NODE_EXTRA_CA_CERTS}"