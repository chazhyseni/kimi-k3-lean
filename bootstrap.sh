#!/usr/bin/env bash
# bootstrap.sh — one-command install: clone, build, start server.
#
#   curl -fsSL https://raw.githubusercontent.com/chazhyseni/litMoE/main/bootstrap.sh | bash
#
# What it does (in order, no detours):
#   1. Clone to $LITMOE_DIR (default ~/.litMoE)
#   2. Build liblitmoe.so + bin/litmoe (~30 sec)
#   3. Start the OpenAI server on http://127.0.0.1:8080
#   4. Wait for /v1/models to return 200
#   5. Install the `litMoE` launcher to ~/.local/bin
#   6. Print the URL, token, and next steps
#
# The server starts with or without model weights. Without weights,
# /v1/models works and /v1/chat/completions returns a clear engine_error.
# Download weights with:  litMoE fetch
#
# Env vars (all optional):
#   LITMOE_DIR       install location     (default ~/.litMoE)
#   LITMOE_PORT      server port          (default 8080)
#   LITMOE_HOST      bind address         (default 127.0.0.1)
#   LITMOE_API_KEY   bearer token         (default random 32 hex)
#   LITMOE_SKIP_DL=1 skip model detection (scaffold-only)
#   LITMOE_NO_HERMES=1 skip Hermes config
#   LITMOE_UNINSTALL=1 kill server + remove launcher

set -euo pipefail

LITMOE_REPO="https://github.com/chazhyseni/litMoE.git"
LITMOE_DIR="${LITMOE_DIR:-$HOME/.litMoE}"
LITMOE_PORT="${LITMOE_PORT:-8080}"
LITMOE_HOST="${LITMOE_HOST:-127.0.0.1}"
LITMOE_API_KEY="${LITMOE_API_KEY:-$(head -c16 /dev/urandom 2>/dev/null | od -An -tx1 | tr -d ' \n' || openssl rand -hex 16 2>/dev/null || echo "k3-$(date +%s)-$$")}"
LITMOE_MODEL_DIR="${LITMOE_MODEL_DIR:-$LITMOE_DIR/checkpoints/k3}"
LITMOE_MODEL_NAME="${LITMOE_MODEL_NAME:-kimi-k3}"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*" >&2; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m  !\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m==> ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- uninstall ---
if [ "${LITMOE_UNINSTALL:-0}" = "1" ]; then
    say "uninstalling litMoE"
    [ -f "$LITMOE_DIR/server.pid" ] && kill "$(cat "$LITMOE_DIR/server.pid" 2>/dev/null)" 2>/dev/null || true
    rm -f "$HOME/.local/bin/litMoE"
    if [ "${LITMOE_NO_HERMES:-0}" != "1" ] && command -v hermes >/dev/null 2>&1; then
        hermes config unset model.base_url model.api_key model.default 2>/dev/null || true
    fi
    ok "done. repo at $LITMOE_DIR kept (rm -rf to wipe)"
    exit 0
fi

# --- prereqs ---
command -v git >/dev/null 2>&1 || die "git not found"
if command -v python3 >/dev/null 2>&1; then LITMOE_PY=python3
elif command -v python >/dev/null 2>&1; then LITMOE_PY=python
else die "Python 3.11+ not found"; fi
ok "python: $LITMOE_PY"

# --- 1. clone ---
if [ -d "$LITMOE_DIR/.git" ]; then
    say "updating $LITMOE_DIR"
    git -C "$LITMOE_DIR" fetch --depth 1 origin main >/dev/null 2>&1
    git -C "$LITMOE_DIR" reset --hard origin/main >/dev/null 2>&1
    ok "updated (HEAD $(git -C "$LITMOE_DIR" rev-parse --short HEAD))"
elif [ -d "$LITMOE_DIR" ]; then
    die "$LITMOE_DIR exists but is not a git repo. Move it or set LITMOE_DIR."
else
    say "cloning to $LITMOE_DIR"
    git clone --depth 1 "$LITMOE_REPO" "$LITMOE_DIR" || die "clone failed"
    ok "cloned"
fi
cd "$LITMOE_DIR"

# --- 2. build ---
if [ ! -f bin/liblitmoe.so ] || [ src -nt bin/liblitmoe.so ]; then
    say "building"
    LDFLAGS="-lm -pthread" make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2) all
    ok "built"
else
    ok "build cached"
fi

# --- 3. model detection (symlink existing weights, don't auto-download) ---
if [ "${LITMOE_SKIP_DL:-0}" != "1" ] && [ ! -d "$LITMOE_MODEL_DIR" -o -z "$(find "$LITMOE_MODEL_DIR" -name '*.safetensors' -print -quit 2>/dev/null)" ]; then
    # Look for existing K3-shaped weights on this host
    found=""
    for d in "$HOME/kimi-local/models/kimi-linear-shards" \
             "$HOME/kimi-local/models/kimi-linear" \
             "$HOME/.litMoE/checkpoints/kimi-linear"; do
        if [ -d "$d" ] && find "$d" -maxdepth 2 -name '*.safetensors' -print -quit 2>/dev/null | grep -q .; then
            found="$d"; break
        fi
    done
    if [ -n "$found" ]; then
        mkdir -p "$(dirname "$LITMOE_MODEL_DIR")"
        ln -sfn "$found" "$LITMOE_MODEL_DIR"
        ok "using existing weights at $found"
    else
        warn "no model weights found. server will start; chat returns engine_error until you fetch:"
        warn "  litMoE fetch"
        mkdir -p "$LITMOE_MODEL_DIR"
    fi
fi

# --- 4. kill prior server on same port ---
if [ -f "$LITMOE_DIR/server.pid" ]; then
    old=$(cat "$LITMOE_DIR/server.pid" 2>/dev/null || echo "")
    [ -n "$old" ] && kill "$old" 2>/dev/null && sleep 0.5 || true
fi
# free the port if anything else is on it (lsof works on macOS + Linux)
if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:"$LITMOE_PORT" 2>/dev/null | xargs kill 2>/dev/null || true
elif command -v fuser >/dev/null 2>&1; then
    fuser -k "${LITMOE_PORT}/tcp" 2>/dev/null || true
fi

# --- 5. start server ---
say "starting server"
export LD_LIBRARY_PATH="$LITMOE_DIR/bin:${LD_LIBRARY_PATH:-}"
cat > "$LITMOE_DIR/server.env" <<EOF
LITMOE_HOST=$LITMOE_HOST
LITMOE_PORT=$LITMOE_PORT
LITMOE_API_KEY=$LITMOE_API_KEY
LITMOE_MODEL_DIR=$LITMOE_MODEL_DIR
LITMOE_MODEL_NAME=$LITMOE_MODEL_NAME
EOF
chmod 600 "$LITMOE_DIR/server.env"

$LITMOE_PY -u serve/__main__.py "$LITMOE_MODEL_DIR" \
    --host "$LITMOE_HOST" --port "$LITMOE_PORT" \
    --api-key "$LITMOE_API_KEY" --model-id "$LITMOE_MODEL_NAME" \
    >>"$LITMOE_DIR/server.log" 2>&1 </dev/null &
echo $! > "$LITMOE_DIR/server.pid"
ok "server PID $(cat "$LITMOE_DIR/server.pid")"

# --- 6. wait for /v1/models ---
ready=0
for i in $(seq 1 30); do
    code=$(curl -sS --max-time 1 -o /dev/null -w '%{http_code}' \
        -H "Authorization: Bearer $LITMOE_API_KEY" \
        "http://$LITMOE_HOST:$LITMOE_PORT/v1/models" 2>/dev/null || echo 000)
    [ "$code" = "200" ] && { ready=1; break; }
    sleep 1
done

if [ "$ready" = "1" ]; then
    ok "server up: http://$LITMOE_HOST:$LITMOE_PORT"
else
    warn "server not ready after 30s; tail $LITMOE_DIR/server.log"
fi

# --- 7. install launcher ---
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR" 2>/dev/null || true
if cp "$LITMOE_DIR/litMoE" "$BIN_DIR/litMoE" 2>/dev/null; then
    chmod +x "$BIN_DIR/litMoE"
    ok "launcher: $BIN_DIR/litMoE"
    case ":$PATH:" in *":$BIN_DIR:"*) ;; *) warn "add to PATH: export PATH=$BIN_DIR:\$PATH" ;; esac
else
    warn "could not install launcher to $BIN_DIR"
fi

# --- 8. Hermes (optional, best-effort) ---
if [ "${LITMOE_NO_HERMES:-0}" != "1" ] && command -v hermes >/dev/null 2>&1; then
    say "registering with Hermes"
    hermes config set model.base_url "http://$LITMOE_HOST:$LITMOE_PORT/v1" >/dev/null 2>&1 || true
    hermes config set model.api_key "$LITMOE_API_KEY" >/dev/null 2>&1 || true
    hermes config set model.default "$LITMOE_MODEL_NAME" >/dev/null 2>&1 || true
    ok "Hermes configured"
fi

# --- 9. handoff ---
cat >&2 <<EOF

done.

  Server:   http://$LITMOE_HOST:$LITMOE_PORT
  Token:    $LITMOE_API_KEY
  Model:    $LITMOE_MODEL_NAME
  Log:      $LITMOE_DIR/server.log

Daily use:
  litMoE chat -m "hello"     # chat (needs weights)
  litMoE fetch               # download K3 weights (~982 GB)
  litMoE stop                 # stop server
  litMoE stack up --webui     # LAN stack with Open WebUI

EOF
exit 0