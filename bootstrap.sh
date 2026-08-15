#!/usr/bin/env bash
# bootstrap.sh — one-command install: clone, build, start server.
#
#   curl -fsSL https://raw.githubusercontent.com/chazhyseni/kimi-k3-lean/main/bootstrap.sh | bash
#
# What it does (in order, no detours):
#   1. Clone to $K3_DIR (default ~/.kimi-k3-lean)
#   2. Build libk3.so + bin/k3 (~30 sec)
#   3. Start the OpenAI server on http://127.0.0.1:8080
#   4. Wait for /v1/models to return 200
#   5. Install the `kimi-k3-lean` launcher to ~/.local/bin
#   6. Print the URL, token, and next steps
#
# The server starts with or without model weights. Without weights,
# /v1/models works and /v1/chat/completions returns a clear engine_error.
# Download weights with:  kimi-k3-lean fetch
#
# Env vars (all optional):
#   K3_DIR       install location     (default ~/.kimi-k3-lean)
#   K3_PORT      server port          (default 8080)
#   K3_HOST      bind address         (default 127.0.0.1)
#   K3_API_KEY   bearer token         (default random 32 hex)
#   K3_SKIP_DL=1 skip model detection (scaffold-only)
#   K3_NO_HERMES=1 skip Hermes config
#   K3_UNINSTALL=1 kill server + remove launcher

set -euo pipefail

K3_REPO="https://github.com/chazhyseni/kimi-k3-lean.git"
K3_DIR="${K3_DIR:-$HOME/.kimi-k3-lean}"
K3_PORT="${K3_PORT:-8080}"
K3_HOST="${K3_HOST:-127.0.0.1}"
K3_API_KEY="${K3_API_KEY:-$(head -c16 /dev/urandom 2>/dev/null | od -An -tx1 | tr -d ' \n' || openssl rand -hex 16 2>/dev/null || echo "k3-$(date +%s)-$$")}"
K3_MODEL_DIR="${K3_MODEL_DIR:-$K3_DIR/checkpoints/k3}"
K3_MODEL_NAME="${K3_MODEL_NAME:-kimi-k3}"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*" >&2; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m  !\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m==> ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- uninstall ---
if [ "${K3_UNINSTALL:-0}" = "1" ]; then
    say "uninstalling kimi-k3-lean"
    [ -f "$K3_DIR/server.pid" ] && kill "$(cat "$K3_DIR/server.pid" 2>/dev/null)" 2>/dev/null || true
    rm -f "$HOME/.local/bin/kimi-k3-lean"
    if [ "${K3_NO_HERMES:-0}" != "1" ] && command -v hermes >/dev/null 2>&1; then
        hermes config unset model.base_url model.api_key model.default 2>/dev/null || true
    fi
    ok "done. repo at $K3_DIR kept (rm -rf to wipe)"
    exit 0
fi

# --- prereqs ---
command -v git >/dev/null 2>&1 || die "git not found"
if command -v python3 >/dev/null 2>&1; then K3_PY=python3
elif command -v python >/dev/null 2>&1; then K3_PY=python
else die "Python 3.11+ not found"; fi
ok "python: $K3_PY"

# --- 1. clone ---
if [ -d "$K3_DIR/.git" ]; then
    say "updating $K3_DIR"
    git -C "$K3_DIR" fetch --depth 1 origin main >/dev/null 2>&1
    git -C "$K3_DIR" reset --hard origin/main >/dev/null 2>&1
    ok "updated (HEAD $(git -C "$K3_DIR" rev-parse --short HEAD))"
elif [ -d "$K3_DIR" ]; then
    die "$K3_DIR exists but is not a git repo. Move it or set K3_DIR."
else
    say "cloning to $K3_DIR"
    git clone --depth 1 "$K3_REPO" "$K3_DIR" || die "clone failed"
    ok "cloned"
fi
cd "$K3_DIR"

# --- 2. build ---
if [ ! -f bin/libk3.so ] || [ src -nt bin/libk3.so ]; then
    say "building"
    LDFLAGS="-lm -pthread" make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2) all
    ok "built"
else
    ok "build cached"
fi

# --- 3. model detection (symlink existing weights, don't auto-download) ---
if [ "${K3_SKIP_DL:-0}" != "1" ] && [ ! -d "$K3_MODEL_DIR" -o -z "$(find "$K3_MODEL_DIR" -name '*.safetensors' -print -quit 2>/dev/null)" ]; then
    # Look for existing K3-shaped weights on this host
    found=""
    for d in "$HOME/kimi-local/models/kimi-linear-shards" \
             "$HOME/kimi-local/models/kimi-linear" \
             "$HOME/.kimi-k3-lean/checkpoints/kimi-linear"; do
        if [ -d "$d" ] && find "$d" -maxdepth 2 -name '*.safetensors' -print -quit 2>/dev/null | grep -q .; then
            found="$d"; break
        fi
    done
    if [ -n "$found" ]; then
        mkdir -p "$(dirname "$K3_MODEL_DIR")"
        ln -sfn "$found" "$K3_MODEL_DIR"
        ok "using existing weights at $found"
    else
        warn "no model weights found. server will start; chat returns engine_error until you fetch:"
        warn "  kimi-k3-lean fetch"
        mkdir -p "$K3_MODEL_DIR"
    fi
fi

# --- 4. kill prior server on same port ---
if [ -f "$K3_DIR/server.pid" ]; then
    old=$(cat "$K3_DIR/server.pid" 2>/dev/null || echo "")
    [ -n "$old" ] && kill "$old" 2>/dev/null && sleep 0.5 || true
fi
# free the port if anything else is on it
if command -v fuser >/dev/null 2>&1; then
    fuser -k "${K3_PORT}/tcp" 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:"$K3_PORT" 2>/dev/null | xargs kill 2>/dev/null || true
fi

# --- 5. start server ---
say "starting server"
export LD_LIBRARY_PATH="$K3_DIR/bin:${LD_LIBRARY_PATH:-}"
cat > "$K3_DIR/server.env" <<EOF
K3_HOST=$K3_HOST
K3_PORT=$K3_PORT
K3_API_KEY=$K3_API_KEY
K3_MODEL_DIR=$K3_MODEL_DIR
K3_MODEL_NAME=$K3_MODEL_NAME
EOF
chmod 600 "$K3_DIR/server.env"

$K3_PY -u serve/__main__.py "$K3_MODEL_DIR" \
    --host "$K3_HOST" --port "$K3_PORT" \
    --api-key "$K3_API_KEY" --model-id "$K3_MODEL_NAME" \
    >>"$K3_DIR/server.log" 2>&1 </dev/null &
echo $! > "$K3_DIR/server.pid"
ok "server PID $(cat "$K3_DIR/server.pid")"

# --- 6. wait for /v1/models ---
ready=0
for i in $(seq 1 30); do
    code=$(curl -sS --max-time 1 -o /dev/null -w '%{http_code}' \
        -H "Authorization: Bearer $K3_API_KEY" \
        "http://$K3_HOST:$K3_PORT/v1/models" 2>/dev/null || echo 000)
    [ "$code" = "200" ] && { ready=1; break; }
    sleep 1
done

if [ "$ready" = "1" ]; then
    ok "server up: http://$K3_HOST:$K3_PORT"
else
    warn "server not ready after 30s; tail $K3_DIR/server.log"
fi

# --- 7. install launcher ---
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR" 2>/dev/null || true
if cp "$K3_DIR/kimi-k3-lean" "$BIN_DIR/kimi-k3-lean" 2>/dev/null; then
    chmod +x "$BIN_DIR/kimi-k3-lean"
    ok "launcher: $BIN_DIR/kimi-k3-lean"
    case ":$PATH:" in *":$BIN_DIR:"*) ;; *) warn "add to PATH: export PATH=$BIN_DIR:\$PATH" ;; esac
else
    warn "could not install launcher to $BIN_DIR"
fi

# --- 8. Hermes (optional, best-effort) ---
if [ "${K3_NO_HERMES:-0}" != "1" ] && command -v hermes >/dev/null 2>&1; then
    say "registering with Hermes"
    hermes config set model.base_url "http://$K3_HOST:$K3_PORT/v1" >/dev/null 2>&1 || true
    hermes config set model.api_key "$K3_API_KEY" >/dev/null 2>&1 || true
    hermes config set model.default "$K3_MODEL_NAME" >/dev/null 2>&1 || true
    ok "Hermes configured"
fi

# --- 9. handoff ---
cat >&2 <<EOF

done.

  Server:   http://$K3_HOST:$K3_PORT
  Token:    $K3_API_KEY
  Model:    $K3_MODEL_NAME
  Log:      $K3_DIR/server.log

Daily use:
  kimi-k3-lean chat -m "hello"     # chat (needs weights)
  kimi-k3-lean fetch               # download K3 weights (~982 GB)
  kimi-k3-lean stop                 # stop server
  kimi-k3-lean stack up --webui     # LAN stack with Open WebUI

EOF
exit 0