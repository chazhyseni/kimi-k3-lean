#!/usr/bin/env bash
# bootstrap.sh — install kimi-k3-lean, start the server in the background,
# register it with Hermes, and hand the user a working local LLM endpoint.
#
# Usage (the headline use case):
#   curl -fsSL https://raw.githubusercontent.com/chazhyseni/kimi-k3-lean/main/bootstrap.sh | bash
#
# What this script ACTUALLY does (and what the previous version lied about):
#
#   1. Detects the platform (Linux/macOS), the Python in use, and the
#      shell that's about to take over (bash/zsh on Linux/macOS,
#      PowerShell on Windows).
#   2. Clones the repo to $K3_DIR (default: ~/.kimi-k3-lean).
#      Re-running over the same dir updates it.
#   3. Builds the C engine if the .so is missing or older than sources.
#      Relies on `make`, `gcc` (or Homebrew `clang` on macOS), and Python.
#   4. If a real K3 checkpoint exists at $K3_DIR/checkpoints/k3, starts
#      the server with that model. Otherwise the server starts empty and
#      reports "engine open failed". The user MUST then either let the
#      full setup-and-serve.sh run (downloads ~982 GB) OR add a model
#      later via --model-dir. The previous version claimed the article's
#      tiny_k3.bin fixture would work for HTTP-layer testing — it 404s
#      because the fixture's vocab=256 doesn't match the article's
#      tokenizer vocab=163584. The fixture only proves the C engine
#      matches the oracle, not that the HTTP layer returns tokens.
#      Don't pretend otherwise.
#   5. Launches the server in the BACKGROUND (writes PID to $K3_DIR/server.pid,
#      log to $K3_DIR/server.log). Blocks until /v1/models returns 200 or
#      up to 30s, whichever comes first, then exits 0. If the engine can't
#      open the model, the script still exits 0 with a clear next-step
#      message — the user shouldn't have to debug bootstrap.sh to find
#      out their model is missing.
#   6. REGISTER THE MODEL WITH HERMES. This is the part the previous
#      version got wrong, and why you saw
#         HTTP 404: model 'kimi-k3' not found
#      Hermes's per-provider model registry (config.yaml > providers.*.models)
#      has to list `kimi-k3` for `-m kimi-k3` to resolve. We add it under
#      the right provider and flip model.default to it.
#   7. Prints the next steps: how to talk to it (curl), how to verify
#      (the 5-demo test block), how to download the real K3 weights, and
#      how to stop the server.
#
# Env vars (optional):
#   K3_DIR         install location                (default: ~/.kimi-k3-lean)
#   K3_PORT        server port                     (default: 8080)
#   K3_HOST        bind address                    (default: 127.0.0.1)
#   K3_API_KEY     bearer token                    (default: random 32 hex)
#   K3_PRESET      memory preset                   (default: auto)
#   K3_MODEL_DIR   model checkpoint dir             (default: $K3_DIR/checkpoints/k3)
#   K3_SKIP_DL=1   don't download the K3 weights    (use existing model)
#   K3_SKIP_BUILD=1   skip make
#   K3_NO_HERMES=1    skip the Hermes config update (CI, servers, etc.)
#   K3_UNINSTALL=1    remove server + Hermes changes

set -euo pipefail

# ----- configuration -----
K3_REPO_URL="${K3_REPO_URL:-https://github.com/chazhyseni/kimi-k3-lean.git}"
K3_BRANCH="${K3_BRANCH:-main}"
K3_DIR="${K3_DIR:-$HOME/.kimi-k3-lean}"
K3_PORT="${K3_PORT:-8080}"
K3_HOST="${K3_HOST:-127.0.0.1}"
K3_PRESET="${K3_PRESET:-auto}"
K3_MODEL_DIR="${K3_MODEL_DIR:-$K3_DIR/checkpoints/k3}"
K3_API_KEY="${K3_API_KEY:-$(openssl rand -hex 16 2>/dev/null || echo "k3-$(date +%s)-$$")}"
K3_MODEL_NAME="${K3_MODEL_NAME:-kimi-k3}"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*" >&2; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m  !\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m==> ERROR:\033[0m %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing: $1  ($2)"; }

# ----- uninstall path -----
if [ "${K3_UNINSTALL:-0}" = "1" ]; then
    say "uninstalling kimi-k3-lean"
    if [ -f "$K3_DIR/server.pid" ]; then
        pid=$(cat "$K3_DIR/server.pid")
        kill "$pid" 2>/dev/null && ok "killed server PID $pid" || warn "PID $pid not running"
    fi
    # Roll back Hermes config: best-effort, never fatal
    if command -v hermes >/dev/null 2>&1; then
        hermes config unset providers.ollama-launch.models 2>/dev/null || true
        hermes config unset model.base_url 2>/dev/null || true
        hermes config unset model.api_key 2>/dev/null || true
        hermes config unset model.default 2>/dev/null || true
        ok "rolled back Hermes config"
    fi
    rm -f "$K3_DIR/server.pid" "$K3_DIR/server.log"
    ok "bootstrap done. repo lives at $K3_DIR (manually 'rm -rf' to remove)"
    exit 0
fi

# ----- 1. platform detect -----
OS="$(uname -s)"
case "$OS" in
    Linux)  PLATFORM=linux ;;
    Darwin) PLATFORM=macos ;;
    *)      die "unsupported platform: $OS. Use bootstrap.ps1 on Windows." ;;
esac
ok "platform: $PLATFORM"

need git   "install git first"
need python3 "install Python 3.11+ first"

# check python version
pyv=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "0.0")
case "$pyv" in
    3.1[1-9]|3.[2-9].*|3.[1-9][0-9].*) ;;  # 3.11+
    *) die "Python $pyv found; need 3.11+." ;;
esac
ok "python $pyv"

# ----- 2. clone or update -----
if [ -d "$K3_DIR/.git" ]; then
    say "updating $K3_DIR"
    git -C "$K3_DIR" fetch --depth 1 origin "$K3_BRANCH" >/dev/null 2>&1 \
        || die "could not fetch $K3_BRANCH from origin"
    git -C "$K3_DIR" reset --hard "origin/$K3_BRANCH" >/dev/null 2>&1 \
        || die "could not reset to origin/$K3_BRANCH"
    ok "updated"
elif [ -d "$K3_DIR" ]; then
    die "$K3_DIR exists but is not a git repo. Move it or set K3_DIR to a different path."
else
    say "cloning to $K3_DIR"
    git clone --depth 1 --branch "$K3_BRANCH" "$K3_REPO_URL" "$K3_DIR"
    ok "cloned"
fi
cd "$K3_DIR"

# ----- 3. build -----
if [ "${K3_SKIP_BUILD:-0}" = "1" ]; then
    ok "skipping build (K3_SKIP_BUILD=1)"
elif [ ! -f bin/libk3.so ] || [ src -nt bin/libk3.so ] || [ include -nt bin/libk3.so ]; then
    say "building libk3.so + bin/k3"
    LDFLAGS="-lm -pthread" make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)" all
    ok "built"
else
    ok "build cached"
fi

# ----- 3b. model dir -----
if [ "${K3_SKIP_DL:-0}" = "1" ]; then
    ok "skipping model download (K3_SKIP_DL=1)"
elif [ -d "$K3_MODEL_DIR" ] && [ -n "$(ls "$K3_MODEL_DIR" 2>/dev/null)" ]; then
    ok "model already exists at $K3_MODEL_DIR"
else
    warn "no model at $K3_MODEL_DIR"
    warn "to download the real Kimi K3 (~982 GB, ~4 hours):"
    warn "  K3_DIR=$K3_DIR bash $K3_DIR/scripts/setup-and-serve.sh --download-only"
    warn "the server will start without a model and report 'engine open failed'."
    warn "you'll still get the HTTP scaffold + Hermes registration,"
    warn "so any future download immediately becomes accessible."
    # create empty placeholder so the rest of the pipeline doesn't 404
    mkdir -p "$K3_MODEL_DIR"
fi

# ----- 4. launch server in background -----
say "launching kimi-k3-lean server (background)"
export LD_LIBRARY_PATH="$K3_DIR/bin:${LD_LIBRARY_PATH:-}"

# Stop any prior server whose PID file points at our repo AND this same
# port. Idempotent. The port match matters because the user might
# legitimately have two k3-lean installs running on different ports
# (e.g. test on 8081 alongside prod on 8080).
if [ -f "$K3_DIR/server.pid" ]; then
    old_pid=$(cat "$K3_DIR/server.pid" 2>/dev/null || echo "")
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
        if ps -p "$old_pid" -o args= 2>/dev/null | grep -q "serve/__main__.py" \
            && ps -p "$old_pid" -o args= 2>/dev/null | grep -q -- "--port $K3_PORT\b"; then
            warn "killing prior server on same port (PID $old_pid)"
            kill "$old_pid" 2>/dev/null || true
            for i in $(seq 1 50); do
                kill -0 "$old_pid" 2>/dev/null || break
                sleep 0.1
            done
        else
            ok "prior server PID $old_pid is alive but on a different port; leaving it alone"
        fi
    fi
    rm -f "$K3_DIR/server.pid"
fi
# Create a small env file the user can `source` to recover the API key
cat > "$K3_DIR/server.env" <<EOF
K3_HOST=$K3_HOST
K3_PORT=$K3_PORT
K3_API_KEY=$K3_API_KEY
K3_PRESET=$K3_PRESET
K3_MODEL_DIR=$K3_MODEL_DIR
K3_MODEL_NAME=$K3_MODEL_NAME
EOF
chmod 600 "$K3_DIR/server.env"

# Start in background. `python3 -u` so prints flush immediately to the
# log (without -u, block-buffered stdout hides startup messages from
# users watching `tail -f`). The backgrounded python inherits its
# stdout/stderr from bootstrap, then bootstrap waits below for the
# listener to come up before exiting.
python3 -u serve/__main__.py "$K3_MODEL_DIR" \
    --host "$K3_HOST" --port "$K3_PORT" --preset "$K3_PRESET" \
    --api-key "$K3_API_KEY" --model-id "$K3_MODEL_NAME" \
    >>"$K3_DIR/server.log" 2>&1 </dev/null &
SERVER_PID=$!
echo "$SERVER_PID" > "$K3_DIR/server.pid"
ok "server PID: $SERVER_PID  (log: $K3_DIR/server.log)"

# Defensive: if for some reason the captured PID isn't the server
# (e.g. python printed an error and forked something else), discover
# the real one. pgrep is racy; the /proc/$p/cwd symlink check avoids
# catching a serve running from some other path.
sleep 1
if ! ps -p "$SERVER_PID" -o args= 2>/dev/null | grep -q "serve/__main__.py"; then
    warn "captured PID $SERVER_PID isn't python; discovering the real one"
    for p in $(pgrep -f "serve/__main__.py" 2>/dev/null); do
        if [ -r "/proc/$p/cwd" ]; then
            cw=$(readlink "/proc/$p/cwd" 2>/dev/null || true)
            case "$cw" in
                "$K3_DIR"*) echo "$p" > "$K3_DIR/server.pid"; SERVER_PID=$p; ok "real server PID: $SERVER_PID"; break ;;
            esac
        fi
    done
fi

# ----- 5. wait for /v1/models (max 30s) -----
# /v1/models is auth-protected, so probe it with the bearer token we just
# generated. A 200 means the server bound the port, the engine created its
# context, and the HTTP layer is reachable. The server prints its own
# startup messages to $K3_DIR/server.log via `python3 -u` so the user can
# tail that for engine errors (model missing, tokenizer mismatch, etc.).
ready=0
for i in $(seq 1 30); do
    code=$(curl -sS --max-time 1 -o /dev/null -w '%{http_code}' \
        -H "Authorization: Bearer $K3_API_KEY" \
        "http://$K3_HOST:$K3_PORT/v1/models" 2>/dev/null || echo 000)
    case "$code" in
        200) ready=1; break ;;
        401) warn "server up but rejected our token; check --api-key vs K3_API_KEY" ;;
        000) : ;;  # not yet listening
        *)   warn "unexpected $code on /v1/models; continuing to poll" ;;
    esac
    sleep 1
done

if [ "$ready" = "1" ]; then
    ok "server up: http://$K3_HOST:$K3_PORT  (model $K3_MODEL_NAME registered)"
else
    warn "server didn't respond on http://$K3_HOST:$K3_PORT within 30s"
    warn "tail $K3_DIR/server.log for details. Most common cause: model missing"
    warn "the server is still running (PID $SERVER_PID); it'll re-try when the model arrives."
fi

# ----- 6. register with Hermes -----
hermes_configure() {
    if [ "${K3_NO_HERMES:-0}" = "1" ]; then
        ok "skipping Hermes (K3_NO_HERMES=1)"
        return 0
    fi
    if ! command -v hermes >/dev/null 2>&1; then
        warn "hermes CLI not on PATH; skipping model registration"
        warn "once you install Hermes, run: hermes config set providers.ollama-launch.models \\"
        warn "  \"['$(hermes config get providers.ollama-launch.default_model 2>/dev/null || echo minimax-m3:cloud)', '$K3_MODEL_NAME']\""
        return 0
    fi

    say "registering $K3_MODEL_NAME with Hermes"

    # 6a. point Hermes at our local server
    hermes config set model.base_url "http://$K3_HOST:$K3_PORT/v1"          >/dev/null 2>&1 && ok "model.base_url set"
    hermes config set model.api_key   "$K3_API_KEY"                          >/dev/null 2>&1 && ok "model.api_key set"

    # 6b. add $K3_MODEL_NAME to the provider's model list WITHOUT
    #     wiping the existing models. Hermes's `config set` for list
    #     values overwrites the entire list, so we read-modify-write.
    # hermes config get emits a YAML-style multi-line list:
    #     - minimax-m3:cloud
    #     - kimi-linear
    # which is neither JSON nor what `json.load` expects. Parse it line by
    # line, keep just "- word" tokens, dedupe, append $K3_MODEL_NAME.
    new=$(hermes config get providers.ollama-launch.models 2>/dev/null | \
        python3 -c "
import re, json, sys
items = []
for line in sys.stdin:
    m = re.match(r'^\s*-\s*(\S+)\s*\$', line.rstrip())
    if m: items.append(m.group(1))
if '$K3_MODEL_NAME' not in items:
    items.append('$K3_MODEL_NAME')
print(json.dumps(items))
")
    hermes config set providers.ollama-launch.models "$new" >/dev/null 2>&1 \
        && ok "providers.ollama-launch.models now: $new"

    # 6c. flip the default
    hermes config set model.default "$K3_MODEL_NAME" >/dev/null 2>&1 \
        && ok "model.default = $K3_MODEL_NAME"

    # 6d. verify: a quick `-m` resolution shouldn't return 404 anymore
    if [ "$ready" = "1" ]; then
        http_code=$(curl -sS --max-time 3 -o /dev/null -w '%{http_code}' \
            -X POST "http://$K3_HOST:$K3_PORT/v1/chat/completions" \
            -H "Authorization: Bearer $K3_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$K3_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
            2>/dev/null || echo 000)
        case "$http_code" in
            200)        ok "Hermes round-trip: 200 OK — model resolves end-to-end" ;;
            401|403)    warn "server responded $http_code; check model.api_key in Hermes config" ;;
            404)        warn "server responded 404; the model name you registered and the one in model.default differ" ;;
            500|503)    warn "server responded $http_code; the server is up but the model isn't loadable. tail server.log" ;;
            *)          warn "server responded $http_code; passing through" ;;
        esac
    fi
}
hermes_configure

# ----- 7. hand-off -----
cat >&2 <<EOF

\033[1;36m==> done\033[0m

Server:      http://$K3_HOST:$K3_PORT
API key:     (in $K3_DIR/server.env, mode 0600)
Model:       $K3_MODEL_NAME   (registered with Hermes as default)
Test it:     see README § "Test it" for five real curl demos
Stop:        kill \$(cat $K3_DIR/server.pid)
Tail log:    tail -f $K3_DIR/server.log
Download K3: K3_DIR=$K3_DIR bash $K3_DIR/scripts/setup-and-serve.sh --download-only
Remove:      curl ... | K3_UNINSTALL=1 bash

If you saw \`warn: server didn't respond\` above, the model wasn't on disk.
The HTTP scaffold and Hermes registration are in place; the round-trip
will start working as soon as the K3 weights finish downloading.

The article's \`tiny_k3.bin\` fixture does NOT give working chat
completions — it's only used by the C engine's 3-GATE oracle tests.
Downloading the real K3 is the only path to a working model server.

EOF
exit 0
