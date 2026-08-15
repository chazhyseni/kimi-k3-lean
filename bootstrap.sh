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
# Cross-platform random hex (POSIX has /dev/urandom; Windows has none).
if [ -r /dev/urandom ]; then
    _RAND_HEX_DEFAULT="$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
elif command -v openssl >/dev/null 2>&1; then
    _RAND_HEX_DEFAULT="$(openssl rand -hex 16 2>/dev/null)"
else
    _RAND_HEX_DEFAULT="k3-$(date +%s)-$$"
fi
K3_API_KEY="${K3_API_KEY:-$_RAND_HEX_DEFAULT}"
K3_MODEL_NAME="${K3_MODEL_NAME:-kimi-k3}"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*" >&2; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m  !\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m==> ERROR:\033[0m %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing: $1  ($2)"; }

# ----- OS / Python detection (cross-platform) -----
case "$(uname -s 2>/dev/null)" in
    Linux|GNU*)  K3_OS=linux  ;;
    Darwin)     K3_OS=macos  ;;
    MINGW*|MSYS*|CYGWIN*) K3_OS=windows ;;
    *)          K3_OS=other  ;;
esac

if command -v python3 >/dev/null 2>&1; then
    K3_PY=python3
elif command -v python >/dev/null 2>&1; then
    K3_PY=python
else
    die "Python 3.11+ not found; install it (apt/dnf/brew/winget) first"
fi
K3_PY_VERSION=$($K3_PY -c "import sys;print('%d.%d'%sys.version_info[:2])")
case "$K3_PY_VERSION" in
    3.1[1-9]|3.[2-9]*) : ;;
    *) die "Python $K3_PY_VERSION found; need 3.11+" ;;
esac
ok "python: $K3_PY $K3_PY_VERSION  ($K3_OS)"

# CPU count with multiple fallbacks (Linux, macOS, git-bash on Windows).
if command -v nproc >/dev/null 2>&1; then
    K3_NPROC=$(nproc)
elif [ -r /proc/cpuinfo ]; then
    K3_NPROC=$(grep -c ^processor /proc/cpuinfo)
elif command -v sysctl >/dev/null 2>&1; then
    K3_NPROC=$(sysctl -n hw.ncpu)
else
    K3_NPROC=2
fi

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

# ----- 1. platform detect (already done above) -----
# K3_OS / K3_PY / K3_PY_VERSION / K3_NPROC are set in the config section.
# git is the only other prereq; check it now since the platform check
# already happened.
need git "install git first (apt/dnf/brew/winget/git-for-windows)"
PLATFORM="$K3_OS"
ok "platform: $PLATFORM (python: $K3_PY $K3_PY_VERSION, nproc: $K3_NPROC)"

if [ -d "$K3_DIR/.git" ]; then
    say "updating $K3_DIR"
    git -C "$K3_DIR" fetch --depth 1 origin "$K3_BRANCH" >/dev/null 2>&1 \
        || die "could not fetch $K3_BRANCH from origin"
    git -C "$K3_DIR" reset --hard "origin/$K3_BRANCH" >/dev/null 2>&1 \
        || die "could not reset to origin/$K3_BRANCH"
    # Sanity check: confirm HEAD == origin/HEAD after the reset. Anything
    # else indicates a race (concurrent editor, lock contention) or partial
    # checkout; surface it loudly so the user knows what HEAD they're running.
    local_head=$(git -C "$K3_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
    origin_head=$(git -C "$K3_DIR" rev-parse --short "origin/$K3_BRANCH" 2>/dev/null || echo unknown)
    if [ "$local_head" = "$origin_head" ] && [ "$local_head" != "unknown" ]; then
        ok "updated (HEAD $local_head)"
    else
        warn "post-reset HEAD is $local_head, origin is $origin_head; retrying once:"
        git -C "$K3_DIR" reset --hard "origin/$K3_BRANCH" >/dev/null 2>&1 \
            || die "could not reset on retry"
        local_head=$(git -C "$K3_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
        ok "retry OK (HEAD $local_head)"
    fi
elif [ -d "$K3_DIR" ]; then
    die "$K3_DIR exists but is not a git repo. Move it or set K3_DIR to a different path."
else
    say "cloning to $K3_DIR"
    git clone --depth 1 --branch "$K3_BRANCH" "$K3_REPO_URL" "$K3_DIR" \
        || die "git clone failed"
    ok "cloned (HEAD $(git -C "$K3_DIR" rev-parse --short HEAD))"
fi
cd "$K3_DIR"

# ----- 3. build -----
if [ "${K3_SKIP_BUILD:-0}" = "1" ]; then
    ok "skipping build (K3_SKIP_BUILD=1)"
elif [ ! -f bin/libk3.so ] || [ src -nt bin/libk3.so ] || [ include -nt bin/libk3.so ]; then
    say "building libk3.so + bin/k3"
    LDFLAGS="-lm -pthread" make -j"$K3_NPROC" all
    ok "built"
else
    ok "build cached"
fi

# ----- 3b. model dir -----
# Default: download K3 weights in the background and start the server.
# The model does not perform without the weights (the C engine returns
# engine_error on /v1/chat/completions with no model on disk). Set
# K3_SKIP_DL=1 to skip the download.
DL_NEEDED=1
if [ "${K3_SKIP_DL:-0}" = "1" ]; then
    ok "skipping model download (K3_SKIP_DL=1)"
    DL_NEEDED=0
elif find "$K3_MODEL_DIR" -maxdepth 2 -name "*.safetensors" -print -quit 2>/dev/null | grep -q .; then
    ok "model already at $K3_MODEL_DIR (has safetensors shards)"
    DL_NEEDED=0
elif [ -f "$K3_DIR/download.pid" ] && kill -0 "$(cat "$K3_DIR/download.pid" 2>/dev/null)" 2>/dev/null; then
    pid=$(cat "$K3_DIR/download.pid")
    ok "model download in progress (pid $pid); tail $K3_DIR/download.log"
    DL_NEEDED=0
fi

if [ "$DL_NEEDED" = "1" ]; then
    # Stale partial download (e.g. crashed earlier with .cache/ but no
    # shards). Warn the user but don't delete -- we don't know if they
    # put something precious there.
    if [ -d "$K3_MODEL_DIR" ] && [ -n "$(ls -A "$K3_MODEL_DIR" 2>/dev/null)" ]; then
        warn "model dir had non-shard content (stale partial download?); keeping it"
        warn "  if the download keeps failing, run:"
        warn "    rm -rf $K3_MODEL_DIR && kimi-k3-lean fetch"
    fi
    mkdir -p "$K3_MODEL_DIR"
    say "downloading Kimi K3 weights in background (~982 GB, ~4 hours, resumable)"
    say "  tail -f $K3_DIR/download.log for progress"
    # Pre-flight: skip the 4-hour transfer if `hf` is broken. Old
    # system huggingface_hub installs report success on `hf --help`
    # but crash with `TypeError: __init__() got an unexpected keyword
    # argument 'mode'` the moment they touch a real file. The fix is
    # in scripts/download-model.sh (isolated venv). If the user has
    # a fresh-enough `hf`, this is a no-op.
    if command -v hf >/dev/null 2>&1; then
        if ! hf cache verify --help >/dev/null 2>&1; then
            warn "system \`hf\` is missing 'cache verify'; download-model.sh will repair"
        fi
    else
        warn "\`hf\` not on PATH; download-model.sh will install a venv"
    fi

    export K3_DIR K3_MODEL_DIR
    nohup bash "$K3_DIR/scripts/download-model.sh" "$K3_MODEL_DIR" \
        >> "$K3_DIR/download.log" 2>&1 &
    echo "$!" > "$K3_DIR/download.pid"
    ok "download started (pid $(cat "$K3_DIR/download.pid"))"
    # Quick pre-flight: if the download subprocess exits within 5
    # seconds it's almost certainly the broken-hf bug. Catch it
    # here so the user doesn't discover it 4 hours from now.
    sleep 5
    if ! kill -0 "$(cat "$K3_DIR/download.pid")" 2>/dev/null; then
        warn "download subprocess exited early; check $K3_DIR/download.log"
        warn "  fix: rm -rf ~/.local/share/kimi-k3-lean/hf-venv and re-run,"
        warn "       or set K3_SKIP_DL=1 and run \`kimi-k3-lean fetch\` separately."
    fi
fi
# ----- 4. launch server in background -----
say "launching kimi-k3-lean server (background)"
export LD_LIBRARY_PATH="$K3_DIR/bin:${LD_LIBRARY_PATH:-}"

# Stop any prior server whose PID file points at our repo AND this same
# port. Idempotent. The port match matters because the user might
# legitimately have two k3-lean installs running on different ports
# (e.g. test on 8081 alongside prod on 8080).
#
# Belt and suspenders: if the PID file is missing (e.g. cleared during
# git reset --hard) but serve/__main__.py is still running anywhere in
# $K3_DIR, scan for it and kill. Without this, an updated bootstrap can't
# free up the port for a brand new server.
_kill_prior_serve() {
    local sig="$1"
    shift
    for p in "$@"; do
        # Match on identity of being our process. We accept a PID if ANY
        # of these hold (cross-platform):
        #   1. /proc/PID/cwd resolves into $K3_DIR (Linux)
        #   2. lsof on its cwd shows $K3_DIR (Linux + macOS w/ lsof)
        #   3. ps command line contains $K3_DIR AND contains
        #      serve/__main__.py (works anywhere ps exists)
        # macOS has NO /proc filesystem; relying on it caused a silent
        # no-op that left stale servers bound to the port.
        if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
            continue
        fi
        local matches=0
        # 1. /proc/PID/cwd (Linux)
        if [ -L "/proc/$p/cwd" ]; then
            local cwd
            cwd=$(readlink "/proc/$p/cwd" 2>/dev/null || echo "")
            case "$cwd" in "$K3_DIR"*) matches=1 ;; esac
        fi
        # 2. lsof cwd (Linux, macOS with lsof)
        if [ "$matches" = "0" ] && command -v lsof >/dev/null 2>&1; then
            local cwd2
            cwd2=$(lsof -p "$p" -a -d cwd -F n 2>/dev/null | head -1 | sed "s/^n//")
            case "$cwd2" in "$K3_DIR"*) matches=1 ;; esac
        fi
        # 3. ps command line contains $K3_DIR + serve/__main__.py
        #    (works on macOS, BSD, anything with ps)
        if [ "$matches" = "0" ] && command -v ps >/dev/null 2>&1; then
            local cmdline
            cmdline=$(ps -p "$p" -o args= 2>/dev/null || true)
            case "$cmdline" in
                *serve/__main__.py*"$K3_DIR"*) matches=1 ;;
                *) : ;;
            esac
        fi
        if [ "$matches" = "1" ]; then
            warn "killing prior server (PID $p, $sig)"
            if [ "$sig" = "SIGKILL" ]; then
                kill -9 "$p" 2>/dev/null || true
            else
                kill "$p" 2>/dev/null || true
            fi
        fi
    done
}

# Source 1: PID file
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

# Source 2: anything serving from our repo, even if the PID file is gone.
# NOTE: Use case instead of `[ -n "$x" ]` because an empty prior_pids under
# `set -euo pipefail` would make the conditional itself exit 1 and abort
# the whole bootstrap. case always evaluates true (matched) so it doesn't trip set -e.
prior_pids=$(ps -ef 2>/dev/null | grep "serve/__main__.py" | grep "$K3_DIR" | awk "{print \$2}" | tr "\n" "," | sed "s/,$//" || true)
case "$prior_pids" in
    "")
        : ;;  # nothing stale; skip
    *)
        warn "found stale serve/__main__.py still running in $K3_DIR (PID(s): $prior_pids)"
        _kill_prior_serve SIGTERM $prior_pids || true
        for i in 1 2 3 4 5 6 7 8 9 10; do
            still=$(echo "$prior_pids" | tr " " "\n" | while read p; do ps -p "$p" -o pid= 2>/dev/null; done || true)
            case "$still" in
                "") break ;;
                *) sleep 0.3 ;;
            esac
        done
        # Recheck; SIGKILL the survivors
        still_pids=$(ps -ef 2>/dev/null | grep "serve/__main__.py" | grep "$K3_DIR" | awk "{print \$2}" || true)
        case "$still_pids" in
            "") : ;;
            *) _kill_prior_serve SIGKILL $still_pids || true ;;
        esac
        ;;
esac
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
$K3_PY -u serve/__main__.py "$K3_MODEL_DIR" \
    --host "$K3_HOST" --port "$K3_PORT" --preset "$K3_PRESET" \
    --api-key "$K3_API_KEY" --model-id "$K3_MODEL_NAME" \
    >>"$K3_DIR/server.log" 2>&1 </dev/null &
SERVER_PID=$!
echo "$SERVER_PID" > "$K3_DIR/server.pid"
ok "server PID: $SERVER_PID  (log: $K3_DIR/server.log)"

# Quick early-exit check: capture the python PID. On Linux $! is python
# itself; on macOS the bash + python wrapper sometimes gives the wrapper
# PID, so we wait 2s and re-resolve via pgrep + ps-cmdline match.
# Cross-platform: works with /proc on Linux or without on macOS.
sleep 2
resolved=""
if ps -p "$SERVER_PID" -o args= 2>/dev/null | grep -q "serve/__main__.py"; then
    resolved="$SERVER_PID"
else
    for p in $(pgrep -f "serve/__main__.py" 2>/dev/null || true); do
        cmd=$(ps -p "$p" -o args= 2>/dev/null || true)
        case "$cmd" in
            *serve/__main__.py*"$K3_MODEL_DIR"*)
                resolved="$p"
                ;;
            *"serve/__main__.py"*) resolved="$p" ;;
        esac
        [ -n "$resolved" ] && break
    done
fi
if [ -n "$resolved" ]; then
    if [ "$resolved" != "$SERVER_PID" ]; then
        warn "captured PID $SERVER_PID is a wrapper; real server PID: $resolved"
    fi
    echo "$resolved" > "$K3_DIR/server.pid"
    SERVER_PID="$resolved"
else
    warn "no serve/__main__.py process found after 2s; surfacing log tail"
    tail -n 20 "$K3_DIR/server.log" 2>/dev/null | sed "s/^/    /" || true
fi

# If the previous run left a non-empty log, surface the last few lines
# during startup so a failure self-diagnoses without the user needing
# to \`tail -f\`.
if [ -s "$K3_DIR/server.log" ]; then
    last_lines=$(tail -n 5 "$K3_DIR/server.log" 2>/dev/null || true)
    if [ -n "$last_lines" ]; then
        echo "last server.log lines from prior run:"
        printf "%s\n" "$last_lines" | sed "s/^/    /"
    fi
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
        $K3_PY -c "
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

# ----- 6. install the launcher -----
# The launcher goes into ~/.local/bin/ on Linux/macOS (PATH-friendly on
# both; works out of the box on any machine that follows XDG Base Dir).
# On Windows under git-bash, this is still usable but the proper Windows
# installer is bootstrap.ps1 (drop into %LOCALAPPDATA%\Programs or similar).
if [ "$K3_OS" = "windows" ]; then
    # git-bash on Windows: ~/.local/bin is fine for command-line use,
    # but for full Windows integration, run bootstrap.ps1 instead.
    K3_BIN_DIR="$HOME/.local/bin"
    warn "Windows detected: bootstrap.sh installed the launcher for git-bash use."
    warn "For full PowerShell integration, also run bootstrap.ps1."
else
    K3_BIN_DIR="$HOME/.local/bin"
fi
K3_BIN="$K3_BIN_DIR/kimi-k3-lean"
mkdir -p "$K3_BIN_DIR" 2>/dev/null || true
if [ -d "$K3_BIN_DIR" ] && [ -w "$K3_BIN_DIR" ]; then
    rm -f "$K3_BIN"
    if cp "$K3_DIR/kimi-k3-lean" "$K3_BIN" 2>/dev/null; then
        chmod +x "$K3_BIN"
        ok "launcher installed: $K3_BIN"
        case ":$PATH:" in
            *":$K3_BIN_DIR:"*) ok "PATH already includes $K3_BIN_DIR" ;;
            *) warn "$K3_BIN_DIR is not on PATH; add this to your shell rc:"
               warn "    export PATH=$K3_BIN_DIR:\$PATH" ;;
        esac
    else
        warn "could not install launcher to $K3_BIN (perms?)"
    fi
else
    warn "skipped launcher install (no write access to $K3_BIN_DIR)"
fi

# ----- 6.5. auto-download (only with K3_DOWNLOAD=1) -----
# If the user sets K3_DOWNLOAD=1, kick off the K3 weights download in the
# background right after the scaffold is up. Resumable.
if [ "${K3_DOWNLOAD:-0}" = "1" ]; then
    if [ -d "$K3_DIR/scripts" ]; then
        say "K3_DOWNLOAD=1: starting K3 weights download in background (~4 hrs)"
        ( cd "$K3_DIR" && nohup bash "$K3_DIR/scripts/setup-and-serve.sh" --download-only             >> "$K3_DIR/download.log" 2>&1 & )
        ok "download started; tail -f $K3_DIR/download.log"
    fi
fi

# ----- 7. hand-off -----
cat >&2 <<EOF
done: $(date -u +%Y-%m-%dT%H:%M:%SZ)

  Server:   http://$K3_HOST:$K3_PORT
  Launcher: $K3_BIN
  Model:    $K3_MODEL_NAME
  Log:      $K3_DIR/server.log
  Token:    (in $K3_DIR/server.env, mode 0600)
  Platform: $K3_OS (python: $K3_PY)

Daily-use (only commands you ever need):

  kimi-k3-lean start            # start the server (skips download; backgound still running)
  kimi-k3-lean stop
  kimi-k3-lean chat -m "hello"  # works once download finishes
  kimi-k3-lean doctor           # print install state

For a full LAN stack with Open WebUI:

  kimi-k3-lean stack up --webui

To uninstall:

  kimi-k3-lean uninstall

---

Done. The scaffold is up and the K3 weights are downloading in the
background. When the download finishes (~4 hours), restart the
server with `kimi-k3-lean stop && kimi-k3-lean start` and chat
completions will work. Tail progress with
`kimi-k3-lean fetch --status` or `tail -f ~/.kimi-k3-lean/download.log`.
EOF

exit 0
