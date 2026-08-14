#!/usr/bin/env bash
# bootstrap.sh — install kimi-k3-lean and start the server in one command.
#
# Usage (the headline use case):
#   curl -fsSL https://raw.githubusercontent.com/chazhyseni/kimi-k3-lean/main/bootstrap.sh | bash
#
# This script does not require sudo, does not touch anything outside
# $K3_DIR, and is safe to pipe from curl. It is the equivalent of:
#   1. git clone https://github.com/chazhyseni/kimi-k3-lean.git ~/.kimi-k3-lean
#   2. cd ~/.kimi-k3-lean
#   3. ./scripts/setup-and-serve.sh --serve-only    (or full setup if no model)
#
# If you want to install the C engine + Python server to a system prefix
# instead, use ./install.sh from inside a clone.
#
# Environment:
#   K3_DIR         install location (default: ~/.kimi-k3-lean)
#   K3_PORT        server port (default: 8080)
#   K3_HOST        bind host (default: 127.0.0.1)
#   K3_API_KEY     optional bearer token for the server
#   K3_PRESET      memory preset: laptop | desktop | workstation | server | max | auto
#                  (default: auto)
#   K3_SKIP_DL     set to 1 to skip the model download (server-only)

set -euo pipefail

REPO_URL="https://github.com/chazhyseni/kimi-k3-lean.git"
BRANCH="main"

K3_DIR="${K3_DIR:-$HOME/.kimi-k3-lean}"
K3_PORT="${K3_PORT:-8080}"
K3_HOST="${K3_HOST:-127.0.0.1}"
K3_PRESET="${K3_PRESET:-auto}"
K3_SKIP_DL="${K3_SKIP_DL:-0}"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*" >&2; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m==> ERROR:\033[0m %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing: $1  ($2)"; }

# ----- 1. preflight -----
say "preflight"
need git     "install with your package manager"
need make    "install with your package manager"
need gcc     "or cc — install build-essential / Xcode CLI tools"
need python3 "Python 3.11+ recommended"

# ----- 2. clone (or update) -----
if [ -d "$K3_DIR/.git" ]; then
    say "updating $K3_DIR"
    git -C "$K3_DIR" fetch --depth 1 origin "$BRANCH" >/dev/null 2>&1 \
        || git -C "$K3_DIR" fetch --depth 1 origin main
    git -C "$K3_DIR" reset --hard "origin/$BRANCH" >/dev/null 2>&1 \
        || git -C "$K3_DIR" reset --hard origin/main
    ok "updated"
elif [ -d "$K3_DIR" ]; then
    die "$K3_DIR exists but is not a git repo. Remove it or set K3_DIR."
else
    say "cloning to $K3_DIR"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$K3_DIR"
    ok "cloned"
fi

cd "$K3_DIR"

# ----- 3. build -----
if [ ! -f bin/libk3.so ] || [ src -nt bin/libk3.so ] || [ include -nt bin/libk3.so ]; then
    say "building libk3.so + bin/k3"
    LDFLAGS="-lm -pthread" make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)" all
    ok "built"
else
    ok "build cached"
fi

# ----- 4. start the server -----
export LD_LIBRARY_PATH="$K3_DIR/bin:${LD_LIBRARY_PATH:-}"

if [ "$K3_SKIP_DL" = "1" ] || [ -d checkpoints/k3 ] && [ -n "$(ls checkpoints/k3 2>/dev/null)" ]; then
    say "starting server (--serve-only)"
    args=(--host "$K3_HOST" --port "$K3_PORT" --preset "$K3_PRESET")
    [ -n "${K3_API_KEY:-}" ] && args+=(--api-key "$K3_API_KEY")
    if [ -d checkpoints/k3 ]; then
        exec python3 serve/__main__.py checkpoints/k3 "${args[@]}"
    else
        # No model yet — start with the article's tiny_k3.bin fixture so the
        # server comes up and the user can see /v1/models and the server is
        # working while they decide whether to commit to the 1.5 TB download.
        exec python3 serve/__main__.py tiny_k3.bin "${args[@]}"
    fi
else
    # Full setup: download + convert + serve.
    exec scripts/setup-and-serve.sh --preset "$K3_PRESET" --port "$K3_PORT" --host "$K3_HOST"
fi
