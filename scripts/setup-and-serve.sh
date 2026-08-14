#!/usr/bin/env bash
# setup-and-serve.sh — one command from clone to running server.
#
# Usage:
#   ./scripts/setup-and-serve.sh                   full path: build + download + convert + serve
#   ./scripts/setup-and-serve.sh --build-only      just build libk3.so and the CLI
#   ./scripts/setup-and-serve.sh --download-only   just download weights
#   ./scripts/setup-and-serve.sh --convert-only    just convert the checkpoint
#   ./scripts/setup-and-serve.sh --serve-only      just start the server (assume model + build exist)
#   ./scripts/setup-and-serve.sh --dry-run         build only, start server in fake-engine mode (no model needed)
#
# Environment variables:
#   K3_MODEL_DIR    checkpoint directory (default: ./checkpoints/k3)
#   K3_PRESET       memory preset: laptop|desktop|workstation|server|max|auto (default: auto)
#   K3_HOST         bind host (default: 127.0.0.1)
#   K3_PORT         bind port (default: 8080)
#   K3_API_KEY      bearer token (default: empty; required if binding non-loopback)
#   K3_MAX_TOKENS   default max_tokens per request (default: 256)
#   K3_LOG_REQUESTS log each request (default: false)
#   SKIP_DOWNLOAD   if 1, do not download weights (assume they exist)
#   SKIP_CONVERT    if 1, do not convert weights (assume already converted)
#
# Examples:
#   # First-time setup with a real K3 checkpoint:
#   ./scripts/setup-and-serve.sh
#
#   # Bring up server only, model already on disk:
#   ./scripts/setup-and-serve.sh --serve-only
#
#   # Quick smoke-test without downloading:
#   ./scripts/setup-and-serve.sh --dry-run
#
# Exit codes:
#   0   success
#   1   build failure
#   2   download failure
#   3   convert failure
#   4   model directory missing (and SKIP_DOWNLOAD=1)
#   5   server failed to start
#   6   server did not respond to health check
#   7   prerequisite tool missing (curl, python3, etc.)

set -euo pipefail

# --------------------------------------------------------------- defaults
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
K3_MODEL_DIR="${K3_MODEL_DIR:-${REPO_ROOT}/checkpoints/k3}"
K3_PRESET="${K3_PRESET:-auto}"
K3_HOST="${K3_HOST:-127.0.0.1}"
K3_PORT="${K3_PORT:-8080}"
K3_API_KEY="${K3_API_KEY:-}"
K3_MAX_TOKENS="${K3_MAX_TOKENS:-256}"
K3_LOG_REQUESTS="${K3_LOG_REQUESTS:-}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"

BUILD_ONLY=0
DOWNLOAD_ONLY=0
CONVERT_ONLY=0
SERVE_ONLY=0
DRY_RUN=0

# --------------------------------------------------------------- args
usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# //'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --build-only)     BUILD_ONLY=1; shift ;;
        --download-only)  DOWNLOAD_ONLY=1; shift ;;
        --convert-only)   CONVERT_ONLY=1; shift ;;
        --serve-only)     SERVE_ONLY=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        usage ;;
        *)                echo "unknown flag: $1" >&2; exit 7 ;;
    esac
done

# --------------------------------------------------------------- preflight
note() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit "${2:-1}"; }

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        die "missing required tool: $1" 7
    fi
}

note "preflight"
require_tool make
require_tool python3
require_tool curl

# Check disk space for the model. The article's K3 needs ~982 GB after convert.
if [ "${DRY_RUN}" -eq 0 ] && [ "${DOWNLOAD_ONLY:-0}" -eq 0 ] && [ "${CONVERT_ONLY:-0}" -eq 0 ]; then
    if [ "${SKIP_DOWNLOAD:-0}" -eq 0 ]; then
        model_parent="$(dirname "${K3_MODEL_DIR}")"
        avail_kb=$(df -Pk "${model_parent}" | awk 'NR==2 {print $4}')
        avail_gb=$((avail_kb / 1024 / 1024))
        if [ "${avail_gb}" -lt 1100 ]; then
            warn "only ${avail_gb} GB free at ${model_parent}; the K3 checkpoint needs ~1100 GB after download+convert"
            warn "set K3_MODEL_DIR to a path with more space, or set SKIP_DOWNLOAD=1 / SKIP_CONVERT=1"
        fi
    fi
fi

# --------------------------------------------------------------- build
build() {
    note "building libk3.so and bin/k3"
    cd "${REPO_ROOT}"
    # The article's Makefile uses LDFLAGS ?= which doesn't append on systems
    # where conda sets LDFLAGS; always pass it.
    LDFLAGS="-lm -pthread" make -j"$(nproc 2>/dev/null || echo 4)" libk3 k3
    if [ ! -f bin/libk3.so ]; then
        die "build did not produce bin/libk3.so" 1
    fi
    if [ ! -f bin/k3 ]; then
        die "build did not produce bin/k3" 1
    fi
    note "build OK: $(file bin/libk3.so | head -1)"
}

# --------------------------------------------------------------- download
download() {
    note "downloading Kimi K3 weights to ${K3_MODEL_DIR}"
    # Use the article's download-model.sh, which is HF-aware and resumable.
    cd "${REPO_ROOT}"
    if [ ! -f scripts/download-model.sh ]; then
        die "scripts/download-model.sh missing; this repo should ship it from the article" 2
    fi
    # The article's downloader expects a relative path under scripts/.
    mkdir -p "${K3_MODEL_DIR}"
    K3_MODEL_DIR="${K3_MODEL_DIR}" bash scripts/download-model.sh
}

# --------------------------------------------------------------- convert
convert() {
    note "converting checkpoint at ${K3_MODEL_DIR} to native format"
    cd "${REPO_ROOT}"
    # The convert step needs PyTorch + safetensors; the article's Dockerfile.convert
    # wraps this. For a one-shot convert on the host, see docs/CONVERT.md.
    if command -v docker >/dev/null 2>&1; then
        note "using Docker for conversion (Dockerfile.convert)"
        docker build -f Dockerfile.convert -t kimi-k3-convert .
        docker run --rm \
            -v "${K3_MODEL_DIR}:/data:rw" \
            -v "${REPO_ROOT}:/out:rw" \
            kimi-k3-convert \
            python3 tools/convert.py /data /out/checkpoints/k3-native
    else
        die "convert step needs Docker (or PyTorch on host with transformers, safetensors). " \
            "see docs/CONVERT.md for the manual path." 3
    fi
}

# --------------------------------------------------------------- serve
serve() {
    local model_path="$1"

    if [ ! -d "${model_path}" ]; then
        die "model directory not found: ${model_path}" 4
    fi

    note "starting kimi-k3-lean OpenAI server"
    note "  model:    ${model_path}"
    note "  preset:   ${K3_PRESET}"
    note "  endpoint: http://${K3_HOST}:${K3_PORT}/v1"
    note "  api key:  ${K3_API_KEY:-(none — server is open. do not expose to network.)}"

    # Build the args.
    local args=(
        "${model_path}"
        "--preset" "${K3_PRESET}"
        "--host" "${K3_HOST}"
        "--port" "${K3_PORT}"
        "--max-tokens" "${K3_MAX_TOKENS}"
    )
    if [ -n "${K3_API_KEY}" ]; then
        args+=("--api-key" "${K3_API_KEY}")
    fi
    if [ -n "${K3_LOG_REQUESTS}" ]; then
        args+=("--log-requests")
    fi
    if [ "${DRY_RUN}" -eq 1 ]; then
        args+=("--dry-run")
    fi

    # If non-loopback and no api key, warn.
    if [ "${K3_HOST}" != "127.0.0.1" ] && [ "${K3_HOST}" != "::1" ] && [ -z "${K3_API_KEY}" ]; then
        warn "binding to non-loopback (${K3_HOST}) without --api-key is unsafe."
        warn "any host on the network can use this server."
        warn "set K3_API_KEY or pass --api-key to require a bearer token."
        sleep 3
    fi

    # Run the server in the foreground. Use LD_LIBRARY_PATH so libk3.so is found.
    cd "${REPO_ROOT}"
    exec env \
        LD_LIBRARY_PATH="${REPO_ROOT}/bin:${LD_LIBRARY_PATH:-}" \
        python3 serve/__main__.py "${args[@]}"
}

# --------------------------------------------------------------- workflow

# In dry-run, build + serve with FakeEngine. Skip download/convert.
if [ "${DRY_RUN}" -eq 1 ]; then
    build
    mkdir -p /tmp/k3-lean-fake
    serve /tmp/k3-lean-fake
    exit 0
fi

# Build unless --serve-only.
if [ "${SERVE_ONLY}" -eq 0 ]; then
    build
fi

# Download unless --build-only or --serve-only or SKIP_DOWNLOAD=1.
if [ "${BUILD_ONLY}" -eq 0 ] && [ "${SERVE_ONLY}" -eq 0 ] && [ "${SKIP_DOWNLOAD}" -eq 0 ]; then
    download
fi

# Convert unless --build-only, --download-only, --serve-only, or SKIP_CONVERT=1.
if [ "${BUILD_ONLY}" -eq 0 ] && [ "${DOWNLOAD_ONLY}" -eq 0 ] && [ "${SERVE_ONLY}" -eq 0 ] && [ "${SKIP_CONVERT}" -eq 0 ]; then
    convert
fi

# Done-with-just-steps.
if [ "${BUILD_ONLY}" -eq 1 ] || [ "${DOWNLOAD_ONLY}" -eq 1 ] || [ "${CONVERT_ONLY}" -eq 1 ]; then
    note "done."
    exit 0
fi

# Otherwise serve.
serve "${K3_MODEL_DIR}"