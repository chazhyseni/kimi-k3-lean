#!/usr/bin/env bash
# setup-and-serve.sh — one command from clone to running server.
#
# Usage:
#   ./scripts/setup-and-serve.sh                    build + download + convert + serve
#   ./scripts/setup-and-serve.sh --build-only       build only
#   ./scripts/setup-and-serve.sh --download-only    download weights only
#   ./scripts/setup-and-serve.sh --convert-only     convert checkpoint only
#   ./scripts/setup-and-serve.sh --serve-only       start the server
#   ./scripts/setup-and-serve.sh --help
#   ./scripts/setup-and-serve.sh --build-only       build liblitmoe.so + the CLI
#   ./scripts/setup-and-serve.sh --download-only    download weights
#   ./scripts/setup-and-serve.sh --convert-only     convert the checkpoint
#   ./scripts/setup-and-serve.sh --serve-only       start the server (assumes model + build exist)
#   ./scripts/setup-and-serve.sh --install-deps    install OS-level prerequisites first
#
# Environment variables:
#   LITMOE_MODEL_DIR    checkpoint directory (default: ./checkpoints/k3)
#   LITMOE_PRESET       memory preset (default: auto)
#   LITMOE_HOST         bind host (default: 127.0.0.1)
#   LITMOE_PORT         bind port (default: 8080)
#   LITMOE_API_KEY      bearer token (default: empty; required if binding non-loopback)
#   LITMOE_MAX_TOKENS   default max_tokens per request (default: 256)
#   LITMOE_LOG_REQUESTS log each request (default: empty)
#   SKIP_DOWNLOAD   if 1, do not download weights
#   SKIP_CONVERT    if 1, do not convert weights
#
# Exit codes:
#   0   success
#   1   build failure
#   2   download failure
#   3   convert failure
#   4   model directory missing
#   5   prerequisite tool missing
#   6   server did not respond to health check

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ----------------------- defaults -----------------------
LITMOE_MODEL_DIR="${LITMOE_MODEL_DIR:-${REPO_ROOT}/checkpoints/k3}"
LITMOE_PRESET="${LITMOE_PRESET:-auto}"
LITMOE_HOST="${LITMOE_HOST:-127.0.0.1}"
LITMOE_PORT="${LITMOE_PORT:-8080}"
LITMOE_API_KEY="${LITMOE_API_KEY:-}"
LITMOE_MAX_TOKENS="${LITMOE_MAX_TOKENS:-256}"
LITMOE_LOG_REQUESTS="${LITMOE_LOG_REQUESTS:-}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"

BUILD_ONLY=0
DOWNLOAD_ONLY=0
CONVERT_ONLY=0
SERVE_ONLY=0
INSTALL_DEPS=0

# ----------------------- logging -----------------------
note() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit "${2:-1}"; }

# ----------------------- arg parsing -----------------------
usage() {
    sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# //'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --build-only)     BUILD_ONLY=1; shift ;;
        --download-only)  DOWNLOAD_ONLY=1; shift ;;
        --convert-only)   CONVERT_ONLY=1; shift ;;
        --serve-only)     SERVE_ONLY=1; shift ;;
        --install-deps)   INSTALL_DEPS=1; shift ;;
        -h|--help)        usage ;;
        *)                die "unknown flag: $1" 5 ;;
    esac
done

# ----------------------- OS detection -----------------------
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case "${ID:-unknown}:${VERSION_CODENAME:-}" in
            debian:*) echo "debian" ;;
            ubuntu:*) echo "ubuntu" ;;
            fedora:*)  echo "fedora" ;;
            rhel:*)    echo "rhel" ;;
            centos:*)  echo "centos" ;;
            *)         echo "linux-${ID:-unknown}" ;;
        esac
    elif command -v lsb_release >/dev/null 2>&1; then
        lsb_release -is | tr '[:upper:]' '[:lower:]'
    elif [ "$(uname -s)" = "Darwin" ]; then
        echo "macos"
    elif [ "$(uname -s)" = "FreeBSD" ]; then
        echo "freebsd"
    else
        echo "unknown"
    fi
}

# ----------------------- tool installer -----------------------
require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        die "missing required tool: $1
install with: ./scripts/setup-and-serve.sh --install-deps
or use your package manager. " 5
    fi
}

install_deps() {
    local os
    os="$(detect_os)"
    note "installing prerequisites for ${os}"

    case "${os}" in
        debian|ubuntu)
            sudo apt-get update
            sudo apt-get install -y \
                build-essential cmake git curl wget \
                python3 python3-pip python3-venv \
                libssl-dev libgomp1 ca-certificates
            ;;
        fedora|rhel|centos)
            sudo dnf install -y \
                gcc gcc-c++ make cmake git curl wget \
                python3 python3-pip \
                openssl-devel libgomp ca-certificates
            ;;
        macos)
            if ! command -v brew >/dev/null 2>&1; then
                die "Homebrew not found. Install from https://brew.sh first." 5
            fi
            brew install python@3.11 cmake git curl wget openssl
            ;;
        *)
            warn "unknown OS: ${os}. Please install manually:"
            warn "  build-essential gcc g++ make cmake git curl python3 python3-pip"
            ;;
    esac

    # Python deps for the convert step.
    python3 -m pip install --user --quiet --break-system-packages \
        safetensors 2>/dev/null \
        || python3 -m pip install --user --quiet safetensors 2>/dev/null \
        || warn "could not install safetensors via pip — convert step will fail"

    ok "prerequisites installed"
}

# ----------------------- subroutines -----------------------

build() {
    note "building liblitmoe.so and bin/litmoe"
    require_tool make
    require_tool gcc
    require_tool python3

    cd "${REPO_ROOT}"

    # macOS: nproc doesn't exist. Use sysctl.
    local jobs
    if command -v nproc >/dev/null 2>&1; then
        jobs=$(nproc)
    elif command -v sysctl >/dev/null 2>&1; then
        jobs=$(sysctl -n hw.ncpu)
    else
        jobs=4
    fi

    LDFLAGS="-lm -pthread" make -j"${jobs}" all

    if [ ! -f bin/liblitmoe.so ]; then
        die "build did not produce bin/liblitmoe.so" 1
    fi
    if [ ! -f bin/litmoe ]; then
        die "build did not produce bin/litmoe" 1
    fi
    ok "build OK"
}

download() {
    note "downloading Kimi K3 weights to ${LITMOE_MODEL_DIR}"
    cd "${REPO_ROOT}"
    mkdir -p "${LITMOE_MODEL_DIR}"
    bash scripts/fetch-model.sh "${LITMOE_MODEL_DIR}"
    ok "download OK"
}

convert() {
    note "converting checkpoint at ${LITMOE_MODEL_DIR}"
    if [ ! -d "${LITMOE_MODEL_DIR}" ]; then
        die "checkpoint directory missing: ${LITMOE_MODEL_DIR}. " \
            "run ./scripts/setup-and-serve.sh --download-only first." 4
    fi
    cd "${REPO_ROOT}"

    local native_dir="${K3_NATIVE_DIR:-${LITMOE_MODEL_DIR}-native}"

    # Prefer Python-native convert (no Docker required).
    if command -v python3 >/dev/null 2>&1 && python3 -c "import safetensors" 2>/dev/null; then
        note "using python3 tools/convert.py (no Docker required)"
        python3 tools/convert.py "${LITMOE_MODEL_DIR}" "${native_dir}"
        ok "convert OK: ${native_dir}"
        # Point LITMOE_MODEL_DIR at the converted dir so the engine loads it.
        echo "${native_dir}" > "${LITMOE_MODEL_DIR}/.native-path"
        return 0
    fi

    # Fallback: Docker.
    if command -v docker >/dev/null 2>&1; then
        note "using Docker (Dockerfile.convert)"
        docker build -f Dockerfile.convert -t kimi-k3-convert .
        docker run --rm \
            -v "${LITMOE_MODEL_DIR}:/data:rw" \
            -v "${REPO_ROOT}:/out:rw" \
            kimi-k3-convert \
            python3 /src/tools/convert.py /data /out/checkpoints/k3-native
        ok "convert OK (via Docker)"
        return 0
    fi

    die "convert needs either:
  1. Python 3 + safetensors (pip install safetensors), or
  2. Docker.

run ./scripts/setup-and-serve.sh --install-deps to install Python deps." 3
}

serve() {
    local model_path="$1"

    # If convert wrote a .native-path file, use that.
    if [ -f "${model_path}/.native-path" ]; then
        local native_path
        native_path=$(cat "${model_path}/.native-path")
        if [ -d "${native_path}" ]; then
            note "using converted checkpoint at ${native_path}"
            model_path="${native_path}"
        fi
    fi

    if [ ! -d "${model_path}" ]; then
        die "model directory not found: ${model_path}" 4
    fi

    note "starting litMoE OpenAI server"
    note "  model:    ${model_path}"
    note "  preset:   ${LITMOE_PRESET}"
    note "  endpoint: http://${LITMOE_HOST}:${LITMOE_PORT}/v1"
    note "  api key:  ${LITMOE_API_KEY:-(none — server is open. do not expose to network.)}"

    # Refuse to bind non-loopback without auth.
    if [ "${LITMOE_HOST}" != "127.0.0.1" ] && [ "${LITMOE_HOST}" != "::1" ] && [ -z "${LITMOE_API_KEY}" ]; then
        warn "binding to non-loopback (${LITMOE_HOST}) without --api-key is unsafe."
        warn "any host on the network can use this server."
        warn "set LITMOE_API_KEY or pass --api-key to require a bearer token."
        sleep 3
    fi

    local args=(
        "${model_path}"
        "--preset" "${LITMOE_PRESET}"
        "--host" "${LITMOE_HOST}"
        "--port" "${LITMOE_PORT}"
        "--max-tokens" "${LITMOE_MAX_TOKENS}"
    )
    [ -n "${LITMOE_API_KEY}" ]      && args+=("--api-key" "${LITMOE_API_KEY}")
    [ -n "${LITMOE_LOG_REQUESTS}" ] && args+=("--log-requests")

    cd "${REPO_ROOT}"
    exec env \
        LD_LIBRARY_PATH="${REPO_ROOT}/bin:${LD_LIBRARY_PATH:-}" \
        python3 serve/__main__.py "${args[@]}"
}

# ----------------------- main flow -----------------------

# Optional: install OS-level deps first.
if [ "${INSTALL_DEPS}" -eq 1 ]; then
    install_deps
    note "rerun this script without --install-deps to continue setup"
    exit 0
fi

# Real-model flow.
if [ "${SERVE_ONLY}" -eq 0 ]; then
    build
fi

if [ "${BUILD_ONLY}" -eq 0 ] && [ "${SERVE_ONLY}" -eq 0 ] && [ "${SKIP_DOWNLOAD}" -eq 0 ]; then
    download
fi

if [ "${BUILD_ONLY}" -eq 0 ] && [ "${DOWNLOAD_ONLY}" -eq 0 ] && [ "${SERVE_ONLY}" -eq 0 ] && [ "${SKIP_CONVERT}" -eq 0 ]; then
    convert
fi

if [ "${BUILD_ONLY}" -eq 1 ] || [ "${DOWNLOAD_ONLY}" -eq 1 ] || [ "${CONVERT_ONLY}" -eq 1 ]; then
    ok "done."
    exit 0
fi

serve "${LITMOE_MODEL_DIR}"