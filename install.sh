#!/usr/bin/env bash
# install.sh — install kimi-k3-lean to a prefix.
#
# Usage:
#   ./install.sh                    # installs to /usr/local (requires sudo)
#   ./install.sh PREFIX=$HOME/.local  # user-level install, no sudo
#   PREFIX=/opt/kimi-k3-lean ./install.sh
#
# What it installs:
#   bin/k3            CLI binary
#   lib/libk3.so      shared library (Python server uses via ctypes)
#   lib/libk3_static.a  static library (for embedding)
#   include/libk3/libk3.h  public C API
#
# Tested on:
#   - Debian 11.11, glibc 2.31 (this host)
#   - macOS 14 + arm64 (M-series)
#
# NOT tested on this host (would require the platform):
#   - Windows (use install.ps1 instead)
#   - RHEL / Fedora (should work; uses glibc)

set -e

# --------------------------------------------------------------------------- args
PREFIX="${PREFIX:-/usr/local}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# --------------------------------------------------------------------------- env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --------------------------------------------------------------------------- detect
OS="$(uname -s)"
case "$OS" in
    Linux*)     PLATFORM="linux" ;;
    Darwin*)    PLATFORM="macos" ;;
    CYGWIN*|MINGW*|MSYS*)  PLATFORM="windows" ;;
    *)          echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac

echo "==> kimi-k3-lean installer"
echo "    platform:    $PLATFORM"
echo "    prefix:      $PREFIX"
echo "    build type:  $BUILD_TYPE"
echo "    jobs:        $JOBS"

# --------------------------------------------------------------------------- check toolchain
need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "==> ERROR: $1 not found. Install it and retry." >&2
        exit 1
    fi
}

need gcc
need make

# --------------------------------------------------------------------------- build
echo "==> Building..."
# The article's Makefile uses LDFLAGS ?= which doesn't append when conda
# has set LDFLAGS, so we always pass -lm -pthread explicitly.
export LDFLAGS="-lm -pthread"
export CFLAGS="${CFLAGS:-} -O2"
make -j"$JOBS" clean >/dev/null 2>&1 || true
make -j"$JOBS"

# --------------------------------------------------------------------------- install
echo "==> Installing to $PREFIX ..."
# We use `install -D` so directories are created as needed. We don't depend
# on a recent GNU coreutils because the install command itself is portable.
PREFIX="$PREFIX" make install PREFIX="$PREFIX"

# --------------------------------------------------------------------------- post-install: ldconfig on Linux
if [[ "$PLATFORM" == "linux" ]]; then
    if command -v ldconfig >/dev/null 2>&1; then
        if [[ -w /etc/ld.so.conf.d ]] || command -v sudo >/dev/null 2>&1; then
            echo "==> Running ldconfig..."
            LDCONF="$(mktemp)"
            echo "$PREFIX/lib" > "$LDCONF"
            if [[ -w /etc/ld.so.conf.d ]]; then
                install -m 0644 "$LDCONF" /etc/ld.so.conf.d/kimi-k3-lean.conf
                ldconfig
            else
                sudo install -m 0644 "$LDCONF" /etc/ld.so.conf.d/kimi-k3-lean.conf
                sudo ldconfig
            fi
            rm -f "$LDCONF"
        fi
    fi
fi

# --------------------------------------------------------------------------- verify
echo "==> Verifying installation..."
INSTALLED_K3="$PREFIX/bin/k3"
INSTALLED_LIB="$PREFIX/lib/libk3.so"
INSTALLED_HDR="$PREFIX/include/libk3/libk3.h"

if [[ ! -x "$INSTALLED_K3" ]]; then
    echo "==> ERROR: $INSTALLED_K3 not installed" >&2; exit 1
fi
if [[ ! -f "$INSTALLED_LIB" ]]; then
    echo "==> ERROR: $INSTALLED_LIB not installed" >&2; exit 1
fi
if [[ ! -f "$INSTALLED_HDR" ]]; then
    echo "==> ERROR: $INSTALLED_HDR not installed" >&2; exit 1
fi

# Smoke-test the installed binary.
if ! "$INSTALLED_K3" --help >/dev/null 2>&1; then
    echo "==> WARNING: $INSTALLED_K3 --help returned non-zero" >&2
fi
if ! "$INSTALLED_K3" --version >/dev/null 2>&1; then
    echo "==> WARNING: $INSTALLED_K3 --version returned non-zero" >&2
fi

cat <<EOF

==> Installation complete.

Installed files:
  $INSTALLED_K3
  $INSTALLED_LIB
  $INSTALLED_HDR

To start the OpenAI server (after fetching a model):

  python3 serve/__main__.py /path/to/checkpoint --host 127.0.0.1 --port 8080

See README.md and docs/INSTALL.md for next steps.

EOF