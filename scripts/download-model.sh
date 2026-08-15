#!/usr/bin/env bash
# download-model.sh, fetch the Kimi K3 checkpoint and verify it byte-exactly.
#
#   scripts/download-model.sh <dest_dir>
#
# The checkpoint is 1.56 TB across 96 safetensors shards. A partial or corrupt download
# does not fail loudly, it produces wrong tokens, so the byte total is checked against
# the published figure before anything else is allowed to use it.
#
# The repository is public, so no token is needed. If you have one, the hf CLI picks it
# up from $HF_TOKEN or ~/.cache/huggingface/token on its own; this script never reads,
# echoes or forwards it.

set -euo pipefail

DEST="${1:?usage: download-model.sh <dest_dir>}"
REPO="moonshotai/Kimi-K3"

# Published totals for the released checkpoint. Verified, not assumed.
EXPECT_SHARDS=96
EXPECT_BYTES=1560936091448

# The downloader is the `hf` CLI from huggingface_hub 1.x. The older entry points are
# gone: `python3 -m huggingface_hub.commands.huggingface_cli` was removed when the CLI
# was renamed, and the `[cli]` extra no longer exists.
#
# We deliberately do NOT attempt a pip install here. On the platform this project
# targets, a system pip install cannot succeed: Ubuntu 24.04 ships no pip module at
# all, and once pip is present PEP 668 marks the interpreter externally-managed and
# refuses. Guessing wrong in an unattended script that is about to move 1.56 TB is
# worse than stopping with an instruction.
if ! command -v hf >/dev/null 2>&1; then
    cat >&2 <<'EOF'
the `hf` CLI is required and was not found on PATH.

Install it one of these ways, then re-run:

  pipx install huggingface_hub            # recommended; what PEP 668 points you to
  uv tool install huggingface_hub
  python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install huggingface_hub
    then add ~/.venvs/hf/bin to PATH

On Debian/Ubuntu, `pipx` is `sudo apt install pipx`.
EOF
    exit 1
fi

# Pre-flight: a broken `hf` (e.g. huggingface_hub from system pip with a
# filelock that doesn't accept mode= kwarg) will report success on `hf --help`
# but crash with 'TypeError: __init__() got an unexpected keyword argument
# "mode"' the moment it hits a real file. Detect that quickly and provide
# a fix: install hf into our own venv at ~/.local/share/kimi-k3-lean/hf-venv
# (or ~/.venvs/hf if no project-local venv is writable), then PATH-prepend
# that venv so this and subsequent calls use a known-good install.
ensure_hf() {
    # 1. Path-local venv under the repo dir (preferred when writable).
    if [ -z "${HF_VENV:-}" ]; then
        for candidate in "${K3_DIR:-}/.hf-venv" "${XDG_DATA_HOME:-$HOME/.local/share}/kimi-k3-lean/hf-venv" "$HOME/.venvs/hf" "$HOME/.local/share/kimi-k3-lean/hf-venv"; do
            if [ -x "$candidate/bin/hf" ]; then
                HF_VENV="$candidate"
                break
            fi
        done
    fi
    if [ -z "${HF_VENV:-}" ]; then
        HF_VENV="$HOME/.local/share/kimi-k3-lean/hf-venv"
        if [ ! -x "$HF_VENV/bin/hf" ]; then
            mkdir -p "$(dirname "$HF_VENV")"
            if command -v python3 >/dev/null 2>&1; then
                python3 -m venv "$HF_VENV" 2>/dev/null || true
            fi
            if [ ! -x "$HF_VENV/bin/pip" ] && command -v uv >/dev/null 2>&1; then
                # Fall back to uv (creates the venv and installs in one go).
                uv venv "$HF_VENV" --python python3 2>/dev/null || true
            fi
            if [ -x "$HF_VENV/bin/pip" ]; then
                # Pin filelock to a version that supports mode= kwarg if needed,
                # and pin huggingface_hub so we don't get a too-new version that
                # abandoned it. >=1.0,<1.2 covers hf CLI rename and still works
                # with pip-system filelock 3.x.
                # --no-deps: huggingface_hub's CLI itself doesn't need
                # markdown-it-py/pygments/mdurl (those are for the
                # transformers+rich Jupyter rendering stack). Installing
                # with deps forces a 5-retry storm on offline hosts that
                # don't have those packages on their local pypi mirror.
                # --timeout 20: fail fast on offline hosts instead of
                # hanging 5x per package.
                # --disable-pip-version-check: skip the "new pip available"
                # nag, which itself is a network call.
                # --no-cache-dir: don't leave partial state.
                "$HF_VENV/bin/pip" install --quiet --no-deps \
                    --timeout 20 --no-cache-dir --disable-pip-version-check \
                    "filelock>=3.12" "huggingface_hub>=1.0,<1.2" 2>>"${DOWNLOAD_LOG:-/dev/null}" || true
            fi
        fi
    fi
    if [ -x "$HF_VENV/bin/hf" ]; then
        # Prepend the venv to PATH so all subsequent hf calls use it.
        export PATH="$HF_VENV/bin:$PATH"
    fi
}
ensure_hf

# An older installation can provide `hf` without `hf cache verify`, which is what turns
# the size check below into a real integrity check.
hf --help >/dev/null 2>&1 || {
    echo "this hf CLI doesn't respond; install huggingface_hub and re-run." >&2
    exit 1
}
hf cache verify --help >/dev/null 2>&1 || {
    echo "this hf CLI has no 'cache verify'; upgrade huggingface_hub and re-run." >&2
    exit 1
}

# Resolve the branch to an immutable commit ONCE, and use it for both the download and
# the verification. Otherwise `main` can move between the two steps and the checksums are
# compared against a different snapshot than the one on disk.
#
# Resolved through the hf CLI rather than `python3 -c "import huggingface_hub"`: the
# recommended installs above put the library in an isolated environment and expose only
# the `hf` executable, so importing it from the system interpreter fails on exactly the
# setups this script just told the user to create.
# NOTE: awk must read to EOF here. Exiting on the first match closes the pipe, `hf` takes
# SIGPIPE, and under `set -o pipefail` that kills the script with 141 before it prints
# anything at all.
REVISION="${K3_REVISION:-}"
if [ -z "$REVISION" ]; then
    REVISION="$(hf models info "$REPO" 2>/dev/null \
                | tr -d ' ",' \
                | awk -F: '!v && /^sha:/ {v = $2} END {print v}')"
fi
case "$REVISION" in
    ????????????????????????????????????????) ;;   # 40 hex characters
    *) echo "could not resolve a commit for $REPO (got '${REVISION:-empty}')." >&2
       echo "  Set K3_REVISION to a commit sha to skip this lookup." >&2
       exit 1 ;;
esac

mkdir -p "$DEST"

# Free-space preflight. Without this the transfer runs until the filesystem fills, which
# takes the machine's logging and package manager with it, and the byte-total check below
# then reports a mismatch -- a true statement that misdiagnoses the actual failure.
AVAIL=$(df -P -k "$DEST" | awk 'NR==2 {print $4 * 1024}')
if [ "$AVAIL" -lt "$EXPECT_BYTES" ]; then
    printf 'FAIL: %s has %s bytes free, the checkpoint needs %s.\n' \
        "$DEST" "$AVAIL" "$EXPECT_BYTES" >&2
    printf '      Short by %s bytes (%.2f TB). Point <dest_dir> at a larger filesystem.\n' \
        "$((EXPECT_BYTES - AVAIL))" \
        "$(awk -v d="$((EXPECT_BYTES - AVAIL))" 'BEGIN{print d/1e12}')" >&2
    printf '      Packing the trunk afterwards needs a further ~109 GB.\n' >&2
    exit 1
fi

echo "downloading $REPO@$REVISION -> $DEST"
echo "  1.56 TB across $EXPECT_SHARDS shards; expect ~30 min at 1 GB/s"
echo

# Xet is the transfer backend in huggingface_hub 1.x. HF_HUB_ENABLE_HF_TRANSFER, which
# this script used to set, is ignored there and was a hard error on 0.x whenever the
# hf_transfer package was absent -- which it always was, since nothing installed it.
#
# A token is not needed: the repository is public. If one is present in the environment
# or in a saved login the CLI uses it for higher rate limits; this script never touches it.
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}" \
hf download "$REPO" --revision "$REVISION" --local-dir "$DEST" --max-workers 16

echo
echo "verifying…"
# find, not `ls | wc -l`: under `set -euo pipefail` a glob that matches nothing makes
# ls exit non-zero and the script dies HERE, so the "download is incomplete" message
# below -- the entire point of this verification block -- would never be reached.
N=$(find "$DEST" -maxdepth 1 -name '*.safetensors' | wc -l)
B=$(find "$DEST" -maxdepth 1 -name '*.safetensors' -printf '%s\n' | awk '{s+=$1} END{print s+0}')

printf '  shards : %s (expect %s)\n' "$N" "$EXPECT_SHARDS"
printf '  bytes  : %s (expect %s)\n' "$B" "$EXPECT_BYTES"

if [ "$N" -ne "$EXPECT_SHARDS" ]; then
    echo "FAIL: wrong shard count, the download is incomplete."
    exit 1
fi
if [ "$B" -ne "$EXPECT_BYTES" ]; then
    echo "FAIL: byte total mismatch (delta $((B - EXPECT_BYTES)))."
    echo "      A partial checkpoint yields wrong output silently. Re-run to resume."
    exit 1
fi

# Per-shard sizes. The total above already proves the download is complete; this proves
# WHICH shard is wrong when it is not, turning "re-download 1.56 TB" into "re-download
# 17 GB". It also catches the one case a total cannot: two shards wrong in opposite
# directions by the same amount.
SIZES="$(dirname "$0")/shard_sizes.txt"
if [ -f "$SIZES" ]; then
    bad=0
    while read -r name want; do
        [ -n "$name" ] || continue
        got=$(stat -c%s "$DEST/$name" 2>/dev/null || echo 0)
        if [ "$got" != "$want" ]; then
            printf '  BAD  %s: %s bytes, expected %s\n' "$name" "$got" "$want"
            bad=$((bad + 1))
        fi
    done < "$SIZES"
    if [ "$bad" -ne 0 ]; then
        echo "FAIL: $bad shard(s) do not match their published sizes."
        echo "      Delete just those files and re-run; the download resumes."
        exit 1
    fi
    printf '  shards : all %s match their published sizes individually\n' "$EXPECT_SHARDS"
fi

# Sizes prove the download is COMPLETE. They cannot prove it is the RIGHT bytes: a
# substituted shard of identical length passes everything above. SECURITY.md names that
# gap; this closes it by comparing against the hashes the Hub published for the exact
# commit resolved earlier. It re-reads all 1.56 TB, so it is not free -- on a machine
# without SHA-NI it can take longer than the download did.
echo
echo "verifying checksums against Hub metadata for $REVISION…"
echo "  (re-reads the full 1.56 TB; set K3_SKIP_CHECKSUM=1 to skip)"
if [ "${K3_SKIP_CHECKSUM:-0}" = "1" ]; then
    echo "  SKIPPED by K3_SKIP_CHECKSUM=1; sizes were still checked above."
else
    hf cache verify "$REPO" --revision "$REVISION" --local-dir "$DEST" \
        --fail-on-missing-files
    echo "  RESULT : checksum-verified snapshot at $REVISION"
fi
echo
echo "next: scripts/pack-trunk.sh $DEST <trunk_dir>"
echo "      packing the trunk is what lets the engine stream it, which is what makes"
echo "      the memory budget a dial instead of a floor."
