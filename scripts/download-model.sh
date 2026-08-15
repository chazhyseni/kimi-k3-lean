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

# The downloader is the `hf` CLI from huggingface_hub 1.x. We do NOT
# install it here -- the script is about to move 1.56 TB and we want
# fail-fast behavior, not a pip-install storm on offline hosts. If hf
# is missing or broken, the user gets a clear message and the script
# exits non-zero.
# Always build our own hf into a venv. The system hf may be broken
# (Debian ships filelock 3.0.12 with no mode= kwarg; the user-installed
# huggingface_hub 1.4.1 calls mode= and crashes; `hf --help` works but
# `hf download` doesn\'t). The fix: an isolated venv with pinned
# huggingface_hub + filelock that we control.
#
# This adds 60-120s of pip install the FIRST time. On reruns, the venv
# is reused and the script proceeds straight to the download.
#
# We do NOT use --quiet: silent pip is what made the user think the
# bootstrap was stuck. Output is visible.

VENV="${HF_VENV:-$HOME/.local/share/kimi-k3-lean/hf-venv}"

hf_works() {
    # Returns 0 if $VENV/bin/hf exists AND `hf download --help` works.
    # The "download --help" check is critical: that is the code path
    # that uses FileLock(mode=...) and would otherwise crash silently.
    [ -x "$VENV/bin/hf" ] && "$VENV/bin/hf" download --help >/dev/null 2>&1
}

if ! hf_works; then
    echo "setting up hf CLI in $VENV" >&2
    echo "(first run: ~60-120s of pip install. reruns reuse the venv.)" >&2
    mkdir -p "$(dirname "$VENV")"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip 2>&1 | tail -2 >&2
    "$VENV/bin/pip" install \
        --no-cache-dir --disable-pip-version-check \
        "filelock>=3.12,<4.0" \
        "click>=8.0" \
        "huggingface_hub>=1.0,<1.3" 2>&1 | tail -5 >&2
    if ! hf_works; then
        echo >&2
        echo "FAILED: $VENV/bin/hf doesn\'t work after install." >&2
        echo "  try: rm -rf $VENV && re-run" >&2
        echo "  or:  pipx install huggingface_hub   (then set HF_VENV=\$(which hf))" >&2
        exit 1
    fi
    echo "hf ready: $VENV/bin/hf" >&2
fi

# Prepend the venv\'s bin to PATH so all subsequent `hf` calls use it.
export PATH="$VENV/bin:$PATH"

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
    # The `hf models info` subcommand was removed in huggingface_hub 1.x;
    # the equivalent is the Python API model_info(). Use the venv\'s
    # python so we don\'t depend on a system pip --user install.
    REVISION="$("$VENV/bin/python" -c \
        "from huggingface_hub import model_info; print(model_info('$REPO').sha)" \
        2>/dev/null)"
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
hf download "$REPO" --revision "$REVISION" --local-dir "$DEST"

echo
echo "verifying…"
# find, not `ls | wc -l`: under `set -euo pipefail` a glob that matches nothing makes
# ls exit non-zero and the script dies HERE, so the "download is incomplete" message
# below -- the entire point of this verification block -- would never be reached.
N=$(find "$DEST" -maxdepth 1 -name '*.safetensors' | wc -l | tr -d ' ')
# Portable size sum: GNU stat -c%s, BSD/macOS stat -f%z.
if stat -c%s /dev/null >/dev/null 2>&1; then
    size_cmd="stat -c%s"
else
    size_cmd="stat -f%z"
fi
B=0
while IFS= read -r f; do
    [ -n "$f" ] || continue
    B=$((B + $("$size_cmd" "$f")))
done < <(find "$DEST" -maxdepth 1 -name '*.safetensors')

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
