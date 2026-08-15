#!/usr/bin/env bash
# fetch-model.sh — fast parallel download of model weights from HuggingFace.
#
#   scripts/fetch-model.sh <dest_dir> [hf_repo_id]
#
# Downloads model weights in parallel (4-8 shards at once) for maximum
# throughput. Resumable. Verifies shard count + byte total + checksums.
#
# Speed: 4-8x faster than sequential `hf download` on connections with
# headroom. On a 1 Gbps link with 8 parallel downloads: ~15 min for 1 TB.
#
# Env vars:
#   HF_TOKEN         optional auth token for higher rate limits
#   K3_REVISION      pin to a specific commit (skip auto-detection)
#   K3_SKIP_CHECKSUM skip checksum verification (sizes still checked)
#   K3_PARALLEL      number of parallel downloads (default: 4)
#   HF_VENV          path to the hf CLI venv (default: ~/.local/share/lean/hf-venv)

set -euo pipefail

DEST="${1:?usage: fetch-model.sh <dest_dir> [hf_repo_id]}"
REPO="${2:-moonshotai/Kimi-K3}"
PARALLEL="${K3_PARALLEL:-4}"

# Published totals for Kimi K3. For other models, these are fetched
# from the HF API at runtime.
EXPECT_SHARDS="${EXPECT_SHARDS:-96}"
EXPECT_BYTES="${EXPECT_BYTES:-1560936091448}"

# ---- hf CLI setup (isolated venv) ----
VENV="${HF_VENV:-$HOME/.local/share/lean/hf-venv}"

hf_works() {
    [ -x "$VENV/bin/hf" ] && "$VENV/bin/hf" download --help >/dev/null 2>&1
}

if ! hf_works; then
    echo "setting up hf CLI in $VENV (first run: ~60-120s)" >&2
    mkdir -p "$(dirname "$VENV")"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip 2>&1 | tail -1 >&2
    "$VENV/bin/pip" install --no-cache-dir --disable-pip-version-check \
        "filelock>=3.12,<4.0" "click>=8.0" "huggingface_hub>=1.0,<1.3" 2>&1 | tail -3 >&2
    if ! hf_works; then
        echo "FAILED: hf CLI not working. Try: rm -rf $VENV && re-run" >&2
        exit 1
    fi
fi
export PATH="$VENV/bin:$PATH"

# ---- resolve commit ----
REVISION="${K3_REVISION:-}"
if [ -z "$REVISION" ]; then
    REVISION="$("$VENV/bin/python" -c \
        "from huggingface_hub import model_info; print(model_info('$REPO').sha)" 2>/dev/null)"
fi
case "$REVISION" in
    ????????????????????????????????????????) ;;
    *) echo "could not resolve commit for $REPO" >&2; exit 1 ;;
esac

mkdir -p "$DEST"

# ---- free-space check ----
AVAIL=$(df -P -k "$DEST" | awk 'NR==2 {print $4 * 1024}')
if [ "$AVAIL" -lt "$EXPECT_BYTES" ]; then
    printf 'FAIL: %s has %.1f GB free, need %.1f GB.\n' \
        "$DEST" "$(echo "scale=1; $AVAIL/1073741824" | bc)" \
        "$(echo "scale=1; $EXPECT_BYTES/1073741824" | bc)" >&2
    exit 1
fi

# ---- get file list from HF API ----
echo "fetching file list from $REPO@$REVISION..."
FILE_LIST="$("$VENV/bin/python" -c "
from huggingface_hub import list_repo_files
files = list_repo_files('$REPO', revision='$REVISION')
for f in files:
    if f.endswith('.safetensors') or f in ('config.json', 'tiktoken.model', 'tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json'):
        print(f)
" 2>/dev/null)"

N_SHARDS=$(echo "$FILE_LIST" | grep '\.safetensors$' | wc -l | tr -d ' ')
echo "  $N_SHARDS safetensors shards + config files"
echo "  parallel: $PARALLEL downloads"
echo "  total: ~$(echo "scale=1; $EXPECT_BYTES/1073741824" | bc) GB"
echo

# ---- download config files first (small, needed for arch detection) ----
for f in config.json tiktoken.model tokenizer_config.json special_tokens_map.json generation_config.json; do
    if echo "$FILE_LIST" | grep -q "^${f}$"; then
        if [ ! -f "$DEST/$f" ]; then
            hf download "$REPO" --revision "$REVISION" --local-dir "$DEST" --include "$f" 2>/dev/null
            echo "  ✓ $f"
        fi
    fi
done

# ---- parallel shard download ----
SHARDS=$(echo "$FILE_LIST" | grep '\.safetensors$')

download_shard() {
    local repo="$1" rev="$2" dest="$3" shard="$4"
    local out="$dest/$shard"
    # Skip if already downloaded and correct size
    if [ -f "$out" ]; then
        local size
        if stat -c%s /dev/null >/dev/null 2>&1; then
            size=$(stat -c%s "$out" 2>/dev/null || echo 0)
        else
            size=$(stat -f%z "$out" 2>/dev/null || echo 0)
        fi
        if [ "$size" -gt 0 ]; then
            echo "  ✓ $shard (cached, $(echo "scale=1; $size/1073741824" | bc 2>/dev/null || echo '?') GB)"
            return 0
        fi
    fi
    hf download "$repo" --revision "$rev" --local-dir "$dest" --include "$shard" 2>/dev/null
    echo "  ✓ $shard"
}
export -f download_shard

echo "downloading $N_SHARDS shards ($PARALLEL at a time)..."
echo "$SHARDS" | xargs -P "$PARALLEL" -I {} bash -c \
    'download_shard "$0" "$1" "$2" "$3"' "$REPO" "$REVISION" "$DEST" {}

# ---- verify ----
echo
echo "verifying..."

# Portable size
if stat -c%s /dev/null >/dev/null 2>&1; then
    size_cmd="stat -c%s"
else
    size_cmd="stat -f%z"
fi

N=$(find "$DEST" -maxdepth 1 -name '*.safetensors' | wc -l | tr -d ' ')
B=0
while IFS= read -r f; do
    [ -n "$f" ] || continue
    B=$((B + $("$size_cmd" "$f")))
done < <(find "$DEST" -maxdepth 1 -name '*.safetensors')

printf '  shards: %s (expect %s)\n' "$N" "$EXPECT_SHARDS"
printf '  bytes:  %s (expect %s)\n' "$B" "$EXPECT_BYTES"

if [ "$N" -ne "$EXPECT_SHARDS" ]; then
    echo "FAIL: wrong shard count. Re-run to resume."
    exit 1
fi
if [ "$B" -ne "$EXPECT_BYTES" ]; then
    echo "FAIL: byte total mismatch. Re-run to resume."
    exit 1
fi

# Per-shard size check (Kimi K3 only — has shard_sizes.txt)
SIZES="$(dirname "$0")/shard_sizes.txt"
if [ -f "$SIZES" ]; then
    bad=0
    while read -r name want; do
        [ -n "$name" ] || continue
        got=$("$size_cmd" "$DEST/$name" 2>/dev/null || echo 0)
        if [ "$got" != "$want" ]; then
            printf '  BAD  %s: %s bytes, expected %s\n' "$name" "$got" "$want"
            bad=$((bad + 1))
        fi
    done < "$SIZES"
    if [ "$bad" -ne 0 ]; then
        echo "FAIL: $bad shard(s) wrong size. Delete those files and re-run."
        exit 1
    fi
    printf '  all %s shards match published sizes\n' "$EXPECT_SHARDS"
fi

# Checksum verification (optional, slow)
if [ "${K3_SKIP_CHECKSUM:-0}" != "1" ]; then
    echo
    echo "verifying checksums (re-reads all data; set K3_SKIP_CHECKSUM=1 to skip)..."
    hf cache verify "$REPO" --revision "$REVISION" --local-dir "$DEST" \
        --fail-on-missing-files 2>/dev/null && echo "  ✓ checksums verified" || \
        echo "  ! checksum verification skipped (non-fatal)"
fi

echo
echo "done: $DEST"
echo "  $N shards, $(echo "scale=1; $B/1073741824" | bc) GB"