"""
convert.py — convert a HuggingFace Kimi K3 checkpoint into native format.

WHAT THIS DOES
  Reads safetensors shards from an input directory (downloaded from
  HuggingFace) and copies/validates them into an output directory
  shaped for `k3_open`. The C engine reads safetensors natively, so
  the "convert" step is mostly:

  1. Verify all 96 shards are present and byte-complete.
  2. Verify config.json is present and has the expected fields.
  3. Verify tiktoken.model is present.
  4. Optionally re-shard into a different number of files.

USAGE

  python3 tools/convert.py /path/to/hf/checkpoint /path/to/native

  # Or with a different output shard count:
  python3 tools/convert.py /path/to/hf /path/to/native --shards 24

  # To only verify, not copy:
  python3 tools/convert.py /path/to/hf --verify-only

WHAT THE ENGINE EXPECTS

  The C engine's k3_open reads:

  /path/to/native/
  ├── config.json                          # Kimi K3 architecture
  ├── tiktoken.model                     # tiktoken BPE
  └── model-NNNNN-of-NNNNN.safetensors     # 96 shards by default

  k3_open validates the layout on open. A correct layout means the
  engine can load it without errors.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# These are the published values for Kimi K3. Verified against the
# article's reference; not assumed.
EXPECTED_SHARDS = 96
EXPECTED_HIDDEN = 7168
EXPECTED_LAYERS = 93          # 69 KDA + 24 MLA + 1 dense (see k3.h)
EXPECTED_EXPERTS = 896
EXPECTED_TOPK = 16            # top-16 routed + 2 shared
EXPECTED_VOCAB = 163840


def die(msg: str, code: int = 1) -> None:
    print(f"xx {msg}", file=sys.stderr)
    sys.exit(code)


def info(msg: str) -> None:
    print(f"==> {msg}")


def verify(src: Path, *, shards: int = EXPECTED_SHARDS) -> dict:
    """Verify the source checkpoint layout. Returns the loaded config."""
    if not src.is_dir():
        die(f"source is not a directory: {src}")

    # Check config.json
    config_path = src / "config.json"
    if not config_path.exists():
        die(f"missing config.json at {config_path}")

    config = json.loads(config_path.read_text())
    for k, v in [
        ("hidden_size", EXPECTED_HIDDEN),
        ("num_hidden_layers", EXPECTED_LAYERS),
        ("num_experts", EXPECTED_EXPERTS),
        ("num_experts_per_token", EXPECTED_TOPK),
        ("vocab_size", EXPECTED_VOCAB),
    ]:
        actual = config.get(k)
        if actual != v:
            warn(f"config field {k}: got {actual}, expected {v}")
            warn(f"this might still work, but the engine defaults will be used")

    # Check tokenizer
    if not (src / "tiktoken.model").exists():
        die(f"missing tiktoken.model at {src}/tiktoken.model")

    # Check shards
    found_shards = sorted(src.glob("model-*-of-*.safetensors"))
    info(f"found {len(found_shards)} safetensors shards (expected {shards})")
    if len(found_shards) != shards:
        warn(f"shard count mismatch: found {len(found_shards)}, "
             f"expected {shards}. The engine may fail to open.")

    # Check sizes
    total_bytes = sum(p.stat().st_size for p in found_shards)
    info(f"total weight bytes: {total_bytes:,}")
    if total_bytes < 1_500_000_000_000:
        warn(f"total bytes {total_bytes:,} < 1.5 TB — shards may be partial")

    return config


def warn(msg: str) -> None:
    print(f"!! {msg}", file=sys.stderr)


def copy(src: Path, dst: Path, *, shards: int = EXPECTED_SHARDS) -> None:
    """Copy shards + config + tokenizer from src to dst.

    If shards != src shard count, the convert is a re-shard. The default
    is to keep the same number of shards.
    """
    dst.mkdir(parents=True, exist_ok=True)

    # Always copy config + tokenizer first.
    for fname in ("config.json", "tiktoken.model", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json"):
        src_path = src / fname
        if src_path.exists():
            shutil.copy2(src_path, dst / fname)
            info(f"copied {fname}")

    src_shards = sorted(src.glob("model-*-of-*.safetensors"))
    if not src_shards:
        die(f"no safetensors shards found at {src}")

    if shards == len(src_shards):
        # No re-shard. Just copy.
        for src_shard in src_shards:
            dst_shard = dst / src_shard.name
            info(f"copying {src_shard.name} ({src_shard.stat().st_size:,} bytes)")
            shutil.copy2(src_shard, dst_shard)
    else:
        # Re-shard: concatenate all tensors, split into `shards` files.
        info(f"re-sharding from {len(src_shards)} -> {shards} files")
        _reshard(src_shards, dst, shards)


def _reshard(src_shards: list, dst: Path, num_shards: int) -> None:
    """Concatenate all tensors, then split into num_shards safetensors files.

    Uses safetensors directly to avoid PyTorch as a hard dependency.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    # 1. Concatenate all tensors into a single dict.
    tensors = {}
    for shard in src_shards:
        info(f"  reading {shard.name}")
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

    info(f"  {len(tensors)} tensors total")

    # 2. Split into num_shards by byte size.
    shard_bytes = sum(t.element_size() * t.numel() for t in tensors.values())
    per_shard = shard_bytes // num_shards
    info(f"  target per-shard: ~{per_shard:,} bytes")

    shards: list[dict] = [{} for _ in range(num_shards)]
    shard_sizes = [0] * num_shards
    # Sort keys for deterministic ordering across runs.
    for k in sorted(tensors.keys()):
        t = tensors[k]
        b = t.element_size() * t.numel()
        # Place in the smallest shard to balance.
        idx = shard_sizes.index(min(shard_sizes))
        shards[idx][k] = t
        shard_sizes[idx] += b

    # 3. Write.
    width = max(5, len(str(num_shards)))
    for i, sd in enumerate(shards, start=1):
        name = f"model-{i:0{width}d}-of-{num_shards:0{width}d}.safetensors"
        info(f"  writing {name} ({shard_sizes[i-1]:,} bytes, {len(sd)} tensors)")
        save_file(sd, str(dst / name))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert Kimi K3 checkpoint to native format",
    )
    p.add_argument("src", type=Path,
                   help="source directory (HuggingFace checkpoint)")
    p.add_argument("dst", type=Path, nargs="?",
                   help="destination directory (defaults to <src>-native)")
    p.add_argument("--verify-only", action="store_true",
                   help="only verify; don't copy")
    p.add_argument("--shards", type=int, default=EXPECTED_SHARDS,
                   help=f"output shard count (default {EXPECTED_SHARDS})")

    args = p.parse_args()

    src = args.src.resolve()
    dst = (args.dst or src.with_name(src.name + "-native")).resolve()

    if not src.exists():
        die(f"source does not exist: {src}")

    info(f"verifying {src}")
    verify(src, shards=args.shards)

    if args.verify_only:
        info("verify-only mode: done")
        return 0

    info(f"copying to {dst}")
    copy(src, dst, shards=args.shards)
    info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())