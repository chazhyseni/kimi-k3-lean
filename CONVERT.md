# CONVERT — turn a HuggingFace Kimi K3 checkpoint into native format

The C engine in `kimi-k3-lean` reads its weights from **safetensors
shards**, not the HuggingFace transformer cache. The shipped `setup-and-serve.sh`
uses Docker to run the conversion; this document covers the other paths.

## What "convert" means

The article's engine expects a directory of safetensors shards matching
the released Kimi K3 layout:

```
checkpoints/k3-native/
├── config.json
├── tokenizer.model
├── model-00001-of-00096.safetensors
├── model-00002-of-00096.safetensors
├── ...
└── model-00096-of-00096.safetensors
```

Each shard is ~10 GB raw bf16. After convert, weights stay bf16 (the
engine's quantization happens at runtime via the LFRU expert cache).
Total disk footprint: ~982 GB.

## Path 1: Docker (default)

The `Dockerfile.convert` image includes PyTorch + transformers +
safetensors. Run:

```bash
docker build -f Dockerfile.convert -t kimi-k3-convert .
docker run --rm \
    -v /path/to/raw/checkpoint:/in:ro \
    -v /path/to/native:/out:rw \
    kimi-k3-convert \
    python3 /opt/kimi-k3-lean/tools/convert.py /in /out
```

This reads from `/in` (the HuggingFace cache) and writes native format
to `/out`. Plan for ~1 hour of CPU time on a workstation.

## Path 2: PyTorch on the host (no Docker)

You need:

- Python 3.11
- `pip install torch transformers safetensors sentencepiece tiktoken`
- ~64 GB RAM (the convert step materializes tensors in memory)

Then:

```bash
git clone https://github.com/chazhyseni/kimi-k3-lean.git
cd kimi-k3-lean
python3 tools/convert.py /path/to/raw/checkpoint /path/to/native
```

## Path 3: Manual (don't do this unless you must)

If neither Docker nor PyTorch is available:

1. Download the 96 safetensors shards manually from HuggingFace using
   `huggingface-cli download moonshotai/Kimi-K3-Instruct` (or `git lfs
   clone`).
2. Make sure `config.json` and `tokenizer.model` are in the same
   directory.
3. The C engine should load directly from the safetensors directory
   **without** a convert step — its native format IS safetensors.

The `convert.py` script's only job is to validate the layout and
optionally re-shard for the engine's expected shard count. If you can
satisfy `k3_open` from raw HuggingFace output, skip the convert entirely.

## What the engine actually reads

`k3_open` looks for:

- One or more `model-*-of-*.safetensors` shards
- `config.json` (with the model's hidden_size, num_hidden_layers,
  num_attention_heads, num_experts, num_experts_per_token, etc.)
- `tokenizer.model` (tiktoken-format BPE)

If all three are present, no convert step is needed. The convert tool
in `tools/convert.py` is for hosts that want a different shard count
or to pre-validate the layout.

## Troubleshooting

**"config.json not found"** — the C engine reads this on open. Make
sure it's in the same directory as the safetensors shards.

**"missing tensor language_model.model.layers.0.input_layernorm.weight"**
— one of the shards is incomplete. Re-download with `huggingface-cli`.

**"k3_open failed: out of memory"** — the article's `k3_open` loads
layer 0 (the MLA layers and the trunk cache) into RAM. If you don't
have ~8 GB free, use `--preset laptop` (which has a 4 GB trunk budget).

**"engine_version mismatch"** — the engine version that produced the
checkpoint isn't the same as the engine you're running. The article
doesn't version-gate yet, but if you see this, rebuild `libk3.so` from
the matching source.