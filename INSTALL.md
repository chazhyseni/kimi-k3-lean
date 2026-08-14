# Installation

Five minutes from clone to a working `k3 --help` on any of the three
target platforms. Choose your OS below.

> **TL;DR for any platform:**
>
> ```bash
> git clone https://github.com/sqliteai/kimi-k3-lean.git
> cd kimi-k3-lean
> ```
>
> Then follow your OS section below.

---

## Linux (glibc 2.31+, x86_64 or aarch64)

Tested on Debian 11.11 (this host), Debian 12, Ubuntu 22.04/24.04.
Should work on RHEL 9+, Fedora, Arch.

### One-shot install (recommended)

```bash
./install.sh                       # system install (needs sudo)
./install.sh PREFIX=$HOME/.local   # user install, no sudo
```

This builds with the Makefile, then runs `make install`. On Linux it
also adds `$PREFIX/lib` to `ldconfig` so `libk3.so` is found at runtime
without `LD_LIBRARY_PATH`.

### Manual install

```bash
# Build.
LDFLAGS="-lm -pthread" make -j$(nproc)

# Install.
sudo make install PREFIX=/usr/local
# or, no sudo:
make install PREFIX=$HOME/.local
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

# Verify.
k3 --help
k3 --version
```

### CMake (alternative)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
sudo cmake --install build --prefix /usr/local
```

### Distribution packages

| distro | status | install command |
|---|---|---|
| Debian 12+ | build from source | (none yet) |
| Ubuntu 22.04+ | build from source | (none yet) |
| Fedora 39+ | build from source | (none yet) |
| Arch | build from source | (none yet) |

A `.deb` / `.rpm` recipe is on the roadmap. The CI matrix runs on
Ubuntu 22.04 and 24.04 so it should "just work" when packaged.

---

## macOS (Sonoma 14+ on M-series, Ventura 13+ on Intel)

Tested in CI on macOS 13 + 14, both arm64 and x86_64.

### Apple Silicon (M1 / M2 / M3 / M4 / M5)

```bash
# Install Apple Command Line Tools if you don't already have them.
xcode-select --install

# Clone and install.
git clone https://github.com/sqliteai/kimi-k3-lean.git
cd kimi-k3-lean
./install.sh PREFIX=$HOME/.local
```

This builds with `clang` (the system default). The engine uses
`armv8.2-a+fp16+dotprod+i8mm` extensions when supported.

### Intel macOS (x86_64)

```bash
git clone https://github.com/sqliteai/kimi-k3-lean.git
cd kimi-k3-lean
./install.sh PREFIX=$HOME/.local
```

### Homebrew (future)

A Homebrew tap is on the roadmap:

```bash
brew tap sqliteai/kimi-k3-lean
brew install kimi-k3-lean
```

The formula is being prepared but not yet published.

---

## Windows (10/11, Server 2019+, x86_64)

Tested in CI on Windows Server 2022 with both MSVC and MinGW.

### PowerShell (recommended)

```powershell
git clone https://github.com/sqliteai/kimi-k3-lean.git
cd kimi-k3-lean
.\install.ps1                                  # system install (admin)
.\install.ps1 -Prefix C:\Users\you\k3lean      # user install
```

This invokes CMake under the hood (the Makefile uses Unix tools that
aren't on Windows by default). If you don't have a C compiler installed
yet, install one of:

- **Visual Studio Build Tools 2022** with the "Desktop development with C++" workload (largest, full features)
- **MinGW-w64** via `winget install mingw-w64` (smaller, MSYS2-style)

The `install.ps1` script will detect whichever is available.

### Manual CMake (Windows)

```powershell
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build --prefix C:\Users\you\k3lean
```

### Windows path / DLL notes

`libk3.dll` is in `C:\Users\you\k3lean\bin\`. Windows resolves DLLs by:

1. The directory of the executable
2. The current working directory
3. Directories in `PATH`

So either put `bin\` in your `PATH`, or copy `libk3.dll` next to the
`k3.exe` you want to run. The PowerShell install adds it to `PATH`
automatically.

### Windows installer (future)

A Windows MSI is on the roadmap. The CI matrix produces a portable ZIP
for each release, which is the canonical Windows distribution for now.

---

## Verifying the install

Every install path should produce a working `k3` binary. The test
gates do NOT need model weights — they use synthetic fixtures that ship
in the repo.

```bash
# All three should succeed.
k3 --help
k3 --version
k3 --list-presets

# The 3-GATE test suite (no model required).
LDFLAGS="-lm -pthread" make test
# or, on Windows:
cd build && ctest --output-on-failure
```

The expected output is:

```
GATE 1  teacher forcing : 32/32 positions match tf_pred
GATE 2  greedy decode   : 20/20 generated tokens match full_ids
GATE 3  incremental    : 20/20 generated tokens match full_ids

VERDICT: ENGINE MATCHES THE REFERENCE EXACTLY
```

If you see this, the engine is built and verified bit-identical to the
reference oracle on the synthetic tiny_k3 fixture.

---

## Docker

For all platforms, the Docker image is the easiest install:

```bash
# Build.
docker build -t kimi-k3-lean:latest .

# Verify.
docker run --rm kimi-k3-lean:latest --help

# Run (after fetching the model).
docker run --rm \
    -v /path/to/checkpoint:/model:ro \
    -p 8080:8080 \
    kimi-k3-lean:latest \
    /model --preset server
```

The convert step (HuggingFace → native format) has its own image:

```bash
docker build -f Dockerfile.convert -t kimi-k3-lean:convert .

# Convert a single shard (or directory of shards).
docker run --rm \
    -v /path/to/shards:/in \
    -v /path/to/output:/out \
    kimi-k3-lean:convert \
    /in --out /out
```

The `docker-compose.yml` brings up the runtime image with sensible
defaults; copy `.env.example` to `.env` and set `K3_MODEL_DIR` first.

---

## Uninstalling

```bash
# Linux / macOS.
make uninstall PREFIX=/usr/local       # or whatever you used

# Windows.
# No uninstaller yet; remove the install prefix and the PATH entry manually.
Remove-Item -Recurse C:\Users\you\k3lean
```

---

## Next steps

- See [`BUILD.md`](BUILD.md) for source-build instructions, including
  cross-compilation.
- See [`COMBINED.md`](COMBINED.md) for the architecture overview and
  the OpenAI-server story.
- See [`docs/TUNING.md`](docs/TUNING.md) for choosing the right preset.
- See [`README.md`](README.md) for the original article.