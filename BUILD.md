# Building from source

This document describes how to build kimi-k3-lean from source on each
supported platform. Most users should use [`install.sh`](install.sh)
or [`install.ps1`](install.ps1) instead — those scripts wrap the
commands below.

## Source layout

```
.
├── Makefile                    # POSIX build (Linux + macOS)
├── CMakeLists.txt              # Cross-platform build (everywhere)
├── install.sh                  # POSIX install wrapper
├── install.ps1                 # Windows install wrapper
├── Dockerfile                  # Multi-arch runtime image
├── Dockerfile.convert          # PyTorch-using convert image
├── docker-compose.yml          # Compose for the runtime image
├── .dockerignore
├── .env.example
│
├── include/
│   ├── k3/                     # article's public headers (k3.h, k3_cfg.h, …)
│   └── libk3/
│       ├── libk3.h             # public C API (the seam)
│       └── k3_internal.h       # private header
│
├── src/
│   ├── core/k3_ops.c           # kernels (1,361 lines)
│   ├── io/                     # safetensors / trunk readers
│   ├── cache/k3_cache.c        # expert LRU
│   ├── model/k3_bind.c         # tensor-name binding
│   ├── tokenizer/              # BPE
│   ├── lib/
│   │   ├── k3_engine.c         # extracted engine
│   │   └── k3_api.c            # public API implementation
│   └── cli/k3_run.c            # thin CLI
│
├── tests/
│   ├── fixtures/               # synthetic tiny_k3, gate reference data
│   └── unit/                   # article's test suite (test_ops, etc.)
│
├── docs/                       # article's docs + ours (combined, design choices, etc.)
│
├── bin/                        # build output
│   ├── k3                      # CLI
│   └── libk3.so                # shared library (Linux)
└── COMBINED.md, README.md, OPTION_C_HANDOFF.md, COMPARISON.md, …
```

---

## Linux x86_64

Tested on Debian 11.11 (this host), Ubuntu 22.04 / 24.04.

### Required toolchain

| tool | version | purpose |
|---|---|---|
| gcc | 9+ (10+ for AVX2 detection) | compiler |
| glibc | 2.31+ | libc |
| make | any | build |
| binutils | any | linker (`ld`, `nm`, `objdump`) |
| pthread | glibc-builtin | threading |

### Build

```bash
# From a fresh clone.
git clone https://github.com/sqliteai/kimi-k3-lean.git
cd kimi-k3-lean
LDFLAGS="-lm -pthread" make -j$(nproc)
```

> **The `LDFLAGS="-lm -pthread"` override is required** on any host
> with a conda environment. The Makefile uses `LDFLAGS ?=`, which does
> not append to a conda-set `LDFLAGS`. Without the override, you get
> `undefined reference to expf`, `sqrtf`, and `pthread_create`.

### Test

```bash
LDFLAGS="-lm -pthread" make test
```

Expected: GATEs 1, 2, 3 all pass; "VERDICT: ENGINE MATCHES THE
REFERENCE EXACTLY".

### Install

```bash
sudo make install PREFIX=/usr/local
# or, no sudo:
make install PREFIX=$HOME/.local
export LD_LIBRARY_PATH=$HOME/.local/lib
```

---

## Linux aarch64 (ARM)

Tested in CI on Ubuntu 24.04 aarch64. Should work on Graviton, Apple
Silicon-via-Linux, RPi 5.

### Required toolchain

| tool | version |
|---|---|
| gcc | 10+ (for `-march=armv8.2-a+dotprod+i8mm`) |
| aarch64-linux-gnu binutils | any |

### Build

Same as x86_64; the Makefile auto-detects the architecture:

```bash
LDFLAGS="-lm -pthread" make -j$(nproc)
```

The engine uses ARM NEON + dotprod + i8mm (not SVE — that's a future
optimization). Performance is roughly 0.7× the M5 Pro numbers in the
README — the Apple M-series has wider NEON units than typical ARM
servers.

---

## macOS arm64 (M-series)

Tested in CI on macOS 14 (Sonoma) on M-series.

### Required toolchain

```bash
xcode-select --install     # installs clang, make, etc.
```

Homebrew is optional — `brew install llvm libomp` only if you want a
newer LLVM than Apple ships.

### Build

```bash
LDFLAGS="-lm -pthread" make -j$(sysctl -n hw.ncpu)
```

The Makefile detects `arm64` and emits
`-march=armv8.2-a+fp16+dotprod+i8mm`.

### Install

```bash
make install PREFIX=$HOME/.local
```

---

## macOS x86_64 (Intel)

Tested in CI on macOS 13 (Ventura). Should work on older Intel Macs.

### Build

```bash
LDFLAGS="-lm -pthread" make -j$(sysctl -n hw.ncpu)
```

Emits `-march=nocona -mavx2 -mfma` — same as Linux x86_64.

---

## Windows (MSVC)

Tested in CI on Windows Server 2022.

### Required toolchain

| tool | install via |
|---|---|
| Visual Studio 2022 Build Tools | winget install Microsoft.VisualStudio.2022.BuildTools |
| CMake | winget install Kitware.CMake |
| Ninja | winget install Ninja-build.Ninja |

### Build

```powershell
# From a Developer PowerShell prompt (vcvars on PATH).
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Test

```powershell
cd build
ctest --output-on-failure -C Release
```

### Install

```powershell
cmake --install build --prefix C:\Users\you\k3lean
```

---

## Windows (MinGW)

Tested in CI on Windows Server 2022 with MSYS2.

### Required toolchain

```powershell
winget install MSYS2.MSYS2
# Inside MSYS2 shell:
pacman -S mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja
```

### Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

---

## Cross-compilation

Cross-compilation works via CMake. The Makefile does not support
cross-compilation out of the box.

### Linux x86_64 → Linux aarch64

```bash
# Install the cross toolchain.
sudo apt-get install gcc-aarch64-linux-gnu

# Configure with the cross compiler.
cmake -S . -B build-aarch64 \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc
cmake --build build-aarch64 --parallel
```

### Linux x86_64 → Windows

```bash
sudo apt-get install mingw-w64

cmake -S . -B build-windows \
    -DCMAKE_SYSTEM_NAME=Windows \
    -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc
cmake --build build-windows --parallel
```

The Windows cross-built binary runs on any Windows x86_64 host. Test
it with `wine build-windows/bin/k3.exe --help` if you have wine
installed.

---

## Build targets

| target | what it does |
|---|---|
| `make` | builds `bin/k3` and `bin/libk3.so` |
| `make libk3` | builds only the shared library |
| `make libk3.so` | builds only the shared library |
| `make libk3.a` | builds only the static library |
| `make test` | runs the 3-GATE oracle test suite |
| `make clean` | removes all build artifacts |
| `make install PREFIX=…` | installs to `$PREFIX/{bin,lib,include}` |
| `make uninstall PREFIX=…` | removes installed files |
| `make portable` | builds a portable tarball (see scripts/) |
| `make debug` | debug build (`-O0 -g3`) |
| `make asan` | AddressSanitizer build |
| `make ubsan` | UndefinedBehaviorSanitizer build |
| `make help` | lists all targets |

CMake equivalents:

```bash
cmake --build build                    # incremental
cmake --build build --target k3        # CLI only
cmake --build build --target k3        # shared library
cmake --install build --prefix …       # install
```

---

## Compiler flags we set

| flag | reason |
|---|---|
| `-O2` | default optimization (override with `-O3` in CFLAGS) |
| `-mavx2 -mfma` | x86_64 SIMD (no AVX-512; not portable to AMD Zen ≤ 4) |
| `-march=armv8.2-a+fp16+dotprod+i8mm` | ARM SIMD on M-series and Graviton 3+ |
| `-ffp-contract=off` | deterministic floating-point across compilers |
| `-fopenmp` | OpenMP for parallel reduction in kernels |
| `-pthread` | thread support (linker flag) |

To override, set `CFLAGS` before invoking `make` (or pass
`-DCMAKE_C_FLAGS` to CMake).

---

## What gets built

| output | description | size |
|---|---|---|
| `bin/k3` | CLI binary | ~170 KB |
| `bin/libk3.so` | shared library (the seam) | ~165 KB |
| `bin/libk3_static.a` | static library (for embedding) | ~600 KB |
| `include/libk3/libk3.h` | public C API header | ~7 KB |
| `include/libk3/k3_internal.h` | private header | ~3 KB |

The engine itself is **176 KB of compiled C**. The 1.45 TB of model
weights are NOT built; they live on disk and are read at runtime.

---

## Sanitizer builds

For debugging, the Makefile has sanitizer targets:

```bash
make asan     # AddressSanitizer: catches out-of-bounds, use-after-free
make ubsan    # UndefinedBehaviorSanitizer: catches signed overflow, etc.
LDFLAGS="-lm -pthread" make test
```

ASan is slower (2-3×) but catches memory bugs that would otherwise
silently corrupt output. If you're debugging an unexpected result, ASan
is the first thing to try.