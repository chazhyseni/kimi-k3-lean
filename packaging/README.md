# Packaging notes for kimi-k3-lean.

This directory contains distribution recipes for the platform-specific
packaging systems Homebrew, Windows MSI, Debian, and RPM do not provide
out-of-the-box.

## What's here

```
packaging/
├── homebrew/
│   └── kimi-k3-lean.rb     # Homebrew formula
├── windows/
│   └── build-msi.ps1       # WiX MSI builder (Windows)
├── debian/
│   └── kimi-k3-lean.spec   # RPM .spec (works on rpm-based; for .deb use alien)
└── README.md               # this file
```

## Homebrew (macOS, Linuxbrew)

**Tap:** `sqliteai/kimi-k3-lean`

**To publish the tap:**

1. Create a new GitHub repo: `sqliteai/homebrew-kimi-k3-lean`
2. Copy `packaging/homebrew/kimi-k3-lean.rb` into `Formula/kimi-k3-lean.rb`
3. Push
4. Users install with:

```bash
brew tap sqliteai/kimi-k3-lean
brew install kimi-k3-lean
```

**SHA256 placeholder:** the formula has `PLACEHOLDER_SHA256` which the
release workflow computes and updates in the tap repo automatically.
For the initial manual publish:

```bash
# From the tap repo root:
sha256sum <(curl -L https://github.com/sqliteai/kimi-k3-lean/archive/refs/tags/v0.6.8.tar.gz)
# Then paste the hash into Formula/kimi-k3-lean.rb
```

## Windows MSI

The MSI is built with the WiX Toolset 3.x. The build script
(`packaging/windows/build-msi.ps1`) requires:

- PowerShell 5.1+
- WiX 3.x installed (via `choco install wixtoolset`)
- The kimi-k3-lean Windows release zip extracted to `dist/`

The script produces `kimi-k3-lean-0.6.8.msi` (~7 MB) which:

- Installs to `C:\Program Files\kimi-k3-lean\`
- Adds `C:\Program Files\kimi-k3-lean\bin` to system PATH
- Registers the `kimi-k3-lean` Add/Remove Programs entry
- Bundles LICENSE, README, INSTALL.md

Tested on Windows Server 2022 with WiX 3.14.

## Debian / Ubuntu (.deb)

Not yet provided. To produce one:

```bash
# In a clean Debian 12 environment:
sudo apt-get install checkinstall
LDFLAGS="-lm -pthread" make install PREFIX=/usr
sudo checkinstall --pkgname=kimi-k3-lean \
    --pkgversion=0.6.8 \
    --pkglicense=Apache-2.0 \
    --maintainer="sqliteai" \
    --description="Lean OpenAI-compatible server for Kimi K3"
```

This generates a `kimi-k3-lean_0.6.8-1_amd64.deb` for upload to a PPA or
direct distribution.

## Fedora / RHEL (RPM)

The `.spec` file in `packaging/debian/kimi-k3-lean.spec` is actually
the RPM spec — the directory name is misleading (rename to `rpm/` if
you want to be tidy).

Build with:

```bash
rpmbuild -ba packaging/rpm/kimi-k3-lean.spec
```

Produces `~/rpmbuild/RPMS/x86_64/kimi-k3-lean-0.6.8-1.x86_64.rpm`.

## Docker

The `Dockerfile` and `docker-compose.yml` at the repo root are the
canonical container packaging. Multi-arch (linux/amd64, linux/arm64) is
built and pushed by `.github/workflows/release.yml` on tag push:

```
ghcr.io/sqliteai/kimi-k3-lean:0.6.8     (runtime)
ghcr.io/sqliteai/kimi-k3-lean:latest    (rolling)
ghcr.io/sqliteai/kimi-k3-lean:convert-0.6.8  (PyTorch convert)
```