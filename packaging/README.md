# Packaging notes for litMoE.

This directory contains distribution recipes for the platform-specific
packaging systems Homebrew, Windows MSI, Debian, and RPM do not provide
out-of-the-box.

## What's here

```
packaging/
├── homebrew/
│   └── litMoE.rb     # Homebrew formula
├── windows/
│   └── build-msi.ps1       # WiX MSI builder (Windows)
├── rpm/
│   └── litMoE.spec   # RPM .spec
└── README.md               # this file
```

## Homebrew (macOS, Linuxbrew)

**Tap:** `chazhyseni/litMoE`

**To publish the tap:**

1. Create a new GitHub repo: `chazhyseni/homebrew-litMoE`
2. Copy `packaging/homebrew/litMoE.rb` into `Formula/litMoE.rb`
3. Push
4. Users install with:

```bash
brew tap chazhyseni/litMoE
brew install litMoE
```

**SHA256 placeholder:** the formula has `PLACEHOLDER_SHA256` which the
release workflow computes and updates in the tap repo automatically.
For the initial manual publish:

```bash
# From the tap repo root:
sha256sum <(curl -L https://github.com/chazhyseni/litMoE/archive/refs/tags/v0.6.8.tar.gz)
# Then paste the hash into Formula/litMoE.rb
```

## Windows MSI

The MSI is built with the WiX Toolset 3.x. The build script
(`packaging/windows/build-msi.ps1`) requires:

- PowerShell 5.1+
- WiX 3.x installed (via `choco install wixtoolset`)
- The litMoE Windows release zip extracted to `dist/`

The script produces `litMoE-0.6.8.msi` (~7 MB) which:

- Installs to `C:\Program Files\litMoE\`
- Adds `C:\Program Files\litMoE\bin` to system PATH
- Registers the `litMoE` Add/Remove Programs entry
- Bundles LICENSE, README, INSTALL.md

Tested on Windows Server 2022 with WiX 3.14.

## Debian / Ubuntu (.deb)

Not yet provided. To produce one:

```bash
# In a clean Debian 12 environment:
sudo apt-get install checkinstall
LDFLAGS="-lm -pthread" make install PREFIX=/usr
sudo checkinstall --pkgname=litMoE \
    --pkgversion=0.6.8 \
    --pkglicense=Apache-2.0 \
    --maintainer="chazhyseni" \
    --description="Lean OpenAI-compatible server for Kimi K3"
```

This generates a `litMoE_0.6.8-1_amd64.deb` for upload to a PPA or
direct distribution.

## Fedora / RHEL (RPM)

The `.spec` file in `packaging/rpm/` is the RPM spec.

Build with:

```bash
rpmbuild -ba packaging/rpm/litMoE.spec
```

Produces `~/rpmbuild/RPMS/x86_64/litMoE-0.6.8-1.x86_64.rpm`.

## Docker

The `Dockerfile` and `docker-compose.yml` at the repo root are the
canonical container packaging. Multi-arch (linux/amd64, linux/arm64) is
built and pushed by `.github/workflows/release.yml` on tag push:

```
ghcr.io/chazhyseni/litMoE:0.6.8     (runtime)
ghcr.io/chazhyseni/litMoE:latest    (rolling)
ghcr.io/chazhyseni/litMoE:convert-0.6.8  (PyTorch convert)
```