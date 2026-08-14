<#
bootstrap.ps1 — install kimi-k3-lean and start the server in one command.

Usage (the headline use case):
  Invoke-Expression (Invoke-WebRequest -UseBasicParsing https://raw.githubusercontent.com/chazhyseni/kimi-k3-lean/main/bootstrap.ps1).Content

What this does:
  1. Clones the repo to $K3_DIR (default: ~/.kimi-k3-lean)
  2. Builds libk3.dll + k3.exe
  3. Starts the OpenAI server with whatever weights are in ./checkpoints/k3
     (downloading and converting them if not yet present)

If you want to install the C engine + Python server to a system prefix,
use .\install.ps1 from inside a clone instead.

Environment:
  $env:K3_DIR         install location (default: ~/.kimi-k3-lean)
  $env:K3_PORT        server port (default: 8080)
  $env:K3_HOST        bind host (default: 127.0.0.1)
  $env:K3_API_KEY     optional bearer token for the server
  $env:K3_PRESET      memory preset: laptop | desktop | workstation | server | max | auto
                      (default: auto)
  $env:K3_SKIP_DL     set to 1 to skip the model download (server-only)
#>

$ErrorActionPreference = "Stop"

$REPO_URL = "https://github.com/chazhyseni/kimi-k3-lean.git"
$BRANCH   = "main"

$K3_DIR    = if ($env:K3_DIR)    { $env:K3_DIR }    else { Join-Path $env:USERPROFILE ".kimi-k3-lean" }
$K3_PORT   = if ($env:K3_PORT)   { [int]$env:K3_PORT } else { 8080 }
$K3_HOST   = if ($env:K3_HOST)   { $env:K3_HOST }   else { "127.0.0.1" }
$K3_PRESET = if ($env:K3_PRESET) { $env:K3_PRESET } else { "auto" }
$K3_SKIP_DL = if ($env:K3_SKIP_DL) { $env:K3_SKIP_DL } else { "0" }

function Need($cmd) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Error "missing: $cmd  (install it first)"
        exit 1
    }
}

Write-Host "==> preflight" -ForegroundColor Cyan
Need git
Need cmake
Need cl       # MSVC: run from a "Developer PowerShell for VS" prompt
              # MinGW: cl not found; the build will still work if ninja is preferred.

# ----- 2. clone (or update) -----
if (Test-Path (Join-Path $K3_DIR ".git")) {
    Write-Host "==> updating $K3_DIR" -ForegroundColor Cyan
    git -C $K3_DIR fetch --depth 1 origin $BRANCH 2>&1 | Out-Null
    git -C $K3_DIR reset --hard "origin/$BRANCH" 2>&1 | Out-Null
    Write-Host "  ✓ updated" -ForegroundColor Green
} elseif (Test-Path $K3_DIR) {
    Write-Error "$K3_DIR exists but is not a git repo. Remove it or set K3_DIR."
    exit 1
} else {
    Write-Host "==> cloning to $K3_DIR" -ForegroundColor Cyan
    git clone --depth 1 --branch $BRANCH $REPO_URL $K3_DIR
    Write-Host "  ✓ cloned" -ForegroundColor Green
}

Set-Location $K3_DIR

# ----- 3. build -----
if (-not (Test-Path "bin/libk3.dll")) {
    Write-Host "==> building" -ForegroundColor Cyan
    cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release
    Write-Host "  ✓ built" -ForegroundColor Green
} else {
    Write-Host "  ✓ build cached" -ForegroundColor Green
}

# ----- 4. start the server -----
if ($K3_SKIP_DL -eq "1" -or (Test-Path "checkpoints/k3") -and (Get-ChildItem "checkpoints/k3" -ErrorAction SilentlyContinue)) {
    Write-Host "==> starting server (--serve-only)" -ForegroundColor Cyan
    $args = @("--host", $K3_HOST, "--port", "$K3_PORT", "--preset", $K3_PRESET)
    if ($K3_API_KEY) { $args += @("--api-key", $K3_API_KEY) }

    $env:PATH = "$K3_DIR\bin;$env:PATH"
    if (Test-Path "checkpoints/k3") {
        & python serve\__main__.py checkpoints/k3 @args
    } else {
        & python serve\__main__.py tiny_k3.bin @args
    }
} else {
    # Full setup: download + convert + serve.
    & scripts\setup-and-serve.ps1 -Preset $K3_PRESET -Port $K3_PORT -Host $K3_HOST
}
