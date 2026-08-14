<#
setup-and-serve.ps1 — one command from clone to running server, Windows.

Usage:
  .\scripts\setup-and-serve.ps1                       full path: build + download + convert + serve
  .\scripts\setup-and-serve.ps1 -BuildOnly            just build libk3.dll and k3.exe
  .\scripts\setup-and-serve.ps1 -DownloadOnly         just download weights
  .\scripts\setup-and-serve.ps1 -ConvertOnly          just convert the checkpoint
  .\scripts\setup-and-serve.ps1 -ServeOnly            just start the server

Environment variables:
  $env:K3_MODEL_DIR    checkpoint directory (default: .\checkpoints\k3)
  $env:K3_PRESET       memory preset (default: auto)
  $env:K3_HOST         bind host (default: 127.0.0.1)
  $env:K3_PORT         bind port (default: 8080)
  $env:K3_API_KEY      bearer token
  $env:K3_MAX_TOKENS   default max_tokens (default: 256)

Exit codes match the bash version (0 success, 1 build fail, 2 download
fail, 3 convert fail, 4 model missing, 5 server fail to start, 7 missing
tool).
#>

[CmdletBinding()]
param(
    [switch]$BuildOnly,
    [switch]$DownloadOnly,
    [switch]$ConvertOnly,
    [switch]$ServeOnly
)

$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO_ROOT  = Split-Path -Parent $SCRIPT_DIR

$K3_MODEL_DIR = if ($env:K3_MODEL_DIR) { $env:K3_MODEL_DIR } else { Join-Path $REPO_ROOT "checkpoints\k3" }
$K3_PRESET    = if ($env:K3_PRESET)    { $env:K3_PRESET }    else { "auto" }
$K3_HOST      = if ($env:K3_HOST)      { $env:K3_HOST }      else { "127.0.0.1" }
$K3_PORT      = if ($env:K3_PORT)      { [int]$env:K3_PORT }  else { 8080 }
$K3_API_KEY   = if ($env:K3_API_KEY)   { $env:K3_API_KEY }   else { "" }
$K3_MAX_TOKENS = if ($env:K3_MAX_TOKENS) { [int]$env:K3_MAX_TOKENS } else { 256 }

function Note($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "!! $msg" -ForegroundColor Yellow }
function Die($msg, $code = 1) {
    Write-Host "xx $msg" -ForegroundColor Red
    exit $code
}

function Require-Tool($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Die "missing required tool: $name" 7
    }
}

function Build {
    Note "building libk3.dll and k3.exe"
    Push-Location $REPO_ROOT
    try {
        # mingw32-make / nmake / msbuild handling is done by the Makefile on
        # POSIX systems; on Windows we use the included Makefile with MinGW
        # by default. MSVC users should run via CMake instead.
        $env:LDFLAGS = "-lm -pthread"
        if (Get-Command mingw32-make -ErrorAction SilentlyContinue) {
            & mingw32-make -j 4 libk3 k3
        } elseif (Get-Command nmake -ErrorAction SilentlyContinue) {
            # MSVC users — CMake is the supported path. See BUILD.md.
            Die "MSVC detected — use 'cmake -B build && cmake --build build --config Release' instead" 7
        } else {
            Die "neither mingw32-make nor nmake found; install one of them (or use cmake)" 7
        }
        if (-not (Test-Path "bin\libk3.dll")) { Die "build did not produce bin\libk3.dll" 1 }
        if (-not (Test-Path "bin\k3.exe"))    { Die "build did not produce bin\k3.exe" 1 }
        Note "build OK"
    } finally {
        Pop-Location
    }
}

function Download {
    Note "downloading Kimi K3 weights to $K3_MODEL_DIR"
    Push-Location $REPO_ROOT
    try {
        if (-not (Test-Path "scripts\download-model.sh")) {
            Die "scripts\download-model.sh missing; use WSL or run download manually" 2
        }
        # The article's downloader is bash; on Windows use WSL.
        if (Get-Command wsl -ErrorAction SilentlyContinue) {
            New-Item -ItemType Directory -Force -Path $K3_MODEL_DIR | Out-Null
            & wsl bash scripts/download-model.sh
        } else {
            Die "downloading on Windows requires WSL with bash. " +
                "Install WSL or download manually from HuggingFace." 2
        }
    } finally {
        Pop-Location
    }
}

function Convert {
    Note "converting checkpoint to native format"
    Push-Location $REPO_ROOT
    try {
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            & docker build -f Dockerfile.convert -t kimi-k3-convert .
            & docker run --rm -v "${K3_MODEL_DIR}:/data:rw" -v "${REPO_ROOT}:/out:rw" kimi-k3-convert `
                python3 tools/convert.py /data /out/checkpoints/k3-native
        } else {
            Die "convert step needs Docker; install Docker Desktop for Windows." 3
        }
    } finally {
        Pop-Location
    }
}

function Serve($modelPath) {
    if (-not (Test-Path $modelPath)) {
        Die "model directory not found: $modelPath" 4
    }

    Note "starting kimi-k3-lean OpenAI server"
    Note "  model:    $modelPath"
    Note "  preset:   $K3_PRESET"
    Note "  endpoint: http://${K3_HOST}:${K3_PORT}/v1"
    Note "  api key:  $(if ($K3_API_KEY) { '(set)' } else { '(none — server is open)' })"

    if (($K3_HOST -ne "127.0.0.1") -and (-not $K3_API_KEY)) {
        Warn "binding to non-loopback without --api-key is unsafe."
        Start-Sleep -Seconds 3
    }

    Push-Location $REPO_ROOT
    try {
        $args = @($modelPath,
                  "--preset", $K3_PRESET,
                  "--host", $K3_HOST,
                  "--port", "$K3_PORT",
                  "--max-tokens", "$K3_MAX_TOKENS")
        if ($K3_API_KEY) { $args += @("--api-key", $K3_API_KEY) }

        $env:PATH = "$REPO_ROOT\bin;$env:PATH"
        & python serve\__main__.py @args
    } finally {
        Pop-Location
    }
}

# --------------------------------------------------------------- workflow

if (-not $ServeOnly) {
    Build
}

if (-not $BuildOnly -and -not $ServeOnly -and -not $env:SKIP_DOWNLOAD) {
    Download
}

if (-not $BuildOnly -and -not $DownloadOnly -and -not $ServeOnly -and -not $env:SKIP_CONVERT) {
    Convert
}

if ($BuildOnly -or $DownloadOnly -or $ConvertOnly) {
    Note "done."
    exit 0
}

Serve $K3_MODEL_DIR