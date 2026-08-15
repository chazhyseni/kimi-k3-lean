<#
setup-and-serve.ps1 — one command from clone to running server, Windows.

Usage:
  .\scripts\setup-and-serve.ps1                       full path: build + download + convert + serve
  .\scripts\setup-and-serve.ps1 -BuildOnly            just build liblitmoe.dll and k3.exe
  .\scripts\setup-and-serve.ps1 -DownloadOnly         just download weights
  .\scripts\setup-and-serve.ps1 -ConvertOnly          just convert the checkpoint
  .\scripts\setup-and-serve.ps1 -ServeOnly            just start the server

Environment variables:
  $env:LITMOE_MODEL_DIR    checkpoint directory (default: .\checkpoints\k3)
  $env:LITMOE_PRESET       memory preset (default: auto)
  $env:LITMOE_HOST         bind host (default: 127.0.0.1)
  $env:LITMOE_PORT         bind port (default: 8080)
  $env:LITMOE_API_KEY      bearer token
  $env:LITMOE_MAX_TOKENS   default max_tokens (default: 256)

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

$LITMOE_MODEL_DIR = if ($env:LITMOE_MODEL_DIR) { $env:LITMOE_MODEL_DIR } else { Join-Path $REPO_ROOT "checkpoints\k3" }
$LITMOE_PRESET    = if ($env:LITMOE_PRESET)    { $env:LITMOE_PRESET }    else { "auto" }
$LITMOE_HOST      = if ($env:LITMOE_HOST)      { $env:LITMOE_HOST }      else { "127.0.0.1" }
$LITMOE_PORT      = if ($env:LITMOE_PORT)      { [int]$env:LITMOE_PORT }  else { 8080 }
$LITMOE_API_KEY   = if ($env:LITMOE_API_KEY)   { $env:LITMOE_API_KEY }   else { "" }
$LITMOE_MAX_TOKENS = if ($env:LITMOE_MAX_TOKENS) { [int]$env:LITMOE_MAX_TOKENS } else { 256 }

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
    Note "building liblitmoe.dll and k3.exe"
    Push-Location $REPO_ROOT
    try {
        # mingw32-make / nmake / msbuild handling is done by the Makefile on
        # POSIX systems; on Windows we use the included Makefile with MinGW
        # by default. MSVC users should run via CMake instead.
        $env:LDFLAGS = "-lm -pthread"
        if (Get-Command mingw32-make -ErrorAction SilentlyContinue) {
            & mingw32-make -j 4 liblitmoe k3
        } elseif (Get-Command nmake -ErrorAction SilentlyContinue) {
            # MSVC users — CMake is the supported path. See BUILD.md.
            Die "MSVC detected — use 'cmake -B build && cmake --build build --config Release' instead" 7
        } else {
            Die "neither mingw32-make nor nmake found; install one of them (or use cmake)" 7
        }
        if (-not (Test-Path "bin\liblitmoe.dll")) { Die "build did not produce bin\liblitmoe.dll" 1 }
        if (-not (Test-Path "bin\k3.exe"))    { Die "build did not produce bin\k3.exe" 1 }
        Note "build OK"
    } finally {
        Pop-Location
    }
}

function Download {
    Note "downloading Kimi K3 weights to $LITMOE_MODEL_DIR"
    Push-Location $REPO_ROOT
    try {
        if (-not (Test-Path "scripts\fetch-model.sh")) {
            Die "scripts\fetch-model.sh missing; use WSL or run download manually" 2
        }
        # The article's downloader is bash; on Windows use WSL.
        if (Get-Command wsl -ErrorAction SilentlyContinue) {
            New-Item -ItemType Directory -Force -Path $LITMOE_MODEL_DIR | Out-Null
            & wsl bash scripts/fetch-model.sh
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
            & docker run --rm -v "${LITMOE_MODEL_DIR}:/data:rw" -v "${REPO_ROOT}:/out:rw" kimi-k3-convert `
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

    Note "starting litMoE OpenAI server"
    Note "  model:    $modelPath"
    Note "  preset:   $LITMOE_PRESET"
    Note "  endpoint: http://${LITMOE_HOST}:${LITMOE_PORT}/v1"
    Note "  api key:  $(if ($LITMOE_API_KEY) { '(set)' } else { '(none — server is open)' })"

    if (($LITMOE_HOST -ne "127.0.0.1") -and (-not $LITMOE_API_KEY)) {
        Warn "binding to non-loopback without --api-key is unsafe."
        Start-Sleep -Seconds 3
    }

    Push-Location $REPO_ROOT
    try {
        $args = @($modelPath,
                  "--preset", $LITMOE_PRESET,
                  "--host", $LITMOE_HOST,
                  "--port", "$LITMOE_PORT",
                  "--max-tokens", "$LITMOE_MAX_TOKENS")
        if ($LITMOE_API_KEY) { $args += @("--api-key", $LITMOE_API_KEY) }

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

Serve $LITMOE_MODEL_DIR