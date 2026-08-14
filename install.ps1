<#
install.ps1 — install kimi-k3-lean to a prefix on Windows.
#
# Usage:
#   .\install.ps1                                       # installs to C:\Program Files\kimi-k3-lean (requires admin)
#   .\install.ps1 -Prefix C:\Users\you\kimi-k3-lean    # user-level install, no admin
#   .\install.ps1 -BuildType Release -Jobs 8
#
# What it installs:
#   bin\k3.exe                       CLI binary
#   lib\k3.dll                       shared library (Python uses via ctypes)
#   lib\k3_static.lib                static library for embedding
#   include\libk3\libk3.h            public C API
#
# Tested:
#   - NOT tested on this host (this host is Linux). The script follows the
#     same logic as install.sh and should work; it has not been exercised.
#
#   - PowerShell 5.1+ is required (ships with Windows 10/11 and Server 2016+).
#
[CmdletBinding()]
param(
    [string]$Prefix = "C:\Program Files\kimi-k3-lean",
    [ValidateSet("Release", "Debug", "RelWithDebInfo", "MinSizeRel")]
    [string]$BuildType = "Release",
    [int]$Jobs = [Environment]::ProcessorCount
)

$ErrorActionPreference = "Stop"

# --------------------------------------------------------------- helpers
function Test-Command($name) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    return $null -ne $cmd
}

function Require-Command($name) {
    if (-not (Test-Command $name)) {
        Write-Host "ERROR: $name not found. Install it and retry." -ForegroundColor Red
        exit 1
    }
}

function Test-MSVC() {
    # MSVC or MinGW both fine. We check for cl.exe or gcc.
    if (Test-Command "cl.exe") { return "msvc" }
    if (Test-Command "gcc.exe") { return "mingw" }
    if (Test-Command "clang.exe") { return "clang" }
    return $null
}

# --------------------------------------------------------------- check toolchain
Write-Host "==> kimi-k3-lean installer (Windows)"
Write-Host "    prefix:     $Prefix"
Write-Host "    build type: $BuildType"
Write-Host "    jobs:       $Jobs"

$compiler = Test-MSVC
if (-not $compiler) {
    Write-Host "ERROR: no C compiler found. Install Visual Studio (Build Tools), MinGW, or clang." -ForegroundColor Red
    exit 1
}
Write-Host "    compiler:   $compiler"

Require-Command "cmake"

# --------------------------------------------------------------- find MSVC environment if needed
if ($compiler -eq "msvc") {
    # When running from a plain PowerShell, MSVC isn't on PATH unless we're
    # in a Developer Command Prompt. Detect vcvars and run it.
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vsPath) {
            $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $vcvars) {
                Write-Host "==> Sourcing vcvars64.bat from $vsPath"
                # Capture vcvars environment by running it in cmd and exporting.
                $vcvarsOutput = & cmd.exe /c "`"$vcvars`" && set" 2>&1
                foreach ($line in $vcvarsOutput) {
                    if ($line -match "^(.+?)=(.*)$") {
                        $name = $matches[1]
                        $value = $matches[2]
                        Set-Item -Path "Env:$name" -Value $value -ErrorAction SilentlyContinue
                    }
                }
            }
        }
    }
}

# --------------------------------------------------------------- build
Write-Host "==> Building..."
$SCRIPT_DIR = $PSScriptRoot
Push-Location $SCRIPT_DIR
try {
    # Clean build dir.
    if (Test-Path build) { Remove-Item -Recurse -Force build }

    # Configure.
    & cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=$BuildType
    if ($LASTEXITCODE -ne 0) {
        Write-Host "==> cmake configure failed, falling back to NMake Makefiles" -ForegroundColor Yellow
        Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
        & cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=$BuildType
        if ($LASTEXITCODE -ne 0) { exit 1 }
    }

    # Build.
    & cmake --build build --config $BuildType --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { exit 1 }
} finally {
    Pop-Location
}

# --------------------------------------------------------------- install
Write-Host "==> Installing to $Prefix ..."
Push-Location $SCRIPT_DIR
try {
    & cmake --install build --prefix "$Prefix"
    if ($LASTEXITCODE -ne 0) { exit 1 }
} finally {
    Pop-Location
}

# --------------------------------------------------------------- verify
Write-Host "==> Verifying installation..."
$installedBin  = Join-Path $Prefix "bin\k3.exe"
$installedLib  = Join-Path $Prefix "lib\k3.dll"
$installedHdr  = Join-Path $Prefix "include\libk3\libk3.h"

if (-not (Test-Path $installedBin)) { Write-Host "ERROR: $installedBin missing" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $installedLib)) { Write-Host "ERROR: $installedLib missing" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $installedHdr)) { Write-Host "ERROR: $installedHdr missing" -ForegroundColor Red; exit 1 }

# Smoke-test.
try { & $installedBin --help | Out-Null } catch { Write-Host "WARNING: $installedBin --help returned non-zero" -ForegroundColor Yellow }

Write-Host ""
Write-Host "==> Installation complete."
Write-Host ""
Write-Host "Installed files:"
Write-Host "  $installedBin"
Write-Host "  $installedLib"
Write-Host "  $installedHdr"
Write-Host ""
Write-Host "To start the OpenAI server (after fetching a model):"
Write-Host ""
Write-Host "  python serve\__main__.py C:\path\to\checkpoint --host 127.0.0.1 --port 8080"
Write-Host ""
Write-Host "See COMBINED.md and INSTALL.md for next steps."