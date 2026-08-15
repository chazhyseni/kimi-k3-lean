<#
bootstrap.ps1 - one-command install on Windows: clone, build, start server.

Usage:
  Invoke-Expression (Invoke-WebRequest -UseBasicParsing https://raw.githubusercontent.com/chazhyseni/litMoE/main/bootstrap.ps1).Content

What it does:
  1. Clones to $LITMOE_DIR (default $HOME\.litMoE)
  2. Builds liblitmoe + k3 via CMake (~30 sec)
  3. Starts the OpenAI server on http://127.0.0.1:8080
  4. Waits for /v1/models to return 200
  5. Prints the URL, token, and next steps

The server starts with or without model weights. Without weights,
/v1/models works and /v1/chat/completions returns a clear engine_error.
Download weights with:  litMoE fetch (requires WSL or git-bash)

Env vars:
  LITMOE_DIR       install location     (default $HOME\.litMoE)
  LITMOE_PORT      server port          (default 8080)
  LITMOE_HOST      bind address         (default 127.0.0.1)
  LITMOE_API_KEY   bearer token         (default random 32 hex)
  LITMOE_SKIP_DL=1 skip model detection
#>

$ErrorActionPreference = "Stop"

$LITMOE_REPO = "https://github.com/chazhyseni/litMoE.git"
$LITMOE_DIR = if ($env:LITMOE_DIR) { $env:LITMOE_DIR } else { Join-Path $HOME ".litMoE" }
$LITMOE_PORT = if ($env:LITMOE_PORT) { [int]$env:LITMOE_PORT } else { 8080 }
$LITMOE_HOST = if ($env:LITMOE_HOST) { $env:LITMOE_HOST } else { "127.0.0.1" }
$LITMOE_API_KEY = if ($env:LITMOE_API_KEY) { $env:LITMOE_API_KEY } else { -join ((1..32) | ForEach-Object { '{0:x}' -f (Get-Random -Max 16) }) }
$LITMOE_MODEL_DIR = if ($env:LITMOE_MODEL_DIR) { $env:LITMOE_MODEL_DIR } else { Join-Path $LITMOE_DIR "checkpoints\k3" }
$LITMOE_MODEL_NAME = if ($env:LITMOE_MODEL_NAME) { $env:LITMOE_MODEL_NAME } else { "kimi-k3" }

function Say($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok($m)  { Write-Host "  + $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  ! $m" -ForegroundColor Yellow }
function Die($m) { Write-Host "==> ERROR: $m" -ForegroundColor Red; exit 1 }

# --- uninstall ---
if ($env:LITMOE_UNINSTALL -eq "1") {
    Say "uninstalling"
    if (Test-Path "$LITMOE_DIR\server.pid") {
        try { Stop-Process -Id (Get-Content "$LITMOE_DIR\server.pid") -Force } catch {}
    }
    if (Get-Command hermes -ErrorAction SilentlyContinue) {
        & hermes config unset model.base_url model.api_key model.default 2>$null
    }
    Ok "done. repo at $LITMOE_DIR kept"
    exit 0
}

# --- prereqs ---
foreach ($c in @("git","python","cmake")) {
    if (-not (Get-Command $c -ErrorAction SilentlyContinue)) { Die "missing: $c" }
}
$pyv = & python -c "import sys;print('%d.%d'%sys.version_info[:2])"
if ($pyv -notmatch '^3\.(1[1-9]|[2-9])') { Die "Python $pyv; need 3.11+" }
Ok "python $pyv"

# --- 1. clone ---
if (Test-Path "$LITMOE_DIR\.git") {
    Say "updating $LITMOE_DIR"
    & git -C $LITMOE_DIR fetch --depth 1 origin main 2>$null
    & git -C $LITMOE_DIR reset --hard origin/main 2>$null
} elseif (Test-Path $LITMOE_DIR) {
    Die "$LITMOE_DIR exists but is not a git repo"
} else {
    Say "cloning to $LITMOE_DIR"
    & git clone --depth 1 $LITMOE_REPO $LITMOE_DIR
}
Set-Location $LITMOE_DIR

# --- 2. build via CMake ---
if (-not (Test-Path "build\bin\Release\k3.exe") -and -not (Test-Path "bin\liblitmoe.dll")) {
    Say "building"
    & cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { Die "cmake configure failed" }
    & cmake --build build --config Release --parallel
    if ($LASTEXITCODE -ne 0) { Die "build failed" }
    Ok "built"
} else {
    Ok "build cached"
}

# --- 3. start server ---
Say "starting server"

# write server.env
@"
LITMOE_HOST=$LITMOE_HOST
LITMOE_PORT=$LITMOE_PORT
LITMOE_API_KEY=$LITMOE_API_KEY
LITMOE_MODEL_DIR=$LITMOE_MODEL_DIR
LITMOE_MODEL_NAME=$LITMOE_MODEL_NAME
"@ | Out-File -FilePath "$LITMOE_DIR\server.env" -Encoding ascii

# stop prior server
if (Test-Path "$LITMOE_DIR\server.pid") {
    try { Stop-Process -Id (Get-Content "$LITMOE_DIR\server.pid") -Force } catch {}
}

$env:PATH = "$LITMOE_DIR\build\bin\Release;$LITMOE_DIR\bin;$env:PATH"
$argList = @($LITMOE_MODEL_DIR, "--host", $LITMOE_HOST, "--port", $LITMOE_PORT, "--api-key", $LITMOE_API_KEY, "--model-id", $LITMOE_MODEL_NAME)
$proc = Start-Process -FilePath "python" -ArgumentList @("-u", "serve\__main__.py") + $argList `
    -RedirectStandardOutput "$LITMOE_DIR\server.log" -RedirectStandardError "$LITMOE_DIR\server.log" `
    -WindowStyle Hidden -PassThru
Set-Content -Path "$LITMOE_DIR\server.pid" -Value $proc.Id
Ok "server PID: $($proc.Id)"

# --- 4. wait for /v1/models ---
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Headers @{ Authorization = "Bearer $LITMOE_API_KEY" } `
            -Uri "http://$LITMOE_HOST`:$LITMOE_PORT/v1/models" -TimeoutSec 1
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if ($ready) { Ok "server up: http://$LITMOE_HOST`:$LITMOE_PORT" }
else { Warn "not ready after 30s; check $LITMOE_DIR\server.log" }

# --- 5. Hermes ---
if ($env:LITMOE_NO_HERMES -ne "1" -and (Get-Command hermes -ErrorAction SilentlyContinue)) {
    Say "registering with Hermes"
    & hermes config set model.base_url "http://$LITMOE_HOST`:$LITMOE_PORT/v1" 2>$null
    & hermes config set model.api_key $LITMOE_API_KEY 2>$null
    & hermes config set model.default $LITMOE_MODEL_NAME 2>$null
    Ok "Hermes configured"
}

# --- 6. handoff ---
Write-Host ""
Write-Host "done."
Write-Host ""
Write-Host "  Server:   http://$LITMOE_HOST`:$LITMOE_PORT"
Write-Host "  Token:    $LITMOE_API_KEY"
Write-Host "  Model:    $LITMOE_MODEL_NAME"
Write-Host "  Log:      $LITMOE_DIR\server.log"
Write-Host ""
Write-Host "Daily use:"
Write-Host "  litMoE chat -m `"hello`"     # chat (needs weights)"
Write-Host "  litMoE stop"
Write-Host ""
exit 0