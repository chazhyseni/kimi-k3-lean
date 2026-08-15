<#
bootstrap.ps1 - one-command install on Windows: clone, build, start server.

Usage:
  Invoke-Expression (Invoke-WebRequest -UseBasicParsing https://raw.githubusercontent.com/chazhyseni/kimi-k3-lean/main/bootstrap.ps1).Content

What it does:
  1. Clones to $K3_DIR (default $HOME\.kimi-k3-lean)
  2. Builds libk3 + k3 via CMake (~30 sec)
  3. Starts the OpenAI server on http://127.0.0.1:8080
  4. Waits for /v1/models to return 200
  5. Prints the URL, token, and next steps

The server starts with or without model weights. Without weights,
/v1/models works and /v1/chat/completions returns a clear engine_error.
Download weights with:  kimi-k3-lean fetch (requires WSL or git-bash)

Env vars:
  K3_DIR       install location     (default $HOME\.kimi-k3-lean)
  K3_PORT      server port          (default 8080)
  K3_HOST      bind address         (default 127.0.0.1)
  K3_API_KEY   bearer token         (default random 32 hex)
  K3_SKIP_DL=1 skip model detection
#>

$ErrorActionPreference = "Stop"

$K3_REPO = "https://github.com/chazhyseni/kimi-k3-lean.git"
$K3_DIR = if ($env:K3_DIR) { $env:K3_DIR } else { Join-Path $HOME ".kimi-k3-lean" }
$K3_PORT = if ($env:K3_PORT) { [int]$env:K3_PORT } else { 8080 }
$K3_HOST = if ($env:K3_HOST) { $env:K3_HOST } else { "127.0.0.1" }
$K3_API_KEY = if ($env:K3_API_KEY) { $env:K3_API_KEY } else { -join ((1..32) | ForEach-Object { '{0:x}' -f (Get-Random -Max 16) }) }
$K3_MODEL_DIR = if ($env:K3_MODEL_DIR) { $env:K3_MODEL_DIR } else { Join-Path $K3_DIR "checkpoints\k3" }
$K3_MODEL_NAME = if ($env:K3_MODEL_NAME) { $env:K3_MODEL_NAME } else { "kimi-k3" }

function Say($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok($m)  { Write-Host "  + $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  ! $m" -ForegroundColor Yellow }
function Die($m) { Write-Host "==> ERROR: $m" -ForegroundColor Red; exit 1 }

# --- uninstall ---
if ($env:K3_UNINSTALL -eq "1") {
    Say "uninstalling"
    if (Test-Path "$K3_DIR\server.pid") {
        try { Stop-Process -Id (Get-Content "$K3_DIR\server.pid") -Force } catch {}
    }
    if (Get-Command hermes -ErrorAction SilentlyContinue) {
        & hermes config unset model.base_url model.api_key model.default 2>$null
    }
    Ok "done. repo at $K3_DIR kept"
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
if (Test-Path "$K3_DIR\.git") {
    Say "updating $K3_DIR"
    & git -C $K3_DIR fetch --depth 1 origin main 2>$null
    & git -C $K3_DIR reset --hard origin/main 2>$null
} elseif (Test-Path $K3_DIR) {
    Die "$K3_DIR exists but is not a git repo"
} else {
    Say "cloning to $K3_DIR"
    & git clone --depth 1 $K3_REPO $K3_DIR
}
Set-Location $K3_DIR

# --- 2. build via CMake ---
if (-not (Test-Path "build\bin\Release\k3.exe") -and -not (Test-Path "bin\libk3.dll")) {
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
K3_HOST=$K3_HOST
K3_PORT=$K3_PORT
K3_API_KEY=$K3_API_KEY
K3_MODEL_DIR=$K3_MODEL_DIR
K3_MODEL_NAME=$K3_MODEL_NAME
"@ | Out-File -FilePath "$K3_DIR\server.env" -Encoding ascii

# stop prior server
if (Test-Path "$K3_DIR\server.pid") {
    try { Stop-Process -Id (Get-Content "$K3_DIR\server.pid") -Force } catch {}
}

$env:PATH = "$K3_DIR\build\bin\Release;$K3_DIR\bin;$env:PATH"
$argList = @($K3_MODEL_DIR, "--host", $K3_HOST, "--port", $K3_PORT, "--api-key", $K3_API_KEY, "--model-id", $K3_MODEL_NAME)
$proc = Start-Process -FilePath "python" -ArgumentList @("-u", "serve\__main__.py") + $argList `
    -RedirectStandardOutput "$K3_DIR\server.log" -RedirectStandardError "$K3_DIR\server.log" `
    -WindowStyle Hidden -PassThru
Set-Content -Path "$K3_DIR\server.pid" -Value $proc.Id
Ok "server PID: $($proc.Id)"

# --- 4. wait for /v1/models ---
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Headers @{ Authorization = "Bearer $K3_API_KEY" } `
            -Uri "http://$K3_HOST`:$K3_PORT/v1/models" -TimeoutSec 1
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if ($ready) { Ok "server up: http://$K3_HOST`:$K3_PORT" }
else { Warn "not ready after 30s; check $K3_DIR\server.log" }

# --- 5. Hermes ---
if ($env:K3_NO_HERMES -ne "1" -and (Get-Command hermes -ErrorAction SilentlyContinue)) {
    Say "registering with Hermes"
    & hermes config set model.base_url "http://$K3_HOST`:$K3_PORT/v1" 2>$null
    & hermes config set model.api_key $K3_API_KEY 2>$null
    & hermes config set model.default $K3_MODEL_NAME 2>$null
    Ok "Hermes configured"
}

# --- 6. handoff ---
Write-Host ""
Write-Host "done."
Write-Host ""
Write-Host "  Server:   http://$K3_HOST`:$K3_PORT"
Write-Host "  Token:    $K3_API_KEY"
Write-Host "  Model:    $K3_MODEL_NAME"
Write-Host "  Log:      $K3_DIR\server.log"
Write-Host ""
Write-Host "Daily use:"
Write-Host "  kimi-k3-lean chat -m `"hello`"     # chat (needs weights)"
Write-Host "  kimi-k3-lean stop"
Write-Host ""
exit 0