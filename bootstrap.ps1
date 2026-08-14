<#
bootstrap.ps1 - install kimi-k3-lean and start the server on Windows.

Usage (the headline use case):
  Invoke-Expression (Invoke-WebRequest -UseBasicParsing https://raw.githubusercontent.com/chazhyseni/kimi-k3-lean/main/bootstrap.ps1).Content

What this does (mirrors bootstrap.sh exactly; both should stay in sync):

  1. Clones the repo to $K3_DIR (default: $HOME\.kimi-k3-lean)
  2. Builds libk3.dll + k3.exe (visual studio build tools required)
  3. If $K3_DIR\checkpoints\k3 exists, starts the server against it.
     Otherwise the server starts without a model and reports "engine
     open failed". Run scripts\setup-and-serve.ps1 --download-only
     to fetch the real K3 weights (~982 GB, ~4 hours).
  4. Writes $K3_DIR\server.env with the bearer token, so any other
     harness on the same machine can pick it up.
  5. WAITS for /v1/models to return 200 (with bearer auth), so the
     user knows the server actually came up before this script exits.
  6. Best-effort Hermes config: same model.base_url, api_key, and
     appends the model to providers.ollama-launch.models. If Hermes
     isn't installed, prints the commands for the user to run by hand.
  7. Prints the next steps: how to talk to it (curl), how to verify,
     how to download real K3, how to stop the server.

Env vars (optional, all-power uppercase form):
  K3_DIR         install location            (default: $HOME\.kimi-k3-lean)
  K3_PORT        server port                 (default: 8080)
  K3_HOST        bind address                (default: 127.0.0.1)
  K3_PRESET      memory preset               (default: auto)
  K3_MODEL_DIR   model checkpoint dir        (default: $K3_DIR\checkpoints\k3)
  K3_SKIP_DL=1   skip the K3 download
  K3_NO_HERMES=1 skip the Hermes edits
  K3_UNINSTALL=1 remove the server + roll back Hermes edits

NOTE on cross-platform coverage: bootstrap.sh (Bash) is fully tested
on Linux + macOS. bootstrap.ps1 (PowerShell) is provided for parity
but each new bootstrap.sh feature requires manual mirroring here
until we have a CI matrix. If you find a bug here, please file it.
#>

$ErrorActionPreference = "Stop"

# --------------- defaults
$K3_REPO_URL = if ($env:K3_REPO_URL) { $env:K3_REPO_URL } else { "https://github.com/chazhyseni/kimi-k3-lean.git" }
$K3_BRANCH   = if ($env:K3_BRANCH)   { $env:K3_BRANCH }   else { "main" }
$K3_DIR      = if ($env:K3_DIR)      { $env:K3_DIR }      else { Join-Path $HOME ".kimi-k3-lean" }
$K3_PORT     = if ($env:K3_PORT)     { $env:K3_PORT }     else { 8080 }
$K3_HOST     = if ($env:K3_HOST)     { $env:K3_HOST }     else { "127.0.0.1" }
$K3_PRESET   = if ($env:K3_PRESET)   { $env:K3_PRESET }   else { "auto" }
$K3_MODEL_DIR= if ($env:K3_MODEL_DIR){ $env:K3_MODEL_DIR} else { Join-Path $K3_DIR "checkpoints\k3" }
$K3_API_KEY  = if ($env:K3_API_KEY)  { $env:K3_API_KEY }  else { -join ((1..32) | ForEach-Object { '{0:x}' -f (Get-Random -Max 16) }) }
$K3_MODEL_NAME = if ($env:K3_MODEL_NAME) { $env:K3_MODEL_NAME } else { "kimi-k3" }
$K3_SKIP_DL  = if ($env:K3_SKIP_DL)  { $env:K3_SKIP_DL }  else { "0" }
$K3_NO_HERMES= if ($env:K3_NO_HERMES){ $env:K3_NO_HERMES}else { "0" }

function Say($m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  + $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  ! $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "==> ERROR: $m" -ForegroundColor Red; exit 1 }

# --------------- uninstall
if ($env:K3_UNINSTALL -eq "1") {
    Say "uninstalling kimi-k3-lean"
    if (Test-Path "$K3_DIR\server.pid") {
        $pid_old = Get-Content "$K3_DIR\server.pid"
        try { Stop-Process -Id $pid_old -Force -ErrorAction Stop; Ok "killed $pid_old" }
        catch { Warn "$pid_old not running" }
    }
    if (Get-Command hermes -ErrorAction SilentlyContinue) {
        & hermes config unset providers.ollama-launch.models 2>$null
        & hermes config unset model.base_url 2>$null
        & hermes config unset model.api_key 2>$null
        & hermes config unset model.default 2>$null
        Ok "rolled back Hermes config"
    }
    Remove-Item "$K3_DIR\server.pid","$K3_DIR\server.log","$K3_DIR\server.env" -ErrorAction SilentlyContinue
    Ok "done. repo lives at $K3_DIR (Remove-Item -Recurse it to wipe)"
    exit 0
}

# --------------- pre-flight
foreach ($c in @("git","python","cl")) {
    if (-not (Get-Command $c -ErrorAction SilentlyContinue)) { Die "missing: $c" }
}
$pyv = & python -c "import sys;print('%d.%d'%sys.version_info[:2])"
if ($pyv -notmatch '^3\.(1[1-9]|[2-9])') { Die "Python $pyv; need 3.11+" }
Ok "python $pyv"

# --------------- clone/update
if (Test-Path "$K3_DIR\.git") {
    Say "updating $K3_DIR"
    & git -C $K3_DIR fetch --depth 1 origin $K3_BRANCH 2>$null | Out-Null
    & git -C $K3_DIR reset --hard "origin/$K3_BRANCH" 2>$null | Out-Null
} elseif (Test-Path $K3_DIR) {
    Die "$K3_DIR exists but is not a git repo."
} else {
    Say "cloning to $K3_DIR"
    & git clone --depth 1 --branch $K3_BRANCH $K3_REPO_URL $K3_DIR
}
Set-Location $K3_DIR

# --------------- build
if (-not (Test-Path "bin\libk3.dll") -or ((Get-Item src).LastWriteTime -gt (Get-Item "bin\libk3.dll").LastWriteTime)) {
    Say "building"
    & cl /I include /I include\k3 /I include\libk3 /I third_party /I src\core /I src\io /I src\cache /I src\model /I src\tokenizer /I src\lib /LD src\lib\k3_api.c src\lib\k3_engine.c src\core\k3_ops.c src\io\k3_st.c src\io\k3_load.c src\io\k3_trunk.c src\cache\k3_cache.c src\model\k3_bind.c src\cli\k3_run.c /Fe:bin\k3.exe /Fo:x86_64-msvc\ /link /OUT:bin\libk3.dll
    if ($LASTEXITCODE -ne 0) { Die "build failed; install Visual Studio Build Tools first" }
} else {
    Ok "build cached"
}

# --------------- model
if ($K3_SKIP_DL -ne "1") {
    if (Test-Path $K3_MODEL_DIR) {
        Ok "model at $K3_MODEL_DIR"
    } else {
        Warn "no model at $K3_MODEL_DIR"
        Warn "download the real Kimi K3 (~982 GB, ~4 hours):"
        Warn "  `$env:K3_DIR='$K3_DIR'; & '$K3_DIR\scripts\setup-and-serve.ps1' --download-only"
        New-Item -ItemType Directory -Force -Path $K3_MODEL_DIR | Out-Null
    }
}

# --------------- start server in background
Say "launching kimi-k3-lean server (background)"
$env:PATH = "$K3_DIR\bin;$env:PATH"
$env:K3_API_KEY = $K3_API_KEY

# write the server.env for the user
@"
K3_HOST=$K3_HOST
K3_PORT=$K3_PORT
K3_API_KEY=$K3_API_KEY
K3_PRESET=$K3_PRESET
K3_MODEL_DIR=$K3_MODEL_DIR
K3_MODEL_NAME=$K3_MODEL_NAME
"@ | Out-File -FilePath "$K3_DIR\server.env" -Encoding ascii

# stop prior same-port server if our pid file points at it
if (Test-Path "$K3_DIR\server.pid") {
    $old = Get-Content "$K3_DIR\server.pid"
    try {
        $proc = Get-Process -Id $old -ErrorAction Stop
        if ($proc.Path -match "serve\\__main__\.py") {
            Warn "killing prior server PID $old"
            Stop-Process -Id $old -Force
            Start-Sleep -Seconds 1
        }
    } catch { }
}

$argList = @($K3_MODEL_DIR,"--host",$K3_HOST,"--port",$K3_PORT,"--preset",$K3_PRESET,"--api-key",$K3_API_KEY,"--model-id",$K3_MODEL_NAME)
$proc = Start-Process -FilePath "python" -ArgumentList @("-u","serve\__main__.py") + $argList `
    -RedirectStandardOutput "$K3_DIR\server.log" -RedirectStandardError "$K3_DIR\server.log" `
    -WindowStyle Hidden -PassThru
Set-Content -Path "$K3_DIR\server.pid" -Value $proc.Id
Ok "server PID: $($proc.Id)  (log: $K3_DIR\server.log)"

# --------------- wait for /v1/models (max 30s)
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Headers @{ Authorization = "Bearer $K3_API_KEY" } `
            -Uri "http://$K3_HOST`:$K3_PORT/v1/models" -TimeoutSec 1
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch {
        $code = $_.Exception.Response.StatusCode.value__
        if ($code -eq 401) { Warn "server up but token rejected"; break }
    }
    Start-Sleep -Seconds 1
}
if ($ready) { Ok "server up: http://$K3_HOST`:$K3_PORT  (model $K3_MODEL_NAME registered)" }
else { Warn "server didn't respond on http://$K3_HOST`:$K3_PORT within 30s; tail `$K3_DIR\server.log`" }

# --------------- Hermes
if ($K3_NO_HERMES -ne "1" -and (Get-Command hermes -ErrorAction SilentlyContinue)) {
    Say "registering $K3_MODEL_NAME with Hermes"
    & hermes config set model.base_url "http://$K3_HOST`:$K3_PORT/v1" | Out-Null
    & hermes config set model.api_key   $K3_API_KEY                       | Out-Null
    # providers.ollama-launch.models is a JSON array; config-set overwrites,
    # so read first then write back with our name appended
    $curJson = & hermes config get providers.ollama-launch.models 2>$null
    try {
        $cur = $curJson | ConvertFrom-Json -ErrorAction Stop
    } catch { $cur = @() }
    if ($cur -notcontains $K3_MODEL_NAME) { $cur += $K3_MODEL_NAME }
    & hermes config set providers.ollama-launch.models ($cur | ConvertTo-Json -Compress) | Out-Null
    & hermes config set model.default $K3_MODEL_NAME | Out-Null
    Ok "Hermes configured"
} elseif ($K3_NO_HERMES -ne "1") {
    Warn "hermes CLI not on PATH; skipping model registration"
}

# --------------- handoff
Write-Host ""
Write-Host "==> done" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Server:      http://$K3_HOST`:$K3_PORT"
Write-Host "  API key:     (in $K3_DIR\server.env)"
Write-Host "  Model:       $K3_MODEL_NAME"
Write-Host "  Tail log:    Get-Content '$K3_DIR\server.log' -Wait"
Write-Host "  Stop:        Stop-Process -Id (Get-Content '$K3_DIR\server.pid')"
Write-Host "  Remove:      `\$env:K3_UNINSTALL=1; Invoke-Expression (Invoke-WebRequest ...).Content"
Write-Host ""
exit 0
