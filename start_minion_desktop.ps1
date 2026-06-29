$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = 11435
$url = "http://127.0.0.1:$port"
$logDir = Join-Path $projectDir "logs"
$profileDir = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "MinionChatLocal\EdgeProfile"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Test-MinionReady {
    try {
        $resp = Invoke-WebRequest -Uri "$url/api/health" -UseBasicParsing -TimeoutSec 2
        return $resp.StatusCode -eq 200 -and $resp.Content -match '"ok"'
    } catch {
        return $false
    }
}

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

if (-not (Test-MinionReady)) {
    $stdoutLog = Join-Path $logDir "server-out.log"
    $stderrLog = Join-Path $logDir "server-err.log"
    Start-Process `
        -FilePath "python" `
        -ArgumentList @("server.py", "--port", "$port") `
        -WorkingDirectory $projectDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog

    $ready = $false
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-MinionReady) {
            $ready = $true
            break
        }
    }

    if (-not $ready) {
        throw "Server Minion chua san sang tai $url. Xem log trong $logDir"
    }
}

$edgePaths = @(
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
    "$env:LOCALAPPDATA\Microsoft\Edge\Application\msedge.exe"
)
$edgePath = $edgePaths | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if ($edgePath) {
    if (-not (Test-Path $profileDir)) {
        New-Item -ItemType Directory -Path $profileDir | Out-Null
    }
    Start-Process -FilePath $edgePath -ArgumentList @(
        "--app=$url",
        "--window-size=1220,860",
        "--user-data-dir=$profileDir"
    )
} else {
    Start-Process $url
}
