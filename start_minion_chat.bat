@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0start_minion_desktop.ps1" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_minion_desktop.ps1"
  endlocal
  exit /b %ERRORLEVEL%
)

rem Chạy server chat local rồi mở giao diện trong trình duyệt.
set "PORT=11435"
set "URL=http://127.0.0.1:%PORT%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='SilentlyContinue';" ^
  "try { $r = Invoke-WebRequest -Uri '%URL%/api/health' -UseBasicParsing -TimeoutSec 2; if ($r.Content -match '\"ok\"') { exit 0 } } catch { exit 1 }; exit 1"

if errorlevel 1 (
  if not exist "logs" mkdir "logs"
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; Start-Process -FilePath 'python' -ArgumentList 'server.py --port %PORT%' -WorkingDirectory '%~dp0' -WindowStyle Hidden -RedirectStandardOutput '%~dp0logs\minion-chat-server.log' -RedirectStandardError '%~dp0logs\minion-chat-server.err.log'"
  timeout /t 4 /nobreak > nul
)

start "" "%URL%"
endlocal
