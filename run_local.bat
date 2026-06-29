@echo off
REM Chay Minion MVP tren Windows. KHONG can torch/fastapi.
REM Yeu cau: Python 3.10+ va mot LLM endpoint (Ollama / LM Studio).

cd /d "%~dp0"

if not exist .env (
  echo - Tao .env tu .env.example ^(hay chinh model/provider neu can^)
  copy .env.example .env >nul
)

set CMD=%1
if "%CMD%"=="" set CMD=serve
shift

if "%CMD%"=="serve" (
  python minion.py serve %1 %2 %3 %4
) else if "%CMD%"=="chat" (
  python minion.py chat %1 %2 %3 %4
) else if "%CMD%"=="agent" (
  python minion.py agent %1 %2 %3 %4 %5 %6
) else (
  python minion.py %CMD% %1 %2 %3 %4
)
