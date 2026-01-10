@echo off
setlocal EnableExtensions

REM Single entrypoint. This file should not contain complex logic.
REM It starts an interactive cmd session so Ctrl+C stops without a Y/N prompt.

set "PORT=39841"
set "MODEL_DIR=pretrained_models\CosyVoice2-0.5B"
set "LOG_FILE=cosyvoice_webui.log"

cd /d "%~dp0"
pushd "%~dp0cosyvoice_launcher"

where go >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Go is not installed or not on PATH.
  echo Install Go from go.dev and try again.
  echo.
  pause
  exit /b 1
)

REM Run via cmd /k so Ctrl+C stops without the batch Y/N prompt.
REM No command chaining here (it was causing cmd parse errors). We run from cosyvoice_launcher so go finds go.mod.
cmd /k go run . --port %PORT% --model-dir %MODEL_DIR% --log-file %LOG_FILE%
