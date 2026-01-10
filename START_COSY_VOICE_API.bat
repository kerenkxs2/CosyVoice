@echo off
setlocal EnableExtensions

REM Start the CosyVoice FastAPI server (API-only; no web UI)

set "PORT=5005"
set "MODEL_DIR=pretrained_models\CosyVoice2-0.5B"
set "PROMPT_WAV=asset\zero_shot_prompt.wav"
set "PROMPT_TEXT=希望你以后能够做的比我还好呦。"

cd /d "%~dp0"

set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" (
  set "PYEXE=python"
)

REM Start server (runs until you press Ctrl+C)
"%PYEXE%" "runtime\python\fastapi\server.py" --port %PORT% --model_dir %MODEL_DIR% --fp16 --default_prompt_wav "%PROMPT_WAV%" --default_prompt_text "%PROMPT_TEXT%" --voices_manifest "voice_presets\voices.json"
