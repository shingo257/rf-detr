@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo [.venv not found] Run: uv sync
  exit /b 1
)
".venv\Scripts\python.exe" scripts\vast_cleanup_orphans.py %*
