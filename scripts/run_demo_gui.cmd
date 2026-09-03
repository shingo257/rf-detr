@echo off
setlocal EnableExtensions

cd /d "%~dp0.."
set "ROOT=%CD%"
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "PYW=%ROOT%\.venv\Scripts\pythonw.exe"
set "LAUNCHER=%ROOT%\scripts\launch_gui.py"
set "LOG=%ROOT%\artifacts\gui_launch_error.log"

if not exist "%PY%" (
  echo [.venv not found] Run: uv sync --extra demo-gui
  pause
  exit /b 1
)

if not exist "%LAUNCHER%" (
  echo [launcher missing] %LAUNCHER%
  pause
  exit /b 1
)

rem Preflight import check (errors visible in this console).
"%PY%" -c "import rfdetr_demo.gui.app" 2>"%LOG%"
if errorlevel 1 (
  echo.
  echo GUI preflight failed. See:
  echo   %LOG%
  echo.
  type "%LOG%"
  pause
  exit /b 1
)

rem Launch GUI without a console window; errors go to MessageBox + log file.
if exist "%PYW%" (
  start "" /D "%ROOT%" "%PYW%" "%LAUNCHER%"
) else (
  start "" /D "%ROOT%" "%PY%" "%LAUNCHER%"
)

exit /b 0
