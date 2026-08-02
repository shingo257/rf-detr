@echo off
setlocal
REM Run from cmd.exe: scripts\run_demo_video.cmd [--max-frames 30] ...
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_demo_video.ps1" %*
exit /b %ERRORLEVEL%
