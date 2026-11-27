@echo off
rem The fly's daily autopilot, for a Windows scheduled task (run it once per trading day after 16:15 New York time).
rem Uses the project environment next to this repository: ..\.venv-lab (lab PC) or ..\.venv (home). Extra arguments are passed on,
rem e.g.  run_autopilot.cmd --preview-only
cd /d "%~dp0"
set "PY=%~dp0..\.venv-lab\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo no Python environment found next to the repository
  exit /b 1
)
"%PY%" -m broker.autopilot %*
