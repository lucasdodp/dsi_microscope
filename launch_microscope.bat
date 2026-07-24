@echo off
REM ---------------------------------------------------------------------------
REM  Launch the event-DSI microscope control GUI from the project's .venv.
REM  Double-click, or use the desktop shortcut created by
REM  tools\create_desktop_shortcut.ps1.
REM
REM  Uses pythonw.exe so no console window stays open. If the app fails to
REM  start, run launch_microscope_debug.bat instead to see the error.
REM ---------------------------------------------------------------------------
setlocal

REM Project root = the folder this script lives in.
set "ROOT=%~dp0"
set "PYW=%ROOT%.venv\Scripts\pythonw.exe"

if not exist "%PYW%" (
    echo Could not find %PYW%
    echo The .venv is missing. Recreate it, then re-run.
    pause
    exit /b 1
)

cd /d "%ROOT%"
start "" "%PYW%" "%ROOT%main.py"
endlocal
