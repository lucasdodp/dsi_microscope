@echo off
REM ---------------------------------------------------------------------------
REM  Launch the DSI Image Lab (two-channel post-processing workbench) from the
REM  project's .venv. Double-click, or use the desktop shortcut created by
REM  tools\create_image_lab_shortcut.ps1.
REM
REM  Uses pythonw.exe so no console window stays open. If the app fails to
REM  start, run launch_image_lab_debug.bat instead to see the error.
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
start "" "%PYW%" "%ROOT%tools\image_lab.py"
endlocal
