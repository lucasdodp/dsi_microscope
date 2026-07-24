@echo off
REM ---------------------------------------------------------------------------
REM  Same as launch_image_lab.bat, but runs python.exe (not pythonw.exe) so the
REM  console window stays open and prints any startup error / traceback.
REM  Use this when troubleshooting on the lab PC.
REM ---------------------------------------------------------------------------
setlocal

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Could not find %PY%
    echo The .venv is missing. Recreate it, then re-run.
    pause
    exit /b 1
)

cd /d "%ROOT%"
"%PY%" "%ROOT%tools\image_lab.py"
echo.
echo --- app exited (code %ERRORLEVEL%) ---
pause
endlocal
