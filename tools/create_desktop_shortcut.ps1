<#
    Create a Desktop shortcut that launches the event-DSI microscope GUI.

    The shortcut runs the project's own virtual-env pythonw.exe directly (no
    console window), with the repo as the working directory, so the app starts
    exactly as `python main.py` does. Re-run this any time to refresh the
    shortcut (e.g. after moving the repo).

    Usage (from the repo root):
        powershell -ExecutionPolicy Bypass -File tools\create_desktop_shortcut.ps1

    Optional: -ShortcutName "DSI Microscope"   (default)
#>

param(
    [string]$ShortcutName = "DSI Microscope"
)

$ErrorActionPreference = "Stop"

# Repo root = parent of this script's folder.
$root    = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
$mainpy  = Join-Path $root "main.py"
$icon    = Join-Path $root "assets\microscope.ico"

if (-not (Test-Path $pythonw)) {
    throw "pythonw.exe not found at $pythonw - is the .venv set up?"
}
if (-not (Test-Path $mainpy)) {
    throw "main.py not found at $mainpy - run this from the repo."
}

$desktop  = [Environment]::GetFolderPath("Desktop")
$lnkPath  = Join-Path $desktop "$ShortcutName.lnk"

$shell    = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($lnkPath)
$shortcut.TargetPath       = $pythonw
$shortcut.Arguments        = "`"$mainpy`""
$shortcut.WorkingDirectory = $root
$shortcut.Description       = "Institut Fresnel event-DSI microscope control"
if (Test-Path $icon) {
    $shortcut.IconLocation = "$icon,0"
}
$shortcut.Save()

Write-Host "Created shortcut: $lnkPath"
Write-Host "  Target : $pythonw"
Write-Host "  Args   : `"$mainpy`""
Write-Host "  WorkDir: $root"
