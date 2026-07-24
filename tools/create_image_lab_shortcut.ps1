<#
    Create a Desktop shortcut that launches the DSI Image Lab.

    The shortcut runs the project's own virtual-env pythonw.exe directly (no
    console window), with the repo as the working directory, so the app starts
    exactly as `python tools\image_lab.py` does. Re-run this any time to refresh
    the shortcut (e.g. after moving the repo).

    Usage (from the repo root):
        powershell -ExecutionPolicy Bypass -File tools\create_image_lab_shortcut.ps1

    Optional: -ShortcutName "DSI Image Lab"   (default)
#>

param(
    [string]$ShortcutName = "DSI Image Lab"
)

$ErrorActionPreference = "Stop"

# Repo root = parent of this script's folder.
$root    = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
$labpy   = Join-Path $root "tools\image_lab.py"
$icon    = Join-Path $root "assets\image_lab.ico"

if (-not (Test-Path $pythonw)) {
    throw "pythonw.exe not found at $pythonw - is the .venv set up?"
}
if (-not (Test-Path $labpy)) {
    throw "image_lab.py not found at $labpy - run this from the repo."
}

$desktop  = [Environment]::GetFolderPath("Desktop")
$lnkPath  = Join-Path $desktop "$ShortcutName.lnk"

$shell    = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($lnkPath)
$shortcut.TargetPath       = $pythonw
$shortcut.Arguments        = "`"$labpy`""
$shortcut.WorkingDirectory = $root
$shortcut.Description       = "DSI Image Lab - two-channel post-processing workbench"
if (Test-Path $icon) {
    $shortcut.IconLocation = "$icon,0"
}
$shortcut.Save()

Write-Host "Created shortcut: $lnkPath"
Write-Host "  Target : $pythonw"
Write-Host "  Args   : `"$labpy`""
Write-Host "  WorkDir: $root"
