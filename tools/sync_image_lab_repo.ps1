<#
    Push the Image Lab's source from THIS (main) repo into the standalone
    dsi-image-lab distribution repo, then commit and push it.

    The main repo is the single source of truth: the Image Lab is developed and
    tested here (ui\main_window.py's "Open Image Lab…" button launches it, and
    it shares config.py + core\ with the acquisition app). The standalone repo
    is a copy that lets the Lab be installed on a machine with no cameras and
    none of the vendor SDKs.

    Workflow:
        1. Edit the Image Lab in this repo; commit as usual.
        2. Run this script. It copies the shared files across, commits them in
           the standalone repo (noting which main-repo commit they came from),
           and pushes to GitHub.

    Usage (from the main repo root):
        powershell -ExecutionPolicy Bypass -File tools\sync_image_lab_repo.ps1

    Options:
        -DestRepo <path>   Location of the standalone repo checkout.
                           Default: C:\lucas_python_scripts\dsi-image-lab
        -NoPush            Commit in the standalone repo but don't push, so you
                           can review the commit and push it yourself.
#>

param(
    [string]$DestRepo = "C:\lucas_python_scripts\dsi-image-lab",
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"

# Main repo root = parent of this script's tools\ folder.
$mainRoot = Split-Path -Parent $PSScriptRoot

# The files that must stay byte-identical in both repos. Deliberately NOT here:
# README.md, requirements.txt, .gitignore — the standalone repo keeps its own
# (slim requirements, its own install README), and syncing would clobber them.
$files = @(
    "config.py",
    "core\__init__.py",
    "core\image_processing.py",
    "tools\image_lab.py",
    "tools\image_ops.py",
    "tools\create_image_lab_shortcut.ps1",
    "tools\make_image_lab_icon.py",
    "launch_image_lab.bat",
    "launch_image_lab_debug.bat",
    "assets\image_lab.ico"
)

# --- validate the destination -------------------------------------------------
if (-not (Test-Path $DestRepo)) {
    throw "Standalone repo not found at $DestRepo. Clone it first, or pass -DestRepo <path>."
}
if (-not (Test-Path (Join-Path $DestRepo ".git"))) {
    throw "$DestRepo is not a git repository."
}

# --- copy the shared files ----------------------------------------------------
Write-Host "Syncing Image Lab -> $DestRepo"
foreach ($rel in $files) {
    $src = Join-Path $mainRoot $rel
    $dst = Join-Path $DestRepo $rel
    if (-not (Test-Path $src)) {
        throw "Source file missing in main repo: $src"
    }
    $dstDir = Split-Path -Parent $dst
    if (-not (Test-Path $dstDir)) {
        New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
    }
    Copy-Item -Path $src -Destination $dst -Force
    Write-Host "  copied $rel"
}

# --- what main-repo commit are we syncing from? -------------------------------
Push-Location $mainRoot
try {
    $mainHash    = (git rev-parse --short HEAD).Trim()
    $mainSubject = (git log -1 --pretty=%s).Trim()
} finally {
    Pop-Location
}

# --- commit + push in the standalone repo -------------------------------------
Push-Location $DestRepo
try {
    # Stage only the synced files, so a standalone-only edit (README etc.) that
    # happens to be uncommitted isn't swept into this commit.
    git add -- $files

    $pending = git status --porcelain -- $files
    if ([string]::IsNullOrWhiteSpace($pending)) {
        Write-Host ""
        Write-Host "Already up to date - the standalone repo matches main. Nothing to commit." -ForegroundColor Green
        return
    }

    $msg = "Sync Image Lab from main repo @ $mainHash`n`nmain: $mainSubject"
    # Keep the standalone repo's history authored consistently; no co-author line.
    git -c user.name="Lucas de Oliveira de Pietro" -c user.email="naoolhelucas@gmail.com" `
        commit -m $msg
    Write-Host ""
    Write-Host "Committed sync (from main @ $mainHash)." -ForegroundColor Green

    if ($NoPush) {
        Write-Host "Skipped push (-NoPush). Review it, then run 'git push' in $DestRepo."
    } else {
        git push
        Write-Host "Pushed to origin." -ForegroundColor Green
    }
} finally {
    Pop-Location
}
