# WrenLift installer for Windows.
#
#   irm https://wrenlift.com/install.ps1 | iex
#
# Fetches the `wlift`, `hatch` and `wlift-lsp` binaries from the
# latest GitHub Release (or the tag in WLIFT_VERSION), verifies the
# SHA256, and drops them in INSTALL_DIR (default
# `%LOCALAPPDATA%\Programs\wlift\bin`), adding that directory to the
# user's PATH when it is not there yet.
#
# Environment knobs
#   WLIFT_VERSION  - pin a tag instead of picking latest. `v0.1.0`.
#   INSTALL_DIR    - where to drop the binaries.
#   WLIFT_REPO     - override the GitHub slug. Default wrenlift/WrenLift.
#
# Supported platforms: Windows x86_64. The binaries need the
# Microsoft Visual C++ Redistributable (2015+).

$ErrorActionPreference = "Stop"

$Repo = if ($env:WLIFT_REPO) { $env:WLIFT_REPO } else { "wrenlift/WrenLift" }
$Version = $env:WLIFT_VERSION
$InstallDir = if ($env:INSTALL_DIR) { $env:INSTALL_DIR } else {
    Join-Path $env:LOCALAPPDATA "Programs\wlift\bin"
}
$Triple = "x86_64-pc-windows-msvc"

function Say($msg) { Write-Host "==> $msg" }
function Die($msg) { Write-Error "error: $msg"; exit 1 }

if (-not [Environment]::Is64BitOperatingSystem) {
    Die "WrenLift ships 64-bit Windows binaries only"
}
$arch = $env:PROCESSOR_ARCHITECTURE
if ($arch -ne "AMD64") {
    Die "unsupported architecture '$arch': WrenLift ships x86_64 Windows binaries (ARM64 can run them under emulation on Windows 11)"
}

# -- Resolve the tag -------------------------------------------

if (-not $Version) {
    Say "Resolving latest release..."
    $latest = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" `
        -Headers @{ "User-Agent" = "wlift-install" }
    $Version = $latest.tag_name
    if (-not $Version) { Die "couldn't resolve the latest release tag" }
}
Say "Target:   $Version ($Triple)"
Say "Dest:     $InstallDir"

# -- Download and verify ---------------------------------------

$archive = "wlift-$Version-$Triple.zip"
$archiveUrl = "https://github.com/$Repo/releases/download/$Version/$archive"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("wlift-install-" + [System.IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Say "Downloading $archive..."
    try {
        Invoke-WebRequest -Uri $archiveUrl -OutFile (Join-Path $tmp $archive) -UseBasicParsing
        Invoke-WebRequest -Uri "$archiveUrl.sha256" -OutFile (Join-Path $tmp "$archive.sha256") -UseBasicParsing
    } catch {
        Die "download failed: $archiveUrl`n  - Is the tag '$Version' a real release? See https://github.com/$Repo/releases`n  - Is your network or proxy blocking github.com?"
    }

    Say "Verifying SHA256..."
    $expected = ((Get-Content (Join-Path $tmp "$archive.sha256") -Raw) -split '\s+')[0].ToLower()
    $actual = (Get-FileHash (Join-Path $tmp $archive) -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $expected) {
        Die "checksum mismatch (got $actual, expected $expected)"
    }

    Say "Extracting..."
    Expand-Archive -Path (Join-Path $tmp $archive) -DestinationPath $tmp -Force
    $staged = Join-Path $tmp "wlift-$Version-$Triple"
    foreach ($exe in "wlift.exe", "hatch.exe") {
        if (-not (Test-Path (Join-Path $staged $exe))) {
            Die "archive didn't contain $exe - did the release format change?"
        }
    }

    # -- Install ----------------------------------------------------

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    foreach ($exe in "wlift.exe", "hatch.exe", "wlift-lsp.exe") {
        $src = Join-Path $staged $exe
        if (Test-Path $src) {
            Move-Item -Force $src (Join-Path $InstallDir $exe)
        }
    }
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Installed $Version to $InstallDir" -ForegroundColor Green

# -- PATH -------------------------------------------------------

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$onPath = ($userPath -split ';') -contains $InstallDir
if (-not $onPath) {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$InstallDir", "User")
    $env:Path = "$env:Path;$InstallDir"
    Write-Host ""
    Write-Host "Added $InstallDir to your user PATH. Open a new terminal for it to take effect."
}
Write-Host ""
Write-Host "Try:  wlift --version"
