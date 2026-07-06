#Requires -Version 5.1
<#
    deskbot installer (Windows / PowerShell)

    - Verifies Python 3.11+
    - Installs Ollama if missing (winget), starts the server
    - Detects RAM, pulls the matching tiered models
    - Installs the deskbot package (editable) and puts `deskbot` on PATH
    - Runs `deskbot doctor` to verify everything

    Run from the project root:
        powershell -ExecutionPolicy Bypass -File .\install.ps1
#>

[CmdletBinding()]
param(
    [switch]$SkipModelPull
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    WARN: $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
Write-Step "Checking Python"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python 3.11+ not found on PATH. Install it from https://python.org and re-run this script."
}
$verOutput = & python --version 2>&1
if ($verOutput -match "Python (\d+)\.(\d+)") {
    $maj = [int]$Matches[1]; $min = [int]$Matches[2]
    if ($maj -lt 3 -or ($maj -eq 3 -and $min -lt 11)) {
        throw "Python 3.11+ required, found $verOutput"
    }
}
Write-Ok "$verOutput"

# ---------------------------------------------------------------------------
Write-Step "Checking Ollama"
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    Write-Warn "Ollama not found. Attempting install via winget..."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "winget not available. Install Ollama manually from https://ollama.com/download and re-run this script."
    }
    winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
    Write-Ok "Ollama installed. You may need to open a new terminal for PATH to refresh."
} else {
    Write-Ok "Ollama already installed"
}

Write-Step "Making sure the Ollama server is running"
$serverUp = $false
try {
    Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 | Out-Null
    $serverUp = $true
} catch { $serverUp = $false }

if (-not $serverUp) {
    Write-Warn "Ollama server not responding — starting it in the background"
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}
Write-Ok "Ollama server reachable"

# ---------------------------------------------------------------------------
Write-Step "Detecting RAM and picking a model tier"
$ramGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
if ($ramGb -ge 28) { $tier = "32gb" }
elseif ($ramGb -ge 14) { $tier = "16gb" }
else { $tier = "8gb" }
Write-Ok "Detected $ramGb GB RAM -> tier '$tier'"

# Model names are the single source of truth in deskbot\defaults\config.yaml.
# Parse them here (best-effort, no external YAML dependency needed yet) so we
# can pull the right models before the Python package itself is installed.
$configText = Get-Content (Join-Path $root "deskbot\defaults\config.yaml") -Raw
function Get-TierModel($tierName, $kind) {
    if ($configText -match "(?s)$tierName`:\s*\r?\n\s*text:\s*""([^""]+)""\s*\r?\n\s*vision:\s*""([^""]+)""") {
        if ($kind -eq "text") { return $Matches[1] } else { return $Matches[2] }
    }
    return $null
}
$textModel = Get-TierModel $tier "text"
$visionModel = Get-TierModel $tier "vision"
Write-Ok "text model: $textModel | vision model: $visionModel"

if (-not $SkipModelPull) {
    Write-Step "Pulling models (this can take a while on first run)"
    & ollama pull $textModel
    & ollama pull $visionModel
    Write-Ok "Models pulled"
} else {
    Write-Warn "Skipping model pull (-SkipModelPull)"
}

# ---------------------------------------------------------------------------
Write-Step "Installing the deskbot Python package"
Push-Location $root
try {
    & python -m pip install --upgrade pip | Out-Null
    & python -m pip install --user -e ".[dev]"
} finally {
    Pop-Location
}
Write-Ok "deskbot package installed (editable, --user)"

# ---------------------------------------------------------------------------
Write-Step "Checking the browser layer (Chrome/Edge)"
# The browser tools launch your ALREADY-INSTALLED Chrome/Edge via Playwright's
# channel= option, so no browser download is normally needed. We only fall
# back to Playwright's bundled Chromium if neither is found.
$hasEdge = (Get-Command msedge -ErrorAction SilentlyContinue) -or (Test-Path "$env:ProgramFiles (x86)\Microsoft\Edge\Application\msedge.exe") -or (Test-Path "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe")
$hasChrome = (Get-Command chrome -ErrorAction SilentlyContinue) -or (Test-Path "$env:ProgramFiles\Google\Chrome\Application\chrome.exe") -or (Test-Path "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe")
if (-not ($hasEdge -or $hasChrome)) {
    Write-Warn "Neither Edge nor Chrome found — installing Playwright's bundled Chromium as a fallback"
    & python -m playwright install chromium
} else {
    Write-Ok "Found an installed browser for the browser layer (Edge: $hasEdge, Chrome: $hasChrome)"
}

# ---------------------------------------------------------------------------
Write-Step "Ensuring 'deskbot' is on PATH"
$userScripts = & python -c "import sysconfig; print(sysconfig.get_path('scripts', 'nt_user'))"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$userScripts*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$userScripts", "User")
    Write-Ok "Added $userScripts to your User PATH (open a new terminal to pick it up)"
} else {
    Write-Ok "$userScripts already on PATH"
}
$env:Path = "$env:Path;$userScripts"

# ---------------------------------------------------------------------------
Write-Step "Running deskbot doctor"
& deskbot doctor

Write-Host ""
Write-Host "Install complete. Open a NEW terminal window, then try:" -ForegroundColor Cyan
Write-Host "    deskbot chat -p friend"
Write-Host "    deskbot persona create"
