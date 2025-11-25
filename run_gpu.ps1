# run_gpu.ps1
# Creates/activates a venv, installs dependencies, downloads model if missing,
# adds local CUDA runtime wheel bins to PATH (if present), then runs depth.py

$ErrorActionPreference = 'Stop'

$projDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $projDir

# Dry-run support: set environment variable DRY_RUN=1 to simulate actions without making changes
$isDry = $false
if ($env:DRY_RUN -eq '1') { $isDry = $true; Write-Host '*** DRY RUN MODE: No changes will be made (simulated) ***' -ForegroundColor Yellow }

# Locate Python
$pythonCandidates = @(
    'C:\Python312\python.exe',
    'C:\Program Files\Python312\python.exe'
)
$python = $null
foreach ($p in $pythonCandidates) { if (Test-Path $p) { $python = $p; break } }
if (-not $python) {
    try { $python = (Get-Command python -ErrorAction Stop).Source } catch {}
}
if (-not $python) { Write-Error 'Python executable not found. Install Python 3.10+ or set PATH.'; exit 1 }

# Create venv if missing
$venvPath = Join-Path $projDir 'venv'
if (-not (Test-Path $venvPath)) {
    if (-not $isDry) {
        Write-Host 'Creating virtual environment...' -ForegroundColor Cyan
        & $python -m venv $venvPath
    } else {
        Write-Host "DRY RUN: would create virtual environment at: $venvPath" -ForegroundColor Yellow
    }
}

# Activate venv (PowerShell)
$activate = Join-Path $venvPath 'Scripts\Activate.ps1'
if (-not (Test-Path $activate)) { Write-Error 'Activation script not found in venv.'; exit 1 }
. $activate

# Ensure we use the venv python executable for all subsequent operations
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
if (Test-Path $venvPython) {
    $python = $venvPython
    Write-Host "Using venv python: $python"
} else {
    Write-Warning "Venv python not found at $venvPython — continuing with detected python: $python"
}

Write-Host 'Upgrading pip/tools...' -ForegroundColor Cyan
# Helper: run a process and tail its output while showing a spinner
function Run-ProcessWithTail($exe, $argList, $title) {
    $outLog = [IO.Path]::GetTempFileName()
    $errLog = [IO.Path]::GetTempFileName()
    Write-Host "`n$title -> $exe $argList" -ForegroundColor Cyan
    $exitCode = 0
    try {
        $startInfo = @{ FilePath = $exe; ArgumentList = $argList; RedirectStandardOutput = $outLog; RedirectStandardError = $errLog; NoNewWindow = $true }
        $proc = Start-Process @startInfo -PassThru

        $spinner = @('|','/','-','\\')
        $idx = 0
        while (-not $proc.HasExited) {
            $ch = $spinner[$idx % $spinner.Length]
            Write-Host -NoNewline "`r[$ch] $title... "
            Start-Sleep -Milliseconds 300
            $idx++
        }
        Write-Host "`r[✓] $title completed." -ForegroundColor Green
        Write-Host "--- Last stdout ---"
        Get-Content $outLog -Tail 30 | ForEach-Object { Write-Host "  $_" }
        Write-Host "--- Last stderr ---"
        Get-Content $errLog -Tail 30 | ForEach-Object { Write-Host "  $_" }
        $exitCode = $proc.ExitCode
    } catch {
        Write-Warning "Start-Process with redirected output failed; falling back to synchronous execution. Error: $($_.Exception.Message)"
        # Fallback: run via cmd.exe and redirect output to the out log file (stderr merged)
        $cmd = "`"$exe`" $argList > `"$outLog`" 2>&1"
        & cmd.exe /c $cmd
        $exitCode = $LASTEXITCODE
        Write-Host "[✓] $title completed (fallback)." -ForegroundColor Green
        Get-Content $outLog -Tail 50 | ForEach-Object { Write-Host "  $_" }
    } finally {
        Remove-Item $outLog -ErrorAction SilentlyContinue
        Remove-Item $errLog -ErrorAction SilentlyContinue
    }
    return $exitCode
}

# Upgrade pip/tools using the active python
if (-not $isDry) {
    & $python -m pip install -U pip setuptools wheel
} else {
    Write-Host 'DRY RUN: would upgrade pip, setuptools, wheel' -ForegroundColor Yellow
}

# Ensure requirements.txt exists
$reqFile = Join-Path $projDir 'requirements.txt'
if (-not (Test-Path $reqFile)) {
    Write-Host 'Writing default requirements.txt' -ForegroundColor Cyan
    @'
onnxruntime-gpu
opencv-python
mediapipe
pillow
numpy
requests
'@ | Out-File -Encoding UTF8 $reqFile
}

Write-Host 'Installing Python packages (this can take several minutes)...' -ForegroundColor Cyan
if (-not $isDry) {
    # Use the venv's python to run pip so output streams reliably
    $rc = Run-ProcessWithTail $python "-m pip install -r `"$reqFile`"" 'pip install requirements'
    if ($rc -ne 0) {
        Write-Warning "pip install returned exit code $rc — attempting per-package installs"
        # Read requirements and try installing packages one by one to surface specific failures
        $reqs = Get-Content $reqFile | Where-Object { $_ -and -not $_.StartsWith('#') } | ForEach-Object { $_.Trim() }
        foreach ($pkg in $reqs) {
            Write-Host "Installing individual package: $pkg" -ForegroundColor Cyan
            $r2 = Run-ProcessWithTail $python "-m pip install --upgrade $pkg" "install $pkg"
            if ($r2 -ne 0) { Write-Warning "Package $pkg failed to install (exit $r2)" }
        }
    }
} else {
    Write-Host "DRY RUN: would run: $python -m pip install -r `"$reqFile`"" -ForegroundColor Yellow
}

# If GPU is present, try to install CUDA runtime wheels (best effort)
$hasGPU = $false
try { & nvidia-smi > $null; $hasGPU = $true } catch {}
if ($hasGPU) {
    Write-Host 'GPU detected — attempting GPU-related wheel installs (best-effort).' -ForegroundColor Cyan
    try {
        $gpuPkgs = @('nvidia-cuda-runtime-cu12','nvidia-cudnn-cu12','nvidia-cublas-cu12','nvidia-cufft-cu12','nvidia-cusparse-cu12','nvidia-cuda-nvrtc-cu12')
        foreach ($pkg in $gpuPkgs) {
            if (-not $isDry) {
                $rc2 = Run-ProcessWithTail $python "-m pip install --upgrade $pkg" "install $pkg"
                if ($rc2 -ne 0) { Write-Warning "install of $pkg returned $rc2" }
            } else {
                Write-Host "DRY RUN: would install GPU wheel: $pkg" -ForegroundColor Yellow
            }
        }
    } catch {
        Write-Warning 'One or more NVIDIA runtime wheels failed to install; continuing — ONNX Runtime may still fall back to CPU.'
    }
}

# Download model if missing
 $modelUrl = 'https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx'
 $modelPath = Join-Path $projDir 'depth_anything_v2_small_f32.onnx'
 if (-not (Test-Path $modelPath)) {
    if (-not $isDry) {
        Write-Host 'Downloading Depth Anything model (~200MB)...' -ForegroundColor Cyan
       try {
           # Try WebClient with progress events; if not supported, fallback to Invoke-WebRequest
           try {
               $wc = New-Object System.Net.WebClient
               $done = $false
               $wc.DownloadProgressChanged += { param($s,$e) Write-Host -NoNewline "`rDownloading model: $($e.ProgressPercentage)% ($([math]::Round($e.BytesReceived/1KB)) KB)" }
               $wc.DownloadFileCompleted += { $done = $true; Write-Host "`nDownload complete." }
               $uri = [System.Uri]::new($modelUrl)
               $wc.DownloadFileAsync($uri, $modelPath)
               while (-not $done) { Start-Sleep -Milliseconds 200 }
           } catch {
               Write-Host 'WebClient progress unavailable; using Invoke-WebRequest fallback...' -ForegroundColor Yellow
               Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath -UseBasicParsing
           }
       } catch {
           Write-Warning "Model download failed: $($_.Exception.Message)"
       }
     } else {
         Write-Host "DRY RUN: would download model from: $modelUrl to: $modelPath" -ForegroundColor Yellow
     }
 }

# Try to add local nvidia wheel bins (if the wheels installed into venv)
$sitePackages = Join-Path $venvPath 'Lib\site-packages'
$nvidiaRoot = Join-Path $sitePackages 'nvidia'
$bins = @('cublas\bin','cuda_runtime\bin','cuda_nvrtc\bin','cudnn\bin','cufft\bin','cusparse\bin')
foreach ($b in $bins) {
    $p = Join-Path $nvidiaRoot $b
    if (Test-Path $p) { $env:PATH = $p + ';' + $env:PATH; Write-Host "Added $p to PATH" }
}

# Validate critical imports inside the venv; attempt minimal fallbacks or report clear instructions
function Test-Import($module) {
    $cmd = "import $module; print('OK')"
    $out = & $python -c $cmd 2>&1
    return $LASTEXITCODE -eq 0
}

Write-Host 'Verifying installed Python packages...' -ForegroundColor Cyan
$missing = @()
$checks = @('cv2','onnxruntime','PIL','numpy','requests')
foreach ($m in $checks) {
    if (-not (Test-Import $m)) { $missing += $m; Write-Warning "Missing module: $m" }
}
if ($missing.Count -gt 0) {
    Write-Warning "Some modules are missing: $($missing -join ', ')"
    if ($missing -contains 'cv2') {
        Write-Host 'Attempting to install opencv-python-headless as fallback...' -ForegroundColor Cyan
        if (-not $isDry) { Run-ProcessWithTail $python "-m pip install --upgrade opencv-python-headless" 'install opencv-headless' }
    }
    if ($missing -contains 'onnxruntime') {
        Write-Host 'Attempting to install onnxruntime (CPU) as fallback...' -ForegroundColor Cyan
        if (-not $isDry) { Run-ProcessWithTail $python "-m pip install --upgrade onnxruntime" 'install onnxruntime' }
    }
    if ($missing -contains 'PIL') {
        Write-Host 'Pillow missing — attempting to install...' -ForegroundColor Cyan
        if (-not $isDry) { $r = Run-ProcessWithTail $python "-m pip install --upgrade pillow" 'install pillow'; if ($r -ne 0) { Write-Warning 'pillow install failed' } }
    }
    if ($missing -contains 'requests') {
        Write-Host 'requests missing — attempting to install...' -ForegroundColor Cyan
        if (-not $isDry) { $r = Run-ProcessWithTail $python "-m pip install --upgrade requests" 'install requests'; if ($r -ne 0) { Write-Warning 'requests install failed' } }
    }
    # Re-run verification for any remaining missing modules
    $stillMissing = @()
    foreach ($m in $checks) { if (-not (Test-Import $m)) { $stillMissing += $m } }
    if ($stillMissing.Count -gt 0) { Write-Warning "Still missing modules after attempts: $($stillMissing -join ', ')" }
}

# Final message and launch
Write-Host 'Launching Depth Studio (this will use the active venv)...' -ForegroundColor Green
& $python (Join-Path $projDir 'depth.py')
