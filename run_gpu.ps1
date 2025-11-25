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

Write-Host 'Upgrading pip/tools...' -ForegroundColor Cyan
# Helper: run a process and tail its output while showing a spinner
function Run-ProcessWithTail($exe, $argList, $title) {
    $log = [IO.Path]::GetTempFileName()
    Write-Host "`n$title -> $exe $argList" -ForegroundColor Cyan
    $exitCode = 0
    try {
        $startInfo = @{ FilePath = $exe; ArgumentList = $argList; RedirectStandardOutput = $log; RedirectStandardError = $log; NoNewWindow = $true }
        $proc = Start-Process @startInfo -PassThru

        $spinner = @('|','/','-','\')
        $idx = 0
        while (-not $proc.HasExited) {
            $ch = $spinner[$idx % $spinner.Length]
            Write-Host -NoNewline "`r[$ch] $title... "
            Start-Sleep -Milliseconds 300
            $idx++
        }
        Write-Host "`r[✓] $title completed." -ForegroundColor Green
        Get-Content $log -Tail 30 | ForEach-Object { Write-Host "  $_" }
        $exitCode = $proc.ExitCode
    } catch {
        Write-Warning "Start-Process with redirected output failed; falling back to synchronous execution. Error: $($_.Exception.Message)"
        # Fallback: run via cmd.exe and redirect output to the log file. This will run synchronously.
        $cmd = "`"$exe`" $argList > `"$log`" 2>&1"
        & cmd.exe /c $cmd
        $exitCode = $LASTEXITCODE
        Write-Host "[✓] $title completed (fallback)." -ForegroundColor Green
        Get-Content $log -Tail 30 | ForEach-Object { Write-Host "  $_" }
    } finally {
        Remove-Item $log -ErrorAction SilentlyContinue
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
    Run-ProcessWithTail $python "-m pip install -r `"$reqFile`"" 'pip install requirements'
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
                Run-ProcessWithTail $python "-m pip install --upgrade $pkg" "install $pkg"
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
            # Use WebClient to show progress
            $wc = New-Object System.Net.WebClient
            $done = $false
            $wc.DownloadProgressChanged += { param($s,$e) Write-Host -NoNewline "`rDownloading model: $($e.ProgressPercentage)% ($([math]::Round($e.BytesReceived/1KB)) KB)" }
            $wc.DownloadFileCompleted += { $done = $true; Write-Host "`nDownload complete." }
            $uri = [System.Uri]::new($modelUrl)
            $wc.DownloadFileAsync($uri, $modelPath)
            while (-not $done) { Start-Sleep -Milliseconds 200 }
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

# Final message and launch
Write-Host 'Launching Depth Studio (this will use the active venv)...' -ForegroundColor Green
& $python (Join-Path $projDir 'depth.py')
