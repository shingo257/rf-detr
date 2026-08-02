# Run RF-DETR video demo via package entry point.

# From cmd.exe: scripts\run_demo_video.cmd

# From PowerShell: .\scripts\run_demo_video.ps1

param(

    [string]$Source = "",

    [string]$Output = "",

    [ValidateSet("detect", "keypoint")]

    [string]$Task = "detect",

    [ValidateSet("nano", "small", "medium", "large")]

    [string]$Model = "nano",

    [double]$Threshold = 0.5,

    [switch]$PersonOnly,

    [int]$FrameStride = 2,

    [int]$MaxFrames = 0,

    [ValidateSet("DEBUG", "INFO", "WARNING", "ERROR")]

    [string]$LogLevel = "INFO"

)



$ErrorActionPreference = "Stop"



Write-Host "[rf-detr] run_demo_video.ps1 starting..." -ForegroundColor Cyan



$scriptsDir = $PSScriptRoot

$repoRoot = Split-Path -Parent $scriptsDir

$confidentialDefault = Join-Path $repoRoot "confidential\media\input\mn1-2.mov"



if (-not (Test-Path -LiteralPath $confidentialDefault) -and [string]::IsNullOrWhiteSpace($Source)) {

    Write-Host "[WARN] Default confidential input not found; rfdetr-demo will fall back to other defaults." -ForegroundColor Yellow

}



if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {

    Write-Host "[ERROR] uv not found on PATH." -ForegroundColor Red

    Write-Host "From rf-detr root run:"

    Write-Host '  uv sync --extra demo'

    exit 1

}



$outputDir = Join-Path $repoRoot "artifacts\demo"

if (-not (Test-Path $outputDir)) {

    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

}



$demoArgs = @(

    "rfdetr-demo",

    "--task", $Task,

    "--model", $Model,

    "--threshold", $Threshold,

    "--frame-stride", $FrameStride,

    "--log-level", $LogLevel

)



if (-not [string]::IsNullOrWhiteSpace($Source)) {

    $demoArgs += @("--source", $Source)

}

if (-not [string]::IsNullOrWhiteSpace($Output)) {

    $demoArgs += @("--output", $Output)

}

if ($PersonOnly) {

    $demoArgs += @("--person-only")

}

if ($MaxFrames -gt 0) {

    $demoArgs += @("--max-frames", $MaxFrames)

}



foreach ($arg in $args) {

    if ($demoArgs -notcontains $arg) {

        $demoArgs += $arg

    }

}



$env:UV_TORCH_BACKEND = "cpu"



Push-Location $repoRoot

try {

    Write-Host "[rf-detr] Running video demo..."

    Write-Host "  uv run $($demoArgs -join ' ')"

    & uv run @demoArgs

    $exitCode = $LASTEXITCODE

} finally {

    Pop-Location

}



if ($exitCode -ne 0) {

    Write-Host "[ERROR] Demo failed (exit $exitCode)." -ForegroundColor Red

} else {

    if (-not [string]::IsNullOrWhiteSpace($Output)) {

        Write-Host "[OK] Demo finished. Output: $Output" -ForegroundColor Green

    } else {

        Write-Host "[OK] Demo finished. See artifacts\demo\" -ForegroundColor Green

    }

}

exit $exitCode

