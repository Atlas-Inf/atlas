# Build Atlas for AMD GPUs on Windows. Verified on gfx1151 / Strix Halo,
# Windows 11, TheRock ROCm 10.0.0. See docs/porting/STRIX_WINDOWS_HIP.md.
#
# Windows twin of build-amd.sh: same env knobs (ATLAS_TARGET_HW stays
# strix-hip; ATLAS_TARGET_MODEL / ATLAS_TARGET_QUANT pass through), same
# cargo invocation, same shims — but compiled by MSVC against the HIP shim
# DLLs that get staged beside spark.exe.
#
#   powershell -ExecutionPolicy Bypass -File .\build-amd.ps1
#
# Needs: MSVC (vcvars64), the ROCm SDK at $env:HIP_PATH, and cargo — the
# underlying script checks all three and the GPU before compiling. Must run
# from PowerShell, NOT Git Bash: under bash, Git's /usr/bin precedes MSVC on
# PATH and rustc invokes the coreutils link.exe instead of the MSVC linker.
#
# Using the prebuilt zip instead? There is nothing to build:
#   $env:ATLAS_BIN = "C:\path\to\unzipped\spark.exe"   # then .\serve-amd.ps1
param(
    # Forwarded to first_run.ps1; a no-op for the build phase but accepted so
    # the wrappers take the same flags.
    [switch]$NoSmokeTest
)
$ErrorActionPreference = 'Stop'
if ($env:ATLAS_BIN) {
    Write-Host "ATLAS_BIN is set ($env:ATLAS_BIN) -- prebuilt binary, nothing to build. Run .\serve-amd.ps1"
    exit 0
}
& (Join-Path $PSScriptRoot 'scripts\strix-windows\first_run.ps1') -Phase build -NoSmokeTest:$NoSmokeTest
if (-not $?) { exit 1 }
