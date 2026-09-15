# Serve a model with Atlas on AMD GPUs under Windows. Verified coherent on
# gfx1151 / Strix Halo with Qwen3.8-27B-NVFP4. Windows twin of serve-amd.sh.
#
#   .\serve-amd.ps1                          # unsloth/Qwen3.8-27B-NVFP4 (default)
#   .\serve-amd.ps1 D:\models\Qwen3.8-27B-NVFP4   # a local weights snapshot
#   $env:ATLAS_PORT=9000; .\serve-amd.ps1
#
# Optional positional arg is the LOCAL weights directory (Windows resolves
# weights by path, not HF cache) — equivalent to setting ATLAS_MODEL_DIR.
# Every env knob first_run.ps1 documents applies here too: ATLAS_MODEL_NAME,
# ATLAS_BIN (prebuilt spark.exe), ATLAS_PORT, ATLAS_BIND, ATLAS_MAX_SEQ_LEN,
# ATLAS_MAX_PREFILL_TOKENS, ATLAS_GPU_UTIL.
#
# The serve recipe is the fingerprinted fp8d configuration behind the
# 2026-09-13 ST-995 record — it lives in first_run.ps1's Phase-Serve, which
# this forwards to (check + serve). The GPU preflight hard-fails on
# non-gfx1151 before any kernel launches. The server owns this console until
# Ctrl-C; the smoke probe runs detached and writes first_run_smoke.log
# beside the exe.
$ErrorActionPreference = 'Stop'
if ($args.Count -gt 0 -and (Test-Path $args[0] -PathType Container)) {
    $env:ATLAS_MODEL_DIR = (Resolve-Path $args[0]).Path
    $args = if ($args.Count -gt 1) { $args[1..($args.Count - 1)] } else { @() }
}
& (Join-Path $PSScriptRoot 'scripts\strix-windows\first_run.ps1') -Phase serve @args
if (-not $?) { exit 1 }
