# Serve a model with Atlas on AMD GPUs under Windows. Verified coherent on
# gfx1151 / Strix Halo with Qwen3.8-27B-NVFP4. Windows twin of serve-amd.sh.
#
#   .\serve-amd.ps1                              # Qwen3.8 27B + MTP on :8081 (default)
#   .\serve-amd.ps1 -Spec dflash2                # Qwen3.8 27B + DFlash2 on :8096
#   .\serve-amd.ps1 -Model qwen3.8-flash-next    # Flash-Next + MTP on :8095 (needs VGM 32 GB)
#   .\serve-amd.ps1 D:\models\Qwen3.8-27B-NVFP4  # a local weights snapshot
#   $env:ATLAS_PORT=9000; .\serve-amd.ps1
#
# (-Model qwen3.8-flash-next -Spec dflash2 is rejected: no DFlash2 drafter
# exists for Flash-Next.)
#
# Optional positional arg is the LOCAL weights directory (Windows resolves
# weights by path, not HF cache) -- equivalent to setting ATLAS_MODEL_DIR.
# Every env knob first_run.ps1 documents applies to the default (27B + MTP)
# path: ATLAS_MODEL_NAME, ATLAS_BIN (prebuilt spark.exe), ATLAS_PORT,
# ATLAS_BIND, ATLAS_MAX_SEQ_LEN, ATLAS_MAX_PREFILL_TOKENS, ATLAS_GPU_UTIL.
# On -Spec dflash2 and -Model qwen3.8-flash-next, ATLAS_PORT and ATLAS_BIND
# forward to the recipe's -Port/-BindHost; the Flash-Next recipe also reads
# its own env knobs (SEQ_LEN, PREFILL_TOKENS, SERIAL, DISABLE_THINKING,
# PREFIX_CACHE, SSM_SLOTS, ATLAS_UTIL).
#
#   .\serve-amd.ps1 -Thinking off    # --disable-thinking on any recipe
#   .\serve-amd.ps1 -Thinking on     # thinking = server default (per-request off still wins)
#
# -Thinking forwards to every recipe as env ATLAS_THINKING (issue #146).
#
# The serve recipe for the Qwen3.8 27B paths is the fingerprinted fp8d
# configuration behind the 2026-09-13 ST-995 record -- it lives in
# first_run.ps1's Phase-Serve, which the default (27B + MTP) mode forwards to
# (check + serve). -Spec dflash2 forwards to
# scripts\strix-windows\win_serve_dflash2_nvfp4.ps1 and
# -Model qwen3.8-flash-next to
# scripts\strix-windows\win_serve_flashnext_nvfp4.ps1. The GPU preflight
# hard-fails on non-gfx1151 before any kernel launches. On the 27B paths the
# server owns this console until Ctrl-C; the smoke probe runs detached and
# writes first_run_smoke.log beside the exe. -Model qwen3.8-flash-next
# instead starts spark with Start-Process and returns once the server is
# up -- stop it with `Get-Process spark | Stop-Process`.
param(
    # Optional positional arg: a LOCAL weights directory (Windows resolves
    # weights by path, not HF cache) -- equivalent to setting ATLAS_MODEL_DIR.
    [Parameter(Position = 0)]
    [string]$ModelDir,
    # Which model to serve: qwen3.8 (the dense 27B) or qwen3.8-flash-next.
    [ValidateSet('qwen3.8', 'qwen3.8-flash-next')]
    [string]$Model = 'qwen3.8',
    # Speculation mode: mtp (default) or dflash2 (Qwen3.8 27B only).
    [ValidateSet('mtp', 'dflash2')]
    [string]$Spec = 'mtp',
    # Thinking switch: 'off' is the engine kill switch (--disable-thinking,
    # outranks every request); 'on' makes thinking the server default
    # (--default-chat-template-kwargs, per-request off still wins); 'default'
    # keeps each recipe's shipped behavior. DISABLE_THINKING=1 stays an alias
    # for 'off'.
    [ValidateSet('default', 'off', 'on')]
    [string]$Thinking = 'default',
    # Forwarded to first_run.ps1: skip the post-startup completion probe.
    [switch]$NoSmokeTest
)
$ErrorActionPreference = 'Stop'
# DISABLE_THINKING=1 aliases -Thinking off when -Thinking wasn't given.
if (-not $PSBoundParameters.ContainsKey('Thinking') -and $env:DISABLE_THINKING -eq '1') { $Thinking = 'off' }
$env:ATLAS_THINKING = $Thinking
# serve-amd.sh parity: DFLASH=1 selects the drafter when -Spec wasn't given.
if ($env:DFLASH -eq '1' -and -not $PSBoundParameters.ContainsKey('Spec')) { $Spec = 'dflash2' }
if ($Model -eq 'qwen3.8-flash-next' -and $Spec -eq 'dflash2') {
    throw 'no DFlash2 drafter exists for Qwen3.8-Flash-Next: use -Spec mtp'
}
if ($Model -eq 'qwen3.8-flash-next') {
    if (-not $ModelDir) { $ModelDir = "$env:USERPROFILE\models\nvidia-Qwen3.8-Flash-Next-NVFP4" }
    if (-not (Test-Path $ModelDir -PathType Container)) {
        throw "no weights at $ModelDir. Fetch with: hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --local-dir `"$ModelDir`""
    }
    Remove-Item Env:SERIAL -ErrorAction SilentlyContinue
    $serve = Join-Path $PSScriptRoot 'scripts\strix-windows\win_serve_flashnext_nvfp4.ps1'
    $recipeArgs = @{ ModelDir = $ModelDir }
    if ($env:ATLAS_PORT) { $recipeArgs.Port = $env:ATLAS_PORT }
    if ($env:ATLAS_BIND) { $recipeArgs.BindHost = $env:ATLAS_BIND }
    & $serve @recipeArgs
    if (-not $?) { exit 1 }
} elseif ($Spec -eq 'dflash2') {
    if (-not $ModelDir) { $ModelDir = "$env:USERPROFILE\models\nvidia-Qwen3.8-27B-NVFP4" }
    $DraftDir = if ($env:DRAFT_MODEL) { $env:DRAFT_MODEL } else { "$env:USERPROFILE\models\dflash2" }
    if (-not (Test-Path $ModelDir -PathType Container)) {
        throw "no weights at $ModelDir. Fetch with: hf download nvidia/Qwen3.8-27B-NVFP4 --local-dir `"$ModelDir`""
    }
    if (-not (Test-Path $DraftDir -PathType Container)) {
        throw "no weights at $DraftDir. Fetch with: hf download incoai/Qwen3.8-27B-DFlash2 --local-dir `"$DraftDir`""
    }
    $serve = Join-Path $PSScriptRoot 'scripts\strix-windows\win_serve_dflash2_nvfp4.ps1'
    $recipeArgs = @{ ModelDir = $ModelDir; DraftDir = $DraftDir }
    if ($env:ATLAS_PORT) { $recipeArgs.Port = $env:ATLAS_PORT }
    if ($env:ATLAS_BIND) { $recipeArgs.BindHost = $env:ATLAS_BIND }
    & $serve @recipeArgs
    if (-not $?) { exit 1 }
} else {
    if ($ModelDir) {
        if (-not (Test-Path $ModelDir -PathType Container)) {
            throw "weights dir not found: $ModelDir"
        }
        $env:ATLAS_MODEL_DIR = (Resolve-Path $ModelDir).Path
    }
    & (Join-Path $PSScriptRoot 'scripts\strix-windows\first_run.ps1') -Phase serve -NoSmokeTest:$NoSmokeTest
    if (-not $?) { exit 1 }
}
